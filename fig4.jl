include("engines/init.jl")
include("engines/data_processing.jl")
include("engines/deep_learning.jl")
include("engines/cross_validation.jl")
include("engines/model_evaluation.jl")
include("engines/figures.jl")

### LGN PCA => LSC17
## run PCA all

tcga_datasets_list = ["Data/TCGA_datasets/$(x)" for x in readdir("Data/TCGA_OV_BRCA_LGG/") ]
TCGA_datasets = load_tcga_datasets(tcga_datasets_list);
BRCA_data = TCGA_datasets["BRCA"]
LGG_data = TCGA_datasets["LGG"]
OV_data = TCGA_datasets["OV"]
TCGALAML_data = TCGA_datasets["LAML"]
LGNAML_data = Dict("name"=>"LgnAML","dataset" => MLSurvDataset("Data/LEUCEGENE/LGN_AML_tpm_n300_btypes_labels_surv.h5")) 

# non dupliques
duplicates_lgnaml = [sum(LGNAML_data["dataset"].genes .== gene) for gene in LGNAML_data["dataset"].genes] .!= 1
duplicates_tcgalaml = [sum(TCGALAML_data["dataset"].genes .== gene) for gene in TCGALAML_data["dataset"].genes] .!= 1
# coding 
keep_tcga_cds = [occursin("protein_coding", bt) for bt in TCGALAML_data["dataset"].biotypes]
# common 
TCGALAML_gene_df = DataFrame("gene"=> TCGALAML_data["dataset"].genes, "biotype"=>  TCGALAML_data["dataset"].biotypes)
LGNAML_gene_df = DataFrame("gene"=>LGNAML_data["dataset"].genes)
intersect_tcgalaml_lgnaml = innerjoin(LGNAML_gene_df[duplicates_lgnaml .== 0,:], TCGALAML_gene_df[(duplicates_tcgalaml .== 0) .&& (keep_tcga_cds .== 1) ,:], on = :gene)
# keep_tcgalaml = [(gene in TCGA_LGN_gene_intersect) && ~(gene in TCGALAML_data["dataset"].genes[duplicates_tcgalaml]) for gene in TCGALAML_data["dataset"].genes]
keep_lgnaml = [gene in intersect_tcgalaml_lgnaml.gene  for gene in LGNAML_data["dataset"].genes]
keep_tcgalaml = [gene in intersect_tcgalaml_lgnaml.gene  for gene in TCGALAML_data["dataset"].genes]
### LSC17 SIGNATURE
LSC17 = CSV.read("Data/SIGNATURES/LSC17.csv", DataFrame);
# LGNAML_data["CDS"] =  [gene in LSC17[:,"alt_name"] for gene in LGNAML_data["dataset"].genes];

lsc17_in_lgnaml = [gene in LSC17[:, "alt_name"] for gene in LGNAML_data["dataset"].genes[keep_lgnaml]]
train_x = LGNAML_data["dataset"].data[:,keep_lgnaml]
yticks = [gene for gene in LGNAML_data["dataset"].genes[keep_lgnaml] if gene in LSC17[:, "alt_name"] ]
P = fit_pca(train_x', size(train_x)[1])
fig = Figure(size = (1024,512));
ax = Axis(fig[1:3,1], 
    title = "LSC17 gene signature weight contributions (in absolute value) to PCA principal components on Leucegene data (n=300)", 
    ylabel = "LSC17 gene name", 
    yticks = (collect(1:size(yticks)[1]), yticks))
# scale = ReversibleScale(x -> asinh(x / 2) / log(10), x -> 2sinh(log(10) * x))
hm = heatmap!(ax, abs.(P[1:300, lsc17_in_lgnaml]), colormap = :Blues)
Colorbar(fig[1:3,2], hm)
ax2 = Axis(fig[4,1],
limits= (1,300,nothing,nothing), ylabel = "Sum of weights (absolute)", xlabel="Principal component")
lines!(ax2, collect(1:300), vec(sum(abs.(P[1:300, lsc17_in_lgnaml]), dims = 2)) )
fig
CairoMakie.save("figures/figure4_lsc17_PCA_contributions.pdf", fig)
### LGN PCA => CLINF 
## X x PC1 => CLINF
X_tr_pca = transform_pca(train_x', P)
lgn_CF = CSV.read("Data/LEUCEGENE/lgn_pronostic_CF", DataFrame)
CF_bin, lnames = numerise_labels(lgn_CF, ["Sex","Cytogenetic risk", "NPM1 mutation", "IDH1-R132 mutation", "FLT3-ITD mutation", ])
lnames
fig = Figure(size = (800,1000));
ax3 = Axis(fig[1,1], xlabel = "Clinical feature", 
    title = "Correlation bewtween PCA first 30 components \nand clinical factors in Leucegene (n=300)",
    xticks = (collect(1:size(lnames)[1]) .- 0.5, lnames),
    xticksmirrored = true,
    xaxisposition = :top,
    yreversed = true,  
    xticklabelrotation = 0.5, 
    # aspect = DataAspect(),
    yticks = (collect(1:30), string.(collect(1:30))),
    ylabel = "Principal component")
MM = [[my_cor(X_tr_pca[j,:], CF_bin[:,i]) for i in 1:size(lnames)[1]] for j in 1:30]
hm2 = heatmap!(ax3, abs.(hcat(MM...) .^ 2), colormap = :Blues)
Colorbar(fig[1,2], hm2, label = "R2 Correlation Coefficient")
fig
CairoMakie.save("figures/figure4_PCA_to_clinical_features_correlations.pdf", fig)
