include("engines/init.jl")
include("engines/data_processing.jl")
include("engines/deep_learning.jl")
include("engines/cross_validation.jl")
include("engines/model_evaluation.jl")

# prediction scores (Accuracy %) for targets in Leucegene 
data_df = CSV.read("figures/LGN_predictions_logistic_regression_sparse_cg.csv", DataFrame)
fig = Figure(size = (512, 512));
ax = Axis(fig[1,1], 
    title = "Prediction accuracy of clinical features in Leucegene data (n=300)\nwith Logistic Regression (python scikit-klearn)",
    yticks = (collect(1:20) / 20 * 100, string.(collect(1:20) / 20 * 100)), 
    xticks = (collect(1:size(data_df)[2]), names(data_df)),
    xticklabelrotation = 0.5,  
    xlabel = "Clinical Feature Prediction Target",
    ylabel = "Accuracy (%)"
);
for (ind, target) in enumerate(names(data_df))
    boxplot!(ax, ones(size(data_df)[1]) * ind, data_df[:,target] *100)
    text!(ax, ind - 0.4, median(data_df[:,target] ) * 100, text = string(round(median(data_df[:,target] ) * 100, digits = 2) ))
end
CairoMakie.save("figures/LGN_predictions_logistic_regression_sparse_cg.pdf",fig)
CairoMakie.save("figures/LGN_predictions_logistic_regression_sparse_cg.png",fig)
CairoMakie.save("figures/LGN_predictions_logistic_regression_sparse_cg.svg",fig)

lgn_CF = CSV.read("Data/LEUCEGENE/lgn_pronostic_CF", DataFrame)
tcga_datasets_list = ["Data/TCGA_datasets/$(x)" for x in readdir("Data/TCGA_OV_BRCA_LGG/") ]
TCGA_datasets = load_tcga_datasets(tcga_datasets_list);
BRCA_data = TCGA_datasets["BRCA"]
LGG_data = TCGA_datasets["LGG"]
OV_data = TCGA_datasets["OV"]

LGNAML_data = Dict("name"=>"LgnAML","dataset" => MLSurvDataset("Data/LEUCEGENE/LGN_AML_tpm_n300_btypes_labels_surv.h5")) 
keep_tcga_cds = [occursin("protein_coding", bt) for bt in BRCA_data["dataset"].biotypes]
keep_lgnaml_common = [gene in BRCA_data["dataset"].genes[keep_tcga_cds] for gene in LGNAML_data["dataset"].genes];
LGNAML_data["CDS"] = keep_lgnaml_common

train_x = Matrix(LGNAML_data["dataset"].data[:,keep_lgnaml_common]')
P = fit_pca(train_x, 300);
train_x_pca = Matrix(transform_pca(train_x, P)');
    
lgn_tsne = tsne(train_x_pca,2, 50, 3000, 30; verbose = true, progress = true)
TSNE_df = DataFrame(:tsne1=>lgn_tsne[:,1], :tsne2=>lgn_tsne[:,2], :label => lgn_CF[:,"Tissue"]);
TSNE_df
p = AlgebraOfGraphics.data(TSNE_df) * mapping(:tsne1,:tsne2, color = :label)
fig = draw(p)
CairoMakie.save("TSNE_LGN_Tissue.svg", fig)
hlsize = 5
nepochs= 5000
lr = 1e-5 
wd = 1e-2  
modeltype = "dnn"
nfolds=5

# targets = Array{Float32}(lgn_CF[:,"Tissue"] .== "Blood")
targets = Array{Int}(lgn_CF[:,"NPM1 mutation"])

folds = split_train_test(LGNAML_data["dataset"].data[:,keep_lgnaml_common], targets;nfolds = nfolds)
print_step = 100
# for fold in folds do train
for (foldn, fold) in enumerate(folds)
    train_x = gpu(Matrix(fold["train_x"]'))
    test_x = gpu(Matrix(fold["test_x"]'))
    
    train_y = gpu(Matrix(fold["train_y"])')
    test_y = gpu(fold["test_y"])
    
    # P = fit_pca(cpu(train_x), 240);
    # train_x = gpu(Matrix(transform_pca(cpu(train_x), P)'))
    # test_x = gpu(Matrix(transform_pca(cpu(test_x), P)))
        
    ## init model and opt
    OPT = Flux.ADAM(lr) 
    # MODEL = gpu(Chain(Dense(size(train_x)[1],hlsize, leakyrelu), 
    # Dense(hlsize, hlsize, leakyrelu), 
    # Dense(hlsize, 1, sigmoid)))
    MODEL = gpu(Chain(Dense(size(train_x)[1], 1, sigmoid)))
    # train loop 
    for i in 1:nepochs
        ps = Flux.params(MODEL)
        gs = gradient(ps) do 
            Flux.binarycrossentropy(MODEL(train_x), train_y) + l2_penalty(MODEL) * wd         
        end 
        lossval = Flux.binarycrossentropy(MODEL(train_x), train_y) + l2_penalty(MODEL) * wd 
        tr_OUTS = MODEL(train_x)
        tst_OUTS = MODEL(test_x)
        train_acc = mean(Array{Int}(cpu(tr_OUTS) .> 0.5) .== cpu(train_y))
        tst_acc = mean(Array{Int}(cpu(tst_OUTS) .> 0.5) .== cpu(test_y))
        if i % print_step ==  0 || i == 1
            println("$foldn $i TRAIN: $lossval $(round(train_acc *100,digits=3))% TEST: $(round(tst_acc *100,digits = 3))% ")
        end 
        Flux.update!(OPT, ps, gs)
    end
end 
c_indices_tst = dump_results!(DS, LOSSES_BY_FOLD, OUTS_TST, Y_T_TST, Y_E_TST, train_cinds, test_cinds)
