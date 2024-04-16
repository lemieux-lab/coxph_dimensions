
include("engines/init.jl")
include("engines/data_processing.jl")
include("engines/deep_learning.jl")
include("engines/cross_validation.jl")
include("engines/model_evaluation.jl")
include("engines/figures.jl")
PARAMS = gather_params("RES_FIG23/")
PARAMS[:,"TYPE"] = replace.(["$x-$y" for (x,y) in  zip(PARAMS[:, "dim_redux_type"], PARAMS[:, "insize"])], "RDM"=>"CDS")
names(PARAMS)
PARAMS[PARAMS[:,"dataset"] .== "LgnAML",["TYPE"]]
PARAMS = PARAMS[PARAMS.TYPE .!= "CLINF-8",:]
to_csv = PARAMS[:,["dataset", "dim_redux_type", "model_type", "cph_test_c_ind"]]
CSV.write("python_scripts/table1.csv", to_csv)
PARAMS[:,"II"] .= 1
medians = combine(groupby(PARAMS[:,["II", "dataset", "TYPE", "model_type", "cph_test_c_ind"]], ["dataset", "TYPE", "model_type"]), "cph_test_c_ind"=>mean, "II"=>sum) 
sort(medians, ["dataset", "TYPE", "model_type"])
    
fig = make_boxplots(PARAMS[PARAMS.TYPE .!= "CLINF-8", :], test_metric = "cph_tst_c_ind_med")
CairoMakie.save("figures/figure2_lgnaml_brca_coxridge_cphdnn_rdm_pca_clinf_sign.svg",fig)
CairoMakie.save("figures/figure2_lgnaml_brca_coxridge_cphdnn_rdm_pca_clinf_sign.png",fig)
CairoMakie.save("figures/PDF/figure2_lgnaml_brca_coxridge_cphdnn_rdm_pca_clinf_sign.pdf",fig)

LC = gather_learning_curves(basedir="RES_FIG23/", skip_steps=100)

LOSSES_DF = innerjoin(LC, sort(PARAMS[PARAMS.TYPE .!= "CLINF-8",:], ["TYPE"]), on = "modelid");
DATA_df = sort(LOSSES_DF[LOSSES_DF[:,"dataset"] .== "BRCA",:], ["TYPE"])
fig = Figure(size = (512,512));
ax = Axis(fig[1,1], xlabel = "steps", ylabel = "Loss value", )
DRD_df = DATA_df[DATA_df[:,"model_type"] .== "cphdnn",:]
        
for modelid in unique(DRD_df[:, "modelid"])
    MOD_df = DRD_df[DRD_df[:,"modelid"] .== modelid, :]
    for foldn in 1:5
        FOLD_data = sort(MOD_df[MOD_df[:,"foldns"] .== foldn,:], "steps")
        lines!(ax, FOLD_data[FOLD_data[:,"tst_train"] .== "train","steps"], FOLD_data[FOLD_data[:,"tst_train"] .== "train","loss_vals"], color = "blue") 
        lines!(ax, FOLD_data[FOLD_data[:,"tst_train"] .== "test","steps"], FOLD_data[FOLD_data[:,"tst_train"] .== "test","loss_vals"], color = "orange")
    end    
end
fig

fig = Figure(size = (800,800));
for (row_id, dataset) in enumerate(["LgnAML", "BRCA"])
    DATA_df = sort(LOSSES_DF[LOSSES_DF[:,"dataset"] .== dataset,:], ["TYPE"])
    for (col_id, TYPE) in enumerate(unique(DATA_df.TYPE))
        DRD_df = DATA_df[DATA_df[:,"TYPE"] .== TYPE,:]
        DRD_df = DRD_df[DRD_df[:,"model_type"] .== "cphdnn",:]
        ax = Axis(fig[row_id,col_id]; 
            xlabel = "steps", ylabel = "Loss value", 
            title = "$dataset - $TYPE");
        println("processing $dataset - $TYPE ...")
        for modelid in unique(DRD_df[:, "modelid"])
            MOD_df = DRD_df[DRD_df[:,"modelid"] .== modelid, :]
            for foldn in 1:5
                FOLD_data = sort(MOD_df[MOD_df[:,"foldns"] .== foldn,:], "steps")
                lines!(ax, FOLD_data[FOLD_data[:,"tst_train"] .== "train","steps"], log10.(FOLD_data[FOLD_data[:,"tst_train"] .== "train","loss_vals"]), color = "blue") 
                lines!(ax, FOLD_data[FOLD_data[:,"tst_train"] .== "test","steps"], log10.(FOLD_data[FOLD_data[:,"tst_train"] .== "test","loss_vals"]), color = "orange")
            end    
        end 
    end 
end 
fig
CairoMakie.save("figures/PDF/ismb_two_pager_1c.pdf", fig)
CairoMakie.save("figures/PDF/ismb_two_pager_1c.svg", fig)
CairoMakie.save("figures/ismb_two_pager_1c.png", fig)

#axislegend(ax1)
# CairoMakie.save("figures/figure3_cphdnn_lgnaml_brca_rdm_pca_clinf_sign_overfit.svg",fig)
# CairoMakie.save("figures/figure3_cphdnn_lgnaml_brca_rdm_pca_clinf_sign_overfit.png",fig)
# CairoMakie.save("figures/PDF/figure3_cphdnn_lgnaml_brca_rdm_pca_clinf_sign_overfit.pdf",fig)

### ISMB two pager fig 1d 

DATA_df = sort(PARAMS[PARAMS.TYPE .!= "CLINF-8",:], ["TYPE"])
DATA_df[:,"TYPE2"] = ["$x-$y-$z" for (x,y, z) in  zip(DATA_df[:, "dataset"], DATA_df[:, "model_type"], DATA_df[:, "TYPE"])]
DATA_df = sort(DATA_df, ["TYPE2"])
DATA_df = innerjoin(DATA_df, DataFrame("TYPE2"=>unique(DATA_df[:,"TYPE2"]), "ID" => collect(1:size(unique(DATA_df[:,"TYPE2"]))[1])), on =:TYPE2)

fig = Figure(size = (1200,500));
ax = Axis(fig[1,1], 
    ylabel = "Concordance",
    xticks = (collect(1:maximum(DATA_df.ID)),string.(unique(DATA_df.TYPE2))),
    xticklabelrotation = 0.5, 
    limits = (0, maximum(DATA_df.ID) + 0.5, nothing, nothing),
    title = "Cross-validation metrics with c-index scores aggregation + bootstrapping vs average. ")
boxplot!(ax, DATA_df.ID .- 0.2, DATA_df.cph_tst_c_ind_med, width = 0.5,  color = (:blue, 0.5), label = "aggregated scores + bootstrapping")
scatter!(ax, DATA_df.ID .- 0.2, DATA_df.cph_tst_c_ind_med, color = :black)
boxplot!(ax, DATA_df.ID .+ 0.2, DATA_df.cph_test_c_ind, width = 0.5,  color = (:orange, 0.5), label = "average scores")
scatter!(ax, DATA_df.ID .+ 0.2, DATA_df.cph_test_c_ind, color = :black)
axislegend(ax, position = :rb)
fig
CairoMakie.save("figures/PDF/ismb_two_pager_1d.pdf", fig)
CairoMakie.save("figures/PDF/ismb_two_pager_1d.svg", fig)
CairoMakie.save("figures/ismb_two_pager_1d.png", fig)

fig = Figure(size = (1600,400));
for (row_id, dataset) in enumerate(["LgnAML", "BRCA"])
    DATA_df = sort(TRUNC_DF[TRUNC_DF[:,"dataset"] .== dataset,:], ["TYPE"])
    for (col_id, TYPE) in enumerate(unique(DATA_df.TYPE))
        DRD_df = DATA_df[DATA_df[:,"TYPE"] .== TYPE,:]
        DRD_df = DRD_df[DRD_df[:,"model_type"] .== "coxridge",:]
        ax = Axis(fig[row_id,col_id]; 
            xlabel = "steps", ylabel = "Loss value", 
            title = "$dataset - $TYPE");
        println("processing $dataset - $TYPE ...")
        for modelid in unique(DRD_df[:, "modelid"])
            MOD_df = DRD_df[DRD_df[:,"modelid"] .== modelid, :]
            for foldn in 1:5
                FOLD_data = sort(MOD_df[MOD_df[:,"foldns"] .== foldn,:], "steps")
                lines!(ax, FOLD_data[FOLD_data[:,"tst_train"] .== "train","steps"], FOLD_data[FOLD_data[:,"tst_train"] .== "train","loss_vals"], color = "blue") 
                lines!(ax, FOLD_data[FOLD_data[:,"tst_train"] .== "test","steps"], FOLD_data[FOLD_data[:,"tst_train"] .== "test","loss_vals"], color = "orange")
            end    
        end 
    end 
end 
fig
#axislegend(ax1)
CairoMakie.save("figures/figure3_coxridge_lgnaml_brca_rdm_pca_clinf_sign_overfit.svg",fig)
CairoMakie.save("figures/figure3_coxridge_lgnaml_brca_rdm_pca_clinf_sign_overfit.png",fig)
CairoMakie.save("figures/PDF/figure3_coxridge_lgnaml_brca_rdm_pca_clinf_sign_overfit.pdf",fig)

fig