
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

make_boxplots(PARAMS, test_metric = "cph_tst_c_ind_med")

LC = gather_learning_curves("RES_FIG23/")
names(LC)
#LC[:,"TYPE"] = ["$x-$y" for (x,y) in  zip(LC[:, "dim_redux_type"], LC[:, "insize"])]

LOSSES_DF = innerjoin(LC, sort(PARAMS, ["TYPE"]), on = "modelid");
TRUNC_DF = LOSSES_DF[(LOSSES_DF.steps .% 100 .== 0) .| (LOSSES_DF.steps .== 1),:]
fig = Figure(size = (1600,400));
for (row_id, dataset) in enumerate(["LgnAML", "BRCA"])
    DATA_df = sort(TRUNC_DF[TRUNC_DF[:,"dataset"] .== dataset,:], ["TYPE"])
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
                lines!(ax, FOLD_data[FOLD_data[:,"tst_train"] .== "train","steps"], FOLD_data[FOLD_data[:,"tst_train"] .== "train","loss_vals"], color = "blue") 
                lines!(ax, FOLD_data[FOLD_data[:,"tst_train"] .== "test","steps"], FOLD_data[FOLD_data[:,"tst_train"] .== "test","loss_vals"], color = "orange")
            end    
        end 
    end 
end 
fig
#axislegend(ax1)
CairoMakie.save("figures/figure3_cphdnn_lgnaml_brca_rdm_pca_clinf_sign_overfit.svg",fig)
CairoMakie.save("figures/figure3_cphdnn_lgnaml_brca_rdm_pca_clinf_sign_overfit.png",fig)
CairoMakie.save("figures/figure3_cphdnn_lgnaml_brca_rdm_pca_clinf_sign_overfit.pdf",fig)

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
CairoMakie.save("figures/figure3_coxridge_lgnaml_brca_rdm_pca_clinf_sign_overfit.pdf",fig)

fig