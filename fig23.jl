
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
PARAMS[:,"II"] .= 1
medians = combine(groupby(PARAMS[:,["II", "dataset", "TYPE", "model_type", "cph_tst_c_ind_med"]], ["dataset", "TYPE", "model_type"]), "cph_tst_c_ind_med"=>mean, "II"=>sum) 
sort(medians, ["dataset", "TYPE"])
    
fig = make_boxplots(PARAMS, test_metric = "cph_tst_c_ind_med")
CairoMakie.save("figures/figure2_lgnaml_brca_coxridge_cphdnn_rdm_pca_clinf_sign.svg",fig)
CairoMakie.save("figures/figure2_lgnaml_brca_coxridge_cphdnn_rdm_pca_clinf_sign.png",fig)
CairoMakie.save("figures/PDF/figure2_lgnaml_brca_coxridge_cphdnn_rdm_pca_clinf_sign.pdf",fig)

LC = gather_learning_curves(basedir="RES_FIG23/", skip_steps=100)

LOSSES_DF = innerjoin(LC, sort(PARAMS, ["TYPE"]), on = "modelid");
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

fig = Figure(size = (1600,400));
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
CairoMakie.save("figures/PDF/figure3_cphdnn_lgnaml_brca_rdm_pca_clinf_sign_overfit.pdf",fig)

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