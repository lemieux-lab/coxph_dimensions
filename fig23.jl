
include("engines/init.jl")
include("engines/data_processing.jl")
include("engines/deep_learning.jl")
include("engines/cross_validation.jl")
include("engines/model_evaluation.jl")
PARAMS = gather_params("RES_FIG23/")
PARAMS[:,"TYPE"] = ["$x-$y" for (x,y) in  zip(PARAMS[:, "dim_redux_type"], PARAMS[:, "insize"])]
PARAMS[PARAMS[:,"dataset"] .== "LgnAML",["TYPE"]]
function make_boxplots(PARAMS)
    fig = Figure(size=(600,512));
    offshift = 0.05
    up_y = 0.80
    DATA_df = PARAMS[PARAMS[:,"dataset"] .== "LgnAML",:]
    DATA_df = innerjoin(DATA_df,DataFrame("TYPE"=>unique(DATA_df[:,"TYPE"]), "ID" => collect(1:size(unique(DATA_df[:,"TYPE"]))[1])), on =:TYPE)
    #dtype_insize = combine(groupby(DATA_df, ["TYPE"]), :insize=>maximum)
    #ticks = ["$dtype\n($ins_max)" for (dtype,ins_max) in zip(dtype_insize[:,1], dtype_insize[:,2])]
    ax1 = Axis(fig[1,1];
                title = "Leucegene",
                limits = (0.5,size(unique(DATA_df[:,"TYPE"]))[1] + 0.5, nothing, up_y),
                xlabel = "Dimensionality reduction",
                ylabel = "Concordance index",
                xticks =  (collect(1:size(unique(DATA_df[:,"TYPE"]))[1]), replace.(unique(DATA_df[:,"TYPE"]), "-"=>"\n")))
    lines!(ax1,[0,size(unique(DATA_df[:,"TYPE"]))[1]],[0.5,0.5], linestyle = :dot)
    boxplot!(ax1, DATA_df.ID[DATA_df[:,"model_type"].=="cphdnn"] .- 0.2, width = 0.5,  DATA_df[DATA_df[:,"model_type"].=="cphdnn", "cph_test_c_ind"], color = "blue", label = "CPHDNN")
    boxplot!(ax1, DATA_df.ID[DATA_df[:,"model_type"].=="coxridge"] .+ 0.2, width = 0.5,  DATA_df[DATA_df[:,"model_type"].=="coxridge", "cph_test_c_ind"], color = "orange", label = "Cox-ridge")
    medians = combine(groupby(DATA_df[:,["ID", "model_type", "cph_test_c_ind"]], ["ID", "model_type"]), :cph_test_c_ind=>median) 
    text!(ax1, medians.ID[medians.model_type .== "cphdnn"].-0.35, medians[medians.model_type .== "cphdnn",:].cph_test_c_ind_median .+ offshift, text= string.(round.(medians[medians.model_type .== "cphdnn",:].cph_test_c_ind_median, digits = 3)))
    text!(ax1, medians.ID[medians.model_type .== "coxridge"] .+ 0.04, medians[medians.model_type .== "coxridge",:].cph_test_c_ind_median .+ offshift, text= string.(round.(medians[medians.model_type .== "coxridge",:].cph_test_c_ind_median, digits = 3)))
    axislegend(ax1, position = :rb)
    fig
    DATA_df = PARAMS[PARAMS[:,"dataset"] .== "BRCA",:]
    DATA_df = innerjoin(DATA_df,DataFrame("TYPE"=>unique(DATA_df[:,"TYPE"]), "ID" => collect(1:size(unique(DATA_df[:,"TYPE"]))[1])), on =:TYPE)
    # dtype_insize = combine(groupby(DATA_df, ["dim_redux_type"]), :insize=>maximum)
    # ticks = ["$dtype\n($ins_max)" for (dtype,ins_max) in zip(dtype_insize[:,1], dtype_insize[:,2])]
    ax2 = Axis(fig[2,1];
                title = "BRCA",
                limits = (0.5,size(unique(DATA_df[:,"TYPE"]))[1] + 0.5, nothing,up_y),
                xlabel = "Dimensionality reduction",
                ylabel = "Concordance index",
                xticks = (collect(1:size(unique(DATA_df[:,"TYPE"]))[1]), unique(DATA_df[:,"TYPE"])))
    lines!(ax2,[0,5],[0.5,0.5], linestyle = :dot)
    boxplot!(ax2, DATA_df.ID[DATA_df[:,"model_type"].=="cphdnn"] .- 0.2, width = 0.5,  DATA_df[DATA_df[:,"model_type"].=="cphdnn", "cph_test_c_ind"], color = "blue", label = "CPHDNN")
    boxplot!(ax2, DATA_df.ID[DATA_df[:,"model_type"].=="coxridge"] .+ 0.2, width = 0.5,  DATA_df[DATA_df[:,"model_type"].=="coxridge", "cph_test_c_ind"], color = "orange", label = "Cox-Ridge")
    medians = combine(groupby(DATA_df[:,["ID", "model_type", "cph_test_c_ind"]], ["ID", "model_type"]), :cph_test_c_ind=>median) 
    text!(ax2, medians.ID[medians.model_type .== "cphdnn"].-0.35, medians[medians.model_type .== "cphdnn",:].cph_test_c_ind_median .+ offshift, text= string.(round.(medians[medians.model_type .== "cphdnn",:].cph_test_c_ind_median, digits = 3)))
    text!(ax2, medians.ID[medians.model_type .== "coxridge"] .+ 0.04, medians[medians.model_type .== "coxridge",:].cph_test_c_ind_median .+ offshift, text= string.(round.(medians[medians.model_type .== "coxridge",:].cph_test_c_ind_median, digits = 3)))
    CairoMakie.save("figures/figure2_lgnaml_brca_coxridge_cphdnn_rdm_pca_clinf_sign.svg",fig)
    CairoMakie.save("figures/figure2_lgnaml_brca_coxridge_cphdnn_rdm_pca_clinf_sign.png",fig)
    CairoMakie.save("figures/figure2_lgnaml_brca_coxridge_cphdnn_rdm_pca_clinf_sign.pdf",fig)
    return fig
end
make_boxplots(PARAMS)

LC = gather_learning_curves("RES_FIG23/")
names(LC)
#LC[:,"TYPE"] = ["$x-$y" for (x,y) in  zip(LC[:, "dim_redux_type"], LC[:, "insize"])]

LOSSES_DF = innerjoin(LC, PARAMS, on = "modelid");
fig = Figure(size = (1024,400));
TRUNC_DF = LOSSES_DF[(LOSSES_DF.steps .% 100 .== 0) .| (LOSSES_DF.steps .== 1),:]
for (row_id, dataset) in enumerate(["LgnAML", "BRCA"])
    DATA_df = TRUNC_DF[TRUNC_DF[:,"dataset"] .== dataset,:]
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

fig = Figure(size = (1024,400));
TRUNC_DF = LOSSES_DF[(LOSSES_DF.steps .% 100 .== 0) .| (LOSSES_DF.steps .== 1),:]
for (row_id, dataset) in enumerate(["LgnAML", "BRCA"])
    DATA_df = TRUNC_DF[TRUNC_DF[:,"dataset"] .== dataset,:]
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
#axislegend(ax1)
CairoMakie.save("figures/figure3_coxridge_lgnaml_brca_rdm_pca_clinf_sign_overfit.svg",fig)
CairoMakie.save("figures/figure3_coxridge_lgnaml_brca_rdm_pca_clinf_sign_overfit.png",fig)
CairoMakie.save("figures/figure3_coxridge_lgnaml_brca_rdm_pca_clinf_sign_overfit.pdf",fig)

fig