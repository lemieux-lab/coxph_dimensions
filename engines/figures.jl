function make_boxplots(PARAMS;test_metric = "cph_test_c_ind")
    fig = Figure(size=(600,512));
    offshift = 0.03
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
    boxplot!(ax1, DATA_df.ID[DATA_df[:,"model_type"].=="cphdnn"] .- 0.2, width = 0.5,  DATA_df[DATA_df[:,"model_type"].=="cphdnn", test_metric], color = "blue", label = "CPHDNN")
    boxplot!(ax1, DATA_df.ID[DATA_df[:,"model_type"].=="coxridge"] .+ 0.2, width = 0.5,  DATA_df[DATA_df[:,"model_type"].=="coxridge", test_metric], color = "orange", label = "Cox-ridge")
    medians = combine(groupby(DATA_df[:,["ID", "model_type", test_metric]], ["ID", "model_type"]), :cph_test_c_ind=>median) 
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
    boxplot!(ax2, DATA_df.ID[DATA_df[:,"model_type"].=="cphdnn"] .- 0.2, width = 0.5,  DATA_df[DATA_df[:,"model_type"].=="cphdnn", test_metric], color = "blue", label = "CPHDNN")
    boxplot!(ax2, DATA_df.ID[DATA_df[:,"model_type"].=="coxridge"] .+ 0.2, width = 0.5,  DATA_df[DATA_df[:,"model_type"].=="coxridge", test_metric], color = "orange", label = "Cox-Ridge")
    medians = combine(groupby(DATA_df[:,["ID", "model_type", test_metric]], ["ID", "model_type"]), :cph_test_c_ind=>median) 
    text!(ax2, medians.ID[medians.model_type .== "cphdnn"].-0.35, medians[medians.model_type .== "cphdnn",:].cph_test_c_ind_median .+ offshift, text= string.(round.(medians[medians.model_type .== "cphdnn",:].cph_test_c_ind_median, digits = 3)))
    text!(ax2, medians.ID[medians.model_type .== "coxridge"] .+ 0.04, medians[medians.model_type .== "coxridge",:].cph_test_c_ind_median .+ offshift, text= string.(round.(medians[medians.model_type .== "coxridge",:].cph_test_c_ind_median, digits = 3)))
    CairoMakie.save("figures/figure2_lgnaml_brca_coxridge_cphdnn_rdm_pca_clinf_sign.svg",fig)
    CairoMakie.save("figures/figure2_lgnaml_brca_coxridge_cphdnn_rdm_pca_clinf_sign.png",fig)
    CairoMakie.save("figures/figure2_lgnaml_brca_coxridge_cphdnn_rdm_pca_clinf_sign.pdf",fig)
    return fig
end

function SMA(DATA_X, DATA_Y; n=10, k=10)
    means_y = []
    means_x = []
    Y_infos =  DATA_Y[sortperm(DATA_X)]
    X_infos = sort(DATA_X)
    step = Int(floor(length(X_infos) / n ))
    for X_id in vcat(collect(1:step:length(X_infos) - step), length(X_infos))
        x_id_min = max(X_id - k, 1)
        x_id_max = min(X_id + k, length(X_infos))
        sma = mean(Y_infos[x_id_min:x_id_max])
        push!(means_y, sma)
        push!(means_x, X_infos[X_id])
    end
    return float.(means_x),float.(means_y)
end 


function draw_scatter_sma!(ax, X, Y, col,label,alpha,ls, SMA_K, SMA_N;text_on=true)
    scatter!(ax, log10.(X), Y, color= col, alpha = alpha)
    sma_x, sma_y = SMA(log10.(X), Y, n=SMA_N, k=SMA_K)
    lines!(ax, sma_x,sma_y, color = col, label = label, linewidth=3, linestyle =ls)
    text_on ? text!(ax, sma_x .- 0.25,sma_y,text = string.(round.(sma_y, digits=2)), color= col) : 0

end 

function add_multi_scatter!(fig, row, col, data_df; SMA_K=10, SMA_N=10)
    ticks = [minimum(data_df[:,"insize"]),100,1000,maximum(data_df[:,"insize"])]
    dname = unique(data_df[:,"dataset"])[1]
    ax = Axis(fig[row,col],
        xticks = (log10.(ticks),string.(ticks)),
        #yticks = (collect(1:20)/20, ["$x" for x in collect(1:20)/20]),
        title = "Survival prediction in $dname with CPH-DNN & COX-ridge \n by input size with random and directed dimensionality reductions",
        xlabel = "Input size",
        ylabel = "concordance index",
        limits = (nothing, nothing, 0.4, 0.75))
    lines!(ax,log10.(ticks[[1,end]]),[0.5,0.5],linetype = "dashed")

    cphdnn = data_df[data_df[:,"model_type"] .== "cphdnn",:]
    coxridge = data_df[data_df[:,"model_type"] .== "cox_ridge",:]

    draw_scatter_sma!(ax,   cphdnn[cphdnn[:,"nb_clinf"] .!= 0 ,"insize"], 
                            cphdnn[cphdnn[:,"nb_clinf"] .!= 0,"cph_test_c_ind"], 
                            "blue", "CPHDNN with clinical factors (16)", 0.5, :dash, SMA_K, SMA_N)
    draw_scatter_sma!(ax,   coxridge[coxridge[:,"nb_clinf"] .!= 0 ,"insize"], 
                            coxridge[coxridge[:,"nb_clinf"] .!= 0,"cph_test_c_ind"], 
                            "orange", "Cox-ridge with clinical factors (16)", 0.5, :dash,SMA_K, SMA_N)
    draw_scatter_sma!(ax,   cphdnn[cphdnn[:,"nb_clinf"] .== 0 ,"insize"], 
                            cphdnn[cphdnn[:,"nb_clinf"] .== 0,"cph_test_c_ind"], 
                            "blue", "CPHDNN no clin. f", 1, :solid,SMA_K, SMA_N)
    draw_scatter_sma!(ax,   coxridge[coxridge[:,"nb_clinf"] .== 0 ,"insize"], 
                            coxridge[coxridge[:,"nb_clinf"] .== 0,"cph_test_c_ind"], 
                            "orange", "Cox-ridge no clin. f", 1, :solid, SMA_K, SMA_N)
    return fig, ax
end 