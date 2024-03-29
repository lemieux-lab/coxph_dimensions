function draw_mean_lines(ax, XY, ls, col, label)
    XY_values = sort(combine(groupby(XY, ["insize"]), "cph_tst_c_ind_med"=>mean), ["insize"])
    lines!(ax, log10.(XY_values[:,1]), XY_values[:,2], linestyle=ls, color = col, label = label, linewidth = 3)
end 
function draw_multi_mean_lines(RES;test_metric = "cph_test_c_ind")
    fig = Figure(size=(700,700));
    for (dname, coords) in zip(unique(RES[:,"dataset"]), [(1,1),(1,2),(2,1),(2,2)])
        data_df = RES[RES[:,"dataset"].== dname .&& RES[:,"dim_redux_type"] .== "STD" ,:]
        ds_size = unique(data_df[:,"nsamples"])[1]
        ticks = [1,10,100,1000,maximum(data_df[:,"insize"])]
        row, col = coords
        SMA_K, SMA_N = 10,10 
        #yticks = (collect(1:100)/100, [x * 100 % 2 == 0 ? "$x" : "" for x in collect(1:100)/100])
        #dname == "LGG" ? yticks =  (collect(1:50)/50, [x * 100 % 4 == 0 ? "$x" : "" for x in collect(1:50)/50]) : yticks
        ax = Axis(fig[row,col],
                xticks = (log10.(ticks),string.(ticks)),
                yticks = (collect(1:100)/100, [Int(round(x * 100)) % 2 == 0 ? "$x" : "" for x in collect(1:100)/100]),
                title = "$dname (n=$ds_size)",
                xlabel = "Input size",
                ylabel = "concordance index",
                limits = (nothing, nothing, 0.48, nothing))
        lines!(ax,log10.(ticks[[1,end]]),[0.5,0.5],linestyle = :dash)
        # BLUE SOLID
        draw_mean_lines(ax,  data_df[data_df[:,"model_type"] .== "cphdnn",["insize", "cph_tst_c_ind_med"]], :solid, :blue, "CPHDNN-STD")
        # BLUE DASH 
        draw_mean_lines(ax,  data_df[data_df[:,"model_type"] .== "coxridge",["insize", "cph_tst_c_ind_med"]], :dash, :blue, "COXRIDGE-STD")
        
        data_df = RES[RES[:,"dataset"].== dname .&& RES[:,"dim_redux_type"] .== "RDM" ,:]
        # ORANGE SOLID
        draw_mean_lines(ax,  data_df[data_df[:,"model_type"] .== "cphdnn",["insize", "cph_tst_c_ind_med"]], :solid, :orange, "CPHDNN-RDM")
        # ORANGE DASH 
        draw_mean_lines(ax,  data_df[data_df[:,"model_type"] .== "coxridge",["insize", "cph_tst_c_ind_med"]], :dash, :orange, "COXRIDGE-RDM")
        
        data_df = RES[RES[:,"dataset"].== dname .&& RES[:,"dim_redux_type"] .== "PCA" ,:]
        # BLACK SOLID
        draw_mean_lines(ax,  data_df[data_df[:,"model_type"] .== "cphdnn",["insize", "cph_tst_c_ind_med"]], :solid, :black, "CPHDNN-PCA")
        # BLACK DASH 
        draw_mean_lines(ax,  data_df[data_df[:,"model_type"] .== "coxridge",["insize", "cph_tst_c_ind_med"]], :dash, :black, "COXRIDGE-PCA")
                        
        coords === (1,2) ? axislegend(ax, framewidth =0, position = :rb, labelsize = 10, patchlabelgap = 0, padding = (0,0,0,0)) : 1   
    end 
    CairoMakie.save("figures/PDF/figure1_lgnaml_brca_ov_lgg_rdm_std_dim_sweep.pdf", fig)
    CairoMakie.save("figures/figure1_lgnaml_brca_ov_lgg_rdm_std_dim_sweep.png", fig)
    CairoMakie.save("figures/figure1_lgnaml_brca_ov_lgg_rdm_std_dim_sweep.svg", fig)

    return fig
end 

function make_multi_scatterplot(RES;test_metric = "cph_test_c_ind")
    fig = Figure(size=(700,700));
    for (dname, coords) in zip(unique(RES[:,"dataset"]), [(1,1),(1,2),(2,1),(2,2)])
        data_df = RES[RES[:,"dataset"].== dname .&& RES[:,"dim_redux_type"] .== "STD" ,:]
        ds_size = unique(data_df[:,"nsamples"])[1]
        ticks = [1,10,100,1000,maximum(data_df[:,"insize"])]
        row, col = coords
        SMA_K, SMA_N = 10,10 
        #yticks = (collect(1:100)/100, [x * 100 % 2 == 0 ? "$x" : "" for x in collect(1:100)/100])
        #dname == "LGG" ? yticks =  (collect(1:50)/50, [x * 100 % 4 == 0 ? "$x" : "" for x in collect(1:50)/50]) : yticks
        ax = Axis(fig[row,col],
                xticks = (log10.(ticks),string.(ticks)),
                yticks = (collect(1:100)/100, [Int(round(x * 100)) % 2 == 0 ? "$x" : "" for x in collect(1:100)/100]),
                title = "$dname (n=$ds_size)",
                xlabel = "Input size",
                ylabel = "concordance index",
                limits = (nothing, nothing, 0.48, nothing))
        lines!(ax,log10.(ticks[[1,end]]),[0.5,0.5],linestyle = :dash)
        draw_scatter_sma!(ax,   data_df[data_df[:,"model_type"] .== "cphdnn","insize"], 
                                data_df[data_df[:,"model_type"] .== "cphdnn",test_metric],
                            "blue", "CPHDNN-STD", 0, :solid, SMA_K,SMA_N,text_on = false)
        draw_scatter_sma!(ax,   data_df[data_df[:,"model_type"] .== "coxridge","insize"], 
                            data_df[data_df[:,"model_type"] .== "coxridge",test_metric],
                        "blue", "COXRIDGE-STD", 0, :dash, SMA_K,SMA_N,text_on = false)
        data_df = RES[RES[:,"dataset"].== dname .&& RES[:,"dim_redux_type"] .== "RDM" ,:]
        draw_scatter_sma!(ax,   data_df[data_df[:,"model_type"] .== "cphdnn","insize"], 
                                data_df[data_df[:,"model_type"] .== "cphdnn",test_metric],
                            "orange", "CPHDNN-RDM", 0, :solid, SMA_K,SMA_N,text_on = false)
        draw_scatter_sma!(ax,   data_df[data_df[:,"model_type"] .== "coxridge","insize"], 
                            data_df[data_df[:,"model_type"] .== "coxridge",test_metric],
                        "orange", "COXRIDGE-RDM", 0, :dash, SMA_K,SMA_N,text_on = false)
        data_df = RES[RES[:,"dataset"].== dname .&& RES[:,"dim_redux_type"] .== "PCA" ,:]
        draw_scatter_sma!(ax,   data_df[data_df[:,"model_type"] .== "cphdnn","insize"], 
                                                data_df[data_df[:,"model_type"] .== "cphdnn",test_metric],
                                            "black", "CPHDNN-PCA", 0, :solid, SMA_K,SMA_N, text_on=false)
        draw_scatter_sma!(ax,   data_df[data_df[:,"model_type"] .== "coxridge","insize"], 
                                            data_df[data_df[:,"model_type"] .== "coxridge",test_metric],
                                        "black", "COXRIDGE-PCA", 0, :dash, SMA_K,SMA_N, text_on=false)
                        
        coords === (1,2) ? axislegend(ax, framewidth =0, position = :rb, labelsize = 10, patchlabelgap = 0, padding = (0,0,0,0)) : 1   
    end 
    CairoMakie.save("figures/figure1_lgnaml_brca_ov_lgg_rdm_std_dim_sweep.pdf", fig)
    CairoMakie.save("figures/figure1_lgnaml_brca_ov_lgg_rdm_std_dim_sweep.png", fig)
    CairoMakie.save("figures/figure1_lgnaml_brca_ov_lgg_rdm_std_dim_sweep.svg", fig)

    return fig
end 

function make_boxplots(PARAMS;test_metric = "cph_test_c_ind")
    fig = Figure(size=(800,512));
    offshift = 0.03
    up_y = 0.80
    DATA_df = sort(PARAMS[PARAMS[:,"dataset"] .== "LgnAML",:], ["TYPE"])
    DATA_df = innerjoin(DATA_df,DataFrame("TYPE"=>unique(DATA_df[:,"TYPE"]), "ID" => collect(1:size(unique(DATA_df[:,"TYPE"]))[1])), on =:TYPE)
    #dtype_insize = combine(groupby(DATA_df, ["TYPE"]), :insize=>maximum)
    #ticks = ["$dtype\n($ins_max)" for (dtype,ins_max) in zip(dtype_insize[:,1], dtype_insize[:,2])]
    ax1 = Axis(fig[1,1];
                title = "Leucegene",
                limits = (0.5,size(unique(DATA_df[:,"TYPE"]))[1] + 0.5, nothing, up_y),
                xlabel = "Dimensionality reduction",
                ylabel = "Concordance index",
                yticks = (collect(1:100)/20, ["$x"  for x in collect(1:100)/20]),                
                xticks =  (collect(1:size(unique(DATA_df[:,"TYPE"]))[1]), replace.(unique(DATA_df[:,"TYPE"]), "-"=>"\n")))
    lines!(ax1,[0,size(unique(DATA_df[:,"TYPE"]))[1] +0.5] ,[0.5,0.5], linestyle = :dot)
    boxplot!(ax1, DATA_df.ID[DATA_df[:,"model_type"].=="cphdnn"] .- 0.2, width = 0.5,  DATA_df[DATA_df[:,"model_type"].=="cphdnn", test_metric], color = "blue", label = "CPHDNN")
    boxplot!(ax1, DATA_df.ID[DATA_df[:,"model_type"].=="coxridge"] .+ 0.2, width = 0.5,  DATA_df[DATA_df[:,"model_type"].=="coxridge", test_metric], color = "orange", label = "Cox-ridge")
    medians = combine(groupby(DATA_df[:,["ID", "model_type", test_metric]], ["ID", "model_type"]), test_metric=>median) 
    text!(ax1, medians.ID[medians.model_type .== "cphdnn"].-0.35, medians[medians.model_type .== "cphdnn","$(test_metric)_median"] .+ offshift, text= string.(round.(medians[medians.model_type .== "cphdnn","$(test_metric)_median"], digits = 3)))
    text!(ax1, medians.ID[medians.model_type .== "coxridge"] .+ 0.04, medians[medians.model_type .== "coxridge","$(test_metric)_median"] .+ offshift, text= string.(round.(medians[medians.model_type .== "coxridge","$(test_metric)_median"], digits = 3)))
    axislegend(ax1, position = :cb)
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
                yticks = (collect(1:100)/20, ["$x"  for x in collect(1:100)/20]), 
                xticks = (collect(1:size(unique(DATA_df[:,"TYPE"]))[1]), unique(DATA_df[:,"TYPE"])))
    lines!(ax2,[0,5],[0.5,0.5], linestyle = :dot)
    boxplot!(ax2, DATA_df.ID[DATA_df[:,"model_type"].=="cphdnn"] .- 0.2, width = 0.5,  DATA_df[DATA_df[:,"model_type"].=="cphdnn", test_metric], color = "blue", label = "CPHDNN")
    boxplot!(ax2, DATA_df.ID[DATA_df[:,"model_type"].=="coxridge"] .+ 0.2, width = 0.5,  DATA_df[DATA_df[:,"model_type"].=="coxridge", test_metric], color = "orange", label = "Cox-Ridge")
    medians = combine(groupby(DATA_df[:,["ID", "model_type", test_metric]], ["ID", "model_type"]), test_metric=>median) 
    text!(ax2, medians.ID[medians.model_type .== "cphdnn"].-0.35, medians[medians.model_type .== "cphdnn","$(test_metric)_median"] .+ offshift, text= string.(round.(medians[medians.model_type .== "cphdnn","$(test_metric)_median"], digits = 3)))
    text!(ax2, medians.ID[medians.model_type .== "coxridge"] .+ 0.04, medians[medians.model_type .== "coxridge","$(test_metric)_median"] .+ offshift, text= string.(round.(medians[medians.model_type .== "coxridge","$(test_metric)_median"], digits = 3)))
    CairoMakie.save("figures/figure2_lgnaml_brca_coxridge_cphdnn_rdm_pca_clinf_sign.svg",fig)
    CairoMakie.save("figures/figure2_lgnaml_brca_coxridge_cphdnn_rdm_pca_clinf_sign.png",fig)
    CairoMakie.save("figures/PDF/figure2_lgnaml_brca_coxridge_cphdnn_rdm_pca_clinf_sign.pdf",fig)
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