function intervals(tr_outs, nbins) 
    [minimum(tr_outs) + i* (maximum(tr_outs)-minimum(tr_outs)) / nbins for i in 1:nbins]
end 
function partition(data, intervals)                                                                                
    ranges = intervals[1:end-1] .=> intervals[2:end]                                                               
    bins = [similar(data, 0) for _ in 1:length(ranges)]                                                                          
    for x in data                                                                                                  
        for (i, (a, b)) in pairs(ranges)                                                                           
            if a <= x < b                                                                                          
                push!(bins[i], x)                                                                                  
                break                                                                                              
            end                                                                                                    
        end                                                                                                        
    end                                                                                                            
    return bins                                                                                                    
end  

function plot_hist_scores(DS, MODEL; log_tr = true)
    tr_outs = MODEL(DS["data_prep"]["train_x"])
    tst_outs = MODEL(DS["data_prep"]["test_x"])
    c_ind_tr, bli, bla, blou = concordance_index(DS["data_prep"]["train_y_t"],DS["data_prep"]["train_y_e"], -1 * tr_outs)
    c_ind_tst, bli, bla, blou = concordance_index(DS["data_prep"]["test_y_t"],DS["data_prep"]["test_y_e"], -1 * tst_outs)
    fig = Figure(size = (1024,512));
    ax = Axis(fig[1,1], 
    # xticks = ([1], ["training set"]),
    title="training set (c index = $(round(c_ind_tr, digits = 3)))")
    log_tr ? tr_scores = log10.(vec(cpu(tr_outs))) : tr_scores = vec(cpu(tr_outs)) 
    log_tr ? tst_scores = log10.(vec(cpu(tst_outs))) : tst_scores = vec(cpu(tst_outs)) 
    
    tr_hist_deceased = [size(CC)[1] for CC in partition(tr_scores[vec(cpu(BRCA_data["data_prep"]["train_y_e"])) .== 1], intervals(tr_scores, 30))]
    tr_hist_alive = [size(CC)[1] for CC in partition(tr_scores[vec(cpu(BRCA_data["data_prep"]["train_y_e"])) .== 0], intervals(tr_scores, 30))]
    x_ticks = intervals(tr_scores, 30)[1:end-1]

    barplot!(ax, x_ticks, tr_hist_deceased .+ tr_hist_alive, color = :blue, strokewidth = 0.5, gap = 0., label = "alive");
    barplot!(ax, x_ticks, tr_hist_deceased, color = :red, gap = 0.,strokewidth = 0.5, label = "deceased");

    axislegend(ax, position = :lc)
    ax = Axis(fig[1,2], 
    # xticks = ([1], ["training set"]),
    title="test set (c index = $(round(c_ind_tst, digits = 3)))")
    tst_hist_deceased = [size(CC)[1] for CC in partition(tst_scores[vec(cpu(BRCA_data["data_prep"]["train_y_e"])) .== 1], intervals(tst_scores, 30))]
    tst_hist_alive = [size(CC)[1] for CC in partition(tst_scores[vec(cpu(BRCA_data["data_prep"]["train_y_e"])) .== 0], intervals(tst_scores, 30))]
    x_ticks = intervals(tst_scores, 30)[1:end-1]

    barplot!(ax, x_ticks, tst_hist_deceased .+ tst_hist_alive, color = :blue, strokewidth = 0.5, gap = 0., label = "alive");
    barplot!(ax, x_ticks, tst_hist_deceased, color = :red, gap = 0.,strokewidth = 0.5, label = "deceased");

    return fig
end 

function plot_hist_scores!(fig, DS, MODEL; log_tr = true)
    tr_outs = MODEL(DS["data_prep"]["train_x"])
    tst_outs = MODEL(DS["data_prep"]["test_x"])
    c_ind_tr, bli, bla, blou = concordance_index(DS["data_prep"]["train_y_t"],DS["data_prep"]["train_y_e"], -1 * tr_outs)
    c_ind_tst, bli, bla, blou = concordance_index(DS["data_prep"]["test_y_t"],DS["data_prep"]["test_y_e"], -1 * tst_outs)
    ax = Axis(fig[1:2,2],  ylabel = "count", xlabel = "output scores",
    # xticks = ([1], ["training set"]),
    title="training set (c index = $(round(c_ind_tr, digits = 3)))")
    log_tr ? tr_scores = log10.(vec(cpu(tr_outs))) : tr_scores = vec(cpu(tr_outs)) 
    log_tr ? tst_scores = log10.(vec(cpu(tst_outs))) : tst_scores = vec(cpu(tst_outs)) 
    
    tr_hist_deceased = [size(CC)[1] for CC in partition(tr_scores[vec(cpu(DS["data_prep"]["train_y_e"])) .== 1], intervals(tr_scores, 30))]
    tr_hist_alive = [size(CC)[1] for CC in partition(tr_scores[vec(cpu(DS["data_prep"]["train_y_e"])) .== 0], intervals(tr_scores, 30))]
    x_ticks = intervals(tr_scores, 30)[1:end-1]

    barplot!(ax, x_ticks, tr_hist_deceased .+ tr_hist_alive, color = :blue, strokewidth = 0.5, gap = 0., label = "alive");
    barplot!(ax, x_ticks, tr_hist_deceased, color = :red, gap = 0.,strokewidth = 0.5, label = "deceased");

    axislegend(ax, position = :rc)
    ax = Axis(fig[1:2,3], ylabel = "count", xlabel = "output scores",
    # xticks = ([1], ["training set"]),
    title="test set (c index = $(round(c_ind_tst, digits = 3)))")
    tst_hist_deceased = [size(CC)[1] for CC in partition(tst_scores[vec(cpu(DS["data_prep"]["test_y_e"])) .== 1], intervals(tst_scores, 30))]
    tst_hist_alive = [size(CC)[1] for CC in partition(tst_scores[vec(cpu(DS["data_prep"]["test_y_e"])) .== 0], intervals(tst_scores, 30))]
    x_ticks = intervals(tst_scores, 30)[1:end-1]

    barplot!(ax, x_ticks, tst_hist_deceased .+ tst_hist_alive, color = :blue, strokewidth = 0.5, gap = 0., label = "alive");
    barplot!(ax, x_ticks, tst_hist_deceased, color = :red, gap = 0.,strokewidth = 0.5, label = "deceased");

    return fig
end 

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