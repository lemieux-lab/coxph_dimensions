include("engines/init.jl")
include("engines/data_processing.jl")
include("engines/deep_learning.jl")
include("engines/cross_validation.jl")
include("engines/model_evaluation.jl")
include("engines/figures.jl")
RES = gather_params("RES_FIG1");
RES = RES[isnan.(RES[:,"cph_test_c_ind"]).== 0,:]
fig = Figure(size=(612,612));
for (dname, coords) in zip(unique(RES[:,"dataset"]), [(1,1),(1,2),(2,1),(2,2)])
    data_df = RES[RES[:,"dataset"].== dname .&& RES[:,"dim_redux_type"] .== "STD" ,:]
    ds_size = unique(data_df[:,"nsamples"])[1]
    ticks = [1,10,100,1000,maximum(data_df[:,"insize"])]
    row, col = coords
    SMA_K, SMA_N = 5,10 
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
                            data_df[data_df[:,"model_type"] .== "cphdnn","cph_test_c_ind"],
                        "blue", "CPHDNN-STD", 0, :solid, SMA_K,SMA_N,text_on = false)
    draw_scatter_sma!(ax,   data_df[data_df[:,"model_type"] .== "coxridge","insize"], 
                        data_df[data_df[:,"model_type"] .== "coxridge","cph_test_c_ind"],
                    "blue", "COXRIDGE-STD", 0, :dash, SMA_K,SMA_N,text_on = false)
    data_df = RES[RES[:,"dataset"].== dname .&& RES[:,"dim_redux_type"] .== "RDM" ,:]
    draw_scatter_sma!(ax,   data_df[data_df[:,"model_type"] .== "cphdnn","insize"], 
                            data_df[data_df[:,"model_type"] .== "cphdnn","cph_test_c_ind"],
                        "orange", "CPHDNN-RDM", 0, :solid, SMA_K,SMA_N,text_on = false)
    draw_scatter_sma!(ax,   data_df[data_df[:,"model_type"] .== "coxridge","insize"], 
                        data_df[data_df[:,"model_type"] .== "coxridge","cph_test_c_ind"],
                    "orange", "COXRIDGE-RDM", 0, :dash, SMA_K,SMA_N,text_on = false)
    # data_df = RES[RES[:,"dataset"].== dname .&& RES[:,"dim_redux_type"] .== "PCA" ,:]
    # draw_scatter_sma!(ax,   data_df[data_df[:,"model_type"] .== "cphdnn","insize"], 
    #                                         data_df[data_df[:,"model_type"] .== "cphdnn","cph_test_c_ind"],
    #                                     "black", "CPHDNN-PCA", 0, :solid, 10,10)
    # draw_scatter_sma!(ax,   data_df[data_df[:,"model_type"] .== "coxridge","insize"], 
    #                                     data_df[data_df[:,"model_type"] .== "coxridge","cph_test_c_ind"],
    #                                 "black", "COXRIDGE-PCA", 0, :dash, 10,10)
                    
    coords === (1,1) ? axislegend(ax, framewidth =0, position = :lt, labelsize = 10, patchlabelgap = 1, padding = (0,0,0,0)) : 1   
end 
CairoMakie.save("figures/figure1_lgnaml_brca_ov_lgg_rdm_std_dim_sweep.pdf", fig)
CairoMakie.save("figures/figure1_lgnaml_brca_ov_lgg_rdm_std_dim_sweep.png", fig)
CairoMakie.save("figures/figure1_lgnaml_brca_ov_lgg_rdm_std_dim_sweep.svg", fig)

fig

SMA_K, SMA_N = 2,4
fig = Figure();
data_df = RES[RES[:,"dataset"].== "LgnAML" .&& RES[:,"dim_redux_type"] .== "STD" ,:]
ds_size = unique(data_df[:,"nsamples"])[1]
ticks = [minimum(data_df[:,"insize"]),100,1000,maximum(data_df[:,"insize"])]
ax = Axis(fig[1,1],
        xticks = (log10.(ticks),string.(ticks)),
        #yticks = (collect(1:20)/20, ["$x" for x in collect(1:20)/20]),
        title = "LGNAML (n=$ds_size)",
        xlabel = "Input size",
        ylabel = "concordance index",
        limits = (nothing, nothing, 0.4, nothing))
lines!(ax,log10.(ticks[[1,end]]),[0.5,0.5],linestyle = :dash)
draw_scatter_sma!(ax,   data_df[data_df[:,"model_type"] .== "cphdnn","insize"], 
                        data_df[data_df[:,"model_type"] .== "cphdnn","cph_test_c_ind"],
                    "blue", "CPHDNN-STD", 1, :solid, SMA_K,SMA_N)
draw_scatter_sma!(ax,   data_df[data_df[:,"model_type"] .== "coxridge","insize"], 
                    data_df[data_df[:,"model_type"] .== "coxridge","cph_test_c_ind"],
                "orange", "COXRIDGE-STD", 1, :dash, SMA_K, SMA_N)
axislegend(ax,position=:rb)
fig