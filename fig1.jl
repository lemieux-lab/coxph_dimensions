include("engines/init.jl")
include("engines/data_processing.jl")
include("engines/deep_learning.jl")
include("engines/cross_validation.jl")
include("engines/model_evaluation.jl")
include("engines/figures.jl")
RES = gather_params("RES_FIG1");
RES = RES[isnan.(RES[:,"cph_test_c_ind"]).== 0,:]
dname = "LgnAML"
draw_multi_mean_lines(RES, test_metric = "cph_tst_c_ind_med")

# make_multi_scatterplot(RES; test_metric = "cph_tst_c_ind_med")

## SPOT CHECK
SMA_K, SMA_N = 5,10
fig = Figure();
dname = "LGG"
data_df = RES[RES[:,"dataset"].== dname .&& RES[:,"dim_redux_type"] .== "PCA" ,:]
ds_size = unique(data_df[:,"nsamples"])[1]
ticks = [1,10,100,1000,maximum(data_df[:,"insize"])]
ax = Axis(fig[1,1],
        xticks = (log10.(ticks),string.(ticks)),
        #yticks = (collect(1:20)/20, ["$x" for x in collect(1:20)/20]),
        title = "$dname (n=$ds_size)",
        xlabel = "Input size",
        ylabel = "concordance index",
        limits = (nothing, nothing, 0.4, nothing))
lines!(ax,log10.(ticks[[1,end]]),[0.5,0.5],linestyle = :dash)
draw_scatter_sma!(ax,   data_df[data_df[:,"model_type"] .== "cphdnn","insize"], 
                        data_df[data_df[:,"model_type"] .== "cphdnn","cph_test_c_ind"],
                    "blue", "CPHDNN", 1, :solid, SMA_K,SMA_N)
draw_scatter_sma!(ax,   data_df[data_df[:,"model_type"] .== "coxridge","insize"], 
                    data_df[data_df[:,"model_type"] .== "coxridge","cph_test_c_ind"],
                "orange", "COXRIDGE", 1, :dash, SMA_K, SMA_N)
axislegend(ax,position=:rb)
fig