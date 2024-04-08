include("engines/init.jl")
include("engines/data_processing.jl")
include("engines/deep_learning.jl")
include("engines/cross_validation.jl")
include("engines/model_evaluation.jl")
outpath, session_id = set_dirs("RES_ISMB") ;

function build_model(modeltype, params; sigmoid_output = true)
    sigmoid_output ? params["output"] = sigmoid : params["output"] = identity 
    if modeltype == "coxridge"
        return Chain(Dense(params["insize"], 1, params["output"], bias = false))
    elseif modeltype == "cphdnn"
        return Chain(Dense(params["insize"],params["cph_hl_size"], leakyrelu), 
        Dense(params["cph_hl_size"], params["cph_hl_size"], leakyrelu), 
        Dense(params["cph_hl_size"], 1, params["output"],  bias = false))
    end 
end 
function evaluate_model(modeltype, DS, dim_redux_size;hlsize = 0, nepochs= 5_000, cph_nb_hl = 0, cph_lr = 1e-4, 
    cph_wd = 1e-4, nfolds =5,dim_redux_type ="RDM", print_step = 500)
folds = prep_data_params_dict!(DS, dim_redux_size, hlsize = hlsize, nepochs= nepochs, 
    cph_nb_hl = cph_nb_hl, cph_lr = cph_lr, cph_wd = cph_wd, nfolds = nfolds,  modeltype = modeltype,
    dim_redux_type =dim_redux_type)
OUTS_TST, Y_T_TST, Y_E_TST, train_cinds, test_cinds = [], [], [], [],[]
LOSSES_BY_FOLD  = []
# for fold in folds do train
for fold in folds
    # format data on GPU 
    train_x, train_y_t, train_y_e, NE_frac_tr, test_x, test_y_t, test_y_e, NE_frac_tst = format_train_test(fold)
    DS["data_prep"] = Dict("train_x"=>train_x, "train_y_t"=>train_y_t,"train_y_e"=>train_y_e,"NE_frac_tr"=>NE_frac_tr, "test_x"=>test_x,
    "test_y_t"=> test_y_t, "test_y_e"=>test_y_e, "NE_frac_tst"=> NE_frac_tst)
    DS["params"]["insize"] = size(train_x)[1]
    
    ## init model and opt
    OPT = Flux.ADAM(DS["params"]["cph_lr"]) 
    MODEL = gpu(build_model(modeltype, DS["params"]))
    
    # train loop 
    LOSS_TR, LOSS_TST = train_model!(MODEL, OPT, DS; print_step=print_step, foldn = fold["foldn"])
    push!(LOSSES_BY_FOLD, (LOSS_TR, LOSS_TST))
    # final model eval 
    OUTS_tst = MODEL(DS["data_prep"]["test_x"])
    OUTS_tr = MODEL(DS["data_prep"]["train_x"])
    cind_test,cdnt_tst, ddnt_tst, tied_tst = concordance_index(DS["data_prep"]["test_y_t"], DS["data_prep"]["test_y_e"], -1 * OUTS_tst)
    cind_tr, cdnt_tr, ddnt_tr, tied_tr  = concordance_index(DS["data_prep"]["train_y_t"],DS["data_prep"]["train_y_e"], -1 * OUTS_tr)
    
    push!(test_cinds, cind_test)
    push!(train_cinds, cind_tr)
    push!(OUTS_TST, vec(cpu(MODEL(DS["data_prep"]["test_x"]))))
    push!(Y_T_TST, vec(cpu(DS["data_prep"]["test_y_t"])))
    push!(Y_E_TST, vec(cpu(DS["data_prep"]["test_y_e"])))
    DS["params"]["nparams"] = size(train_x)[1]
end 
c_indices_tst = dump_results!(DS, LOSSES_BY_FOLD, OUTS_TST, Y_T_TST, Y_E_TST, train_cinds, test_cinds)
return c_indices_tst
end 
# Cox NL loss and concordance index do not always concord. (1A)
# LGN clinf 8 CPHDNN / Cox-ridge training report learning curve (c-index, loss)
# sigmoid, adam, lr=1e-6, cph_wd=1e-3, 350K steps.
# * BRCA clinf 

# Linear output (output range is unbound) vs sigmoid (output clamped between 0 and 1). (1B)
# Using convergence criterion for unbiased determination of L2, learning rate, optimization steps hyper-parameters. (1C)
# Reporting cross-validation metrics with c-index scores aggregation + bootstrapping is more consistent than using the average. (1D) 
