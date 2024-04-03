function fit_transform_pca(X, outdim)
    x_means =  mean(X, dims =2 )
    Z = X .- x_means
    U, S, V = svd(Z,full=true);
    Matrix(U[:, sortperm(S, rev=true)[1:outdim]]') * Z
end
function fit_pca(X, outdim)
    x_means =  mean(X, dims =2 )
    Z = X .- x_means
    U, S, V = svd(Z,full=true);
    return Matrix(U[:, sortperm(S, rev=true)[1:outdim]]') 
end
function transform_pca(X, P)
    x_means =  mean(X, dims =2 )
    Z = X .- x_means
    P * Z
end 

function prep_data_params_dict!(DataSet, dim_redux_size;
    hlsize = 0, nepochs= 5_000, cph_nb_hl = 2, cph_lr = 1e-4, cph_wd = 1e-4,  
    nfolds = 5, modeltype = "coxridge",dim_redux_type ="RDM")
    dataset_name = DataSet["name"]
    DATA = DataSet["dataset"]
    CDS_data = DATA.data[:,DataSet["CDS"]]
    clinical_factors = DataSet["CF"]
    if dim_redux_type == "CLINF"
        X_data = DataSet["CF"]
    elseif dim_redux_type == "STD"
        VARS = var(CDS_data, dims = 1)
        genes = reverse(sortperm(vec(VARS)))[1:dim_redux_size]
        X_data = CDS_data[:,genes] 
    elseif occursin(dim_redux_type, "PCA")
        X_data = CDS_data
    else ## dim redux type is RDM 
        X_data =  CDS_data[:,sample(collect(1:size(CDS_data)[2]), dim_redux_size, replace=false)]    
    end 
    nb_clinf = size(clinical_factors)[2]
    # init params dict
    DataSet["params"] = Dict(
            ## run infos 
            "session_id" => session_id, "nfolds" =>nfolds, "modelid" =>  "$(bytes2hex(sha256("$(now())$(dataset_name)"))[1:Int(floor(end/3))])",
            "outpath"=>outpath, "machine_id"=>strip(read(`hostname`, String)), "device" => "$(device())",
            ## data infos 
            "dim_redux" => dim_redux_size, "nb_clinf" => nb_clinf, "dim_redux_type"=>dim_redux_type, 
            "dataset" => dataset_name,  "nsamples" => size(X_data)[1], "nsamples_test" => Int(round(size(X_data)[1] / nfolds)), 
            "ngenes" => size(X_data)[2] - nb_clinf, "nsamples_train" =>  size(X_data)[1] - Int(round(size(X_data)[1] / nfolds)),
            ## optim infos 
            "nepochs" => nepochs, "cph_lr" => cph_lr, "cph_wd" => cph_wd,
            ## model infos
            "model_type"=> modeltype, 
            "cph_nb_hl" => cph_nb_hl, "cph_hl_size" => hlsize
            ## metrics
    )
    # split folds 
    folds = split_train_test(X_data, DATA.survt, DATA.surve, DATA.samples;nfolds = nfolds)
    return folds
end 
lossfn_tr(MODEL, DS, cph_wd) = cox_nll_vec(MODEL, DS["data_prep"]["train_x"], DS["data_prep"]["train_y_e"], DS["data_prep"]["NE_frac_tr"]) + l2_penalty(MODEL) * cph_wd
lossfn_tst(MODEL, DS, cph_wd) = cox_nll_vec(MODEL, DS["data_prep"]["test_x"], DS["data_prep"]["test_y_e"], DS["data_prep"]["NE_frac_tst"]) + l2_penalty(MODEL) * cph_wd

lossfn_tr_dev(MODEL, DS, cph_wd, tr_size) = cox_nll_vec(MODEL, DS["data_prep"]["train_x"], DS["data_prep"]["train_y_e"], DS["data_prep"]["NE_frac_tr"]) / tr_size + l2_penalty(MODEL) * cph_wd
lossfn_tst_dev(MODEL, DS, cph_wd, tst_size) = cox_nll_vec(MODEL, DS["data_prep"]["test_x"], DS["data_prep"]["test_y_e"], DS["data_prep"]["NE_frac_tst"]) / tst_size + l2_penalty(MODEL) * cph_wd

function train_model!(MODEL, OPT, DS;foldn=0, print_step = 500)
    cph_wd = DS["params"]["cph_wd"]
    LOSS_TR, LOSS_TST = [],[]
    for i in 1:DS["params"]["nepochs"]
        cph_ps = Flux.params(MODEL)
        cph_gs = gradient(cph_ps) do 
            lossfn_tr(MODEL, DS, cph_wd)
        end 
        #meta_eval(cphdnn, tcga_datasets, base_params, verbose = i, verbose_step = 1)
        OUTS_tst = MODEL(DS["data_prep"]["test_x"])
        OUTS_tr = MODEL(DS["data_prep"]["train_x"])
        lossval_tr = round(lossfn_tr_dev(MODEL, DS, cph_wd, DS["params"]["nsamples_train"]), digits = 6)
        lossval_tst = round(lossfn_tst_dev(MODEL, DS, cph_wd, DS["params"]["nsamples_test"]), digits = 6)
        push!(LOSS_TR, lossfn_tr_dev(MODEL, DS, cph_wd, DS["params"]["nsamples_train"]))
        push!(LOSS_TST, lossfn_tst_dev(MODEL, DS, cph_wd, DS["params"]["nsamples_test"]))        
        cind_test,cdnt_tst, ddnt_tst, tied_tst = concordance_index(DS["data_prep"]["test_y_t"], DS["data_prep"]["test_y_e"], -1 * OUTS_tst)
        cind_tr, cdnt_tr, ddnt_tr, tied_tr  = concordance_index(DS["data_prep"]["train_y_t"],DS["data_prep"]["train_y_e"], -1 * OUTS_tr)
        Flux.update!(OPT, cph_ps, cph_gs)
        if i % print_step ==  0 || i == 1
            println("$(DS["params"]["dataset"]) $session_id - $(DS["params"]["model_type"]) - $(DS["params"]["dim_redux_type"]) $(DS["params"]["dim_redux"]) ($(size(DS["data_prep"]["train_x"])[1])) FOLD $foldn - $i : TRAIN c-ind: $(round(cind_tr, digits = 5)) ($lossval_tr)\tTEST c-ind: $(round(cind_test,digits =5)) ($lossval_tst)")
        end 
    end
    return LOSS_TR, LOSS_TST
end


function evaluate_cphdnn_debug(DS, dim_redux_size;hlsize = 512, nepochs= 5_000, cph_nb_hl = 2, cph_lr = 1e-6, 
    cph_wd = 1e-4, nfolds=5, modeltype = "cphdnn",dim_redux_type ="RDM", print_step = 500)
    folds = prep_data_params_dict!(DS, dim_redux_size, hlsize = hlsize, nepochs= nepochs, 
        cph_nb_hl = cph_nb_hl, cph_lr = cph_lr, cph_wd = cph_wd,  modeltype = modeltype,
        nfolds=nfolds, dim_redux_type =dim_redux_type)
    OUTS_TST, Y_T_TST, Y_E_TST, train_cinds, test_cinds = [], [], [], [],[]
    LOSSES_BY_FOLD  = []
    # for fold in folds do train
    fold = folds[1]
    train_x, train_y_t, train_y_e, NE_frac_tr, test_x, test_y_t, test_y_e, NE_frac_tst = format_train_test(fold)
    DS["data_prep"] = Dict("train_x"=>train_x, "train_y_t"=>train_y_t,"train_y_e"=>train_y_e,"NE_frac_tr"=>NE_frac_tr, "test_x"=>test_x,
    "test_y_t"=> test_y_t, "test_y_e"=>test_y_e, "NE_frac_tst"=> NE_frac_tst)
    DS["params"]["insize"] = size(train_x)[1]

    ## init model and opt
    OPT = Flux.ADAM(DS["params"]["cph_lr"]) 
    MODEL = gpu(Chain(Dense(DS["params"]["insize"],DS["params"]["cph_hl_size"], leakyrelu), 
    Dense(DS["params"]["cph_hl_size"], DS["params"]["cph_hl_size"], leakyrelu), 
    Dense(DS["params"]["cph_hl_size"], 1, sigmoid, bias = false)))
    
    # train loop 
    prev_m, curr_m = train_model_debug!(MODEL, OPT, DS; print_step=print_step, foldn = fold["foldn"])
    return prev_m, curr_m
end 

function train_model_debug!(MODEL, OPT, DS;foldn=0, print_step = 500)
    last_model = deepcopy(MODEL)
    # curr_model = deepcopy(MODEL) 
    cph_wd = DS["params"]["cph_wd"]
    OUTS_tst = MODEL(DS["data_prep"]["test_x"])
    OUTS_tr = MODEL(DS["data_prep"]["train_x"])
    lossval_tr = round(lossfn_tr_dev(MODEL, DS, cph_wd, DS["params"]["nsamples_train"]), digits = 4)
    lossval_tst = round(lossfn_tst_dev(MODEL, DS, cph_wd, DS["params"]["nsamples_test"]), digits = 4)
    cind_test,cdnt_tst, ddnt_tst, tied_tst = concordance_index(DS["data_prep"]["test_y_t"], DS["data_prep"]["test_y_e"], -1 * OUTS_tst)
    cind_tr, cdnt_tr, ddnt_tr, tied_tr  = concordance_index(DS["data_prep"]["train_y_t"],DS["data_prep"]["train_y_e"], -1 * OUTS_tr)
    println("$(DS["params"]["dataset"]) $session_id - $(DS["params"]["model_type"]) - $(DS["params"]["dim_redux_type"]) $(DS["params"]["dim_redux"]) ($(size(DS["data_prep"]["train_x"])[1])) FOLD $foldn - 0 : TRAIN c-ind: $(round(cind_tr, digits = 3)) ($lossval_tr)\tTEST c-ind: $(round(cind_test,digits =5)) ($lossval_tst)")
    LOSS_TR, LOSS_TST = [],[]
    push!(LOSS_TR, lossfn_tr_dev(MODEL, DS, cph_wd, DS["params"]["nsamples_train"]))
    push!(LOSS_TST, lossfn_tst_dev(MODEL, DS, cph_wd, DS["params"]["nsamples_test"]))        
    cind_test,cdnt_tst, ddnt_tst, tied_tst = concordance_index(DS["data_prep"]["test_y_t"], DS["data_prep"]["test_y_e"], -1 * OUTS_tst)
    for i in 1:DS["params"]["nepochs"]
        last_model = deepcopy(MODEL)
        cph_ps = Flux.params(MODEL)
        cph_gs = gradient(cph_ps) do 
            lossfn_tr(MODEL, DS, cph_wd)
        end 
        Flux.update!(OPT, cph_ps, cph_gs)
        #meta_eval(cphdnn, tcga_datasets, base_params, verbose = i, verbose_step = 1)
        OUTS_tst = MODEL(DS["data_prep"]["test_x"])
        OUTS_tr = MODEL(DS["data_prep"]["train_x"])
        lossval_tr = round(lossfn_tr_dev(MODEL, DS, cph_wd, DS["params"]["nsamples_train"]), digits = 4)
        lossval_tst = round(lossfn_tst_dev(MODEL, DS, cph_wd, DS["params"]["nsamples_test"]), digits = 4)
        push!(LOSS_TR, lossfn_tr_dev(MODEL, DS, cph_wd, DS["params"]["nsamples_train"]))
        push!(LOSS_TST, lossfn_tst_dev(MODEL, DS, cph_wd, DS["params"]["nsamples_test"]))        
        cind_test,cdnt_tst, ddnt_tst, tied_tst = concordance_index(DS["data_prep"]["test_y_t"], DS["data_prep"]["test_y_e"], -1 * OUTS_tst)
        cind_tr, cdnt_tr, ddnt_tr, tied_tr  = concordance_index(DS["data_prep"]["train_y_t"],DS["data_prep"]["train_y_e"], -1 * OUTS_tr)
        if i % print_step ==  0 || i == 1
            println("$(DS["params"]["dataset"]) $session_id - $(DS["params"]["model_type"]) - $(DS["params"]["dim_redux_type"]) $(DS["params"]["dim_redux"]) ($(size(DS["data_prep"]["train_x"])[1])) FOLD $foldn - $i : TRAIN c-ind: $(round(cind_tr, digits = 3)) ($lossval_tr)\tTEST c-ind: $(round(cind_test,digits =5)) ($lossval_tst)")
        end 
        if ~(lossval_tr == lossval_tr) || ~(cind_tr == cind_tr)
            return last_model, deepcopy(MODEL)
        end 
    end
    return last_model, deepcopy(MODEL)
end


function dump_results!(DS, LOSSES_BY_FOLD, OUTS_TST, Y_T_TST, Y_E_TST, train_cinds, test_cinds)
    med_c_ind, lo_ci, up_ci, c_indices= bootstrap_c_ind(OUTS_TST, Y_T_TST, Y_E_TST)
    DS["params"]["cph_tst_c_ind_med"] = med_c_ind
    DS["params"]["cph_tst_c_ind_up_ci"] = up_ci
    DS["params"]["cph_tst_c_ind_lo_ci"] = lo_ci
    DS["params"]["cph_train_c_ind"] = mean(train_cinds)
    DS["params"]["cph_test_c_ind"] = mean(test_cinds)
    DS["params"]["model_cv_complete"] = true
    DS["params"]["model_title"] = "$(DS["params"]["dataset"])_$(DS["params"]["model_type"])_$(DS["params"]["dim_redux_type"])_$(DS["params"]["insize"])_$(DS["params"]["nb_clinf"])CF"
            
    println("TEST bootstrap c-index : $(med_c_ind) ($up_ci - $lo_ci 95% CI)")
    println("TEST average c-index ($(DS["params"]["nfolds"]) folds): $(round(DS["params"]["cph_test_c_ind"],digits=3))")
    
    ### LOSSES 
    nsteps = size(LOSSES_BY_FOLD[1][1])[1]
    steps = vcat([vcat(collect(1:nsteps), collect(1:nsteps)) for i in 1:size(LOSSES_BY_FOLD)[1]]...)
    model_ids = ["$(DS["params"]["modelid"])" for x in 1:nsteps*2*size(LOSSES_BY_FOLD)[1]];
    foldns = vcat([(ones(nsteps * 2 ) * i) for i in 1:size(LOSSES_BY_FOLD)[1]]...)
    tst_train = vcat([vcat(["train" for i in 1:nsteps], ["test" for i in 1:nsteps]) for i in 1:size(LOSSES_BY_FOLD)[1]]...);
    loss_vals = Float64.(vcat([vcat(LOSSES[1], LOSSES[2]) for LOSSES in LOSSES_BY_FOLD]...));
    LOSSES_dict = Dict("modelid" => model_ids, "foldns" => foldns, "steps" => steps, "tst_train"=> tst_train, "loss_vals"=>loss_vals)

    outfpath = "$(DS["params"]["model_title"])_$(DS["params"]["modelid"])"
    mkdir("$(DS["params"]["outpath"])/$outfpath")
    bson("$(DS["params"]["outpath"])/$outfpath/params.bson",DS["params"])
    bson("$(DS["params"]["outpath"])/$outfpath/learning_curves.bson", LOSSES_dict)
    return c_indices
end 

function evaluate_coxridge(DS, dim_redux_size;hlsize = 0, nepochs= 5_000, cph_nb_hl = 0, cph_lr = 1e-4, 
        cph_wd = 1e-4, nfolds =5, modeltype = "coxridge",dim_redux_type ="RDM", print_step = 500)
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
        MODEL = gpu(Chain(Dense(DS["params"]["insize"], 1, sigmoid, bias = false)))
        
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

function evaluate_cphdnn(DS, dim_redux_size;hlsize = 512, nepochs= 5_000, cph_nb_hl = 2, cph_lr = 1e-6, 
    cph_wd = 1e-4, nfolds=5, modeltype = "cphdnn",dim_redux_type ="RDM", print_step = 500)
    folds = prep_data_params_dict!(DS, dim_redux_size, hlsize = hlsize, nepochs= nepochs, 
        cph_nb_hl = cph_nb_hl, cph_lr = cph_lr, cph_wd = cph_wd,  modeltype = modeltype,
        nfolds=nfolds, dim_redux_type =dim_redux_type)
    OUTS_TST, Y_T_TST, Y_E_TST, train_cinds, test_cinds = [], [], [], [],[]
    LOSSES_BY_FOLD  = []
    # for fold in folds do train
    for fold in folds
        
        train_x, train_y_t, train_y_e, NE_frac_tr, test_x, test_y_t, test_y_e, NE_frac_tst = format_train_test(fold)
        DS["data_prep"] = Dict("train_x"=>train_x, "train_y_t"=>train_y_t,"train_y_e"=>train_y_e,"NE_frac_tr"=>NE_frac_tr, "test_x"=>test_x,
        "test_y_t"=> test_y_t, "test_y_e"=>test_y_e, "NE_frac_tst"=> NE_frac_tst)
        DS["params"]["insize"] = size(train_x)[1]

        ## init model and opt
        OPT = Flux.ADAM(DS["params"]["cph_lr"]) 
        MODEL = gpu(Chain(Dense(DS["params"]["insize"],DS["params"]["cph_hl_size"], leakyrelu), 
        Dense(DS["params"]["cph_hl_size"], DS["params"]["cph_hl_size"], leakyrelu), 
        Dense(DS["params"]["cph_hl_size"], 1, sigmoid,  bias = false)))
        
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

        DS["params"]["nparams"] = sum([*(size(x.weight)...) for x in MODEL]) +  DS["params"]["cph_nb_hl"] * DS["params"]["cph_hl_size"]
    end 
    c_indices_tst = dump_results!(DS, LOSSES_BY_FOLD, OUTS_TST, Y_T_TST, Y_E_TST, train_cinds, test_cinds)
    return c_indices_tst
end 


function evaluate_coxridge_pca(DS, dim_redux_size;hlsize = 0, nepochs= 5_000, cph_nb_hl = 0, cph_lr = 1e-4, 
        cph_wd = 1e-4, nfolds =5, modeltype = "coxridge",dim_redux_type ="PCA", print_step = 500)
    folds = prep_data_params_dict!(DS, dim_redux_size, hlsize = hlsize, nepochs= nepochs, 
        cph_nb_hl = cph_nb_hl, cph_lr = cph_lr, cph_wd = cph_wd, nfolds = nfolds,  modeltype = modeltype,
        dim_redux_type =dim_redux_type)
    OUTS_TST, Y_T_TST, Y_E_TST, train_cinds, test_cinds = [], [], [], [],[]
    LOSSES_BY_FOLD = []
    # for fold in folds do train
    for fold in folds
        # format data on GPU 
        train_x, train_y_t, train_y_e, NE_frac_tr, test_x, test_y_t, test_y_e, NE_frac_tst = format_train_test(fold)
        DS["data_prep"] = Dict("train_x"=>train_x, "train_y_t"=>train_y_t,"train_y_e"=>train_y_e,"NE_frac_tr"=>NE_frac_tr, "test_x"=>test_x,
        "test_y_t"=> test_y_t, "test_y_e"=>test_y_e, "NE_frac_tst"=> NE_frac_tst)
        # fit PCA 
        # fit PCA 
        P = fit_pca(cpu(train_x), dim_redux_size);
        train_x = gpu(Matrix(hcat(transform_pca(cpu(train_x), P)', DS["CF"][fold["train_ids"],:])'))
        test_x = gpu(Matrix(hcat(transform_pca(cpu(test_x), P)', DS["CF"][fold["test_ids"],:])'))
        DS["params"]["insize"] = size(train_x)[1] + DS["params"]["nb_clinf"]
        DS["data_prep"] = Dict("train_x"=>train_x, "train_y_t"=>train_y_t,"train_y_e"=>train_y_e,"NE_frac_tr"=>NE_frac_tr, "test_x"=>test_x,
        "test_y_t"=> test_y_t, "test_y_e"=>test_y_e, "NE_frac_tst"=> NE_frac_tst)

        ## init model 
        ## init model and opt
        OPT = Flux.ADAM(DS["params"]["cph_lr"]) 
        MODEL = gpu(Chain(Dense(DS["params"]["insize"], 1, sigmoid, bias = false)))
        
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

function evaluate_cphdnn_pca(DS, dim_redux_size;hlsize = 512, nepochs= 5_000, cph_nb_hl = 2, cph_lr = 1e-6, 
        cph_wd = 1e-4, nfolds =5, modeltype = "cphdnn",dim_redux_type ="PCA", print_step = 500)
    folds = prep_data_params_dict!(DS, dim_redux_size, hlsize = hlsize, nepochs= nepochs, 
        cph_nb_hl = cph_nb_hl, cph_lr = cph_lr, cph_wd = cph_wd, nfolds = nfolds,  modeltype = modeltype,
        dim_redux_type =dim_redux_type)
    OUTS_TST, Y_T_TST, Y_E_TST, train_cinds, test_cinds = [], [], [], [],[]
    LOSSES_BY_FOLD = []
    # for fold in folds do train
    for fold in folds
    # format data on GPU 
    train_x, train_y_t, train_y_e, NE_frac_tr, test_x, test_y_t, test_y_e, NE_frac_tst = format_train_test(fold)
    DS["data_prep"] = Dict("train_x"=>train_x, "train_y_t"=>train_y_t,"train_y_e"=>train_y_e,"NE_frac_tr"=>NE_frac_tr, "test_x"=>test_x,
    "test_y_t"=> test_y_t, "test_y_e"=>test_y_e, "NE_frac_tst"=> NE_frac_tst)
    # fit PCA 
    P = fit_pca(cpu(train_x), dim_redux_size);
    train_x = gpu(Matrix(hcat(transform_pca(cpu(train_x), P)', DS["CF"][fold["train_ids"],:])'))
    test_x = gpu(Matrix(hcat(transform_pca(cpu(test_x), P)', DS["CF"][fold["test_ids"],:])'))
    DS["params"]["insize"] = size(train_x)[1] + DS["params"]["nb_clinf"]
    DS["data_prep"] = Dict("train_x"=>train_x, "train_y_t"=>train_y_t,"train_y_e"=>train_y_e,"NE_frac_tr"=>NE_frac_tr, "test_x"=>test_x,
    "test_y_t"=> test_y_t, "test_y_e"=>test_y_e, "NE_frac_tst"=> NE_frac_tst)

    ## init model and opt
    OPT = Flux.ADAM(DS["params"]["cph_lr"]) 
    MODEL = gpu(Chain(Dense(DS["params"]["insize"],DS["params"]["cph_hl_size"], leakyrelu), 
        Dense(DS["params"]["cph_hl_size"], DS["params"]["cph_hl_size"], leakyrelu), 
        Dense(DS["params"]["cph_hl_size"], 1, sigmoid, bias = false)))
        
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
    DS["params"]["nparams"] = size(DS["dataset"].data)[2]
    end 
    c_indices_tst = dump_results!(DS, LOSSES_BY_FOLD, OUTS_TST, Y_T_TST, Y_E_TST, train_cinds, test_cinds)
    return c_indices_tst
end