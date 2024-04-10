import h5py 
import numpy as np
import pdb
import lifelines
import pandas as pd 
# BRCA data
nfolds = 5 
BRCA_data = h5py.File("../Data/TCGA_OV_BRCA_LGG/TCGA_BRCA_tpm_n1049_btypes_labels_surv.h5", "r")
X_data = np.array(BRCA_data["data"][:,:])
biotypes = np.array(BRCA_data["biotypes"][:], dtype = str)
survt = np.array(BRCA_data["survt"][:])
surve = np.array(BRCA_data["surve"][:])
cds = biotypes == "protein_coding"
X_data = X_data[cds,:]


#### PCA transformation
def fit_pca(X, outdim):
    x_means = X.mean(1)
    Z = X - x_means.reshape(x_means.shape[0],1)
    U, S, Vh = np.linalg.svd(Z, full_matrices = True, compute_uv=True)
    return U[:,np.argsort(S)[::-1][:outdim]].T

def transform_pca(X, P):
    x_means = X.mean(1)
    Z = X - x_means.reshape(x_means.shape[0],1)
    return np.matmul(P, Z)

def train_coxridge_pca(X_train, Y_t_train, Y_e_train,
                       X_test, Y_t_test, Y_e_test):    
    print("Fitting PCA ...")
    train_size = X_train.shape[1]
    dim_redux = 30
    P = fit_pca(X_train, dim_redux)
    X_tr_pca = transform_pca(X_train, P)
    X_tst_pca = transform_pca(X_test, P)

    train_ds = pd.DataFrame(X_tr_pca.T)
    train_ds["T"] = Y_t_train
    train_ds["E"] = Y_e_train
    cph = lifelines.CoxPHFitter(penalizer = 1e-6,l1_ratio=0.)
    print("Fitting CPH ...")
    model = cph.fit(train_ds, duration_col="T", event_col="E")
    
    tr_c_ind = lifelines.utils.concordance_index(Y_t_train, -model.predict_log_partial_hazard(X_tr_pca.T), Y_e_train)
    tst_c_ind = lifelines.utils.concordance_index(Y_t_test, -model.predict_log_partial_hazard(X_tst_pca.T), Y_e_test)
    
    print(f"FOLD{fold_id + 1} DREDUX: {dim_redux} TRAIN c-ind {tr_c_ind} TEST c-ind {tst_c_ind}")
    return tst_c_ind

def split_train_test(X_data, Y_t, Y_e, nfolds):
    nsamples = X_data.shape[1]
    ids = np.arange(nsamples)
    np.random.shuffle(ids)
    fold_size = int(np.ceil(nsamples / nfolds))
    folds = []
    for fold_id in range(nfolds):
        test_ids = ids[fold_id * fold_size: min((fold_id + 1) * fold_size, nsamples - 1)]
        train_ids = np.setdiff1d(ids, test_ids)
        folds.append([X_data[:,train_ids], Y_t[train_ids], Y_e[train_ids],
                      X_data[:,test_ids], Y_t[test_ids], Y_e[test_ids]])  
    return folds 

#### repeat 10 times.
#### split train test 5-fold
c_inds_rep =  []
for rep_i in range(10):
    folds = split_train_test(X_data, survt, surve, nfolds)
    c_inds = []
    for fold_id, fold in enumerate(folds):
        print(f"REP {rep_i + 1} FOLD {fold_id + 1}")
        c_ind = train_coxridge_pca(fold[0], fold[1], fold[2],
                                fold[3], fold[4], fold[5])
        c_inds.append(c_ind)
    print(f"REP{rep_i +1 } {c_inds} - mean {np.mean(c_inds)}")
    c_inds_rep.append(np.mean(c_inds))
print(f"{c_inds_rep} - mean {np.mean(c_inds_rep)}")
pdb.set_trace()
resdf = pd.DataFrame(dict([("replicate",np.arange(10)),("c_ind-average", c_inds_rep)]))
resdf.to_csv("../figures/python_lifelines_brca_pca30_coxridge.csv")
#### DOES NOT WORK, model does not converge under current parameters
def train_coxridge_cds(X_data, survt, surve):
    train_ds = pd.DataFrame(X_data.T)
    train_ds["T"] = survt 
    train_ds["E"] = surve
    cph = lifelines.CoxPHFitter(penalizer = 1e-3,l1_ratio=0.)
    model = cph.fit(train_ds, duration_col="T", event_col="E")
    c_ind = lifelines.utils.concordance_index(survt, model.predict_log_partial_hazard(X_data.T), surve)
    print(c_ind)

