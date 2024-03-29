include("engines/init.jl")
include("engines/data_processing.jl")
include("engines/deep_learning.jl")
include("engines/cross_validation.jl")
include("engines/model_evaluation.jl")
outpath, session_id = set_dirs("RES_FIG1") ;
## import Leucegene, BRCA

##### Different datasets - dim-redux.
## Leucegene AML / BRCA - RDM / LSC17 / PCA 

##### Explored Features 
## CLINF ON/OFF
## Modeltype DNN/Cox-ridge
## Dim redux size 0-ngenes

##### HOW 
## 5-fold cross-validated c-index (95% CI)
## Learning curves => overfitting
## Survival curves

#tcga_datasets_list = ["Data/TCGA_datasets/$(x)" for x in readdir("Data/TCGA_datasets") ]
#tcga_datasets_list = ["Data/TCGA_datasets/$(x)" for x in readdir("Data/TCGA_test") ]
#DataDict = load_tcga_datasets(tcga_datasets_list)["BRCA"]
#clinical_factors = Matrix(CSV.read("Data/GDC_processed/TCGA_BRCA_clinical_bin.csv", DataFrame))
lgn_CF = CSV.read("Data/LEUCEGENE/lgn_pronostic_CF", DataFrame)
# clinical_factors = Matrix(tmp_df)
tcga_datasets_list = ["Data/TCGA_datasets/$(x)" for x in readdir("Data/TCGA_OV_BRCA_LGG/") ]
TCGA_datasets = load_tcga_datasets(tcga_datasets_list);
BRCA_data = TCGA_datasets["BRCA"]
LGG_data = TCGA_datasets["LGG"]
OV_data = TCGA_datasets["OV"]

LGNAML_data = Dict("name"=>"LgnAML","dataset" => MLSurvDataset("Data/LEUCEGENE/LGN_AML_tpm_n300_btypes_labels_surv.h5")) 
keep_tcga_cds = [occursin("protein_coding", bt) for bt in BRCA_data["dataset"].biotypes]
keep_lgnaml_common = [gene in BRCA_data["dataset"].genes[keep_tcga_cds] for gene in LGNAML_data["dataset"].genes];

BRCA_data["CDS"] = keep_tcga_cds
BRCA_data["CF"] = zeros(size(BRCA_data["dataset"].data)[1],0)  
LGG_data["CDS"] = keep_tcga_cds  
LGG_data["CF"] = zeros(size(LGG_data["dataset"].data)[1],0) 
OV_data["CDS"] = keep_tcga_cds  
OV_data["CF"] = zeros(size(OV_data["dataset"].data)[1],0) 

LGNAML_data["CDS"] = keep_lgnaml_common 
LGNAML_data["CF"] = zeros(size(LGNAML_data["dataset"].data)[1],0) 

### Use LGG, BRCA, AML, OV 
### Use random vs pca 
### Use CPHDNN, Cox-ridge 
### 10 replicates
nepochs, nfolds = 5000, 5
DS_list = shuffle([BRCA_data, LGNAML_data, LGG_data, OV_data]);
# DataSet = LGG_data
# train_size = size(DataSet["dataset"].data)[1] - Int(round(size(DataSet["dataset"].data)[1]    / nfolds))
# evaluate_cphdnn_pca(LGG_data, train_size, nepochs=nepochs, cph_wd= 1e-2);
# evaluate_coxridge_pca(LGG_data, train_size, nepochs=nepochs, cph_lr = 1e-4);
    
for DataSet in DS_list

    

    ### EVAL WITH PCA 
    train_size = size(DataSet["dataset"].data)[1] - Int(round(size(DataSet["dataset"].data)[1]    / nfolds))
    #dim_redux_list = shuffle(vcat(collect(20:20:train_size-20), train_size))
    dim_redux_list = shuffle(collect(1:19))
    for dim_redux_size in dim_redux_list
        evaluate_cphdnn_pca(DataSet, dim_redux_size, nepochs=nepochs, cph_wd= 1e-2);
        evaluate_coxridge_pca(DataSet, dim_redux_size, nepochs=nepochs, cph_lr = 1e-4);
    end  

    ### EVAL WITH RANDOM SIGN
    # ngenes = sum(DataSet["CDS"])
    # dim_redux_list = shuffle([1,2,3,4,5,10,15,20,25,50,75,100,125,250,375,500,1_000,1_500,2_000,2_500,3000,5_000,6000,7000,8000,9000,10_000,11_000,12000,13000,14000,ngenes])
    # for dim_redux_size in dim_redux_list
    #     evaluate_cphdnn(DataSet, dim_redux_size, nepochs =nepochs, cph_wd =1e-2);
    #     evaluate_coxridge(DataSet, dim_redux_size, nepochs =nepochs, cph_lr = 1e-5);
    # end

    # ### EVAL with genes ranked by variance
    # dim_redux_list = shuffle([1,2,3,4,5,10,15,20,25,50,75,100,125,250,375,500,1_000,1_500,2_000,2_500,3000,5_000,6000,7000,8000,9000,10_000,11_000,12000,13000,14000,ngenes])
    # for dim_redux_size in dim_redux_list
    #     evaluate_cphdnn(DataSet, dim_redux_size, nepochs=nepochs, dim_redux_type="STD", cph_wd= 1e-2);
    #     evaluate_coxridge(DataSet, dim_redux_size, nepochs=nepochs, dim_redux_type="STD", cph_lr = 1e-4);
    # end
      
end 
