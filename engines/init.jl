# module Init
using Pkg
Pkg.activate(".")
# imports
#using cuDNN  
using CSV
using DataFrames
using JSON
using HDF5
using Statistics
using Flux 
using CairoMakie 
using Random
using Dates
using AlgebraOfGraphics
using CUDA
using SHA
using BSON 
using Distributions
using TSne
using MultivariateStats
using XLSX
using UMAP 
using LinearAlgebra

# using JuBox 
function set_dirs(basedir="./RES")
    session_id = "$(now())"
    outpath = "$basedir/$session_id"
    mkdir(outpath)

    return outpath, session_id
end

# end 