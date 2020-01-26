using Pkg
for p in ("Flux", "Plots")
    haskey(Pkg.installed(), p) || Pkg.add(p)
end

using ArgParse, Knet

include("processing.jl")

# addparser

function main(model_path::String)
    @load model_path weights
    Flux.loadparams!(generator, weights)
end
