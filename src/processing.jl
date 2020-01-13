using Flux, ImageQualityIndexes



include("data_preparation.jl")

global atype = gpu() >= 0 ? KnetArray{Float32} :  Array{Float32}
const ε = Float32(1e-8)
IMAGE_ZOOM = 0.25
IMAGE_SIZE = 64  # e.g. 64 stands for 64x64 pixels (always square images)
IMAGE_CHANNELS = 3
EPOCHS = 10^4
BATCH_SIZE = 100
SEED = 123
SAVE_FREQ = 100  # ?
α = 0.2
η = 10^(-4) # learning rate for optimizer (Adam)
β1, β2 = 0.9, 0.999  # Adam parametetrs for bias corrected moments
NOISE_DIM = 100


# one-liners
load_image(filename) = Float32.(channelview(load(filename)))
get_images_names(HR_path::String, LR_path::String) = [name for name in readdir(HR_path)],
                                                     [name for name in readdir(LR_path)]
bin_cross_entropy(ŷ, y) = -y .* log.(ŷ .+ 1f-10) -
                          (1  .- y) .* log.(1 .- ŷ .+ 1f-10)  # SPRAWDZ!
load_generator() = generator(blocks_count) |> gpu
load_discriminator() = discriminator() |> gpu
wrap_batchnorm(out_ch) = Chain(x -> expand_dims(x, 2),
                         BatchNorm(out_ch),
                         x -> squeeze(x))
normalize(x) = convert(CuArray{Float32}, 2.0 .* x .- 1.0)
denormalize(x) = convert(CuArray{Float32}, ((x .+ 1.0) ./ 2.0))
squeeze_dims(x) = dropdims(x, dims=tuple(findall(size(x) .== 1)...))
expand_dims(x, n::Int) = reshape(x, ones(Int64, n)..., size(x)...)
flatten(x) = reshape(x, prod(size(x)[1:end-1]), size(x)[end])
optimizer = ADAM(η, (β1, β2))


function load_vgg()
    vgg = VGG19() |> gpu  # Metalhead.jl
    vgg = Chain(vgg.layers[1:20]...) # triple dots include Dropout, Dense, Softmax
    @info "VGG net loaded."
    vgg
end

function discriminator()
end


function dloss()
end


function generator()
end


function gloss()
end

function save_model()
end
