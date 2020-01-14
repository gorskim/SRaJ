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

# util functions and layers
function load_vgg()
    vgg = VGG19() |> gpu  # Metalhead.jl
    vgg = Chain(vgg.layers[1:20]...) # triple dots include Dropout, Dense, Softmax
    @info "VGG net loaded."
    vgg
end


# PRelu
mutable struct PRelu{T}
    α::T
end

@treelike PRelu

PRelu(img_channels::Int; init=Flux.glorot_uniform) = PRelu(param(init(img_channels)))

function (m::PRelu)(x)
	if size(x)[end - 1] == length(m.α)
		return max.(0.0f0, x) .+
			  (reshape(m.α, ones(Int64, length(size(x)) - 2)...,
	                   length(m.α), 1) .* min.(0.0f0, x))
    else
		error("Dimensions mismatch.\n
		       length of α: $(length(m.α))\nnumber of channels: $(size(x)[end-1])")
	end
end


# pixel shuffler
function _partition_channels(x::AbstractArray, n::Int)
	# see PR157 for model-zoo
    indices = collect(1:size(x)[end - 1])
    partitioned = partition(indices, div(size(x)[end-1], n))
    partitions = []
    for c in partitioned
       c = [c_ for c_ in c]
       push!(partitions, x[:, :, c, :])
    end
    partitions
end

function _phase_shift(x, r),
	# https://arxiv.org/pdf/1609.05158.pdf
    w, h, c, n = size(x)
    x = reshape(x, W, H, r, r, N)
    x = [x[i, :, :, :, :] for i in 1:w]
    x = cat([t for t in x]..., dims=2)
    x = [x[i,:,:,:] for i in 1:size(x)[1]]
    x = cat([t for t in x]..., dims=2)
    x
end

function shuffle_pixels(x, r=3)
	dims_expected = 4
	length(size(x)) == 4 ||
		error("Dimensions mismatch.\nexpected: $dims_expected\ngot:$length(size(x))")
	size(x)[end - 1] % (r^2) == 0 ||
		error("Number of channels is not divisable by r^2")
    c_out = div(size(x)[end-1], r^2)
    p = _partition_channels(x, c_out)
    out = cat([_phase_shift(c, r) for c in p]..., dims=3)
    reshape(out, size(out)[1], size(out)[2], c_out, div(size(out)[end], c_out))
end


# networks definition
function discriminator()
end

function generator()
end


# losses definition
function dloss()
end

function gloss()
end
