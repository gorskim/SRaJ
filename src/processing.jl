using Flux
using Distributions
using NNlib: leakyrelu

include("data_preparation.jl")

CuArrays.allowscalar(false)
@info "Scalar operations for CuArrays ale disabled."

# constants
α = 0.2f0  # leakyReLU activation
η = 10^(-4) # learning rate for optimizer (Adam)
β1, β2 = 0.9f0, 0.999f0  # Adam parametetrs for bias corrected moments
ϵ = 10f-10

# one-liners

load_image(filename) = Float32.(channelview(RGB.(load(filename))))
get_images_names(HR_path::String, LR_path::String) = [name for name in readdir(HR_path)[1:3000]],
                                                     [name for name in readdir(LR_path)[1:3000]]
bin_cross_entropy(ŷ, y) = -y .* log.(ŷ .+ ϵ) -
                          (1 .- y) .* log.(1 .- ŷ .+ ϵ)
normalize(x) = convert(CuArray{Float32}, 2.0f0 .* x .- 1.0f0)
denormalize(x) = convert(CuArray{Float32}, ((x .+ 1.0f0) ./ 2.0f0))
squeeze_dims(x) = dropdims(x, dims=tuple(findall(size(x) .== 1)...))
wrap_batchnorm(out_ch) = Chain(x -> expand_dims(x, 2),
                         BatchNorm(out_ch),
                         x -> squeeze_dims(x))
expand_dims(x, n::Int) = reshape(x, ones(Int64, n)..., size(x)...)
flatten(x) = reshape(x, prod(size(x)[1:end-1]), size(x)[end])
same_padding(in_dim::Int, k::Int, s::Int) = Int(0.5 * ((in_size) - 1) * s + k - in_size)


initialize_weights(shape...) = map(Float32, rand(Normal(0, 0.02f0), shape...))
optimizer = ADAM(η, (β1, β2))

# util functions and layers
function load_vgg()
    vgg = VGG19() |> gpu  # Metalhead.jl
    vgg = Chain(vgg.layers[1:20]...) # triple dots include Dropout, Dense, Softmax
    @info "VGG net loaded."
    vgg
end


# PReLU
mutable struct PReLU{T}
    α::T
end

@treelike PReLU

PReLU(img_channels::Int; init=Flux.glorot_uniform) = PReLU(param(init(img_channels)))

function (m::PReLU)(x)
	channels_count = size(x)[end - 1]
	if channels_count == length(m.α)
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

function _phase_shift(x, r)
	# https://arxiv.org/pdf/1609.05158.pdf
    w, h, c, n = size(x)
    x = reshape(x, w, h, r, r, n)
    x = [x[i, :, :, :, :] for i in 1:w]
    x = cat([t for t in x]..., dims=2)
    x = [x[i,:,:,:] for i in 1:size(x)[1]]
    x = cat([t for t in x]..., dims=2)
    x
end

function _shuffle_pixels(x, r=3)
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


# discriminator definition
_dconv(in_size::Int, out_size::Int, k=3, s=1, p=1) =
	Chain(Conv((k, k), in_size=>out_size, stride=(s,s), pad=(p, p); init=initialize_weights), x -> leakyrelu.(x, α))

_dconvBN(in_size::Int, out_size::Int, k=3, s=1, p=1) =
	Chain(Conv((k, k), in_size=>out_size, stride=(s,s), pad=(p ,p); init=initialize_weights), wrap_batchnorm(out_size)...,
		  x -> leakyrelu.(x, α))

function Discriminator()
	Chain(_dconv(3, 64, 3, 1),
		  _dconvBN(64, 64, 3, 2),
		  _dconvBN(64, 128, 3, 1),
		  # _dconvBN(128, 128, 3, 2),
		  _dconvBN(128, 256, 3, 1),
		  # _dconvBN(256, 256, 3, 2),
		  _dconvBN(256, 512, 3, 1),
		  _dconvBN(512, 512, 3, 2),
		  x -> flatten(x),
		  Dense(8 * 8 * 512, 1024),
		  x -> leakyrelu.(x, α),
		  Dense(1024, 1),
		  x -> σ.(x))
end


# generator definition
_gconv(in_size::Int, out_size::Int, k=3, s=1, p=1) =
	Chain(Conv((k, k), in_size=>out_size, stride=(s, s), pad=(p, p); init=initialize_weights))

_gconvBN(in_size::Int, out_size::Int, k=3, s=1, p=1) =
	Chain(Conv((k, k), in_size=>out_size, stride=(s, s), pad=(p, p); init=initialize_weights), wrap_batchnorm(out_size)...)

_conv_block(in_size=64, out_size=64, k=3, s=1, p=1) =
	Chain(Conv((k, k), in_size=>out_size, stride=(s, s), pad=(p, p); init=initialize_weights), wrap_batchnorm(out_size)..., PReLU(out_size))

mutable struct ResidualBlock
	conv_blocks
end

@treelike ResidualBlock

ResidualBlock() = ResidualBlock((_conv_block(), _gconvBN(64, 64)))

function (m::ResidualBlock)(x)
	res = m.conv_blocks[1](x)
	out = m.conv_blocks[2](res)
	out .+ x
end

_upsample_block(in_size::Int, out_size::Int) =
	Chain(_gconv(in_size, out_size, 3, 1),
		  x->_shuffle_pixels(x, 2),
		  PReLU(div(out_size, UP_FACTOR)))

mutable struct Generator
	conv_initial
	residual_blocks
	conv_blocks
	upsample_blocks
end

@treelike Generator

function Gen(blocks::Int)
	conv_initial = Chain(_gconv(3, 64, 9, 1, 4), PReLU(64))

	residual_blocks = []
	for block in 1:blocks
		push!(residual_blocks, ResidualBlock())
	end

	residual_blocks = tuple(residual_blocks...)
	conv_blocks = (_gconvBN(64, 64), _gconv(64, 3, 9, 1, 4))
	upsample_blocks = (_upsample_block(64, 256), _upsample_block(64, 256))
	Generator(conv_initial, residual_blocks, conv_blocks, upsample_blocks)
end

function (gen::Generator)(x)
	x = gen.conv_initial(x)
	x_initial_conv = x

	for residual_block in gen.residual_blocks
		x = residual_block(x)
	end

	x = gen.conv_blocks[1](x)
	x = x .+ x_initial_conv

	@info "block upsampling"
	for upsample_block in gen.upsample_blocks
		x = upsample_block(x)
	end
	@info "upsampling done"
	x = gen.conv_blocks[2](x)
	tanh.(x)
end
