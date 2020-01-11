using Knet

include("data_preparation.jl")

global atype = gpu() >= 0 ? KnetArray{Float32} :  Array{Float32}
const ε = Float32(1e-8)
IMAGE_ZOOM = 0.25
IMAGE_SIZE = 64  # e.g. 64 stands for 64x64 pixels (always square images)
IMAGE_CHANNELS = 3
EPOCHS = 10^4
BATCH_SIZE = 100
SEED = 123
SAVE_FREQ = 100  # ? is it necessary?
ALPHA = 0.2
LEARNING_RATE = 0.0002  # for optimizer (Adam)
BETA1 = 0.5  # Adam parametetrs for bias corrected moments
NOISE_DIM = 100

xtrn,ytrn,_,_= mnist()

xtrn[:,:,1,1]
xtrn
dtrn = minibatch(lr, hr, 100; shuffle=true, xtype=Array{Float32})

###### main
wd, wg, md, mg = load_weights

function D()
end

function G()
end

function init_weights(atype=atype, noise_dim=NOISE_DIM, winit)
    w = Any[]
    m = Any[]
end


function read_weights()
end


function save_weights()
end


function modified_loss()
end

function train_discriminator!()
end

function train_generator!()
end
