using Pkg
Pkg.activate(".")
Pkg.instantiate()

using Images, ImageMagick, Colors
using Flux, Tracker
# using CuArrays
using BSON: @load

include("src/processing.jl")


model_path = "models/model20.bson"
@load model_path model


function load_LR_image(filepath)
    img = [load_image(filepath)]
    out = reshape(cat(img..., dims=1), 32, 32, 3, 1)
    out
end


function save_SR_image(img, image_name)
    @info "$(size(img))"
    SR = img[:, :, :, 1]
    SR = reshape(SR, 3, size(img)[1], size(img)[2])
    out = colorview(RGB, SR[1,:,:], SR[2,:,:], SR[3,:,:])
    out_name = "SR-$(image_name)"
    save(out_name, out)
end


function enhance(image_name)
    LR = load_LR_image(image_name)
    SR = model(LR).data
    save_SR_image(SR, image_name)
end
