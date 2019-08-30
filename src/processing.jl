using Knet, Colors, Images, Statistics

include("data_preparation.jl")


"""
    prepare_data(dir, zoom, new_size)

Prepare data (images) in a given directory for the training process.

# Examples
prepare_data("data/dataset", 0.25, 64)
"""
function prepare_data(dir::String, zoom::Float64, new_size::Int64)
    # prepare the set of images of the default quality
    Y = []
    for img_file in readdir(dir)
        img = load(img_file)
        cropped_images = crop_image(img, new_size)
        Y = vcat(Y, cropped_images)
    end

    # prepare the set of images of lower quality
    X = []
    for img in Y
        resized = imresize(img, zoom)
        X = vcat(X, resized)
    end
    X, Y
end
