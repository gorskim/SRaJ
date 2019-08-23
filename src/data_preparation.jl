"""
This file contains all the functions needed for data preparation.
"""

using Images, TestImages, Colors

# constants definition
TEST_IMAGES_DIR = "data/test_data"
TEST_IMAGES = ["lighthouse", "mandril_color", "lena_color_512", "peppers_color",
               "woman_blonde"]


"""
    crop_image(image, dims)

Crop a smaller image out of an input one.

# Examples
crop_image(img, [1:64, 1:64])
"""
function crop_image(image, dims::Array{UnitRange{Int64},1})
    img = load(image_file)
    img[dims]
end


"""
    decrease_image(image, factor)

Decrease an image by a given factor.
"""
function decrease_image(image, factor::Int)
    for num in 1:(factor - 1)
        image = restrict(image)
    end
    image
end


"""
    prepare_test_data(test_images, dir)

Prepare images for testing, demonstration, etc. 'test_images' vector must
contain names of the images available via TestImages package.

# Examples
prepare_test_data(TEST_IMAGES, "data/test_data")
"""
function prepare_test_data(test_images::Vector{String}, dir::String)
    for image_name in test_images
        img = testimage(image_name)
        save("$dir/$image_name.png", img)
    end
end


"""
    prepare_data(dir)

Prepare data (images) in a given directory for the training process.
This is the main and only function to be used in training.jl
"""
function prepare_data(dir::String, decrease_factor::Int8,
                      dims::Array{UnitRange{Int64},1})
    for img_file in readdir(dir)
        img = load(img_file)

        # TODO
    end
end
