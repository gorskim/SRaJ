using Images, Colors, JLD2, ProgressMeter, ImageFiltering
using TestImages: testimage


# constants definition
TRAINING_DATASET_DIR = "data/dataset"
TEST_IMAGES_DIR = "data/test_data"  # testing = playing with new methods, etc.
TEST_IMAGES = ["lighthouse", "mandril_color", "lena_color_512", "peppers_color",
               "woman_blonde"]
IMAGE_ZOOM = 0.25
IMAGE_SIZE = 64  # e.g. 64 stands for 64x64 pixels (always square images)


"""
    crop_image(image, new_size)

Crop smaller images (square) out of an input one.

# Examples
crop_image(img, 64)
"""
function crop_image(image::Array, new_size::Int64)
    # get rid of the residual pixels
    new_height = height(image) - height(image) % new_size
    new_width = width(image) - width(image) % new_size
    cropped_size = min(new_height, new_width)
    image = image[1:cropped_size, 1:cropped_size]

    # create the array for new images
    cropped_images = Array[]
    horizontal_start = 1
    for i in 1:Int(cropped_size / new_size)
        vertical_start = 1
        for j in 1:Int(cropped_size / new_size)
            cropped_img = image[vertical_start:(j*new_size),
                                horizontal_start:(i*new_size)]
            push!(cropped_images, cropped_img)
            vertical_start += new_size
        end
        horizontal_start += new_size
    end
    cropped_images
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
    prepare_data(dir, zoom, new_size)

SIGNIFICANT: Don't use in this form, super-inefficient code (TODO: improvement)
Prepare data (images) in a given directory for the training process.

# Examples
prepare_data("data/dataset", 0.50, 64)
"""
function prepare_data(dir::String, zoom::Float64, new_size::Int64)
    # prepare the set of images of the default quality
    x = Matrix{Float32}[]
    y = Matrix{Float32}[]
    @info "Cropping, blurring, resizing, creating matrices of arrays:"
    @showprogress for img_file in readdir(dir)
        img = load(joinpath(dir, img_file))
        cropped_images = crop_image(RGB.(float.(img)), new_size)
        for cropped in cropped_images
            blurred = imfilter(cropped, Kernel.gaussian(3))
            resized = imresize(blurred, ratio=zoom)
            x = vcat(x, [resized])
        end
        y = vcat(y, cropped_images)
    end

    @info "Converting low-resolution data to 4D array (channelview)..."
    @time lr = reduce((a, b) -> cat(a, b, dims=4), permutedims(channelview(img),
               (2, 3, 1)) for img in x)
    @info "Converting high-resolution data to 4D array (channelview):"
    @time hr = reduce((a, b) -> cat(a, b, dims=4), permutedims(channelview(img),
               (2, 3, 1)) for img in y)

    lr, hr
end


function main()
    # x, y = prepare_data(TEST_IMAGES_DIR, IMAGE_ZOOM, IMAGE_SIZE)
    # @info "Saving to JLD2 format..."
    # @save "data/test_dataset.jld2" x y

    x, y = prepare_data(TRAINING_DATASET_DIR, IMAGE_ZOOM, IMAGE_SIZE)
    @info "Saving to JLD2 format..."
    @save "data/training_dataset.jld2" x y
end

# main()
