using Images, Colors, JLD2, ImageFiltering
using ProgressMeter: @showprogress
using Random:shuffle! # for shuffle
using Base.Iterators:partition
using CuArrays

# constants definition
TRAINING_DATASET_DIR = "data/dataset"
LR_DIR = "data/prepared_dataset/LR"
HR_DIR = "data/prepared_dataset/HR"
PLAYGROUND_DIR = "data/test_data"  # testing = playing with new methods, etc.
UP_FACTOR = 3
IMAGE_ZOOM = UP_FACTOR ^ (-1)
IMAGE_SIZE = 64  # e.g. 64 stands for 64x64 pixels (always square images)
LR_IMAGE_SIZE = Int(ceil(IMAGE_ZOOM * IMAGE_SIZE))


"""
    _crop_image(image, new_size)

Crop smaller images (square) out of an input one.

# Examples
_crop_image(img, 64)
"""
function _crop_image(image::Array, new_size::Int64)
    # get rid of the residual pixels
    new_height = height(image) - height(image) % new_size
    new_width = width(image) - width(image) % new_size
    cropped_size = min(new_height, new_width)
    image = image[1:cropped_size, 1:cropped_size]

    # create the array for new images
    cropped_images = []
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
    _prepare_test_data(test_images, dir)

Prepare images for testing, demonstration, etc. 'test_images' vector must
contain names of the images available via TestImages package.

# Examples
_prepare_test_data(TEST_IMAGES, "data/test_data")
"""
function _prepare_test_data(test_images::Vector{String}, dir::String)
    for image_name in test_images
        img = testimage(image_name)
        save("$dir/$image_name.png", img)
    end
end


"""
    _prepare_data(dir, zoom, new_size)

Prepare data (images) in a given directory for the training process.

# Examples
_prepare_data("data/dataset", "HR", "LR", 0.50, 64)
"""
function _prepare_data(dir::String, HR_dir::String, LR_dir::String,
					  zoom::Float64, new_size::Int64)
    @info "Cropping, blurring, resizing, creating arrays:"
    @showprogress for img_file in readdir(dir)
        img = RGB.(load(joinpath(dir, img_file)))
        cropped_images = _crop_image(img, new_size)
        i = 1
        for cropped in cropped_images
            i += 1
			out_file = "$i.png"
            save(joinpath(HR_dir, out_file), cropped)
            blurred = imfilter(cropped, Kernel.gaussian(3))
            downsampled = imresize(blurred, ratio=zoom)
            save(joinpath(LR_dir, "d$out_file"), downsampled)
        end
    end
end


function prepare_dataset()
    _prepare_data(TRAINING_DATASET_DIR, HR_DIR, LR_DIR, IMAGE_ZOOM, IMAGE_SIZE)
	@info "Dataset prepared."
end
