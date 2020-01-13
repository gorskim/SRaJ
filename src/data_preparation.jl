using Images, Colors, JLD2, ProgressMeter, ImageFiltering
using TestImages: testimage
using Random:shuffle! # for shuffle
using Base.Iterators:partition
using CuArrays

# constants definition
TRAINING_DATASET_DIR = "data/dataset"
LR_IMAGES = "data/prepared_dataset/LR"
HR_IMAGES = "data/prepared_dataset/HR"
PLAYGROUND_DIR = "data/test_data"  # testing = playing with new methods, etc.
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

Prepare data (images) in a given directory for the training process.

# Examples
prepare_data("data/dataset", "HR", "LR", 0.50, 64)
"""
function prepare_data(dir::String, HR_dir::String, LR_dir::String,
					  zoom::Float64, new_size::Int64)
    @info "Cropping, blurring, resizing, creating arrays:"
    @showprogress for img_file in readdir(dir)
        img = RGB.(load(joinpath(dir, img_file)))
        cropped_images = crop_image(img, new_size)
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

    # @info "Converting data to 4D array"
    # LR_images = reshape(cat(LR_images..., dims=4),
    #                     Int(new_size * zoom), Int(new_size * zoom), 3, length(LR_images))
    # HR_images = reshape(cat(HR_images..., dims=4),
    #                     Int(new_size), Int(new_size), 3, length(HR_images))
	#
    # LR_images, HR_images
end



function prepare_dataset()
    prepare_data(TRAINING_DATASET_DIR, HR_IMAGES, LR_IMAGES, IMAGE_ZOOM, IMAGE_SIZE)
	@info "Dataset prepared."
end
