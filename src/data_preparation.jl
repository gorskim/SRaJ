using Images, Colors
using TestImages: testimage


# constants definition
TEST_IMAGES_DIR = "data/test_data"
TEST_IMAGES = ["lighthouse", "mandril_color", "lena_color_512", "peppers_color",
               "woman_blonde"]


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
