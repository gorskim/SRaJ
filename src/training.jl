using Knet, JLD2, Dates

include("processing.jl")

# constants and parameters definition
DATA_FILE = "..."
IMAGES_COUNT = length(readdir(INPUT_DATA_DIR))
const ε = Float32(1e-8)
IMAGE_ZOOM = 0.25
IMAGE_SIZE = 64  # e.g. 64 stands for 64x64 pixels (always square images)
IMAGE_CHANNELS = 3
EPOCHS = 10^4
BATCH_SIZE = 64  # ?
SEED_SIZE = 100  # ? size vector to generate images from?
SAVE_FREQ = 100  # ? is it necessary?
ALPHA = 0.2


function main()
    @assert gpu() == -1 "There is no active GPU device."  # change to '!='

    @info "Loading data..."
    @time @load DATA_FILE x y

    @info "Data loaded. Training process in progress..." now()

    @info "Training process completed. The model saved at [location here!]" now()
end

main()
