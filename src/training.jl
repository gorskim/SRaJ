using Knet, JLD2, Dates, Metalhead
using Flux.Tracker:update!


include("data_preparation.jl")
include("processing.jl")

# constants and parameters definition
MODELS_PATH = "models/"
ε = Float32(1e-8)
IMAGE_CHANNELS = 3
EPOCHS = 5 * 10^4
MINIBATCH_SIZE = 32  # 32 - 128
SEED = 123
NOISE_DIM = 128  # ? size vector to generate images from
SAVE_FREQ = 100  # ? is it necessary?
α = 0.2
SEED = 123
GENERATOR_BLOCKS_COUNT = 16


function get_minibatch(HR_names::Vector{String}, LR_names::Vector{String})
    HR_batch, LR_batch = [], []
    @info "Loading minibatch."
    @showprogress for (i, HR_name) in enumerate(HR_names)
        push!(HR_batch, load_image(HR_name))
        push!(LR_batch, load_image(LR_names[i]))
    end
    return reshape(cat(HR_batch..., dims=4), IMAGE_SIZE, IMAGE_SIZE,
                   IMAGE_CHANNELS, length(HR_batch)),
           reshape(cat(LR_batch..., dims=4), LR_IMAGE_SIZE, LR_IMAGE_SIZE,
                   IMAGE_CHANNELS, length(LR_batch))
end


function train_step(HR, LR)  # CHECK IT
    HR = normalize(HR)
    d_gs = Tracker.gradient(() -> dloss(HR, LR), params(discriminator))
    update!(optimizer, params(discriminator), d_gs)
    g_gs = Tracker.gradient(() -> gloss(HR, LR), params(generator))
    update!(optimizer, params(generator), g_gs)
end


function train(;prepare_dataset=false)
    current_time = now()
    @info "$current_time\nTraining process has started"

    if prepare_dataset == true
        prepare_dataset()
    end

    HR_names, LR_names = get_images_names(HR_IMAGES, LR_IMAGES)
    dataset_count = length(HR_names)
    @info "Training dataset count: $dataset_count"
    minibatch_indices = partition(shuffle!(collect(1:length(HR_names))),
                                  MINIBATCH_SIZE)
    HR_batches, LR_batches = [HR_names[i] for i in minibatch_indices],
                             [LR_names[i] for i in minibatch_indices]

    @showprogress for epoch in 1:EPOCHS
        @info "---Epoch: $epoch---"
        for batch_num in 1:length(HR_batches)
            HR, LR = get_minibatch(HR_batches[batch_num], LR_batches[batch_num])
            weights = train_step(HR |> gpu, LR |> gpu)
        end
    end

    model = model |> cpu  # super important!
    @save "MODELS_PATH/final_model.jld2" model
    current_time = now()
    @info "$current_time\nTraining process completed. The model saved at: $MODELS_PATH"
end

# train(prepare_dataset=true)
