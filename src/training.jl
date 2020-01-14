using Knet, JLD2, Dates, Metalhead
using Flux.Tracker:update!
using Flux: throttle


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
# smoke variables - to test if everything works fine
N_SMOKE_SAMPLES = 6
SMOKE_MINIBATCH = 2
SMOKE_EPOCHS = 10


function _get_minibatch(HR_names::Vector{String}, LR_names::Vector{String})
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


function _train_step(HR, LR)
    HR = normalize(HR)
    # taking gradients with respect to the loss
    # https://github.com/FluxML/Flux.jl/blob/master/docs/src/models/basics.md
    d_gs = Tracker.gradient(() -> dloss(HR, LR), params(discriminator))
    # https://github.com/FluxML/Flux.jl/blob/master/docs/src/training/optimisers.md
    update!(optimizer, params(discriminator), d_gs)
    g_gs = Tracker.gradient(() -> gloss(HR, LR), params(generator))
    update!(optimizer, params(generator), g_gs)
end


function _save_weights(model, out_filename)
    @info "Saving model..."
    model = model |> cpu  # super important to work on machines with no GPU
    weights = params(model)
    @save joinpath("$MODELS_PATH", out_filename) generator
    current_time = now()
    @info "$current_time\nModel saved at: $MODELS_PATH"
end


function train(;prepare_dataset=false; smoke_run=false; checkpoint_frequency=2500)
    current_time = now()
    @info "$current_time\nTraining process has started"

    if prepare_dataset == true
        prepare_dataset()
    end

    HR_names, LR_names = get_images_names(HR_DIR, LR_DIR)

    if smoke_run != false
        @info "This is a smoke run on a small amount of data."
        HR_names, LR_names = HR_names[1:N_SMOKE_SAMPLES], LR_names[1:N_SMOKE_SAMPLES]
        MINIBATCH_SIZE = SMOKE_MINIBATCH
        EPOCHS = SMOKE_EPOCHS
    end

    dataset_count = length(HR_names)
    @info "Training dataset count: $dataset_count"
    minibatch_indices = partition(shuffle!(collect(1:length(HR_names))),
                                  MINIBATCH_SIZE)
    HR_batches, LR_batches = [HR_names[i] for i in minibatch_indices],
                             [LR_names[i] for i in minibatch_indices]

    @showprogress for epoch in 1:EPOCHS
        @info "---Epoch: $epoch---"
        for batch_num in 1:length(HR_batches)
            HR, LR = _get_minibatch(HR_batches[batch_num], LR_batches[batch_num])
            weights = _train_step(HR |> gpu, LR |> gpu)
            if epoch % checkpoint_frequency == 0
                @info "CHECKPOINT!"
                model_name = "model-$(now()).jld2"
                _save_weights(generator, model_name)
            end
        end
    end

    @info "Training process completed."
    _save_weights(generator, "final_model.jld2")
end

train(prepare_dataset=true, smoke_run=true)
# train()
