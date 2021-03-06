using Pkg
Pkg.activate(".")
Pkg.instantiate()

using Dates, Metalhead, Flux, Tracker
using BSON: @save
using Tracker:update!
using Flux: @treelike, param, params

include("data_preparation.jl")
include("processing.jl")

@info "All the necessary libs imported."

# constants and parameters definition
MODELS_PATH = "models/"
IMAGE_CHANNELS = 3
EPOCHS = 1000
MINIBATCH_SIZE = 32  # 32 - 128
GENERATOR_BLOCKS_COUNT = 8
CHECKPOINT_FREQUENCY = 5

# smoke variables - to test if everything works fine
N_SMOKE_SAMPLES = 6
SMOKE_MINIBATCH = 2
SMOKE_EPOCHS = 10

gen = Gen(GENERATOR_BLOCKS_COUNT) |> gpu
dis = Discriminator() |> gpu
vgg = load_vgg()

# losses definition
losses = Dict("discriminator" => [], "generator" => [], "adv" => [], "content" => [])

function dloss(HR, LR)
	@info "generating images (dloss calculation started)"
    SR = gen(LR)
	@info "done"
    fake_prob = dis(SR)
    fake_labels = zeros(size(fake_prob)...) |> gpu
    fake_dis_loss = bce_with_logits(fake_prob, fake_labels)
    real_prob = dis(HR)
    real_labels = ones(size(real_prob)...)
    fill!(real_labels, 0.9f0) |> gpu
    real_dis_loss = bce_with_logits(real_prob, real_labels)
    output = mean(fake_dis_loss .+ real_dis_loss)
	@info "dloss calculated"
	push!(losses["discriminator"], output |> cpu)
	output
end

function gloss(HR, LR)
	@info "gloss calculation started"
	SR = gen(LR)
	fake_prob = dis(SR)
	real_labels = ones(size(fake_prob)...) |> gpu
	loss_adv = mean(bce_with_logits(fake_prob, real_labels))
	HR_features = vgg(HR)
	SR_features = vgg(SR)
	content_loss = mean(((HR_features .- SR_features)) .^2) ./ 12.75f0
	output = 10f-3 * loss_adv + content_loss
	@info "gloss calculated"
	push!(losses["generator"], output |> cpu)
	push!(losses["adv"], 10f-3 * loss_adv |> cpu)
	push!(losses["content"], content_loss |> cpu)
	output
end


function _get_minibatch(HR_names::Vector{String}, LR_names::Vector{String})
    HR_batch, LR_batch = [], []
    # @info "Loading minibatch."
    for (i, HR_name) in enumerate(HR_names)
        push!(HR_batch, load_image(joinpath(HR_DIR, HR_name)))
        push!(LR_batch, load_image(joinpath(LR_DIR, LR_names[i])))
    end
    return reshape(cat(HR_batch..., dims=4), IMAGE_SIZE, IMAGE_SIZE,
                   IMAGE_CHANNELS, length(HR_batch)),
           reshape(cat(LR_batch..., dims=4), LR_IMAGE_SIZE, LR_IMAGE_SIZE,
                   IMAGE_CHANNELS, length(LR_batch))
end


function _train_step(HR, LR)
    HR = normalize(HR)
    @info "Gradient of discriminator..."
    d_gs = Tracker.gradient(() -> dloss(HR, LR), params(dis))
    @info "Updating discriminator..."
    update!(dis_optimizer, params(dis), d_gs)
    @info "Gradient of generator..."
    g_gs = Tracker.gradient(() -> gloss(HR, LR), params(gen))
    @info "Updating generator..."
    update!(gen_optimizer, params(gen), g_gs)
end


function _save_model(model, out_filename)
    @info "Saving model..."
    model = model |> cpu  # super important to work on machines with no GPU
    @save joinpath(MODELS_PATH, out_filename) model
    current_time = now()
    @info "$current_time\nModel saved at: $MODELS_PATH"
end


function train(;prepare_dataset=false, smoke_run=false,
               checkpoint_frequency=CHECKPOINT_FREQUENCY)
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
    else
		MODELS_PATH = "models/"
		IMAGE_CHANNELS = 3
		EPOCHS = 5000
		MINIBATCH_SIZE = 32  # 32 - 128
		GENERATOR_BLOCKS_COUNT = 8
		CHECKPOINT_FREQUENCY = 5
	end

    dataset_count = length(HR_names)
    @info "Training dataset count: $dataset_count"
    minibatch_indices = partition(shuffle!(collect(1:length(HR_names))),
                                  MINIBATCH_SIZE)
    HR_batches, LR_batches = [HR_names[i] for i in minibatch_indices],
                             [LR_names[i] for i in minibatch_indices]


	@info "minibatches count: $(length(HR_batches))"
    for epoch in 1:EPOCHS
        @info "---Epoch: $epoch---"
        for batch_num in 1:length(HR_batches)
			HR, LR = _get_minibatch(HR_batches[batch_num], LR_batches[batch_num])
            # HR, LR = HRdata[batch_num], LRdata[batch_num]
            @info "$batch_num - training..................................."
            _train_step(HR |> gpu, LR |> gpu)
        end
        if epoch % checkpoint_frequency == 0
            @info "CHECKPOINT!"
            model_name = "model$(epoch).bson"
            _save_model(gen, model_name)
			@save joinpath(MODELS_PATH, "losses$(epoch).bson") losses
        end
    end

    @info "Training process completed."
    _save_model(gen, "final_model.bson")
    @info "Saving losses..."
    @save joinpath(MODELS_PATH, "losses.bson") losses
    @info "COMPLETED"
end

train(prepare_dataset=false, smoke_run=false)
