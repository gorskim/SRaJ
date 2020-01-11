# SRaJ
This project is to be created within MSc thesis concerning image quality enhancement with deep neural networks. It will be based on super-resolution and Julia programming language.

## Project structure
The content of /src directory is as follows:
* ***data_preparation.jl*** - every data preparation related functions. The objective is to use just *prepare_data* as public.
* ***processing.jl*** - every code related to processing and computations. It includes utilities, layers, networks, loss calculation, etc. 
* ***training.jl*** - the actual training script. 
* ***SRaJ.jl*** - contains the code used for super resolution based on previously trained models.

## Credits/sources
* (Flux.jl)[https://arxiv.org/pdf/1811.01457.pdf]
* The orignator of [Generative Adversarial Networks](https://arxiv.org/pdf/1406.2661.pdf) - Ian Goodfellow.
* The architecture of the networks was inspired by [this paper](https://arxiv.org/pdf/1609.04802.pdf).
* [Shreyas Kowshik](https://shreyas-kowshik.github.io/) - noticeable contributor of [Flux.jl](https://fluxml.ai/Flux.jl/stable/)