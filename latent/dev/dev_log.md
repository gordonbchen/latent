12/19/2024
* Added KL Divergence sparsity regularization
* Added Contractive regularization (does not enforce sparsity). Contractive loss is similar to denoising, makes latent robust to small deviations.
* Next: Denoising, then VAE

12/18/2024
* Implemented vanilla autoencoders, L1 regularization
* Next: KL-Divergence and contractive regularization, denoising autoencoder, then VAE
* Added useful class to override hyperparams from the CLI
* Switched from flake8 + black + isort to ruff
