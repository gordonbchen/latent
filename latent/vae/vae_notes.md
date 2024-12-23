# [Variational Autoencoders: Jeremy Jordan](https://www.jeremyjordan.me/variational-autoencoders/)
* Autoencoder: input --> vector of compressed features --> reconstruction
* VAE: input --> prob dist for each feature, sample from prob dist --> reconstruction
* Smooth latent space
* $x$ (input) --$q(z|x)$ (encoder)--> z --$p(x|z)$ (decoder)--> $\hat{x}$
* Loss = $L(x, \hat{x}) + D_{KL}(q(z|x), p(z))$
    * $L(x, \hat{x})$: reconstruction loss
    * $D_{KL}(q(z|x), p(z))$: KL divergence b/t learned distribution and true prior (unit Gaussian with maximum entropy and information)

## Broken implementation
* Encoder: $x$ --NN--> $(\mu, \sigma)$ for every latent space dimension 
* Decoder: $(\mu, \sigma)$ --sample--> $z$ latent space features --NN--> $\hat{x}$
* Gradient flow is broken when sampling from gaussian parameterized by $(\mu, \sigma)$
* Sampling is non-differentiable
* Will be able to train decoder on recon loss, encoder will get no gradient

## Reparameterization trick
* Encoder: $x$ --NN--> $(\mu, \sigma)$ for every latent space dimension 
* Decoder
    * $\epsilon ~N(0, 1)$: sample from unit gaussian
    * $z = \mu + (\sigma * \epsilon)$
    * $z$ latent space features --NN--> $\hat{x}$
* Only sampling from $~N(0, 1)$ unit gaussian is non-differentiable
* Preserves gradient flow to latent space parameterization $(\mu, \sigma)$
* Careful when learning $\sigma$, must be positive

## KL Divergence Regularization
* No KL, only recon loss: gaps latent space, memorizes by learning narrow distributions
* With KL: pushes distributions toward unit gaussian, increases variance, smoother latent space
* Weighting KL Divergence: $L(x, \hat{x}) + \beta D_{KL}(q(z|x), N(0, 1))$
* Disentangled VAEs: larger $\beta$ prevents correlation b/t latent dimensions
    * Will only deviate from prior $N(0, 1)$ when latent dimension is necessary

## Generative
* Sample from prior $p(z) = N(0, 1)$
* Calculate latent features $z$
* Decode into $\hat{x}$
