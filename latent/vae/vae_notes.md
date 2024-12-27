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


# Bayes' Theorem
* $P(A|B) = \frac{P(B|A) P(A)}{P(B)}$
* $P(A)$: prior, initial belief about probability of event A
* Then we observe event B
    * $P(B|A)$: likeihood, probability of event B given event A
    * $P(B)$: known probability of event B
* $P(A|B)$: posterior, updated probability of event A after observing event B

## Example: lateness of a friend
* Prior: $P(L) = \frac{1}{2}$, our assumption about our friend without any additional information
* Observe evidence $B$: friend calls and is stuck in traffic, get $P(B|L)$ and $P(L)$
* Update belief (posterior): $P(L|B)$


# [VAE KL-Divergence Loss: Kevin Frans](https://kvfrans.com/deriving-the-kl/)
## ELBO Derivation
Given an image dataset $x$, where images are generated from latent codes $z$, where $p(z)$ is the latent distribution. $p(z)$ is our prior, our assumption of how the latents are distributed.

Goal: maximize $p_{\theta}(x|z)$ under the data: probability of reconstructing images x from their corresponding latents z, with neural net parameters $\theta$

$p_{\theta}(x|z) = \frac{p(z|x) p(x)}{p(z)}$  
Introduce $q_{\phi}(z|x)$ to approximate $p(z|x)$, probability of encoding image x as latent z, with neural net parameters $\phi$

> $KL(q_{\phi}(z|x), p(z|x)) = E_q[log(q_{\phi}(z|x)) - log(p(z|x))]$  
> $ = E_q[log(q_{\phi}(z|x)) - log(p(z,x)) + log(p(x))]$  
> $ = E_q[log(q_{\phi}(z|x)) - log(p(z,x))] + log(p(x))$  
> $KL(q_{\phi}(z|x), p(z|x)) = E_q[log(q_{\phi}(z|x)) - log(p_{\theta}(x|z)p(z)))] + log(p(x))$  

> $log(p(x)) = KL(q_{\phi}(z|x), p(z|x)) - E_q[log(q_{\phi}(z|x)) - log(p_{\theta}(x|z)p(z)))]$  
> $log(p(x)) = E_q[log(p_{\theta}(x|z)p(z))) - log(q_{\phi}(z|x))] + KL(q_{\phi}(z|x), p(z|x))$  

We want to maximize $log(p(x))$, the log probability of the data. $KL(q_{\phi}(z|x), p(z|x))$ is intractable since it depends on $p(z|x)$ which we don't have. We can maximize our ELBO (Evidence Lower Bound), which is a proxy to maximizing $log(p(x))$): $E_q[log(p_{\theta}(x|z)p(z))) - log(q_{\phi}(z|x))]$.

> $E_q[log(p_{\theta}(x|z)p(z))) - log(q_{\phi}(z|x))] = E_q[log(p_{\theta}(x|z)) + log(p(z)) - log(q_{\phi}(z|x))]$  
> $ = E_q[log(p_{\theta}(x|z))] + E_q[log(p(z)) - log(q_{\phi}(z|x))]$  
> $ = E_q[log(p_{\theta}(x|z))] - KL(q_{\phi}(z|x), p(z))$

To maximize $log(p(x))$, we have to maximize $E_q[log(p_{\theta}(x|z))]$, the log prob of reconstructing images from latents, and minimize $KL(q_{\phi}(z|x), p(z))$, the KL-Divergence b/t the encoding prior distributions.

## Analytical KL-Divergence b/t Gaussians
Goal: minimize $KL(q_{\phi}(z|x), p(z))$. $q_{\phi}(z|x) = N(\mu, \sigma^2)$ determined by neural net. $p(z) = N(0, 1)$.

Gaussian: $p(x) = \frac{1}{\sigma \sqrt{2\pi}} e^{-\frac{1}{2} (\frac{x - \mu}{\sigma})^2}$  
$p(z) = N(0, 1) = \frac{1}{\sqrt{2\pi}} e^{-\frac{1}{2} x^2}$  

> $KL(q_{\phi}(z|x), p(z)) = E_q[log(q_{\phi}(z|x)) - log(p(z))]$  
> $ = E_q[log(\frac{1}{\sigma \sqrt{2\pi}} e^{-\frac{1}{2} (\frac{x - \mu}{\sigma})^2}) - log(\frac{1}{\sqrt{2\pi}} e^{-\frac{1}{2} x^2})]$  
> $ = E_q[log(\frac{1}{\sigma \sqrt{2\pi}}) -\frac{1}{2} (\frac{x - \mu}{\sigma})^2 - log(\frac{1}{\sqrt{2\pi}}) + \frac{1}{2} x^2]$  
> $ = E_q[log(\frac{\sqrt{2\pi}}{\sigma \sqrt{2\pi}}) -\frac{1}{2} (\frac{x - \mu}{\sigma})^2 + \frac{1}{2} x^2]$  
> $ = E_q[-log(\sigma) - \frac{1}{2} (\frac{x - \mu}{\sigma})^2 + \frac{1}{2} x^2]$  
> $ = \frac{1}{2} E_q[-log(\sigma^2) -(\frac{x - \mu}{\sigma})^2 + x^2]$

> $E_q[-log(\sigma^2)] = -log(\sigma^2)$  

> $E_q[(x - \mu)^2] = \sigma^2$  
> $E_q[-(\frac{x - \mu}{\sigma})^2] = -E_q[(\frac{(x - \mu)^2}{\sigma^2})] = -\frac{\sigma^2}{\sigma^2} = -1$  

> $E_q[x^2] = \mu^2 + \sigma^2$

> $KL(q_{\phi}(z|x), p(z)) = \frac{1}{2} (-log(\sigma^2) - 1 + \mu^2 + \sigma^2)$

Checks:
> $\frac{\partial KL}{\partial \mu} = \mu$, min @ $\mu = 0$  
> $\frac{\partial KL}{\partial \sigma} = -\frac{1}{\sigma} + \sigma$, min @ $\sigma = 1$
