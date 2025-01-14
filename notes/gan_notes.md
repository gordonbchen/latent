# [From GAN to WGAN: Lilian Weng](https://lilianweng.github.io/posts/2017-08-20-gan/)
## KL Divergence  
$D_{KL}(p, q) = \int_x p(x) \log(\frac{p(x)}{q(x)}) dx$ 
Minimized when $p(x) = q(x)$  
Asymmetric, $D_{KL}(p, q) \neq D_{KL}(q, p)$  
Extra uncertainty when observing $p(x)$ under the belief of $q(x)$  

## Jensen-Shannon Divergence  
$D_{JS}(p, q) = \frac{1}{2} D_{KL}(p, \frac{p+q}{2}) + \frac{1}{2} D_{KL}(q, \frac{p+q}{2})$

Discriminator $D$ estimates probability of sample coming from real dataset.  
Generator $G$ converts noise $z$ into synthetic samples imitating the real dataset.  
$G$ is trying to trick $D$, and $D$ is trying to discriminate b/t real and fake.  

Discriminator maximizes $E_{x}[\log(D(x))] + E_{z}[\log(1 - D(G(z)))]$  
Generator minimizes $E_{z}[\log(1-D(G(z)))]$  
Loss: $\underset{G}{\min} \ \underset{D}{\max} \ L(D, G) = E_x[\log(D(x))] + E_z[\log(1-D(G(z)))]$  

## Optimal value  
$L(D, G) = E_x[\log(D(x))] + E_z[\log(1-D(G(z)))]$  
$L(D, G) = \int_x (p_r(x) \log(D(x)) + p_g(x) \log(1-D(x))) dx$  

$\frac{dL}{dD(x)} = \frac{p_r(x)}{D(x)} - \frac{p_g(x)}{1-D(x)} = 0$  
$\frac{p_r(x)}{D(x)} = \frac{p_g(x)}{1-D(x)}$  
$p_r(x)(1-D(x)) = p_g(x)D(x)$  
$p_r(x) - p_r(x)D(x) = p_g(x)D(x)$  
$p_r(x) = p_g(x)D(x) + p_r(x)D(x)$  
$D^*(x) = \frac{p_r(x)}{p_g(x) + p_r(x)}$  
$p_r(x) = p_g(x)$ when trained, $D^*(x) = \frac{1}{2}$

$L(D^*, G) = \int_x (p_r(x) \log(D^*(x)) + p_g(x) \log(1-D^*(x))) dx$  
$ = \log(\frac{1}{2}) (\int_x p_r(x) dx + \int_x p_r(x) dx)$  
$ = 2 \log(\frac{1}{2}) = -2 \log(2)$  

## Relation to JS Divergence  
$D_{JS}(p_r, p_g) = \frac{1}{2} D_{KL}(p_r, \frac{p_r + p_g}{2}) + \frac{1}{2} D_{KL}(p_g, \frac{p_r + p_g}{2})$  
$ = \frac{1}{2} \int_x [p_r(x) \log(\frac{p_r(x)}{\frac{p_r(x) + p_g(x)}{2}})] dx + \frac{1}{2} \int_x [p_g(x) \log(\frac{p_g(x)}{\frac{p_r(x) + p_g(x)}{2}})] dx$  
$ = \frac{1}{2} (\log(2) + \int_x [p_r(x) \log(\frac{p_r(x)}{p_r(x) + p_g(x)})] dx) + \frac{1}{2} (\log(2) + \int_x [p_g(x) \log(\frac{p_g(x)}{p_r(x) + p_g(x)})] dx)$  
$ = \log(2) + \frac{1}{2} (\int_x [p_r(x) \log(D^*(x) + p_g(x) \log(1-D^*(x)] dx)$  
$ D_{JS}(p_r, p_g) = \log(2) + \frac{1}{2} L(D^*, G)$  
$ L(D^*, G) = 2D_{JS}(p_r, p_g) -2\log(2)$  
GAN loss is JS-divergence b/t $p_r$ real data distribution and $p_g$ generated data distribution.  

## Problems  
Generator and Discriminator are trying to find Nash equilibrium in 2-player non-cooperative game.  

$p_r$ and $p_g$ are low dimensional. Seemingly high dimensional b/c there are many pixels, but coherent images are strongly constrained (faces are very structured), and the noise distriubtion the generator is fed is usually low dimensional. Low dimensional $p_g$ and $p_r$ means that they are far more likely to be disjoint (think lines in 3d space are probably not going to intersect, but planes and higher dimensional manifold will), so a perfect discriminator can be found.  

Perfect discriminator ($D(x) = 1, \forall x \in p_r$ and $D(x) = 0, \forall x \in p_g$) perfectly separates $p_r$ and $p_g$ manifolds, vanishing gradient for generator (no nudges can cross the boundary into real data distribution). Bad discriminator means inaccurate feedback for generator, too good discriminator means vanishing gradient. Sigmoid output is used for binary cross entropy loss, saturates close to 0 or 1.  

Mode collapse: generator produces same outputs. Exploits small space of examples that trick the discriminator. Discriminator catches on and learns that the point is generated, but the gradients from the discriminator will act the same on all the points, and the generator can no longer make the outputs diverse.

## Training improvements  
Non-saturating loss function. $L_G = -\log(D(G(z))$  

[Improved Techniques for Training GANs: Salimans, et al.](https://proceedings.neurips.cc/paper_files/paper/2016/file/8a3363abe792db2d8761d6403605aeb7-Paper.pdf)  
**Feature matching**: $L_F = ||E_{p_r}[f(x)] - E_{p_g}[f(G(z))]||_2 ^ 2$, where $f(x)$ are the activations of an intermediate layer in the discriminator. Prevents generator from overtraining on discriminator by forcing generated data to match distribution of training data. Not only optimizing tricking discriminator, also optimizing to match data.  

**Minibatch discrimination**: Adding information about other training samples in batch (related to batch norm). In some intermediate layer, compute similarity of activations across examples in batch, sum activations weighted by similarity and concat as another feature. Allows discriminator to use information about other examples when classifying every single example. Using cross-example info from the minibatch, the discriminator should hopefully be able to avoid getting tricked by one specific repeated example and avoid mode collapse.  

**Historical averaging**: $L_H = ||\theta - \frac{1}{t} \sum_t \theta||^2$. Adding cost to prevent large deviations from past parameters.  

**One-sided label smoothing**: smooth positive labels to $\alpha = 1 \to 0.9$.  
Label smoothing (for binary classification) replaces $\alpha = 1 \to 0.9, \beta = 0 \to 0.1$.

Unsmoothed  
$L(D, G) = E_{x \in p_r} [(1) \log(D(x)) + (0) \log(1 - D(x))] + E_{x \in p_g} [(0) \log(D(x)) + (1) \log(1 - D(x))]$  
$L(D, G) = E_{x \in p_r} [\log(D(x))] + E_{x \in p_g} [\log(1 - D(x))]$  

Smoothed  
$L(D, G) = E_{x \in p_r} [\alpha \log(D(x)) + (1 - \alpha) \log(1 - D(x))] + E_{x \in p_g} [\beta \log(D(x)) + (1 - \beta) \log(1 - D(x))]$  
$L(D, G) = \int_x [p_r(x) (\alpha \log(D(x)) + (1 - \alpha) \log(1 - D(x))) + p_g(x)(\beta \log(D(x)) + (1 - \beta) \log(1 - D(x)))]$  

$\frac{dL}{dD(x)} = p_r(x)(\frac{\alpha}{D(x)} - \frac{1-\alpha}{1-D(x)}) + p_g(x)(\frac{\beta}{D(x)} - \frac{1-\beta}{1-D(x)})$  
$ = p_r(x)(\frac{\alpha - \alpha D(x) - D(x) + \alpha D(x)}{D(x)(1-D(x))}) + p_g(x)(\frac{\beta - \beta D(x) - D(x) + \beta D(x)}{D(x)(1-D(x))})$  
$ \frac{dL}{dD(x)} = \frac{p_r(x)(\alpha - D(x)) + p_g(x)(\beta - D(x))}{D(x)(1-D(x))} = 0$  
$p_r(x)(\alpha - D(x)) + p_g(x)(\beta - D(x)) = 0$  
$\alpha p_r(x) - p_r(x)D^*(x) + \beta p_g(x) - p_g(x)D^*(x) = 0$  
$D^*(x) = \frac{\alpha p_r(x) + \beta p_g(x)}{p_r(x) + p_g(x)}$  

$D^*(x) \approx \beta$ when $p_r(x) \approx 0$. This means that when the generator is creating samples that are out of distribution (samples don't match the data), the optimal discriminator value $D^*(x) \neq 0$, and the generator will continue creating samples in that area. Since the goal is to generate samples that match the data distribution, we want $D^*(x) = 0$ when $p_d(x) \approx 0$. So we keep $\beta = 0$ and only smooth $\alpha = 1 \to 0.9$.  

$L(D, G) = E_{x \in p_r} [\alpha \log(D(x)) + (1 - \alpha) \log(1 - D(x))] + E_{x \in p_g} [\log(1 - D(x))]$  

**Virtual Batch Normalization (VBN)**: alternative to Batch Norm. Batch norm (used in DC-GAN) makes normalizaiton dependent on other samples in mini-batch. VBN proposes normalizing using fixed reference batch chosen at the start of training. Computationally expensive since we have to forward both the mini-batch and the reference batch to normalize the mini-batch.

[Towards Principled Methods for Training Generative Adversarial Networks: Arjovsky & Bottou](https://arxiv.org/pdf/1701.04862)  
**Adding noise to discriminator inputs**: $p_r$ and $p_g$ are disjoint. Add noise to $p_r$ to artificially spread out $p_r$ and make it more likely that $p_r$ and $p_g$ overlap.

**Wasserstein loss**

## Wasserstein GAN (WGAN)
Wasserstein distance: measure of distance b/t probability distributions.  

**Discrete**  
$P = (3, 2, 1, 4), Q = (1, 2, 4, 3)$  
Move 2 from $P_1$ to $P_2$. $P = (1, 4, 1, 4), Q = (1, 2, 4, 3)$  
Move 2 from $P_2$ to $P_3$. $P = (1, 2, 3, 4), Q = (1, 2, 4, 3)$  
Move 1 from $Q_3$ to $Q_4$. $P = (1, 2, 3, 4), Q = (1, 2, 3, 4)$  

$\delta_i = \delta_{i-1} + P_i - Q_i$

$\delta_0 = 0$  
$\delta_1 = \delta_0 + P_1 - Q_1 = 0 + 3 - 1 = 2$  
$\delta_2 = \delta_1 + P_2 - Q_2 = 2 + 2 - 2 = 2$  
$\delta_3 = \delta_2 + P_3 - Q_3 = 2 + 1 - 4 = -1$  
$\delta_4 = \delta_3 + P_4 - Q_4 = -1 + 4 - 3 = 0$

$W = \sum |\delta| = 2 + 2 + 1 = 5$  

**Continuous**  
$W(p_r, p_g) = \inf_{\gamma \in \Pi(p_r, p_g)} E_{(x, y) \in \gamma} [||x - y||]$   
Minimum ($\inf$) cost across $\Pi(p_r, p_g)$, the set of all joint probability distributions b/t $p_r$ and $p_g$. The cost $E_{(x, y) \in \gamma} [||x - y||] = \sum \gamma(x, y) ||x - y||$ is the total amount of dirt moved $\gamma(x, y)$ multiplied by the travelling distance $||x-y||$.  

**Improvement over $D_{KL}$ and $D_{JS}$**  
$\forall (x, y) \in P: x=0,  y \sim U(0, 1)$ and $\forall (x, y) \in Q: x=\theta,  y \sim U(0, 1)$  

When $\theta \neq 0$, $P$ and $Q$ are disjoint:   
$D_{KL}(P, Q) = \int_x p(x) \log(\frac{p(x)}{q(x)}) dx$  
$D_{KL}(P, Q) = 1 \cdot \log(\frac{1}{0}) = + \infty$  
$D_{KL}(Q, P) = 1 \cdot \log(\frac{1}{0}) = + \infty$  

$D_{JS}(P, Q) = \frac{1}{2} D_{KL}(P, \frac{P+Q}{2}) + \frac{1}{2} D_{KL}(Q, \frac{P+Q}{2})$  
$D_{JS}(P, Q) = \frac{1}{2} (1 \cdot \log(\frac{1}{1 / 2}) + 1 \cdot \log(\frac{1}{1 / 2})) = \log(2)$  

$W(P, Q) = |\theta|$  

When $\theta = 0$, $P$ and $Q$ are equal:  
$D_{KL} = D_{JS} = 0$  
$W(P, Q) = |\theta| = 0$  

$D_{KL}$ gives $\infty$ when disjoint. $D_{JS}$ has a sudden jump at $\theta=0$, not differentiable. $W$ is smooth and differentiable.  

**Wasserstein distance as GAN loss**  
$W(p_r, p_g) = \frac{1}{K} \sup_{||f||_L \le K} E_{x \sim p_r} [f(x)] - E_{x \sim p_g} [f(x)]$  
Function $f$ is a form of the Wasserstein metric such that $|f|_L \le K$, must be K-Lipschitz continuous.  
$L(D, G) = W(p_r, p_g) = E_{x \sim p_r} [D(x)] - E_{z \sim p_z} [D(G(z))]$  
Discriminator no longer categorizes real data and generated data, but outputs a metric that must be maximized for real data and minimized for fake data. Discriminator now is called a critic instead, since it doesn't output the probability of the sample coming from the real dataset, but a metric instead. Wasserstein loss is better because it measures distance between $p_r$ and $p_g$ distributions, which is much more informative than when the discriminator learns to perfectly separate the distributions and the probability is 0.  

Maintaining K-Lipschitz continuity in the critic is important because Wasserstein distance should approximately continuous so that the generator can learn. We don't want the critic to output extremely different values for inputs close together.  

Weight clipping: $w \leftarrow clip(w, -c, c)$, $c \approx 0.01$. Enforces Lipschitz continuity by ensuring that weight updates, and the rate of change of the critic, are small.  

Gradient penalty:  
$GP = E_x [(||\nabla_x D(x)||_2 - 1) ^ 2]$  
$L(D, G) = E_{x \in p_r} [D(x)] - E_{z \in p_z} [D(G(z))] + \lambda GP$  
Similar to contractive autoencoder loss. Penalized gradient norm deviating from 1. $D(x)$ should change smoothly wrt input $x$.  

# [Generative Adversarial Nets: Ian Goodfellow, et al.](https://arxiv.org/pdf/1406.2661v1)
Generatior $G$ that mimics the data distribution. Discriminator $D$ that estimates the probability a sample is from the data distribution, not generated by $G$. $G$ is optimized to make $D$ make a mistake. Formulated as a 2-player minimax game. The optimal solution is $D(x) = \frac{1}{2}$ everywhere.  

$\underset{G}{\min} \ \underset{D}{\max} \ V(D, G) = E_{x \sim p_{data}(x)} [\log(D(x))] + E_{z \sim p_z(z)} [\log(1 - D(G(z)))]$  

$L_G = \log(1 - D(G(z)))$ saturates when $D$ confidently rejects generated samples (in early training), providing very little gradient for $G$ to improve. Instead, $G$ should maximize $L'_G = \log(D(G(z)))$, the log probability of $D$ believing the generated sample is real.  
  
$\frac{dL_G}{dD(G(z))} = \frac{-1}{1 - D(G(z))}$  
$\frac{dL_G}{dD(G(z))} |_{D(G(x)) = 0} = -1$  
$\frac{dL_G}{dD(G(z))} |_{D(G(x)) = 1} = -\infty$  

$\frac{dL'_G}{dD(G(z))} = \frac{1}{D(G(z))}$  
$\frac{dL'_G}{dD(G(z))} |_{D(G(x)) = 0} = +\infty$  
$\frac{dL'_G}{dD(G(z))} |_{D(G(x)) = 1} = 1$  

$L'_G$ gives much stronger gradients when $D(G(z)) \approx 0$, when the generator is doing very poorly, as opposed to $L_G$, which gives much stronger gradients when $D(G(z)) \approx 1$, when the generator is already doing very well.  

$V(G, D) = E_{x \sim p_d(x)} [\log(D(x))] + E_{z \sim p_z(z)} [\log(1 - D(G(z)))]$  
$ = \int_x [p_d(x) \log(D(x)) + p_g(x) \log(1 - D(x))]$

$\frac{dV}{dD(x)} = \frac{p_d(x)}{D(x)} - \frac{p_g(x)}{1 - D(x)} = 0$  
$\frac{p_d(x) (1-D(x)) - p_g(x) D(x)}{D(x) (1-D(x))} = 0$  
$p_d(x) - p_d(x)D(x) - p_g(x) D(x) = 0$  
$D^*(x) = \frac{p_d(x)}{p_d(x) + p_g(x)}$  

$C(G) = V(G, D*) = E_{x \sim p_d(x)} [\log(D^*(x))] + E_{z \sim p_z(z)} [\log(1 - D^*(G(z)))]$  
$ = E_{x \sim p_d(x)} [\log(D^*(x))] + E_{x \sim p_g(x)} [\log(1 - D^*(x))]$  
$ = E_{x \sim p_d(x)} [\log(\frac{p_d(x)}{p_d(x) + p_g(x)})] + E_{x \sim p_g(x)} [\log(1 - \frac{p_d(x)}{p_d(x) + p_g(x)})]$  
$ = E_{x \sim p_d(x)} [\log(\frac{p_d(x)}{p_d(x) + p_g(x)})] + E_{x \sim p_g(x)} [\log(\frac{p_g(x)}{p_d(x) + p_g(x)})]$  
$ = E_{x \sim p_d(x)} [\log(\frac{1}{2} \cdot \frac{p_d(x)}{\frac{p_d(x) + p_g(x)}{2}})] + E_{x \sim p_g(x)} [\log(\frac{1}{2} \cdot \frac{p_g(x)}{\frac{p_d(x) + p_g(x)}{2}})]$  
$ = -2\log2 + E_{x \sim p_d(x)} [\log(\frac{p_d(x)}{\frac{p_d(x) + p_g(x)}{2}})] + E_{x \sim p_g(x)} [\log(\frac{p_g(x)}{\frac{p_d(x) + p_g(x)}{2}})]$  
$ = -2\log2 + D_{KL}(p_d, \frac{p_d + p_g}{2}) + D_{KL}(p_g, \frac{p_d + p_g}{2})$  
$C(G) = -2\log2 + 2 D_{JS}(p_d, p_g)$

Since $D_{JS} \ge 0$ and $D_{JS} = 0 \iff p_d = p_g \ $: minimum $C^* = -2\log2$ when $p_d = p_g$. $G$ is trained to minimize JS-Divergence b/t $p_d$ and $p_g$.  

# [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks: Radford, et al.](https://arxiv.org/pdf/1511.06434)
Using GANs to build unsupervised image representations, then using parts of the discriminator as a feature extractor for supervised tasks. GAN loss is more expressive than MSE.  

Explore further: model interpretability with CNNs.  

## DCGAN architecture guidlines
Replace pooling (max pool) with strided convs in the Discriminator and fractionally strided convs (aka transposed convs or misnomer deconvs) in the Generator. Allows the network to learn spatial up/down-sampling. [Convolution Types Explained: Paul-Louis Prove](https://towardsdatascience.com/types-of-convolutions-in-deep-learning-717013397f4d).  

Remove fully connected hidden layers. Only 1 linear layer on noise for Generator, then reshape into image. Flatten final conv layer, then 1 linear layer into sigmoid for Discriminator.  

Batch Norm for all layers except for Generator output and Discriminator input. [Batch Norm Explained: Ketan Doshi](https://towardsdatascience.com/batch-norm-explained-visually-how-it-works-and-why-neural-networks-need-it-b18919692739).  

RELU for all Generator layers, except Tanh in output. Leaky ReLU for Discriminator.  

# [Improved Training of Wasserstein GANs](https://arxiv.org/pdf/1704.00028)
Improves WGAN by enforcing 1-Lipschitz continuity (derivative is bounded, ensures smoothness of critic function) using gradient penalty instead of weight clipping.

## Classic GANs
$\underset{G}{\min} \ \underset{D}{\max} \ E_{x \sim p_r} [log(D(x))] + E_{z \sim p_z} [log(1 - D(G(z))]$

Better objective for generator is to maximize $E_{z \in p_z} [log(D(G(z)))]$.  

Training discriminator (or critic in WGANs) to optimality, then taking optimization step for generator is equivalent to minimizing $D_{JS}$.  

## WGANs
$\underset{G}{\min} \ \underset{C}{\max} \ E_{x \sim p_r} [C(x)] - E_{z \sim p_z} [C(G(z))]$

Critic outputs metric, not probability of real/fake. Critic must be 1-Lipschitz continuous for smooth generator loss landscape. Optimizing $G$ for optimal $C$ minimizes Wasserstein distance b/t $p_r$ and $p_g$.

1-Lipschitz continuity enforced through weight-clipping.

Optimal WGAN Critic has gradient norm 1 almost everywhere under $p_r$ and $p_g$.

See [Wasserstein GAN](https://arxiv.org/pdf/1701.07875).

## WGAN-GP
* Gradient penalty coeff: $\lambda = 10$
* Critic optimization steps before generator step: $n_{critic} = 5$
* Learning rate: $\alpha = 0.0001$
* Momentum: $\beta_1 = 0, \beta_2 = 0.9$

```
# Critic optim.
for i in range(n_critic):
    real = get_real_batch()
    z = get_latent_batch()
    fake = generator(z)

    interps = (alpha * real) + ((1 - alpha) * fake)
    grad_norm = norm(d(C(interps)) \ d(interps)).mean(across samples)
    grad_penalty = grad_penalty_coeff * ((grad_norm - 1) ** 2)

    L = C(fake) - C(real) + grad_penalty

    Adam optim step

# Generator optim.
z = get_latent_batch()
fake = generator(z)
L = -C(fake)

Adam optim step
```

Differentiable function is 1-Lipschitz iff grad_norm <= 1. Because it is intractable to constrain the gradient norm everywhere, sample uniformly along lines b/t $p_r$ and $p_g$ and add gradient penalty for those samples.

No critic BatchNorm since critic gradients are calculated wrt each individual sample, and not the entire batch. Use LayerNorm (maybe? Isn't LayerNorm "bad" for CNNs?).

2-sided gradient penalty, gradient norm pushed towards 1.

## Inception Score: [A Simple Explanation of the Inception Score: David Mack](https://medium.com/octavian-ai/a-simple-explanation-of-the-inception-score-372dff6a8c7a)
* Inception score to evaluate generated image quality.
* Ideal generator model
    * Class probability of individual images are very sharp (peak at one class)
    * Sum of class probs of all images gives uniform distribution (diverse outputs)
    * $D_{KL}$ of class probs of individual image, sum of class probs will be very high
* Inception score: $\exp(E_x [D_{KL}(p(y|x), p(y))])$

## Character GAN Language Model
* Generator: 1D CNN, latent velctor -> 32 1-hot char vectors, output softmax
* Critic: 1D CNN
* Argmax (deterministic, no random sampling) to decode generator output -> chars

# Other Resources
* [Many Paths to Equilibrium: GANs Do Not Need to Decrease a Divergence at Every Step" Fedus, et at.](https://arxiv.org/pdf/1710.08446) 

# Reference Implementations
* [PyTorch DCGAN Tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)
* [DCGANs: Dive into DL](https://d2l.ai/chapter_generative-adversarial-networks/dcgan.html)
* [PyTorch-GAN: eriklindernoren](https://github.com/eriklindernoren/PyTorch-GAN)
