# [Autoencoders: Jeremy Jordan](https://www.jeremyjordan.me/autoencoders/)
* Unsupervised learning (no labels)
* Learns compressed representation of data (bottleneck layer)
* Input features are not independent, some correlations and structure
* Bottleneck is like the essential cols of matrix
* Reconstruction loss $L(x, \hat{x})$ measures diff b/t input and recon
* Discourage memorization and overfitting using regularization
* Linear network for autoencoder = PCA
* PCA learns hyperplane, autoencoder learns non-linear manifold
* Even 1 hidden node can allow memorization (think about the number of possible values 1 float32 can represent)

## Sparse autoencoders
* Alternative bottleneck
* Penalize activation, not weights
* Separates latent state representation size and regularization
* L1 regularization: $\lambda \sum |a|$, sum of abs(activation)
* KL Divergence = \sum_s p_s log(\frac{p_s}{q_s})
    * $\hat{p}_j = \frac{1}{m} \sum |a|$, average activation of neuron j over m input samples
    * Bernoulli distribution: 1 with p, 0 with 1-p
    * Regularization = $D_{KL}(p, \hat{p}) = \sum_j (p log(\frac{p}{\hat{p}}) + (1 - p) log(\frac{1 - p}{1 - \hat{p}}))$
    * Divergence from desired sparsity

## Denoising autoencoders
* Autoencoder(noised image) = original image
* Model can no longer just memorize

## Contractive autoencoders
* Force learned encoding to be similar for similar inputs
* Regularize using derivative of activations (hidden layer)
* Penalize large changes in encoding for small perturbations (also noising) in input
* Regularization = $\lambda \sum ||\nabla_x a(x)||$, sum of L2 norm of gradient of activations with respect to input x

# [Probability: Artem Kirsanov](https://youtu.be/KHVR587oW8I?feature=shared)
* Probability distribution P: $\sum_s p_s = 1$
* Surprise of event: $h(s) = log(\frac{1}{p_s}) = -log(p_s)$
* Entropy = average (expected) surprise of prob dist: $H = \sum_s p_s h(s) = -\sum_s p_s *log(p_s)$
* Cross-entropy = surprise when observing P and believing Q: $H(P, Q) = \sum_s p_s h(q_s) = -\sum_s p_s log(q_s)$
* KL-Divergence = extra surprise when observing P and believing Q, beyond inherent surprise in P: $D_{KL} = H(P, Q) - H(P) = -\sum_s p_s log(q_s) - (-\sum_s p_s log(p_s)) = \sum_s p_s log(\frac{p_s}{q_s})$
* Minimizing KL-Divergence = minimizing Cross-Entropy
    * $H(P, Q) = D_{KL} + H(P)$, improving model cannot change $H(P)$
* Classification: $H(P, Q) = -\sum_s p_s log(q_s) = -log(q_s)$
    * $p_s$ is 1 where class is correct, 0 everywhere else
    * $q_s$ is model's predicted prob
    * Same as Negative Log Likelihood loss
