"""Dev script for multi-run hyperparam tuning."""

import subprocess

kl_div_coeff = 0.01

for latent_dim in [3, 4, 6, 8]:
    args = [
        "python3",
        "latent/vae.py",
        "--epochs=128",
        f"--latent_dim={latent_dim}",
        f"--kl_div_coeff={kl_div_coeff}",
        f"--output_dir=outputs/vae/latent_tune/latent{latent_dim}",
    ]
    print(" ".join(args))
    subprocess.run(args)
