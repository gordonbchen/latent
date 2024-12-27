from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from cli_params import CLIParams
from sklearn.decomposition import PCA
from torch import nn
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import MNIST


@dataclass
class HyperParams(CLIParams):
    latent_dim: int = 8
    kl_div_coeff: float = 0.01

    batch_size: int = 128
    lr: float = 3e-4
    epochs: int = 128

    output_dir: str = "outputs/vae/test"


class VAE(nn.Module):
    def __init__(self, latent_dim: int, kl_div_coeff: float) -> None:
        super().__init__()
        self.kl_div_coeff = kl_div_coeff

        self.encoder = nn.Sequential(
            nn.Flatten(), nn.LazyLinear(256), nn.ReLU(), nn.LazyLinear(256), nn.ReLU()
        )
        self.encoder_mean = nn.LazyLinear(latent_dim)
        self.encoder_std = nn.LazyLinear(latent_dim)

        self.decoder = nn.Sequential(
            nn.LazyLinear(256), nn.ReLU(), nn.LazyLinear(256), nn.ReLU(), nn.LazyLinear(28 * 28)
        )

    def encode(self, xb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(xb)
        mu = self.encoder_mean(z)
        std = self.encoder_std(z).exp()
        return mu, std

    def forward(
        self, xb: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        mu, std = self.encode(xb)

        z = mu + (std * torch.randn_like(std))
        recon = self.decoder(z).reshape(xb.shape)

        recon_loss = (0.5 * ((recon - xb) ** 2.0)).mean()

        var = std**2.0
        kl_div = (0.5 * (-var.log() - 1 + (mu**2.0) + var)).mean()

        loss = recon_loss + (self.kl_div_coeff * kl_div)

        metrics = {"recon_loss": recon_loss.item(), "kl_div": kl_div.item()}
        return recon, loss, metrics


class MNISTDataset(Dataset):
    def __init__(self) -> None:
        mnist_ds = MNIST("data", train=True, download=True)
        self.data = mnist_ds.data.to(dtype=torch.float32).unsqueeze(1) / 255.0
        self.targets = mnist_ds.targets

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]


def train(
    vae: VAE,
    train_dl: DataLoader,
    optim: Optimizer,
    epochs: int,
    output_dir: Path,
    n_plot_images: int = 4,
) -> None:
    vae.train()
    writer = SummaryWriter(output_dir)

    for epoch in range(epochs):
        for xb in train_dl:
            xb = xb.to("cuda")
            recon, loss, metrics = vae(xb)

            optim.zero_grad()
            loss.backward()
            optim.step()

        # Logging.
        writer.add_scalar("loss", loss.item(), epoch)
        writer.add_scalars("metrics", metrics, epoch)
        recon_images = torch.cat((xb[:n_plot_images], recon[:n_plot_images]), dim=-1)
        writer.add_images("recon", recon_images.detach().to("cpu"), epoch)
        print(f"Epoch={epoch}: loss={loss.item()}")

    writer.close()


@torch.no_grad()
def show_latent_space(
    vae: VAE, train_ds: MNISTDataset, output_dir: Path, n_samples: int = 10_000
) -> None:
    # Calculate latents and the PCA into 2D.
    idx = torch.randint(0, len(train_ds), size=(n_samples,))
    images = train_ds.data[idx].to("cuda")
    labels = train_ds.targets[idx]

    vae.eval()
    mu, _ = vae.encode(images)
    mu = mu.to("cpu")

    pca = PCA(n_components=2)
    latents_2D = pca.fit_transform(mu)

    # Plot the latent space.
    cmap = plt.get_cmap("jet", 10)
    fig, ax = plt.subplots(figsize=(14, 12))
    img = ax.scatter(latents_2D[:, 0], latents_2D[:, 1], c=labels, cmap=cmap)
    plt.colorbar(img, ax=ax)
    fig.savefig(output_dir / "latent_space.png")

    # Plot reconstructions of the latent space.
    n_steps = 50
    x_vals = torch.linspace(latents_2D[:, 0].min(), latents_2D[:, 0].max(), steps=n_steps)
    y_vals = torch.linspace(latents_2D[:, 1].min(), latents_2D[:, 1].max(), steps=n_steps)

    latent_grid = torch.cartesian_prod(x_vals, y_vals)
    new_latents = pca.inverse_transform(latent_grid).to("cuda", dtype=torch.float32)

    new_images = vae.decoder(new_latents)
    new_images = new_images.reshape((n_steps * n_steps, 28, 28))

    combined_image = torch.empty((n_steps * 28, n_steps * 28), dtype=torch.float32)
    for i in range(n_steps):
        for j in range(n_steps):
            combined_image[
                -(j + 1) * 28 : (n_steps * 28) - j * 28,
                i * 28 : (i + 1) * 28,
            ] = new_images[(i * n_steps) + j]

    combined_image = combined_image.clip(0, 1)
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(combined_image, cmap="bone", origin="upper")

    ax.set_xticks(28 * torch.arange(0, n_steps, 2), [round(x.item(), 4) for x in x_vals[::2]])
    ax.set_yticks(
        28 * torch.arange(0, n_steps, 2),
        reversed([round(y.item(), 4) for y in y_vals[::2]]),
    )
    fig.savefig(output_dir / "latent_space_recon.png")


if __name__ == "__main__":
    HP = HyperParams()

    train_ds = MNISTDataset()
    train_dl = DataLoader(
        train_ds, batch_size=HP.batch_size, shuffle=True, num_workers=2, pin_memory=True
    )

    vae = VAE(HP.latent_dim, HP.kl_div_coeff).to("cuda")
    optim = Adam(vae.parameters(), lr=HP.lr)

    train(vae, train_dl, optim, HP.epochs, HP.output_dir)
    show_latent_space(vae, train_ds, HP.output_dir)
