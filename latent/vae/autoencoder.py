from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from cli_params import CLIParams
from sklearn.decomposition import PCA
from torch import nn
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import MNIST
from torchvision.transforms import v2


class SparsityType:
    L1 = "L1"
    KL = "KL"
    NONE = "NONE"


@dataclass
class HyperParams(CLIParams):
    batch_size: int = 128

    epochs: int = 64
    lr: float = 1e-3

    latent_dim: int = 8

    sparsity_coeff: float = 0.1
    sparsity_type: str = SparsityType.NONE
    target_sparsity: float = 0.01  # Only used for KL sparsity loss.

    contractive: bool = False
    contractive_coeff: float = 0.1

    output_dir: str = "outputs/autoencoder/test"


class AutoEncoder(nn.Module):
    def __init__(self, HP: HyperParams) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(256),
            nn.ReLU(),
            nn.LazyLinear(256),
            nn.ReLU(),
            nn.LazyLinear(HP.latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.LazyLinear(256),
            nn.ReLU(),
            nn.LazyLinear(256),
            nn.ReLU(),
            nn.LazyLinear(28 * 28),
        )

        self.HP = HP

    def forward(
        self, xb: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if self.HP.contractive:
            xb.requires_grad = True

        z = self.encoder(xb)
        recon = self.decoder(z).reshape(xb.shape)

        recon_loss = (0.5 * ((recon - xb) ** 2.0)).mean()
        sparsity_loss = self.calc_sparsity_loss(z)
        contractive_loss = self.contractive_loss(z, xb) if self.HP.contractive else 0.0

        loss = (
            recon_loss
            + (self.HP.sparsity_coeff * sparsity_loss)
            + (self.HP.contractive_coeff * contractive_loss)
        )

        metrics = {"recon_loss": recon_loss.item()}
        if self.HP.sparsity_type != SparsityType.NONE:
            metrics["sparsity_loss"] = sparsity_loss.item()
        if self.HP.sparsity_type == SparsityType.KL:
            metrics["z_l1"] = self.l1_sparsity_loss(z).item()
        if self.HP.contractive:
            metrics["contractive_loss"] = contractive_loss.item()

        return recon, loss, metrics

    def calc_sparsity_loss(self, z: torch.Tensor) -> torch.Tensor:
        match self.HP.sparsity_type:
            case SparsityType.L1:
                return self.l1_sparsity_loss(z)
            case SparsityType.KL:
                return self.kl_sparsity_loss(z)
            case _:
                return 0.0

    def l1_sparsity_loss(self, z: torch.Tensor) -> torch.Tensor:
        return z.abs().mean()

    def kl_sparsity_loss(self, z: torch.Tensor) -> torch.Tensor:
        l1 = torch.clamp(z.abs().mean(dim=0), min=1e-6, max=1 - 1e-6)
        kl1 = self.HP.target_sparsity * torch.log(self.HP.target_sparsity / l1)
        kl2 = (1 - self.HP.target_sparsity) * torch.log((1 - self.HP.target_sparsity) / (1 - l1))
        return (kl1 + kl2).mean()

    def contractive_loss(self, z: torch.Tensor, xb: torch.Tensor) -> torch.Tensor:
        jacobian = torch.autograd.grad(outputs=z.sum(), inputs=xb, retain_graph=True)[0]
        return (jacobian**2).sum() / xb.shape[0]


def get_mnist_data(batch_size: int) -> tuple[MNIST, DataLoader]:
    transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    train_ds = MNIST("data", train=True, download=True, transform=transform)

    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    return train_ds, train_dl


def train(
    autoencoder: AutoEncoder,
    train_dl: DataLoader,
    optim: Optimizer,
    epochs: int,
    output_dir: Path,
    n_plot_images: int = 4,
) -> None:
    autoencoder.train()
    writer = SummaryWriter(output_dir)

    for epoch in range(epochs):
        for xb, _ in train_dl:
            xb = xb.to("cuda")
            recon, loss, metrics = autoencoder(xb)

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
    autoencoder: AutoEncoder, train_ds: MNIST, output_dir: Path, n_samples: int = 2048
) -> None:
    # Calculate latents and the PCA into 2D.
    idx = torch.randint(0, len(train_ds), size=(n_samples,))
    images = train_ds.data[idx].to("cuda", dtype=torch.float32) / 255.0
    labels = train_ds.targets[idx]

    autoencoder.eval()
    latents = autoencoder.encoder(images).to("cpu")

    pca = PCA(n_components=2)
    latents_2D = pca.fit_transform(latents)

    # Plot the latent space.
    cmap = plt.get_cmap("jet", 10)
    fig, ax = plt.subplots(figsize=(10, 10))
    img = ax.scatter(latents_2D[:, 0], latents_2D[:, 1], c=labels, cmap=cmap)
    plt.colorbar(img, ax=ax)
    fig.savefig(output_dir / "latent_space.png")

    # Plot reconstructions of the latent space.
    n_steps = 30
    x_vals = torch.linspace(latents_2D[:, 0].min(), latents_2D[:, 0].max(), steps=n_steps)
    y_vals = torch.linspace(latents_2D[:, 1].min(), latents_2D[:, 1].max(), steps=n_steps)

    latent_grid = torch.cartesian_prod(x_vals, y_vals)
    new_latents = pca.inverse_transform(latent_grid).to("cuda", dtype=torch.float32)

    new_images = autoencoder.decoder(new_latents)
    new_images = new_images.reshape((n_steps * n_steps, 28, 28))

    combined_image = torch.empty((n_steps * 28, n_steps * 28), dtype=torch.float32)
    for i in range(n_steps):
        for j in range(n_steps):
            combined_image[
                -(j + 1) * 28 : (n_steps * 28) - j * 28,
                i * 28 : (i + 1) * 28,
            ] = new_images[(i * n_steps) + j]

    combined_image = combined_image.clip(0, 1)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(combined_image, cmap="bone", origin="upper")

    ax.set_xticks(28 * torch.arange(0, n_steps, 2), [round(x.item(), 4) for x in x_vals[::2]])
    ax.set_yticks(
        28 * torch.arange(0, n_steps, 2),
        reversed([round(y.item(), 4) for y in y_vals[::2]]),
    )
    fig.savefig(output_dir / "latent_space_recon.png")


if __name__ == "__main__":
    HP = HyperParams()

    train_ds, train_dl = get_mnist_data(HP.batch_size)
    autoencoder = AutoEncoder(HP).to("cuda")
    optim = Adam(autoencoder.parameters(), lr=HP.lr)

    train(autoencoder, train_dl, optim, HP.epochs, HP.output_dir)
    show_latent_space(autoencoder, train_ds, HP.output_dir)
