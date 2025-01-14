from __future__ import annotations

from dataclasses import dataclass

import torch
from cli_params import CLIParams
from torch import nn
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import MNIST
from torchvision.transforms import v2


@dataclass
class HyperParams(CLIParams):
    latent_dim: int = 128
    relu_leak: float = 0.2
    conv_filters: int = 128

    lr: float = 1e-4
    batch_size: int = 128
    train_epochs: int = 64

    output_dir: str = "outputs/gan/test"


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        transposed: bool = False,
        relu_leak: float | None = None,
    ) -> None:
        super().__init__()

        conv = nn.ConvTranspose2d if transposed else nn.Conv2d
        self.conv = conv(in_channels, out_channels, kernel_size, stride, padding, bias=False)

        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(relu_leak) if relu_leak is not None else nn.ReLU()

    def forward(self, xb: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(xb)))


class GAN(nn.Module):
    def __init__(
        self, latent_dim: int, data_channels: int, conv_filters: int, relu_leak: float
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim

        self.generator = nn.Sequential(
            # (latent_dim, 1, 1).
            ConvBlock(
                latent_dim,
                conv_filters * 4,
                kernel_size=4,
                stride=1,
                padding=0,
                transposed=True,
            ),
            # (conv_filters*4, 4, 4).
            ConvBlock(conv_filters * 4, conv_filters * 2, transposed=True),
            # (conv_filters*2, 8, 8).
            ConvBlock(conv_filters * 2, conv_filters, transposed=True),
            # (conv_filters, 16, 16).
            nn.ConvTranspose2d(
                conv_filters,
                data_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=True,
            ),
            nn.Sigmoid(),
            # (data_channels, 32, 32).
        )
        self.discriminator = nn.Sequential(
            # (data_channels, 32, 32).
            nn.Conv2d(data_channels, conv_filters, kernel_size=4, stride=2, padding=1, bias=True),
            nn.LeakyReLU(relu_leak),
            # (conv_filters, 16, 16).
            ConvBlock(conv_filters, conv_filters * 2, relu_leak=relu_leak),
            # (conv_filters*2, 8, 8).
            ConvBlock(conv_filters * 2, conv_filters * 4, relu_leak=relu_leak),
            # (conv_filters*4, 4, 4).
            nn.Conv2d(conv_filters * 4, 1, kernel_size=4, stride=1, padding=0, bias=True),
            # (1, 1, 1).
            nn.Flatten(),
            nn.Sigmoid(),
        )

        self.apply(self.weight_init)

    def optim_step(
        self, xb: torch.Tensor, discriminator_optim: Optimizer, generator_optim: Optimizer
    ) -> tuple[float, float, float, float, torch.Tensor]:
        # Generate new images.
        z = torch.rand((xb.shape[0], self.latent_dim, 1, 1), dtype=torch.float32, device="cuda")
        generated = self.generator(z)

        # Optimize discriminator.
        discriminator_optim.zero_grad()

        real_probs = self.discriminator(xb)
        generated_probs = self.discriminator(generated.detach())

        eps = torch.finfo(torch.float32).eps
        discriminator_loss = (
            -torch.log(real_probs + eps).mean() - torch.log(1.0 - generated_probs + eps).mean()
        )

        discriminator_loss.backward()
        discriminator_optim.step()

        # Optimize generator.
        generator_optim.zero_grad()

        generated_probs = self.discriminator(generated)  # Recalc b/c discriminator was optimized.
        generator_loss = -torch.log(generated_probs + eps).mean()

        generator_loss.backward()
        generator_optim.step()

        return (
            discriminator_loss.item(),
            generator_loss.item(),
            real_probs.mean().item(),
            generated_probs.mean().item(),
            generated.detach().to("cpu"),
        )

    def weight_init(self, module: nn.Module) -> None:
        name = module.__class__.__name__
        if name in ("Conv2d", "Conv2dTranspose"):
            nn.init.normal_(module.weight.data, 0.0, 0.02)
        elif "BatchNorm" in name:
            nn.init.normal_(module.weight.data, 1.0, 0.02)
            nn.init.zeros_(module.bias.data)


if __name__ == "__main__":
    HP = HyperParams()

    transform = v2.Compose([v2.PILToTensor(), v2.ToDtype(torch.float32, scale=True), v2.Resize(32)])
    train_ds = MNIST("data", train=True, download=True, transform=transform)
    train_dl = DataLoader(
        train_ds, batch_size=HP.batch_size, shuffle=True, num_workers=2, pin_memory=True
    )

    gan = (
        GAN(HP.latent_dim, train_ds[0][0].shape[0], HP.conv_filters, HP.relu_leak)
        .to("cuda")
        .train()
    )
    discriminator_optim = Adam(gan.discriminator.parameters(), lr=HP.lr, betas=(0.5, 0.999))
    generator_optim = Adam(gan.generator.parameters(), lr=HP.lr, betas=(0.5, 0.999))

    logger = SummaryWriter(HP.output_dir)

    for epoch in range(HP.train_epochs):
        for xb, _ in train_dl:
            xb = xb.to("cuda")

            discriminator_loss, generator_loss, mean_real_prob, mean_generated_prob, generated = (
                gan.optim_step(xb, discriminator_optim, generator_optim)
            )

        total_loss = discriminator_loss + generator_loss
        loss_vals = {
            "total": total_loss,
            "discriminator": discriminator_loss,
            "generator": generator_loss,
        }
        logger.add_scalars("loss", loss_vals, epoch)

        logger.add_scalars(
            "probs", {"real": mean_real_prob, "generated": mean_generated_prob}, epoch
        )

        logger.add_images("generated", generated[:24], epoch)

        print(f"Epoch {epoch}: loss={total_loss:.6f}")

    logger.close()
