from __future__ import annotations

from dataclasses import dataclass

import torch
from cli_params import CLIParams
from torch import nn
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import CIFAR10, MNIST
from torchvision.transforms import v2


@dataclass
class HyperParams(CLIParams):
    latent_dim: int = 64
    relu_leak: float = 0.2
    conv_filters: int = 32
    grad_penalty_coeff: float = 10.0

    lr: float = 1e-4
    batch_size: int = 128
    train_steps: int = 64_000
    critic_steps: int = 5
    log_steps: int = 500

    momentum_beta_1: float = 0.0
    momentum_beta_2: float = 0.9

    cifar_10: bool = False

    output_dir: str = "outputs/gan/wgan/test"


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
        self.conv = conv(in_channels, out_channels, kernel_size, stride, padding, bias=True)
        self.act = nn.LeakyReLU(relu_leak) if relu_leak is not None else nn.ReLU()

    def forward(self, xb: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(xb))


def weight_init(module: nn.Module) -> None:
    name = module.__class__.__name__
    if name in ("Conv2d", "Conv2dTranspose"):
        nn.init.kaiming_normal_(module.weight.data, mode="fan_in")


def Critic(HP: HyperParams, data_channels: int) -> nn.Module:
    critic = nn.Sequential(
        # (data_channels, 32, 32).
        ConvBlock(data_channels, HP.conv_filters, relu_leak=HP.relu_leak),
        # (conv_filters, 16, 16).
        ConvBlock(HP.conv_filters, HP.conv_filters * 2, relu_leak=HP.relu_leak),
        # (conv_filters*2, 8, 8).
        ConvBlock(HP.conv_filters * 2, HP.conv_filters * 4, relu_leak=HP.relu_leak),
        # (conv_filters*4, 4, 4).
        nn.Flatten(),
        # (conv_filters*4*4*4,).
        nn.Linear(HP.conv_filters * 4 * 4 * 4, 1),
    )
    critic.apply(weight_init)
    return critic


def Generator(HP: HyperParams, data_channels: int) -> nn.Module:
    generator = nn.Sequential(
        # (latent_dim, 1, 1).
        ConvBlock(
            HP.latent_dim,
            HP.conv_filters * 4,
            kernel_size=4,
            stride=1,
            padding=0,
            transposed=True,
        ),
        # (conv_filters*4, 4, 4).
        ConvBlock(HP.conv_filters * 4, HP.conv_filters * 2, transposed=True),
        # (conv_filters*2, 8, 8).
        ConvBlock(HP.conv_filters * 2, HP.conv_filters, transposed=True),
        # (conv_filters, 16, 16).
        nn.ConvTranspose2d(
            HP.conv_filters,
            data_channels,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=True,
        ),
        nn.Sigmoid(),
        # (data_channels, 32, 32).
    )
    generator.apply(weight_init)
    return generator


def optim_step(
    xb: torch.Tensor,
    critic: nn.Module,
    generator: nn.Module,
    critic_optim: Optimizer,
    generator_optim: Optimizer,
    step: int,
    HP: HyperParams,
) -> None:
    # Generate new images.
    z = torch.rand((HP.batch_size, HP.latent_dim, 1, 1), dtype=torch.float32, device="cuda")
    fake_xb = generator(z)

    # Optimize critic.
    critic_optim.zero_grad()

    real_loss = -critic(xb).mean()
    fake_loss = critic(fake_xb.detach()).mean()

    alpha = torch.rand((HP.batch_size, 1, 1, 1), dtype=torch.float32, device="cuda")
    interps = (alpha * xb) + ((1 - alpha) * fake_xb.detach())
    interps.requires_grad = True
    jacobian = torch.autograd.grad(
        outputs=critic(interps).sum(), inputs=interps, create_graph=True
    )[0]
    grad_norm = jacobian.view(jacobian.shape[0], -1).norm(dim=-1).mean()
    grad_penalty = HP.grad_penalty_coeff * ((grad_norm - 1.0) ** 2.0)

    critic_loss = real_loss + fake_loss + grad_penalty
    critic_loss.backward()
    critic_optim.step()

    # Optimize generator.
    is_last_step = step == (HP.train_steps - 1)
    if ((step % HP.critic_steps) == 0) or is_last_step:
        generator_optim.zero_grad()
        generator_loss = -1.0 * critic(fake_xb).mean()
        generator_loss.backward()
        generator_optim.step()

    # Log metrics.
    if ((step % HP.log_steps) == 0) or is_last_step:
        total_loss = critic_loss.item() + generator_loss.item()
        loss_vals = {
            "total": total_loss,
            "critic": critic_loss.item(),
            "generator": generator_loss.item(),
        }
        logger.add_scalars("loss", loss_vals, step)
        logger.add_scalar("grad_norm", grad_norm.item(), step)
        logger.add_images("generated", fake_xb.detach().to("cpu")[:24], step)
        print(f"Step {step}: loss={total_loss:.6f}")


if __name__ == "__main__":
    HP = HyperParams()
    assert (HP.train_steps % HP.critic_steps) == 0

    transform = v2.Compose([v2.PILToTensor(), v2.ToDtype(torch.float32, scale=True), v2.Resize(32)])
    if HP.cifar_10:
        train_ds = CIFAR10("data", train=True, download=True, transform=transform)
    else:
        train_ds = MNIST("data", train=True, download=True, transform=transform)

    train_dl = DataLoader(
        train_ds,
        batch_size=HP.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    data_channels = train_ds[0][0].shape[0]
    critic = Critic(HP, data_channels).to("cuda").train()
    generator = Generator(HP, data_channels).to("cuda").train()

    betas = (HP.momentum_beta_1, HP.momentum_beta_2)
    critic_optim = Adam(critic.parameters(), lr=HP.lr, betas=betas)
    generator_optim = Adam(generator.parameters(), lr=HP.lr, betas=betas)

    logger = SummaryWriter(HP.output_dir)

    dl_iter = iter(train_dl)
    for step in range(HP.train_steps):
        try:
            xb, _ = next(dl_iter)
        except StopIteration:
            dl_iter = iter(train_dl)
            xb, _ = next(dl_iter)
        xb = xb.to("cuda")

        optim_step(xb, critic, generator, critic_optim, generator_optim, step, HP)

    del dl_iter  # HACK: Error with infinite data loader and threads.
    logger.close()
