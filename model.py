"""Student-facing NFE-specific model definitions and checkpoint utilities."""

import torch
from torch import nn

from src.unet import ConditionalUNet
from src.utils import count_parameters, parameter_summary

class Model(nn.Module): # DO NOT MODIFY BASE CLASS
    """DO NOT MODIFY: Common parent class for both NFE-specific models."""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def count_parameters(self) -> int:
        """Return the total number of parameters, including frozen ones."""
        return count_parameters(self, trainable_only=False)

    @property
    def num_parameters(self) -> int:
        return self.count_parameters()

    def parameter_summary(self) -> dict[str, int]:
        return parameter_summary(self)

    def forward(self, x: torch.Tensor, timestep: torch.Tensor, **kwargs):
        return self.backbone(x, timestep, **kwargs)

    def sample(
        self,
        shape: tuple[int, ...],
        *,
        device: torch.device | str = "cuda",
        category: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        raise NotImplementedError("Implement sample() in the selected model class.")

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint_path: str,
        evaluate_mode: str,
        device="cpu",
        **kwargs,
    ) -> "Model":
        """Instantiate the selected model class and load its checkpoint."""
        model_classes = {
            "one_nfe": ModelOneNFE,
            "few_nfe": ModelFewNFE,
        }
        try:
            model_class = model_classes[evaluate_mode]
        except KeyError as error:
            raise ValueError(
                f"Unsupported evaluate_mode={evaluate_mode!r}; "
                "choose 'one_nfe' or 'few_nfe'."
            ) from error

        model = model_class(device=device, **kwargs)
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        return model.to(device).eval()


class ModelOneNFE(Model):
    """Student model evaluated with exactly one model function evaluation."""

    def __init__(self, device="cpu", base_ch: int = 128, **kwargs):
        backbone = ConditionalUNet(base_ch=base_ch)
        super().__init__(backbone=backbone)
        self.to(device)

    @torch.no_grad()
    def sample(
        self,
        shape: tuple[int, ...],
        *,
        device: torch.device | str = "cuda",
        category: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        device = torch.device(device)
        batch_size = shape[0]
        # Start from pure Gaussian noise at t = 1.0
        z = torch.randn(shape, device=device)
        t = torch.ones(batch_size, device=device)

        # Exactly 1 NFE: x_0 = z_1 - v(z_1, t=1, c)
        v = self.forward(z, t, category=category)
        x_0 = z - v
        return x_0.clamp(-1.0, 1.0)


class ModelFewNFE(Model):
    """Student model evaluated with no more than four model evaluations."""

    def __init__(self, device="cpu", base_ch: int = 128, num_steps: int = 4, **kwargs):
        backbone = ConditionalUNet(base_ch=base_ch)
        super().__init__(backbone=backbone)
        self.num_steps = min(num_steps, 4)  # Strict <= 4 NFE limit
        self.to(device)

    @torch.no_grad()
    def sample(
        self,
        shape: tuple[int, ...],
        *,
        device: torch.device | str = "cuda",
        category: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        device = torch.device(device)
        batch_size = shape[0]
        x = torch.randn(shape, device=device)

        # Euler ODE solver from t=1.0 down to t=0.0 in 4 steps (4 NFEs)
        dt = 1.0 / self.num_steps
        for step in range(self.num_steps):
            t_val = 1.0 - step * dt
            t = torch.full((batch_size,), t_val, device=device)
            v = self.forward(x, t, category=category)
            x = x - dt * v

        return x.clamp(-1.0, 1.0)
