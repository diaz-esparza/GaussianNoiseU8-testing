import torch
import torchvision.transforms.v2 as v2

from typing import Protocol

__all__ = [
    "GaussianNoiseU8",
    "FloatAndBack",
    "IntermediateInt",
]


class GaussianNoiseU8(Protocol):
    def __call__(
        self,
        inpt: torch.Tensor,
        mean: float = 0,
        sigma: float = 0.1,
        clip: bool = True,
    ) -> torch.Tensor: ...


class FloatAndBack(GaussianNoiseU8):
    def __init__(
        self,
        dtype_f: torch.dtype,
    ) -> None:
        super().__init__()
        if not dtype_f.is_floating_point:
            raise ValueError(f"dtype_f is expected to be float, got {dtype_f=}")

        self.to_float = v2.ToDtype(dtype_f, scale=True)
        self.to_uint8 = v2.ToDtype(torch.uint8, scale=True)
        self.dtype_f = dtype_f

    def __call__(
        self,
        inpt: torch.Tensor,
        mean: float = 0,
        sigma: float = 0.1,
        clip: bool = True,
    ) -> torch.Tensor:
        if sigma < 0:
            raise ValueError(f"sigma shouldn't be negative. Got {sigma}")

        inpt_f = self.to_float(inpt)

        noise = mean + torch.randn_like(inpt, dtype=self.dtype_f) * sigma
        out = inpt_f + noise

        if clip:
            out = torch.clamp(out, 0, 1)
        return self.to_uint8(out)


class IntermediateInt(GaussianNoiseU8):
    def __init__(
        self,
        dtype_i: torch.dtype,
        dtype_f: torch.dtype,
        implicit: bool = False,
        early_conversion: bool = False,
        rounding: bool = False,
    ) -> None:
        if dtype_i.is_floating_point or dtype_i.is_complex or (not dtype_i.is_signed):
            raise ValueError(f"dtype_i is expected to be a signed int, got {dtype_i=}")
        if not dtype_f.is_floating_point:
            raise ValueError(f"dtype_f is expected to be float, got {dtype_f=}")

        self.to_inter = v2.ToDtype(dtype_i, scale=False)
        self.to_uint8 = v2.ToDtype(torch.uint8, scale=False)
        self.dtype_i = dtype_i
        self.dtype_f = dtype_f
        self.implicit = implicit
        self.early_conversion = early_conversion
        self.rounding = rounding

    def __call__(
        self,
        inpt: torch.Tensor,
        mean: float = 0,
        sigma: float = 0.1,
        clip: bool = True,
    ) -> torch.Tensor:
        if sigma < 0:
            raise ValueError(f"sigma shouldn't be negative. Got {sigma}")

        if self.implicit:
            inpt_i = inpt
        else:
            inpt_i = self.to_inter(inpt)

        noise = torch.randn_like(inpt, dtype=self.dtype_f) * (sigma * 255)

        if self.early_conversion:
            if self.rounding:
                noise = noise.round()
            noise = self.to_inter(noise)
            noise += round(mean * 255)

        else:
            noise += mean * 255
            if self.rounding:
                noise = noise.round()
            noise = self.to_inter(noise)

        out = inpt_i + noise

        if clip:
            out = torch.clamp(out, 0, 255)
        return self.to_uint8(out)
