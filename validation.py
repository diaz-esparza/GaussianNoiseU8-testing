from os.path import dirname, join as jo
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import os
import torch

from core import GaussianNoiseU8, FloatAndBack, IntermediateInt

from typing import Any

ROOT = dirname(__file__)
SAVE_DIR = jo(ROOT, "output_validation")
os.makedirs(SAVE_DIR, exist_ok=True)


MATSIZE = 1000
MEAN = 0.25
SIGMA = 0.5
CONFIGS: list[tuple[str, type[GaussianNoiseU8], dict[str, Any]]] = [
    *[
        (
            f"FAB{str(dtype_f)[-2:]}",
            FloatAndBack,
            {"dtype_f": dtype_f},
        )
        for dtype_f in [torch.float32, torch.float16]
    ],
    *[
        (
            f"II{str(dtype_i)[-2:]}{str(dtype_f)[-2:]}{str(implicit)[0]}{str(rounding)[0]}",
            IntermediateInt,
            {
                "dtype_i": dtype_i,
                "dtype_f": dtype_f,
                "implicit": implicit,
                "rounding": rounding,
            },
        )
        for dtype_f in [torch.float32]
        for dtype_i in [torch.int16, torch.int32]
        for implicit in [True, False]
        for rounding in [True, False]
    ],
]


def plot_matrix(
    x: npt.NDArray,
    names: list[str],
    cmap: str = "Reds",
    filename: str = "mat.png",
):
    assert x.ndim == 2
    assert x.shape[0] == x.shape[1] == len(names)
    fig, ax = plt.subplots(figsize=(8, 8))

    cax = ax.matshow(x, cmap=cmap)
    plt.title("Avg per-pixel difference (scale [0, 255])", pad=20)
    fig.colorbar(cax)

    ax.set_xticks(np.arange(len(x)))
    ax.set_yticks(np.arange(len(x)))

    ax.set_xticklabels(names, rotation=45, ha="left")
    ax.set_yticklabels(names)

    # Annotate each cell with the number
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            ax.text(
                j,
                i,
                str(x[i, j]),
                va="center",
                ha="center",
                color="black",
            )

    plt.tight_layout()
    plt.savefig(jo(SAVE_DIR, filename))
    plt.close()


def image_diff(__x1: torch.Tensor, __x2: torch.Tensor):
    return torch.mean(
        torch.abs(__x1.to(torch.float64) - __x2.to(torch.float64))
    ).numpy()


def benchmark_precision(inpt: torch.Tensor) -> npt.NDArray[np.float64]:
    torch.use_deterministic_algorithms(True)

    ref_outputs: list[torch.Tensor] = []
    for _, transform_cls, kwargs in CONFIGS:
        f = transform_cls(**kwargs)
        torch.manual_seed(33)
        ref_outputs.append(f(inpt, mean=MEAN, sigma=SIGMA))

    diff_mat = np.zeros((len(CONFIGS),) * 2, dtype=np.float64)
    for i in range(diff_mat.shape[0]):
        for j in range(diff_mat.shape[1]):
            if i == j:
                # Make another sample to compare intra-config variance
                _, transform_cls, kwargs = CONFIGS[i]
                f = transform_cls(**kwargs)
                torch.manual_seed(33)
                another_output = f(inpt, mean=MEAN, sigma=SIGMA)
                diff_mat[i][j] = image_diff(ref_outputs[i], another_output)
            else:
                if CONFIGS[i][0] == "FAB32":
                    print(
                        f"Stat diff between {{'{CONFIGS[i][0]}', '{CONFIGS[j][0]}'}}"
                        + " (scale [0, 255])"
                    )
                    print(f"mean\tstd")
                    print(
                        f"{
                    (
                        torch.mean(ref_outputs[i].to(torch.float64))
                        - torch.mean(ref_outputs[j].to(torch.float64))
                    ).abs()
                    .item():.2f}"
                        "\t"
                        f"{
                    (
                        torch.std(ref_outputs[i].to(torch.float64))
                        - torch.std(ref_outputs[j].to(torch.float64))
                    ).abs()
                    .item():.2f}"
                        "\n\n",
                    )

                diff_mat[i][j] = image_diff(ref_outputs[i], ref_outputs[j])
    return diff_mat.round(2)


def main():

    names = list(map(lambda x: x[0], CONFIGS))

    # Cool gradient!!11
    inpt_usual = torch.zeros((3, MATSIZE, MATSIZE), dtype=torch.uint8)
    i = 0
    j = 0
    k = 0
    for i in range(MATSIZE // 3):
        inpt_usual[:, i + j + k, :] = torch.tensor(
            [
                (i - j) * 255 // (MATSIZE // 3 - 1),
                (j - k) * 255 // (MATSIZE // 3 - 1),
                k * 255 // (MATSIZE // 3 - 1),
            ],
            dtype=torch.uint8,
        ).reshape((3, 1))
    for j in range(MATSIZE // 3):
        inpt_usual[:, i + j + k, :] = torch.tensor(
            [
                (i - j) * 255 // (MATSIZE // 3 - 1),
                (j - k) * 255 // (MATSIZE // 3 - 1),
                k * 255 // (MATSIZE // 3 - 1),
            ],
            dtype=torch.uint8,
        ).reshape((3, 1))
    while (i + j + k) < MATSIZE:
        inpt_usual[:, i + j + k, :] = torch.tensor(
            [
                (i - j) * 255 // (MATSIZE // 3 - 1),
                (j - k) * 255 // (MATSIZE // 3 - 1),
                min(k * 255 // (MATSIZE // 3 - 1), 255),
            ],
            dtype=torch.uint8,
        ).reshape((3, 1))
        k += 1

    diff_mat_usual = benchmark_precision(inpt_usual)
    plot_matrix(
        diff_mat_usual,
        names,
        filename="diff_usual",
    )
    plt.imshow(np.moveaxis(inpt_usual.numpy(), 0, -1))
    plt.savefig(jo(SAVE_DIR, "noise_input.png"))
    plt.close()

    transform_cls = None
    for _, transform_cls, kwargs in CONFIGS:
        if transform_cls == FloatAndBack:
            break
    else:
        raise AssertionError()
    plt.imshow(
        np.moveaxis(
            transform_cls(**kwargs)(
                inpt_usual,
                mean=MEAN,
                sigma=SIGMA,
            ).numpy(),
            0,
            -1,
        )
    )
    plt.savefig(jo(SAVE_DIR, f"noise_output_float.png"))
    plt.close()

    for _, transform_cls, kwargs in CONFIGS:
        if transform_cls == IntermediateInt:
            break
    else:
        raise AssertionError()
    plt.imshow(
        np.moveaxis(
            transform_cls(**kwargs)(
                inpt_usual,
                mean=MEAN,
                sigma=SIGMA,
            ).numpy(),
            0,
            -1,
        )
    )
    plt.savefig(jo(SAVE_DIR, f"noise_output_int.png"))
    plt.close()

    inpt_max = torch.full_like(inpt_usual, 255, dtype=torch.uint8)
    diff_mat_max = benchmark_precision(inpt_max)
    plot_matrix(
        diff_mat_max,
        names,
        filename="diff_max",
    )

    inpt_min = torch.full_like(inpt_usual, 0, dtype=torch.uint8)
    diff_mat_min = benchmark_precision(inpt_min)
    plot_matrix(
        diff_mat_min,
        names,
        filename="diff_min",
    )


if __name__ == "__main__":
    main()
