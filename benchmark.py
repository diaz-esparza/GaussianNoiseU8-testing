from argparse import ArgumentParser
from os.path import dirname, join as jo

import functools
import os
import torch
import torch.utils.benchmark as benchmark
import yaml

from typing import Any

from core import GaussianNoiseU8, FloatAndBack, IntermediateInt

try:
    ROOT = dirname(__file__)
except NameError:
    ROOT = "."

METRICS = jo(ROOT, "metrics")

PARSER = ArgumentParser(
    description="Benchmark different ways to gaussian noise to an image. "
    "All configs test converting to a float and back and converting the noise to an intermediate int dtype."
)

PARSER.add_argument(
    "sizes",
    type=int,
    nargs="+",
    help="Image sizes to test (one or more). Must input the side's length, as all test images will be square.",
)

PARSER.add_argument(
    "-o",
    "--output",
    type=str,
    help="Optional output dir. Default dir is '__root__/metrics', "
    "where '__root__' is the directory containing the current script.",
)

PARSER.add_argument(
    "-b",
    "--budget",
    type=float,
    default=300,
    help="Minimum time budget in seconds. Actual completion time might be greater.",
)

PARSER.add_argument(
    "-g",
    "--gpu",
    action="store_true",
    help="Test GPU (cuda) performance.",
)

FLOAT_DTYPES = {
    str(i).replace("torch.", ""): i
    for i in [torch.float16, torch.float32, torch.float64]
}
PARSER.add_argument(
    "-f",
    "--float-dtypes",
    choices=FLOAT_DTYPES.keys(),
    nargs="+",
    default=FLOAT_DTYPES.keys(),
    help="Dtypes to generate the noise in (and convert the image to in the case of the 'FloatAndBack' method).",
)

INT_DTYPES = {
    str(i).replace("torch.", ""): i for i in [torch.int16, torch.int32, torch.int64]
}
PARSER.add_argument(
    "-i",
    "--int-dtypes",
    choices=INT_DTYPES.keys(),
    nargs="+",
    default=INT_DTYPES.keys(),
    help="Int dtypes to tranform the noise into (as an intermediate step) to add to the image.",
)

PARSER.add_argument(
    "-R",
    "--rounding",
    action="store_true",
    help="Test performance on rounding the noise (on the 'IntermediateInt' method). "
    "By default it does not round.",
)
PARSER.add_argument(
    "-I",
    "--implicit",
    action="store_true",
    help="Test performance on implicit conversion of the input (on both methods). "
    "By default said conversion is explicit.",
)
PARSER.add_argument(
    "-E",
    "--early-conversion",
    action="store_true",
    help="Test performance on early conversion of the noise to the intermediate (int) dtype before adding the mean (on the 'IntermediateInt' method). "
    "By default said early conversion does not happen.",
)

PARSER.add_argument(
    "-v",
    "--verbose",
    action="store_true",
    help="Enable verbose output.",
)
VERBOSE = False


def dtype_yaml(dumper: yaml.Dumper, data: torch.dtype) -> yaml.Node:
    return dumper.represent_scalar("tag:yaml.org,2002:str", str(data))


def execute_benchmark(
    targets: dict[str, torch.Tensor],
    config: list[tuple[type[GaussianNoiseU8], dict[str, Any]]],
    log_dir: str,
    budget: float = 5,
) -> None:
    os.makedirs(log_dir, exist_ok=True)

    exception = None
    try:
        for t_name, t in targets.items():
            output: list[dict[str, Any]] = []
            print(f"Running {t_name}: ", end="")
            try:
                for idx, (t_class, kwargs) in enumerate(config):
                    print(f"{idx}, ", end="", flush=True)
                    global t_loaded
                    t_loaded = functools.partial(
                        t_class(**kwargs),
                        t,
                        mean=0.1,
                        sigma=2,
                        clip=True,
                    )

                    timings = benchmark.Timer(
                        stmt="t_loaded()",
                        setup="from __main__ import t_loaded",
                        num_threads=1,
                    ).blocked_autorange(min_run_time=budget)

                    if VERBOSE:
                        print(f"\nDone @ {timings.mean} seconds. {t_class=} {kwargs=}")

                    output.append(
                        {
                            "method": t_class.__name__,
                            "mean": timings.mean,
                            "stats": {
                                "iqr": timings.iqr,
                                "median": timings.median,
                            },
                            "args": kwargs,
                        }
                    )
                print("Done!")

            except BaseException as e:
                exception = e
                # Exits loop but still executes the finally block if it was
                # a KeyboardInterrupt

            finally:

                match exception:
                    case None:
                        ...
                    case KeyboardInterrupt():
                        print("\nKeyboardInterrupt detected, saving data...")
                    case e:
                        raise e

                # Dtypes needs to be turned into strings in the yaml
                yaml.add_multi_representer(torch.dtype, dtype_yaml)
                yaml.Dumper.ignore_aliases = lambda *args: True  # type:ignore
                # Save metrics from most to least performant
                with open(jo(log_dir, t_name + ".yaml"), "wt") as f:
                    for i in sorted(output, key=lambda x: x["mean"]):
                        f.write(
                            yaml.dump(
                                [i],
                                sort_keys=False,
                                indent=4,
                                line_break="\n" * 2,
                            )
                            + "\n"
                        )

                if isinstance(exception, KeyboardInterrupt):
                    raise KeyboardInterrupt

    except KeyboardInterrupt:
        return


def main() -> None:
    args = PARSER.parse_args()
    global VERBOSE
    VERBOSE = args.verbose
    global METRICS
    METRICS = METRICS if args.output is None else args.output

    benchmark_suite: list[tuple[type[GaussianNoiseU8], dict[str, Any]]] = [
        *[
            (FloatAndBack, {"dtype_f": dtype_f})
            for dtype_f in (torch.float16, torch.float32, torch.float64)
        ],
        *[
            (
                IntermediateInt,
                {
                    "dtype_i": dtype_i,
                    "dtype_f": dtype_f,
                    "implicit": implicit,
                    "early_conversion": early_conversion,
                    "rounding": rounding,
                },
            )
            for dtype_i in map(INT_DTYPES.__getitem__, args.int_dtypes)
            for dtype_f in map(FLOAT_DTYPES.__getitem__, args.float_dtypes)
            for implicit in {PARSER.get_default("implicit"), args.implicit}
            for early_conversion in {
                PARSER.get_default("early_conversion"),
                args.early_conversion,
            }
            for rounding in {PARSER.get_default("rounding"), args.rounding}
        ],
    ]

    benchmark_targets = {
        f"size_{length}": (torch.rand((3, length, length)) * 255)
        .round()
        .to(torch.uint8)
        for length in args.sizes
    }

    budget_per_test = args.budget / (len(benchmark_suite) * len(args.sizes))

    print(f"Got {len(benchmark_suite)} configs.")
    print(
        f"Got {len(args.sizes)} image sizes:\n"
        f"    [{', '.join(f'{i}x{i}' for i in args.sizes)}]"
    )
    print(f"Total number of tests: {len(benchmark_suite)*len(args.sizes)}")
    print(
        f"    @ a budget of {args.budget/60:.2f} minutes: "
        f"> {budget_per_test:.0f} seconds/test"
    )
    print(f"Using {'GPU' if args.gpu else 'CPU'}")

    if args.gpu:
        for k, v in benchmark_targets.items():
            benchmark_targets[k] = v.to("cuda")

        execute_benchmark(
            benchmark_targets,
            benchmark_suite,
            jo(METRICS, "gpu"),
            budget_per_test,
        )
    else:
        execute_benchmark(
            benchmark_targets,
            benchmark_suite,
            jo(METRICS, "cpu"),
            budget_per_test,
        )


if __name__ == "__main__":
    main()
