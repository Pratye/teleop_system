#!/usr/bin/env python3
"""Quick checks: MPS availability, optional Depth Pro initialization from checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--depth-pro-weights",
        type=str,
        default="",
        help="Path to depth_pro.pt (apple/ml-depth-pro)",
    )
    args = parser.parse_args()

    import torch  # noqa: PLC0415

    print("torch:", torch.__version__)
    print("torch.backends.mps.is_available:", torch.backends.mps.is_available())
    if torch.backends.mps.is_available():
        print("torch.backends.mps.is_built:", torch.backends.mps.is_built())

    if not args.depth_pro_weights:
        print(
            "Skipping Depth Pro (no checkpoint path).\n"
            "Clone https://github.com/apple/ml-depth-pro, pip install -e .\n"
            "Download depth_pro.pt, then:\n"
            "  python scripts/smoke_benchmark.py --depth-pro-weights PATH/depth_pro.pt",
        )
        return

    ck = Path(args.depth_pro_weights)
    if not ck.is_file():
        raise SystemExit(f"checkpoint not found: {ck}")

    try:
        from dataclasses import replace  # noqa: PLC0415

        from depth_pro.depth_pro import (  # noqa: PLC0415
            DEFAULT_MONODEPTH_CONFIG_DICT,
            create_model_and_transforms,
        )
    except ImportError as e:
        raise SystemExit(
            "depth_pro not importable — install apple/ml-depth-pro editable — see requirements.txt.",
        ) from e

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    cfg = replace(DEFAULT_MONODEPTH_CONFIG_DICT, checkpoint_uri=str(ck.resolve()))
    model, _transform = create_model_and_transforms(cfg, device=device)
    print("Loaded Depth Pro — device:", device, "checkpoint:", ck)
    model.eval()


if __name__ == "__main__":
    main()
