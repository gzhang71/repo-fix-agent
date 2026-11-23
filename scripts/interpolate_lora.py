#!/usr/bin/env python3
"""Scale a LoRA adapter toward the base model (WiSE-FT style weight interpolation).

A LoRA adapter adds delta W = B @ A (times alpha/r) to each base weight. Scaling every lora_B
matrix by `alpha` gives exactly base + alpha * delta: alpha=1 is the fine-tuned model, alpha=0 is
the base. Values in between usually keep most of the task gain while giving back much of what
fine-tuning overwrote, so this is the cheap fix when a fine-tune improves the task but fails
`repo-fix forgetting-check`. No model weights are loaded; only the adapter file is rewritten.

  python scripts/interpolate_lora.py --adapter runs/train/r1/lora_dpo --alpha 0.5 --out runs/train/r1/lora_dpo_a050
  # sweep: evaluate alpha in {0.25, 0.5, 0.75, 1.0} on the dev split, keep the best that passes the gate
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

WEIGHT_FILES = ("adapter_model.safetensors", "adapter_model.bin")


def scale_state(state: dict, alpha: float) -> tuple[dict, int]:
    out, n = {}, 0
    for k, v in state.items():
        if "lora_B" in k:
            out[k] = v * alpha
            n += 1
        else:
            out[k] = v
    return out, n


def interpolate(adapter: str | Path, alpha: float, out: str | Path) -> int:
    adapter, out = Path(adapter), Path(out)
    src = next((adapter / f for f in WEIGHT_FILES if (adapter / f).exists()), None)
    if src is None:
        raise FileNotFoundError(f"no adapter weights in {adapter} ({' / '.join(WEIGHT_FILES)})")
    if out.resolve() == adapter.resolve():
        raise ValueError("--out must differ from --adapter")
    out.mkdir(parents=True, exist_ok=True)
    for f in adapter.iterdir():          # config, tokenizer files, README
        if f.is_file() and f.name not in WEIGHT_FILES:
            shutil.copy2(f, out / f.name)
    if src.suffix == ".safetensors":
        from safetensors.torch import load_file, save_file
        state, n = scale_state(load_file(str(src)), alpha)
        save_file(state, str(out / src.name))
    else:
        import torch
        state, n = scale_state(torch.load(src, map_location="cpu"), alpha)
        torch.save(state, out / src.name)
    if n == 0:
        raise ValueError(f"{src} has no lora_B tensors; is this a LoRA adapter?")
    (out / "interpolation.json").write_text(json.dumps({"from": str(adapter), "alpha": alpha, "scaled": n}, indent=2))
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--alpha", type=float, required=True, help="0 = base model, 1 = the fine-tuned adapter")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if not 0 <= a.alpha <= 1.5:
        raise SystemExit("--alpha should be in [0, 1] (up to 1.5 to extrapolate)")
    n = interpolate(a.adapter, a.alpha, a.out)
    print(f"scaled {n} lora_B tensors by {a.alpha} -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
