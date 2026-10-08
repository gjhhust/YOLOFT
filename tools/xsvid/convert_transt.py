"""Convert the SHA-pinned official TransT pickle into portable CPU tensors."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

SOURCE_SHA256 = "b4460d6fa4e3b4ab0bc119602953e3cacdfa770907e13ff1398f652145cd8fc0"
HEAD_PREFIXES = ("featurefusion_network.", "class_embed.", "bbox_embed.")


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def portable_state(checkpoint):
    state = checkpoint.get("net") if isinstance(checkpoint, dict) else None
    if not isinstance(state, dict) or len(state) != 430:
        raise ValueError("Expected the official 430-tensor TransT net")
    if not all(isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in state.items()):
        raise ValueError("TransT net must contain only named tensors")
    if sum(key.startswith(HEAD_PREFIXES) for key in state) != 170:
        raise ValueError("Expected all 170 TransT head tensors")
    if any(not torch.isfinite(value).all().item() for value in state.values()):
        raise ValueError("Non-finite TransT tensor")
    return {key: value.detach().cpu().clone() for key, value in state.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--upstream-source", type=Path, required=True,
                        help="Local clone of https://github.com/chenxin-dlut/TransT")
    args = parser.parse_args()
    if args.source.resolve() == args.output.resolve() or args.output.exists():
        raise ValueError("Output must be new and distinct from the source")
    if sha256(args.source) != SOURCE_SHA256:
        raise ValueError("Official TransT source SHA mismatch; pickle was not loaded")
    if not (args.upstream_source / "ltr/admin/model_constructor.py").is_file():
        raise ValueError("Missing official TransT source checkout")
    sys.path.insert(0, str(args.upstream_source.resolve()))
    # The legacy constructor pickle is loaded only after checking its exact published-source bytes.
    original = torch.load(args.source, map_location="cpu", weights_only=False)
    state = portable_state(original)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"net": state}, args.output)
    restored = torch.load(args.output, map_location="cpu", weights_only=True)["net"]
    if state.keys() != restored.keys() or any(not torch.equal(state[key], restored[key]) for key in state):
        raise RuntimeError("Portable tensor round-trip mismatch")
    print(json.dumps({"source_sha256": SOURCE_SHA256, "output_sha256": sha256(args.output),
                      "tensor_count": len(state), "head_tensor_count": 170,
                      "all_tensors_bit_identical": True}))


if __name__ == "__main__":
    main()
