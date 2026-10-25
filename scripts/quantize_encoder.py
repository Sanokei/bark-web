#!/usr/bin/env python3
"""Quantize the Bark fp32 encoder to int8 (dynamic) as a SINGLE .onnx file.

Why: the fp32 encoder is 2.4GB with external data (encoder-model.onnx.data).
onnxruntime-web in Node cannot load external-data models
("Module.MountedFiles is not available"), and 2.4GB is unusable as a browser
download anyway. Dynamic int8 quantization shrinks it to ~600MB, under the
2GB single-file protobuf limit, so it loads everywhere with no .data sidecar.

Usage: python scripts/quantize_encoder.py \
    --input models/bark-onnx/encoder-model.onnx \
    --output models/bark-onnx/encoder-model.int8.onnx
"""
import argparse
import sys
from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="models/bark-onnx/encoder-model.onnx")
    ap.add_argument("--output", default="models/bark-onnx/encoder-model.int8.onnx")
    args = ap.parse_args()
    inp, out = Path(args.input), Path(args.output)
    if not inp.exists():
        print(f"input not found: {inp}", file=sys.stderr)
        return 1
    print(f"quantizing {inp} -> {out} (dynamic int8, this takes a few minutes)...")
    quantize_dynamic(
        model_input=str(inp),
        model_output=str(out),
        weight_type=QuantType.QInt8,
    )
    size_mb = out.stat().st_size / 1e6
    print(f"wrote {out} ({size_mb:.1f} MB)")
    # Verify: no external data sidecar, and the file loads standalone.
    import onnx

    model = onnx.load(str(out))
    onnx.checker.check_model(model)
    external = [t for t in onnx.external_data_helper._get_all_tensors(model) if t.data_location == onnx.TensorProto.EXTERNAL]
    print(f"onnx.checker OK; external tensors: {len(external)} (must be 0)")
    return 0 if not external else 2


if __name__ == "__main__":
    raise SystemExit(main())
