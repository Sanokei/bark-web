#!/usr/bin/env python3
"""Weight-only int8 quantization of the Bark fp32 encoder via direct ONNX graph surgery.

Why this exists (instead of onnxruntime's ``quantize_dynamic``):
- ``quantize_dynamic`` on the 2.4GB fp32 encoder OOMs reliably (exit 137,
  twice) on this 8GB VM, and a swap file is not available (the container
  blocks ``swapon``).
- Quantizing the 1.2GB fp16 encoder instead produces an INVALID model:
  ``DynamicQuantizeLinear`` rejects fp16 inputs.
- This script is memory-light (peak ~3GB, fits): it loads the model once
  with ``onnx.load`` (the external ``.data`` weights pulled into memory,
  ~2.4GB), then rewrites MatMul/Gemm weights in place — no second full-model
  copy, no activation calibration, no per-node quantization passes.

What it does: for every MatMul/Gemm weight initializer that is fp32 with at
least ``--min-elements`` elements, it replaces the weight with SYMMETRIC
per-tensor int8 storage plus a ``DequantizeLinear`` node::

    scale = max(abs(W)) / 127
    W_int8 = clip(round(W / scale), -127, 127).astype(int8)

Compute stays fp32 (the ``DequantizeLinear`` output feeds the MatMul/Gemm
unchanged; Gemm attributes such as ``transB`` are untouched because the
dequant output has the original weight's shape); only storage shrinks to
int8, which onnxruntime-web WASM supports via ``DequantizeLinear``. Weights
shared by several MatMul/Gemm nodes are quantized once and all consumers
rewired to the same dequant output. Tiny tensors (below ``--min-elements``)
stay fp32. The output embeds everything (``data_location`` DEFAULT, no
``external_data``), so it is a single ~600MB file under the 2GB protobuf
limit — NO ``.data`` sidecar.

Usage: python scripts/quantize_weights.py \
    --input models/bark-onnx/encoder-model.onnx \
    --output models/bark-onnx/encoder-model.int8.onnx
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper


def _num_elements(init: onnx.TensorProto) -> int:
    """Number of elements in an initializer (1 for a scalar)."""
    if not init.dims:
        return 1
    n = 1
    for d in init.dims:
        n *= d
    return n


def _model_size_on_disk(input_path: Path, model: onnx.ModelProto) -> int:
    """Size of the input model including its external-data sidecars.

    The ``.onnx`` graph file alone is only a few MB; the real input
    footprint is the graph plus the ``.data`` file(s) its initializers
    reference, resolved relative to the input directory.
    """
    total = input_path.stat().st_size
    seen: set[str] = set()
    for init in model.graph.initializer:
        for entry in init.external_data:
            if entry.key == "location" and entry.value not in seen:
                seen.add(entry.value)
                sidecar = input_path.parent / entry.value
                if sidecar.exists():
                    total += sidecar.stat().st_size
    return total


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="models/bark-onnx/encoder-model.onnx")
    ap.add_argument("--output", default="models/bark-onnx/encoder-model.int8.onnx")
    ap.add_argument(
        "--min-elements",
        type=int,
        default=1024,
        help="only quantize weight tensors with at least this many elements "
        "(default: 1024; tiny tensors stay fp32)",
    )
    args = ap.parse_args()
    inp, out = Path(args.input), Path(args.output)
    if not inp.exists():
        print(f"input not found: {inp}", file=sys.stderr)
        return 1

    print(f"Quantizing {inp} -> {out} (weight-only symmetric int8, "
          f"min-elements {args.min_elements})...")
    # onnx.load pulls the external .data weights into memory (~2.4GB, OK).
    model = onnx.load(str(inp))
    graph = model.graph
    input_size = _model_size_on_disk(inp, model)

    initializers = {init.name: init for init in graph.initializer}
    # Names already taken (initializers, node outputs, graph I/O) so the
    # generated _int8/_scale/_zero_point/_dequant names can never collide.
    taken_names: set[str] = set(initializers)
    for node in graph.node:
        taken_names.update(node.output)
    taken_names.update(i.name for i in graph.input)
    taken_names.update(o.name for o in graph.output)

    quantized: dict[str, str] = {}  # weight name -> dequant output name
    new_initializers: list[onnx.TensorProto] = []
    new_nodes: list[onnx.NodeProto] = []

    for node in graph.node:
        if node.op_type not in ("MatMul", "Gemm"):
            continue
        if len(node.input) < 2:
            continue
        weight_name = node.input[1]
        if not weight_name or weight_name in quantized:
            continue  # shared weights are quantized once (see below)
        init = initializers.get(weight_name)
        if init is None or init.data_type != TensorProto.FLOAT:
            continue
        if _num_elements(init) < args.min_elements:
            continue
        weights = numpy_helper.to_array(init)
        scale = float(np.max(np.abs(weights)) / 127.0) if weights.size else 0.0
        if scale == 0.0:
            continue  # all-zero weights have no valid scale; leave as-is
        q = np.clip(np.round(weights / scale), -127, 127).astype(np.int8)
        int8_name = f"{weight_name}_int8"
        scale_name = f"{weight_name}_scale"
        zp_name = f"{weight_name}_zero_point"
        dequant_name = f"{weight_name}_dequant"
        if any(n in taken_names for n in (int8_name, scale_name, zp_name, dequant_name)):
            print(f"warning: skipping {weight_name}: generated tensor/node "
                  f"name would collide with an existing name", file=sys.stderr)
            continue
        taken_names.update((int8_name, scale_name, zp_name, dequant_name))
        new_initializers.append(numpy_helper.from_array(q, name=int8_name))
        new_initializers.append(
            numpy_helper.from_array(np.array(scale, dtype=np.float32), name=scale_name)
        )
        new_initializers.append(
            numpy_helper.from_array(np.array(0, dtype=np.int8), name=zp_name)
        )
        new_nodes.append(
            onnx.helper.make_node(
                "DequantizeLinear",
                inputs=[int8_name, scale_name, zp_name],
                outputs=[dequant_name],
                name=dequant_name,
            )
        )
        quantized[weight_name] = dequant_name

    # Rewire every MatMul/Gemm consumer of a quantized weight to the shared
    # dequant output (a shared weight is quantized once, above).
    for node in graph.node:
        if node.op_type in ("MatMul", "Gemm") and len(node.input) >= 2:
            if node.input[1] in quantized:
                node.input[1] = quantized[node.input[1]]

    # Remove the original fp32 initializers, but only where no node still
    # references them: a weight also consumed by a non-MatMul/Gemm node must
    # stay, or the graph would break. (Encoder weights are MatMul-only in
    # practice, so normally every quantized weight is removed.)
    still_referenced: set[str] = set()
    for node in graph.node:
        still_referenced.update(node.input)
    removed = {name for name in quantized if name not in still_referenced}

    kept = [init for init in graph.initializer if init.name not in removed]
    kept.extend(new_initializers)
    del graph.initializer[:]
    graph.initializer.extend(kept)

    # DequantizeLinear nodes depend only on initializers, so putting them
    # first keeps the graph topologically ordered (producers before consumers).
    existing_nodes = list(graph.node)
    del graph.node[:]
    graph.node.extend(new_nodes)
    graph.node.extend(existing_nodes)

    # Embed everything: no external data, so the save is a single file.
    for init in graph.initializer:
        init.data_location = TensorProto.DEFAULT
        del init.external_data[:]

    left_fp32 = sum(
        1 for init in graph.initializer if init.data_type == TensorProto.FLOAT
    )
    external_count = sum(
        1
        for init in graph.initializer
        if init.data_location == TensorProto.EXTERNAL
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(out))
    try:
        onnx.checker.check_model(model)
    except onnx.checker.ValidationError as exc:
        print(f"onnx.checker FAILED: {exc}", file=sys.stderr)
        return 1

    output_size = out.stat().st_size
    print(f"Input size: {input_size / 1e6:.1f} MB ({input_size} bytes)")
    print(f"Output size: {output_size / 1e6:.1f} MB ({output_size} bytes)")
    print(f"Tensors quantized: {len(quantized)}")
    print(f"Tensors left fp32: {left_fp32}")
    print(f"External tensors: {external_count} (must be 0)")
    if external_count != 0:
        print("error: external tensors remain; output is not self-contained",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
