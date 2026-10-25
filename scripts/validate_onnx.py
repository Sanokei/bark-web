#!/usr/bin/env python3
"""Validate Bark 3.0 ONNX exports against the PyTorch model (parity check).

Feeds the SAME random mel input through the PyTorch encoder (via the exact
wrappers in ``convert.py``) and the exported ONNX encoder, then feeds the
SAME encoder frame + token + zero LSTM states through the PyTorch decoder
path and the exported ONNX decoder_joint, and asserts the outputs match.

Tolerances: rtol 1e-3 / atol 1e-3 for fp32 exports; looser rtol 1e-2 /
atol 1e-2 for ``--fp16`` exports (fp16 rounding). The script reports the max
abs diff for every compared tensor and exits non-zero on any mismatch.

Usage (run from the bark-web directory, after ``convert.py``)::

    python scripts/validate_onnx.py --model-dir models/bark-pro --onnx-dir models/bark-onnx

This only validates numerics; it does not run the full conversion.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from convert import ParakeetParts, build_wrappers, load_model  # noqa: E402


def run_ort(session, inputs: dict[str, "object"]) -> dict[str, torch.Tensor]:
    """Run an ONNX Runtime session and return outputs keyed by output name."""
    feed = {
        name: (value.numpy() if isinstance(value, torch.Tensor) else value)
        for name, value in inputs.items()
    }
    names = [o.name for o in session.get_outputs()]
    outs = session.run(names, feed)
    return {name: torch.from_numpy(out) for name, out in zip(names, outs)}


def check(
    label: str,
    pt: torch.Tensor,
    onnx_out: torch.Tensor,
    rtol: float,
    atol: float,
    failures: list[str],
) -> float:
    """Compare one tensor pair, report max abs diff, record failures."""
    if pt.shape != onnx_out.shape:
        failures.append(f"{label}: shape mismatch pt {tuple(pt.shape)} vs onnx {tuple(onnx_out.shape)}")
        print(f"{label}: SHAPE MISMATCH pt {tuple(pt.shape)} vs onnx {tuple(onnx_out.shape)}")
        return float("nan")
    diff = (pt.float() - onnx_out.float()).abs()
    max_diff = float(diff.max()) if diff.numel() else 0.0
    ok = bool(torch.allclose(pt.float(), onnx_out.float(), rtol=rtol, atol=atol))
    status = "OK " if ok else "FAIL"
    print(f"{label}: max abs diff {max_diff:.6g} (rtol {rtol:g}, atol {atol:g}) [{status}]")
    if not ok:
        failures.append(f"{label}: max abs diff {max_diff:.6g} exceeds rtol {rtol:g} / atol {atol:g}")
    return max_diff


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="models/bark-pro")
    parser.add_argument("--onnx-dir", default="models/bark-onnx")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Validate an fp16 encoder export (looser tolerances).",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rtol, atol = (1e-2, 1e-2) if args.fp16 else (1e-3, 1e-3)
    print(f"tolerances: rtol {rtol:g}, atol {atol:g} ({'fp16' if args.fp16 else 'fp32'} mode)")

    import onnxruntime as ort

    model_dir = Path(args.model_dir)
    onnx_dir = Path(args.onnx_dir)
    for name in ("encoder-model.onnx", "decoder_joint-model.onnx"):
        if not (onnx_dir / name).exists():
            print(f"error: {onnx_dir / name} not found; run convert.py first", file=sys.stderr)
            sys.exit(2)

    print(f"loading PyTorch model from {model_dir} ...")
    model = load_model(model_dir, torch.bfloat16)
    model = model.to(torch.float32)
    if args.fp16:
        ParakeetParts(model).encoder.to(torch.float16)
    encoder_wrapper, decoder_wrapper = build_wrappers(model, use_fp16_encoder=args.fp16)
    encoder_wrapper.eval()
    decoder_wrapper.eval()

    enc_sess = ort.InferenceSession(
        str(onnx_dir / "encoder-model.onnx"), providers=["CPUExecutionProvider"]
    )
    dec_sess = ort.InferenceSession(
        str(onnx_dir / "decoder_joint-model.onnx"), providers=["CPUExecutionProvider"]
    )

    torch.manual_seed(args.seed)
    time_frames = 128
    audio = torch.randn(1, 128, time_frames)
    length = torch.tensor([time_frames])
    failures: list[str] = []

    print("--- encoder ---")
    with torch.no_grad():
        pt_hidden, pt_lengths = encoder_wrapper(audio, length)
    onnx_enc = run_ort(enc_sess, {"audio_signal": audio, "length": length})
    check("encoder outputs", pt_hidden, onnx_enc["outputs"], rtol, atol, failures)
    if not torch.equal(pt_lengths, onnx_enc["encoded_lengths"].to(pt_lengths.dtype)):
        failures.append(
            f"encoder encoded_lengths mismatch: pt {pt_lengths.tolist()} "
            f"vs onnx {onnx_enc['encoded_lengths'].tolist()}"
        )
        print(f"encoder encoded_lengths: FAIL pt {pt_lengths.tolist()} "
              f"vs onnx {onnx_enc['encoded_lengths'].tolist()}")
    else:
        print(f"encoder encoded_lengths: OK {pt_lengths.tolist()}")

    print("--- decoder_joint ---")
    token_id = 0
    targets = torch.tensor([[token_id]], dtype=torch.int32)
    target_length = torch.tensor([1], dtype=torch.int32)
    h0 = torch.zeros(2, 1, 640)
    c0 = torch.zeros(2, 1, 640)
    # Same encoder frame for both paths: last frame of the PyTorch encoder output.
    enc_frame = pt_hidden[:, :, -1:]
    with torch.no_grad():
        pt_logits, pt_h, pt_c = decoder_wrapper(
            enc_frame, targets, target_length, h0, c0
        )
    onnx_dec = run_ort(
        dec_sess,
        {
            "encoder_outputs": enc_frame,
            "targets": targets,
            "target_length": target_length,
            "input_states_1": h0,
            "input_states_2": c0,
        },
    )
    check("decoder outputs (logits)", pt_logits, onnx_dec["outputs"], rtol, atol, failures)
    check("decoder output_states_1", pt_h, onnx_dec["output_states_1"], rtol, atol, failures)
    check("decoder output_states_2", pt_c, onnx_dec["output_states_2"], rtol, atol, failures)

    if failures:
        print(f"\nVALIDATION FAILED ({len(failures)} mismatch(es)):")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("\nVALIDATION PASSED: PyTorch and ONNX outputs match within tolerance.")


if __name__ == "__main__":
    main()
