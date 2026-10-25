#!/usr/bin/env python3
"""Convert Bark 3.0 (Parakeet TDT) weights to the ONNX format parakeet.js expects.

Bark 3.0 Pro weights (``models/bark-pro/``: ``model.safetensors`` ~1.25GB,
``config.json``, ``tokenizer.json``) are a Hugging Face ``ParakeetForTDT``
model. parakeet.js cannot consume them directly: it expects two ONNX files
with a fixed I/O contract (see below), so this script builds small wrapper
modules around the transformers submodules and exports them with the classic
``torch.onnx`` exporter (``dynamo=False``) at opset 17.

ONNX contract (verified against ``research/onnx-ref/decoder_joint.int8.onnx``
and the ``ysdede/parakeet-tdt-0.6b-v3-onnx`` repo; opset 17, ir_version 8):

* Encoder — inputs ``audio_signal`` [B, 128, T] (mel frames, channel-major as
  parakeet.js produces them) and ``length`` [B] (valid frames per utterance);
  outputs ``outputs`` [B, 1024, T'] and ``encoded_lengths`` [B].
  parakeet.js calls the encoder with ``{audio_signal: [1, 128, T], length: [1]}``.
* Decoder-joint — inputs ``encoder_outputs`` [B, 1024, T], ``targets`` [B, U],
  ``target_length`` [B], ``input_states_1`` [2, B, 640], ``input_states_2``
  [2, B, 640]; outputs ``outputs`` [..., 8198] (8193 vocab logits followed by
  5 duration logits; parakeet.js splits by vocab size) and
  ``output_states_1`` / ``output_states_2`` [2, B, 640].
  parakeet.js calls the decoder per-frame with ``encoder_outputs`` [1, 1024, 1].

Architecture notes (verified by inspecting transformers' Parakeet code):

* The transformers encoder expects time-major features [B, T, 128], NOT
  [B, 128, T]. The encoder wrapper transposes in and out.
* Subsampling factor 8, so T' ~= ceil(T / 8). ``encoded_lengths`` is computed
  as ``(length + 7) // 8`` clamped to T'.
* The decoder wrapper calls the submodules directly: embedding -> LSTM ->
  projector, then the joint head, taking the last VALID step's hidden state
  (incremental single-frame decode; the LSTM states carry the history).
* ``target_length`` selects that step per element (``target_length - 1``).
  This keeps it live in the exported graph: torch.onnx prunes dead inputs,
  but parakeet.js always feeds it.

Precision / size tradeoff: ONNX Runtime Web handles fp32 best, so the model
is cast to float32 and exported in fp32 by default. That makes the fp32
encoder ~2.5GB (vs 1.25GB bf16 source). ``--fp16`` exports the encoder in
fp16 instead (halves encoder size; fp32 I/O is preserved by casts in the
wrapper). int8 quantization of the exported files is a separate follow-up.

Usage (run from the bark-web directory; the parent runs the full conversion
separately — do NOT run this casually, it loads the 1.25GB model)::

    python scripts/convert.py --model-dir models/bark-pro --out-dir models/bark-onnx

Outputs in ``--out-dir``: ``encoder-model.onnx``, ``decoder_joint-model.onnx``,
``vocab.txt``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Submodule resolution
# ---------------------------------------------------------------------------

def _resolve(root: nn.Module, *paths: str) -> nn.Module:
    """Return the first attribute path that exists under ``root``.

    transformers' Parakeet module layout has shifted across versions
    (submodules sometimes live under ``.model`` and sometimes at top level),
    so every lookup tries the candidates in order and raises a clear error
    listing the available children if none match.
    """
    for path in paths:
        obj: nn.Module = root
        ok = True
        for attr in path.split("."):
            if not hasattr(obj, attr):
                ok = False
                break
            obj = getattr(obj, attr)
        if ok and isinstance(obj, nn.Module):
            return obj
    children = sorted({name for name, _ in root.named_children()})
    raise AttributeError(
        f"None of {paths!r} found under {type(root).__name__}; "
        f"top-level children: {children}"
    )


class ParakeetParts:
    """The transformers submodules the ONNX wrappers need, resolved once."""

    def __init__(self, model: nn.Module) -> None:
        self.encoder = _resolve(model, "model.encoder", "encoder")
        self.encoder_projector = _resolve(
            model, "model.encoder_projector", "encoder_projector"
        )
        self.embedding = _resolve(
            model,
            "model.decoder.embed_tokens",
            "model.decoder.embedding",
            "decoder.embed_tokens",
            "decoder.embedding",
        )
        self.lstm = _resolve(model, "model.decoder.lstm", "decoder.lstm")
        self.decoder_projector = _resolve(
            model,
            "model.decoder.decoder_projector",
            "model.decoder.projector",
            "decoder.decoder_projector",
            "decoder.projector",
        )
        self.joint = _resolve(model, "model.joint", "joint")


# ---------------------------------------------------------------------------
# ONNX wrappers
# ---------------------------------------------------------------------------

class EncoderWrapper(nn.Module):
    """parakeet.js encoder contract: [B, 128, T] in, [B, 1024, T'] out."""

    def __init__(self, parts: ParakeetParts, use_fp16: bool = False) -> None:
        super().__init__()
        self.encoder = parts.encoder
        self.use_fp16 = use_fp16

    def forward(
        self, audio_signal: torch.Tensor, length: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # transformers wants time-major [B, T, 128]; parakeet.js gives [B, 128, T].
        feats = audio_signal.transpose(1, 2)
        if self.use_fp16:
            feats = feats.to(torch.float16)
        batch, frames, _ = feats.shape
        # Padding mask: 1 for valid frames, 0 for padding (length is in frames).
        positions = torch.arange(frames, device=feats.device).unsqueeze(0)
        attention_mask = (positions < length.unsqueeze(1)).to(torch.long)
        out = self.encoder(feats, attention_mask=attention_mask)
        hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        # Back to channel-major [B, 1024, T'] for parakeet.js.
        hidden = hidden.transpose(1, 2)
        if self.use_fp16:
            hidden = hidden.float()
        # Subsampling factor 8; clamp to the actual encoded length.
        encoded_lengths = ((length + 7) // 8).clamp(max=hidden.shape[2])
        return hidden, encoded_lengths


class DecoderJointWrapper(nn.Module):
    """Fused decoder-joint: projector + embedding/LSTM/projector + joint head.

    Single-frame incremental decode: the LSTM states passed in carry the
    history, and the last VALID step's hidden state is projected and joined,
    selected per element by ``target_length`` (step ``target_length - 1``).
    Using ``target_length`` this way also keeps it declared in the exported
    graph: torch.onnx prunes dead inputs, but parakeet.js always feeds it.
    """

    def __init__(self, parts: ParakeetParts) -> None:
        super().__init__()
        self.encoder_projector = parts.encoder_projector
        self.embedding = parts.embedding
        self.lstm = parts.lstm
        self.decoder_projector = parts.decoder_projector
        self.joint = parts.joint

    def forward(
        self,
        encoder_outputs: torch.Tensor,
        targets: torch.Tensor,
        target_length: torch.Tensor,
        input_states_1: torch.Tensor,
        input_states_2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # parakeet.js feeds targets and target_length as int32; cast to
        # int64 up front so the embedding lookup and step gather see
        # LongTensor while the exported graph keeps int32 inputs.
        targets = targets.to(torch.int64)
        target_length = target_length.to(torch.int64)
        # Last encoder frame only: [B, 1024, T] -> [B, 1024].
        enc_frame = encoder_outputs[:, :, -1]
        enc_hidden = self.encoder_projector(enc_frame)  # [B, 640]
        emb = self.embedding(targets)  # [B, U, 640]
        lstm_out, (h_n, c_n) = self.lstm(emb, (input_states_1, input_states_2))
        # Select the last VALID step per element via target_length. This keeps
        # target_length live in the exported graph: torch.onnx prunes dead
        # inputs, but parakeet.js always feeds target_length and
        # onnxruntime-web rejects feeds with undeclared input names. For the
        # single-token decode parakeet.js uses (target_length = 1) this
        # selects step 0, identical to [:, -1, :]; it also stays correct for
        # multi-token inputs.
        step = (target_length - 1).clamp(min=0)  # [B]
        batch_idx = torch.arange(lstm_out.shape[0], device=lstm_out.device)
        dec_hidden = self.decoder_projector(lstm_out[batch_idx, step, :])  # [B, 640]
        if hasattr(self.joint, "head"):
            # joint(dec, enc) = head(relu(enc + dec)); addition commutes.
            joint_out = self.joint.head(F.relu(enc_hidden + dec_hidden))
        else:
            joint_out = self.joint(dec_hidden, enc_hidden)
        # joint_out: [B, 8198] = 8193 vocab logits + 5 duration logits.
        return joint_out, h_n, c_n


def build_wrappers(
    model: nn.Module, use_fp16_encoder: bool = False
) -> tuple[EncoderWrapper, DecoderJointWrapper]:
    """Build the export wrappers around a loaded Parakeet model."""
    parts = ParakeetParts(model)
    return EncoderWrapper(parts, use_fp16=use_fp16_encoder), DecoderJointWrapper(parts)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_dir: Path, dtype: torch.dtype) -> nn.Module:
    """Load Bark 3.0 weights on CPU via transformers."""
    try:
        from transformers import AutoModelForTDT
    except ImportError:  # Older transformers without the TDT auto class.
        from transformers import ParakeetForTDT as AutoModelForTDT  # type: ignore[no-redef]
    try:
        model = AutoModelForTDT.from_pretrained(str(model_dir), dtype=dtype)
    except TypeError:  # transformers < 5 uses torch_dtype instead of dtype.
        model = AutoModelForTDT.from_pretrained(str(model_dir), torch_dtype=dtype)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Vocab export
# ---------------------------------------------------------------------------

def export_vocab(tokenizer_path: Path, out_path: Path) -> None:
    """Write ``vocab.txt`` as one ``token id`` line per token, sorted by id.

    ``tokenizer.json`` is the standard HF tokenizers format: the bulk of the
    vocabulary lives in ``model.vocab`` (token -> id) and special tokens in
    ``added_tokens`` (id -> content). Both are merged; an id claimed by two
    different tokens is a hard error, since parakeet.js indexes tokens by id.
    Tokens containing spaces (none expected for SentencePiece pieces, but
    possible for added tokens) round-trip because readers split on the LAST
    space. The result is verified by loading it back: exactly 8193 lines
    with ``<blk>`` at 8192.
    """
    with tokenizer_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    vocab: dict[str, int] = dict(data.get("model", {}).get("vocab", {}))
    id_to_token = {tok_id: token for token, tok_id in vocab.items()}
    for entry in data.get("added_tokens", []):
        token, tok_id = entry["content"], int(entry["id"])
        if token in vocab and vocab[token] != tok_id:
            raise ValueError(
                f"tokenizer.json: token {token!r} has id {vocab[token]} in "
                f"model.vocab but id {tok_id} in added_tokens"
            )
        if tok_id in id_to_token and id_to_token[tok_id] != token:
            raise ValueError(
                f"tokenizer.json: id {tok_id} claimed by both "
                f"{id_to_token[tok_id]!r} and {token!r}"
            )
        vocab[token] = tok_id
        id_to_token[tok_id] = token
    # The blank token's NAME differs across sources: Bark's added_tokens
    # claims id 8192 as "<blank>", but parakeet.js and the ysdede vocab use
    # "<blk>". Ensure id 8192 maps to exactly one token, named "<blk>":
    # evict any other claimant first (unconditional append created the
    # 8194-line duplicate), and leave an existing "<blk>" untouched.
    if id_to_token.get(8192) not in (None, "<blk>"):
        del vocab[id_to_token[8192]]
        del id_to_token[8192]
    vocab["<blk>"] = 8192
    id_to_token[8192] = "<blk>"
    by_id = sorted(vocab.items(), key=lambda kv: kv[1])
    with out_path.open("w", encoding="utf-8") as f:
        for token, tok_id in by_id:
            if "\n" in token or "\r" in token:
                raise ValueError(f"tokenizer.json: token {token!r} contains a newline")
            f.write(f"{token} {tok_id}\n")
    # Verify by loading back. Note the old check counted dict entries (tokens),
    # which hid duplicate ids — two tokens sharing one id still counted twice.
    # Ids are now tracked separately: any duplicate id is a hard error, as is
    # any deviation from exactly 0..8192.
    reloaded: dict[str, int] = {}
    seen_ids: set[int] = set()
    duplicate_ids: set[int] = set()
    with out_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            idx = line.rfind(" ")
            tok_id = int(line[idx + 1 :])
            reloaded[line[:idx]] = tok_id
            if tok_id in seen_ids:
                duplicate_ids.add(tok_id)
            seen_ids.add(tok_id)
    if len(reloaded) != 8193:
        raise ValueError(f"vocab.txt has {len(reloaded)} entries, expected 8193")
    if reloaded.get("<blk>") != 8192:
        raise ValueError(f"vocab.txt: <blk> id is {reloaded.get('<blk>')}, expected 8192")
    if duplicate_ids:
        raise ValueError(f"vocab.txt: duplicate ids {sorted(duplicate_ids)}")
    if sorted(seen_ids) != list(range(8193)):
        raise ValueError("vocab.txt: ids are not exactly 0..8192")


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_onnx(
    wrapper: nn.Module,
    dummy_inputs: tuple[torch.Tensor, ...],
    input_names: list[str],
    output_names: list[str],
    dynamic_axes: dict[str, dict[int, str]],
    out_path: Path,
    opset: int,
    consolidate_external: bool = False,
    external_location: str | None = None,
) -> None:
    torch.onnx.export(
        wrapper,
        dummy_inputs,
        str(out_path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        dynamo=False,  # Classic exporter: required for opset 17 compat here.
    )
    import onnx

    data_path: Path | None = None
    if consolidate_external:
        # The classic exporter scatters large tensors across one external
        # file per tensor (~200 files for the encoder). Browsers expect a
        # single model file plus at most one .data file, so consolidate:
        # pull any already-external data into memory, re-point every tensor
        # at one file, and save (which writes that file).
        location = external_location or f"{out_path.name}.data"
        model = onnx.load(str(out_path))
        onnx.external_data_helper.load_external_data_for_model(
            model, str(out_path.parent)
        )
        onnx.external_data_helper.convert_model_to_external_data(
            model, all_tensors_to_one_file=True, location=location
        )
        onnx.save(model, str(out_path))
        data_path = out_path.parent / location
    onnx.checker.check_model(str(out_path))
    total_mb = out_path.stat().st_size / 1e6
    if data_path is not None and data_path.exists():
        total_mb += data_path.stat().st_size / 1e6
    print(f"wrote {out_path} ({total_mb:.1f} MB total, onnx.checker OK)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="models/bark-pro")
    parser.add_argument("--out-dir", default="models/bark-onnx")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Export the encoder in fp16 (halves encoder size; decoder stays fp32).",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading model from {model_dir} (bfloat16, CPU) ...")
    model = load_model(model_dir, torch.bfloat16)
    # ONNX Runtime Web handles fp32 best: export in fp32 by default.
    # (Size tradeoff: the fp32 encoder is ~2.5GB vs 1.25GB bf16 source;
    # int8 quantization of the exported files is a follow-up.)
    model = model.to(torch.float32)
    if args.fp16:
        parts = ParakeetParts(model)
        parts.encoder.to(torch.float16)
        print("fp16 mode: encoder cast to float16 (decoder stays float32)")

    encoder_wrapper, decoder_wrapper = build_wrappers(model, use_fp16_encoder=args.fp16)
    encoder_wrapper.eval()
    decoder_wrapper.eval()

    with torch.no_grad():
        print("exporting encoder ...")
        export_onnx(
            encoder_wrapper,
            (torch.randn(1, 128, 80), torch.tensor([80])),
            input_names=["audio_signal", "length"],
            output_names=["outputs", "encoded_lengths"],
            dynamic_axes={
                "audio_signal": {0: "batch", 2: "time"},
                "length": {0: "batch"},
                "outputs": {0: "batch", 2: "time"},
                "encoded_lengths": {0: "batch"},
            },
            out_path=out_dir / "encoder-model.onnx",
            opset=args.opset,
            consolidate_external=True,
            external_location="encoder-model.onnx.data",
        )
        print("exporting decoder_joint ...")
        export_onnx(
            decoder_wrapper,
            (
                torch.randn(1, 1024, 1),
                torch.zeros(1, 1, dtype=torch.int32),
                torch.tensor([1], dtype=torch.int32),
                torch.zeros(2, 1, 640),
                torch.zeros(2, 1, 640),
            ),
            input_names=[
                "encoder_outputs",
                "targets",
                "target_length",
                "input_states_1",
                "input_states_2",
            ],
            output_names=["outputs", "output_states_1", "output_states_2"],
            dynamic_axes={
                "encoder_outputs": {0: "batch", 2: "time"},
                "targets": {0: "batch", 1: "seq"},
                "target_length": {0: "batch"},
                "input_states_1": {1: "batch"},
                "input_states_2": {1: "batch"},
                "outputs": {0: "batch"},
                "output_states_1": {1: "batch"},
                "output_states_2": {1: "batch"},
            },
            out_path=out_dir / "decoder_joint-model.onnx",
            opset=args.opset,
        )

    print("exporting vocab ...")
    export_vocab(model_dir / "tokenizer.json", out_dir / "vocab.txt")
    print("done.")


if __name__ == "__main__":
    main()
