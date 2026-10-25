# bark-web

TypeScript glue to run Bark 3.0 speech-to-text in the browser: a small
`BarkTranscriber` facade over [parakeet.js](https://github.com/ysdede/parakeet.js)
and [onnxruntime-web](https://onnxruntime.ai/docs/tutorials/web/).
Bark 3.0 (Conway/Underdog) is a Parakeet TDT model; this package replaces a
Whisper-based transcription path with it.

## Status

Verified 2026-10-25: the full test suite passes (8/8, `tsc` clean),
**including real end-to-end inference** — the Barcelona weather sample
transcribes correctly through the JS mel preprocessor, the int8 encoder,
the TDT decode loop, and the Bark vocab.

What is verified: the **Node / WASM path** (vitest, onnxruntime-web WASM
backend).

What is **not** yet verified: the **real browser + WebGPU path** — that
needs a GPU machine, and this VM has none.

## How it differs from Whisper

- **English-only.** Bark 3.0 was trained on English speech; other languages
  are out of scope and will produce unreliable output.
- **No word timestamps.** Bark 3.0 does not emit timing information, so this
  wrapper deliberately does not surface word timings.
- **No language detection.** There is nothing to detect: input is assumed to
  be English.

## Usage

```sh
npm install
npm run build   # tsc -> dist/
```

```ts
import { BarkTranscriber } from './dist/index.js';

// pcm: Float32Array of mono audio resampled to 16 kHz beforehand.
// (BarkTranscriber rejects any other sample rate rather than resampling
// silently.)
const transcriber = await BarkTranscriber.fromUrls(
  {
    encoderUrl: 'https://example.com/encoder-model.int8.onnx',
    decoderUrl: 'https://example.com/decoder_joint-model.onnx',
    tokenizerUrl: 'https://example.com/vocab.txt',
  },
  {
    // Node only (see below); omit in browsers.
    wasmPaths: 'file:///path/to/node_modules/onnxruntime-web/dist/',
  },
);
const result = await transcriber.transcribe(pcm, 16000);
console.log(result.text); // plus result.tokens and result.confidence when available
transcriber.dispose();
```

`BarkTranscriber.fromModelDir(dir)` is the local-directory convenience
loader (Node): `dir` must contain `encoder-model.onnx`,
`decoder_joint-model.onnx`, and `vocab.txt`.

### Node note: `wasmPaths`

In Node, pass `wasmPaths` pointing at a local onnxruntime-web `dist/`
directory (as a `file://` URL with a trailing slash). This is required in
Node because parakeet.js's default CDN URL cannot be ESM-imported by Node;
in browsers it can be omitted (the CDN default works). Note also that
`onnxruntime-web` is pinned to exactly `1.24.1` in `package.json` so that
bark-web and parakeet.js share one module instance (parakeet.js declares
the same version); with two instances, pre-setting the WASM path on one
would not affect the one parakeet.js loads.

## Model files

This repo does **not** ship weights (GitHub file-size limits). Obtain them
in three steps:

**(a) Conversion** — from
[ConwayResearch/Underdog-Bark-3.0-Pro](https://huggingface.co/ConwayResearch/Underdog-Bark-3.0-Pro)
on Hugging Face:

```sh
python scripts/convert.py --model-dir <bark-pro dir> --out-dir models/bark-onnx
```

This produces the fp32 `encoder-model.onnx` (plus its external
`encoder-model.onnx.data`, ~2.4GB), `decoder_joint-model.onnx`, and
`vocab.txt`.

**(b) Quantization (required)** — the fp32 encoder cannot be loaded by
onnxruntime-web, which cannot read its external `.data` sidecar. Quantize
it to a single self-contained file with weight-only int8:

```sh
python scripts/quantize_weights.py
```

This produces `encoder-model.int8.onnx` (~838MB, 0 external tensors).
`scripts/quantize_encoder.py` (dynamic quantization) also exists but is
**not** the recommended path: it OOMs on the fp32 model on small machines
and produces invalid output from fp16 input.

**(c) Stage the test fixtures** — place `encoder-model.int8.onnx`,
`decoder_joint-model.onnx`, and `vocab.txt` in `tests/fixtures/models/`
(plus `nemo128.onnx` from
[ysdede/parakeet-tdt-0.6b-v3-onnx](https://huggingface.co/ysdede/parakeet-tdt-0.6b-v3-onnx)
if using the ONNX preprocessor) to run the real-inference test. The staged
model files are git-ignored (see `.gitignore`); the test skips gracefully
when they are absent. The small fixtures `tests/fixtures/bcn_weather.mp3`
and `tests/fixtures/vocab.txt` are committed.

## Architecture notes

- **Encoder** — mel input `[1, 128, T]` (128 mel bins) → encoder output
  `[1, 1024, T']`, subsampling factor 8, plus per-utterance encoded lengths.
- **Decoder-joint** — one TDT step per frame: it takes the last encoder
  frame, int32 token inputs (`targets`, `target_length`), and the LSTM
  states, and outputs 8198 logits per step: 8193 vocab logits followed by
  5 duration logits. The blank token `<blk>` is at id 8192.
- **Vocab** — `vocab.txt` is one `token id` pair per line, 8193 lines,
  with `<blk>` at 8192.
- **Preprocessor** — by default the pure-JS mel preprocessor is used (no
  `preprocessorUrl`); passing one loads the ONNX preprocessor instead.

## License attribution

- Bark 3.0 weights: **CC-BY 4.0** — credit Conway Research / Underdog when
  redistributing the model or building a product on it.
- Built on NVIDIA Parakeet TDT 0.6B.
- parakeet.js: MIT.
- The glue code in this repo: MIT. No `LICENSE` file exists yet — adding
  one is a follow-up.

## Development

```sh
npm test        # vitest run
npm run build   # tsc
```

Scripts:

- `scripts/convert.py` — Hugging Face weights → ONNX (encoder +
  decoder-joint + vocab).
- `scripts/quantize_weights.py` — weight-only int8 quantization of the
  encoder (recommended; produces the single-file `encoder-model.int8.onnx`).
- `scripts/quantize_encoder.py` — dynamic quantization (not recommended;
  see above).
- `scripts/validate_onnx.py` — PyTorch vs ONNX parity check for the
  converted models.
