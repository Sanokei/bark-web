/**
 * Bark 3.0 web transcriber.
 *
 * Wraps parakeet.js (which implements the full Parakeet TDT decode loop on
 * onnxruntime-web) with a small Bark-specific facade. This module does NOT
 * copy the reference decoder: it imports `ParakeetModel` from the
 * `parakeet.js` dependency and only adapts its inputs/outputs.
 *
 * Notes for consumers:
 * - English-only. Bark 3.0 was trained on English speech; other languages
 *   are out of scope and will produce unreliable output.
 * - No word timestamps. Bark provides no timing information, so this wrapper
 *   deliberately does not surface word timings (parakeet.js can estimate
 *   them for other models, but they would not be Bark timings).
 * - CC-BY 4.0 attribution required. Bark 3.0 (Conway/Underdog) is licensed
 *   CC-BY 4.0: credit the model authors when redistributing the model or
 *   building a product on it.
 */

import { ParakeetModel } from 'parakeet.js';
import type {
  BarkExecutionProvider,
  BarkLoadOptions,
  BarkModelUrls,
  BarkTranscriptionResult,
} from './types.js';

/** Bark 3.0 model constants (see also BARK_MODEL_CONFIG in index.ts). */
const BARK_FEATURES_SIZE = 128;
const BARK_SUBSAMPLING = 8;

/** Conventional on-disk filenames for a local Bark model directory. */
const ENCODER_FILENAME = 'encoder-model.onnx';
const DECODER_FILENAME = 'decoder_joint-model.onnx';
const TOKENIZER_FILENAME = 'vocab.txt';

/** Sample rate Bark expects. Anything else must be resampled first. */
const EXPECTED_SAMPLE_RATE = 16000;

/**
 * Map the caller's preferred execution providers onto a parakeet.js backend.
 * `webgpu` prefers the WebGPU-hybrid backend (GPU encoder, WASM decoder);
 * `wasm` forces the pure-WASM backend. Defaults to WebGPU-hybrid.
 */
function backendForProviders(
  providers: BarkExecutionProvider[] | undefined,
): 'webgpu-hybrid' | 'wasm' {
  if (providers !== undefined && providers.includes('wasm') && !providers.includes('webgpu')) {
    return 'wasm';
  }
  return 'webgpu-hybrid';
}

/**
 * Best-effort release of an ONNX Runtime inference session. The session
 * object is reached through `unknown` narrowing because parakeet.js does not
 * expose a public dispose/release API on the model itself.
 */
function releaseSession(session: unknown): void {
  if (typeof session !== 'object' || session === null) {
    return;
  }
  const candidate = session as { release?: unknown; dispose?: unknown };
  try {
    if (typeof candidate.release === 'function') {
      (candidate.release as () => void).call(session);
    } else if (typeof candidate.dispose === 'function') {
      (candidate.dispose as () => void).call(session);
    }
  } catch {
    // Releasing is best-effort; a failed release must not mask caller errors.
  }
}

/** Read a possible session field off the wrapped model via narrowing. */
function sessionField(model: ParakeetModel, field: string): unknown {
  const record = model as unknown as Record<string, unknown>;
  return record[field];
}

/** Turn a local filesystem path into a `file:` URL (POSIX). */
function toFileUrl(path: string): string {
  if (/^[a-zA-Z][a-zA-Z0-9+.-]*:\/\//.test(path)) {
    return path;
  }
  const normalized = path.startsWith('/') ? path : `/${path}`;
  const encoded = normalized
    .split('/')
    .map((segment) => encodeURIComponent(segment))
    .join('/');
  return `file://${encoded}`;
}

function joinDir(dir: string, filename: string): string {
  const trimmed = dir.endsWith('/') ? dir.slice(0, -1) : dir;
  return `${trimmed}/${filename}`;
}

/**
 * Transcribes 16 kHz mono PCM with Bark 3.0.
 *
 * English-only, no word timestamps (Bark provides no timings), CC-BY 4.0
 * attribution required (see the module docstring).
 */
export class BarkTranscriber {
  private readonly model: ParakeetModel;
  private disposed = false;

  private constructor(model: ParakeetModel) {
    this.model = model;
  }

  /**
   * Load Bark from remote asset URLs via parakeet.js.
   *
   * Bark's fixed config is applied here: vocab 8193, 128 mel bins,
   * subsampling factor 8 (predictor hidden 640 / 2 layers are baked into
   * parakeet.js's combined decoder for this model family).
   */
  static async fromUrls(
    urls: {
      encoderUrl: string;
      decoderUrl: string;
      tokenizerUrl: string;
      preprocessorUrl?: string;
    },
    options?: Pick<BarkLoadOptions, 'executionProviders' | 'onProgress' | 'wasmPaths'>,
  ): Promise<BarkTranscriber> {
    if (!urls.encoderUrl || !urls.decoderUrl || !urls.tokenizerUrl) {
      throw new Error(
        'BarkTranscriber.fromUrls requires encoderUrl, decoderUrl and tokenizerUrl.',
      );
    }
    options?.onProgress?.(0, 1, 'loading');
    // Pre-set the WASM binary location on the shared onnxruntime-web
    // instance BEFORE parakeet.js runs initOrt: initOrt accepts a wasmPaths
    // parameter but never assigns it — its only assignment is the jsDelivr
    // CDN fallback when ort.env.wasm.wasmPaths is unset, which Node cannot
    // ESM-import. package.json pins onnxruntime-web to exactly 1.24.1 (the
    // version parakeet.js declares) so npm hoists a single shared instance
    // that both this module and parakeet.js import; with wasmPaths already
    // set, initOrt skips its CDN default. The wasmPaths forward below is
    // kept as well (harmless, and future-proof if upstream fixes initOrt).
    // Dynamic import works in both Node and browsers.
    if (options?.wasmPaths !== undefined) {
      const ortModule = await import('onnxruntime-web');
      const ort = ortModule.default ?? ortModule;
      ort.env.wasm.wasmPaths = options.wasmPaths;
    }
    const model = await ParakeetModel.fromUrls({
      encoderUrl: urls.encoderUrl,
      decoderUrl: urls.decoderUrl,
      tokenizerUrl: urls.tokenizerUrl,
      preprocessorUrl: urls.preprocessorUrl,
      backend: backendForProviders(options?.executionProviders),
      wasmPaths: options?.wasmPaths,
      subsampling: BARK_SUBSAMPLING,
      nMels: BARK_FEATURES_SIZE,
    });
    options?.onProgress?.(1, 1, 'ready');
    return new BarkTranscriber(model);
  }

  /**
   * Convenience loader for local model files (Node).
   *
   * `dir` must contain `encoder-model.onnx`, `decoder_joint-model.onnx` and
   * `vocab.txt`. The paths are passed to parakeet.js as `file:` URLs.
   */
  static async fromModelDir(
    dir: string,
    options?: Pick<BarkLoadOptions, 'executionProviders' | 'onProgress' | 'wasmPaths'>,
  ): Promise<BarkTranscriber> {
    if (!dir) {
      throw new Error('BarkTranscriber.fromModelDir requires a directory path.');
    }
    const urls: BarkModelUrls = {
      encoderUrl: toFileUrl(joinDir(dir, ENCODER_FILENAME)),
      decoderUrl: toFileUrl(joinDir(dir, DECODER_FILENAME)),
      tokenizerUrl: toFileUrl(joinDir(dir, TOKENIZER_FILENAME)),
    };
    return BarkTranscriber.fromUrls(urls, options);
  }

  /**
   * Transcribe 16 kHz mono PCM audio.
   *
   * @param audio - Mono PCM samples as a Float32Array.
   * @param sampleRate - Must be 16000. Audio at any other rate must be
   *   resampled to 16 kHz by the caller first (resampling is not done here,
   *   so a mismatched rate is rejected rather than silently mis-decoded).
   * @returns `{ text }`, plus `tokens` (decoder token ids) when available
   *   and `confidence` (overall log-probability) when reported.
   * @throws If the audio is empty, contains non-finite samples, or the
   *   sample rate is not 16000 — or if this transcriber was disposed.
   */
  async transcribe(
    audio: Float32Array,
    sampleRate: number = EXPECTED_SAMPLE_RATE,
  ): Promise<BarkTranscriptionResult> {
    if (this.disposed) {
      throw new Error('BarkTranscriber has been disposed.');
    }
    if (!(audio instanceof Float32Array) || audio.length === 0) {
      throw new Error('transcribe requires a non-empty Float32Array of audio.');
    }
    for (let i = 0; i < audio.length; i += 1) {
      if (!Number.isFinite(audio[i])) {
        throw new Error(
          `transcribe requires finite audio samples (non-finite sample at index ${i}).`,
        );
      }
    }
    if (sampleRate !== EXPECTED_SAMPLE_RATE) {
      throw new Error(
        `Bark expects 16 kHz audio, got sampleRate ${sampleRate}. ` +
          'Resample to 16000 Hz before calling transcribe.',
      );
    }

    const result = await this.model.transcribe(audio, sampleRate, {
      returnTokenIds: true,
    });
    const out: BarkTranscriptionResult = { text: result.utterance_text };
    if (result.tokenIds !== undefined && result.tokenIds.length > 0) {
      out.tokens = [...result.tokenIds];
    }
    const logProb = result.confidence_scores?.overall_log_prob;
    if (typeof logProb === 'number') {
      out.confidence = logProb;
    }
    return out;
  }

  /**
   * Free the underlying ONNX sessions. Best-effort: session release failures
   * are swallowed. The transcriber must not be used after dispose().
   */
  dispose(): void {
    if (this.disposed) {
      return;
    }
    this.disposed = true;
    releaseSession(sessionField(this.model, 'encoderSession'));
    releaseSession(sessionField(this.model, 'joinerSession'));
  }
}
