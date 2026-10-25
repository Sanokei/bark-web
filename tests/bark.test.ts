/**
 * Tests for BarkTranscriber (bark-web).
 *
 * IMPORTANT — READ THIS FIRST:
 * These tests run against the REAL converted Bark 3.0 ONNX weights
 * (encoder-model.int8.onnx, decoder_joint-model.onnx, vocab.txt, nemo128.onnx),
 * produced by scripts/convert.py and staged in tests/fixtures/models/.
 * The encoder is the weight-only int8 build (scripts/quantize_weights.py),
 * a single file with no .data sidecar — required because onnxruntime-web
 * in Node cannot load external-data models.
 * What is under test is the BarkTranscriber code path itself, exercised
 * end-to-end through onnxruntime-web (via parakeet.js).
 *
 * No mocks of the model itself: the inference test below runs REAL inference
 * through onnxruntime-web (via parakeet.js) whenever the model files are
 * present. Model files are NOT downloaded by this suite — they are staged
 * locally. Tests that need model files skip gracefully (with a clear
 * message) when they are absent.
 *
 * The input-validation tests do not load a model: they construct a
 * BarkTranscriber around a minimal stand-in object solely to reach the
 * validation code (which runs before any model call). The stand-in is never
 * invoked on the passing paths, and the real-inference test below covers the
 * actual model path.
 */

// @vitest-environment node

import { execFileSync } from 'node:child_process';
import { existsSync, readFileSync } from 'node:fs';
import { readFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { describe, expect, test } from 'vitest';
import { BarkTranscriber } from '../src/bark.js';
import { BARK_MODEL_CONFIG } from '../src/index.js';

const FIXTURES_DIR = join(dirname(fileURLToPath(import.meta.url)), 'fixtures');
const MODELS_DIR = join(FIXTURES_DIR, 'models');
const VOCAB_PATH = join(FIXTURES_DIR, 'vocab.txt');
const MP3_PATH = join(FIXTURES_DIR, 'bcn_weather.mp3');

/** Model files required for the real-inference test. */
const MODEL_FILES = {
  encoder: join(MODELS_DIR, 'encoder-model.int8.onnx'),
  decoder: join(MODELS_DIR, 'decoder_joint-model.onnx'),
  vocab: join(MODELS_DIR, 'vocab.txt'),
  preprocessor: join(MODELS_DIR, 'nemo128.onnx'),
};

const modelFilesPresent = Object.values(MODEL_FILES).every((p) => existsSync(p));
const vocabPresent = existsSync(VOCAB_PATH);
const mp3Present = existsSync(MP3_PATH);

/**
 * Build a BarkTranscriber without loading a model, so the input-validation
 * code can be exercised without weights. The stand-in is never called on
 * these paths: validation throws before any model invocation.
 */
function makeUnloadedTranscriber(): BarkTranscriber {
  const standIn = {
    transcribe: async (): Promise<never> => {
      throw new Error('stand-in model must never be invoked by validation tests');
    },
  };
  const Ctor = BarkTranscriber as unknown as new (model: unknown) => BarkTranscriber;
  return new Ctor(standIn);
}

describe('BarkTranscriber input validation', () => {
  test('transcribe rejects empty audio', async () => {
    const t = makeUnloadedTranscriber();
    await expect(t.transcribe(new Float32Array(0))).rejects.toThrow(/non-empty/);
  });

  test('transcribe rejects non-finite samples', async () => {
    const t = makeUnloadedTranscriber();
    const audio = new Float32Array([0.1, Number.NaN, 0.2]);
    await expect(t.transcribe(audio)).rejects.toThrow(/finite/);
    const inf = new Float32Array([0.1, Number.POSITIVE_INFINITY]);
    await expect(t.transcribe(inf)).rejects.toThrow(/finite/);
  });

  test('transcribe rejects wrong sample rate', async () => {
    const t = makeUnloadedTranscriber();
    const audio = new Float32Array([0.1, 0.2, 0.3]);
    await expect(t.transcribe(audio, 8000)).rejects.toThrow(/16000/);
    await expect(t.transcribe(audio, 44100)).rejects.toThrow(/16000/);
  });

  test('fromUrls throws when URLs are missing', async () => {
    await expect(
      BarkTranscriber.fromUrls({ encoderUrl: '', decoderUrl: '', tokenizerUrl: '' }),
    ).rejects.toThrow(/encoderUrl|decoderUrl|tokenizerUrl/);
  });

  test('dispose is idempotent and transcribe-after-dispose throws', async () => {
    const t = makeUnloadedTranscriber();
    expect(() => {
      t.dispose();
      t.dispose();
    }).not.toThrow();
    await expect(t.transcribe(new Float32Array([0.1]))).rejects.toThrow(/disposed/);
  });
});

describe('BARK_MODEL_CONFIG', () => {
  test('matches the Bark 3.0 architecture', () => {
    expect(BARK_MODEL_CONFIG.vocabSize).toBe(8193);
    expect(BARK_MODEL_CONFIG.featuresSize).toBe(128);
    expect(BARK_MODEL_CONFIG.subsampling).toBe(8);
    expect(BARK_MODEL_CONFIG.predHidden).toBe(640);
    expect(BARK_MODEL_CONFIG.predLayers).toBe(2);
  });
});

describe.skipIf(!vocabPresent)('tokenizer vocab', () => {
  test('vocab.txt has 8193 entries with <blk> at a valid index', () => {
    const text = readFileSync(VOCAB_PATH, 'utf-8');
    const lines = text.split('\n').map((l) => l.trim()).filter((l) => l.length > 0);
    expect(lines).toHaveLength(8193);
    const entries = lines.map((line) => {
      const idx = line.lastIndexOf(' ');
      return { token: line.slice(0, idx), id: Number(line.slice(idx + 1)) };
    });
    const blank = entries.find((e) => e.token === '<blk>');
    expect(blank).toBeDefined();
    expect(blank?.id).toBeGreaterThanOrEqual(0);
    expect(blank?.id).toBeLessThan(8193);
  });
});

if (!vocabPresent) {
  test('tokenizer vocab (skipped: tests/fixtures/vocab.txt absent)', () => {
    console.info('SKIP: tokenizer vocab test needs tests/fixtures/vocab.txt (fetch it first).');
  });
}

/** Decode an MP3 fixture to 16kHz mono Float32 PCM via ffmpeg, if available. */
function decodeMp3ToPcm(mp3Path: string): Float32Array | null {
  try {
    execFileSync('ffmpeg', ['-version'], { stdio: 'ignore' });
  } catch {
    return null;
  }
  try {
    const raw = execFileSync(
      'ffmpeg',
      ['-v', 'error', '-i', mp3Path, '-ac', '1', '-ar', '16000', '-f', 'f32le', '-'],
      { maxBuffer: 512 * 1024 * 1024 },
    ) as Buffer;
    const samples = new Float32Array(raw.length / 4);
    for (let i = 0; i < samples.length; i += 1) {
      samples[i] = raw.readFloatLE(i * 4);
    }
    return samples;
  } catch {
    return null;
  }
}

describe.skipIf(!modelFilesPresent || !mp3Present)('real inference (bark-3.0)', () => {
  test(
    'transcribes the Barcelona weather sample',
    async () => {
      const pcm = decodeMp3ToPcm(MP3_PATH);
      if (pcm === null || pcm.length === 0) {
        console.info('SKIP: could not decode tests/fixtures/bcn_weather.mp3 (ffmpeg missing or failed).');
        return;
      }
      // Node splits the two loaders by scheme: onnxruntime-web 1.24.1's WASM
      // backend reads models via fs (an http:// URL is treated as a
      // filesystem path and fails with ENOENT), while parakeet.js's
      // tokenizer loader is fetch-only and Node's fetch (undici) does not
      // support file:// URLs. So ORT gets plain absolute filesystem paths
      // (the int8 encoder is a single file with no .data sidecar — that is
      // why the int8 build is staged instead of the fp32 encoder)
      // while the tokenizer gets a file:// URL served by the temporary fetch
      // wrapper below. No preprocessorUrl is passed: omitting it preserves
      // the JS mel preprocessor; nemo128.onnx stays in the presence check
      // above as part of the staged set, but is not loaded on this path.
      // The WASM binaries must match the loaded runtime version: prefer
      // parakeet.js's nested onnxruntime-web dist/ when present, else the
      // top-level dist/ (1.24.1 after the package.json pin + dedupe). Fail
      // loudly if neither exists (never silently skip).
      const testDir = dirname(fileURLToPath(import.meta.url));
      const wasmCandidates = [
        join(testDir, '..', 'node_modules', 'parakeet.js', 'node_modules', 'onnxruntime-web', 'dist'),
        join(testDir, '..', 'node_modules', 'onnxruntime-web', 'dist'),
      ];
      const wasmDir = wasmCandidates.find((dir) =>
        existsSync(join(dir, 'ort-wasm-simd-threaded.wasm')),
      );
      if (wasmDir === undefined) {
        throw new Error(
          'Could not find an onnxruntime-web dist/ directory (looked for ' +
            "parakeet.js's nested onnxruntime-web and the top-level one). " +
            'Run npm install first.',
        );
      }
      const wasmPaths = `${pathToFileURL(wasmDir).href}/`;
      // Temporary fetch wrapper: serve file:// URLs from disk (the tokenizer
      // path) and delegate everything else to the original fetch. Restored
      // in the finally below, after transcriber.dispose(). Any failure here
      // or in model loading fails the test loudly.
      const originalFetch = globalThis.fetch;
      globalThis.fetch = async (input, init) => {
        const url = typeof input === 'string' ? input : input.url;
        if (url.startsWith('file://')) {
          const bytes = await readFile(fileURLToPath(url));
          return new Response(bytes, { status: 200 });
        }
        return originalFetch(input, init);
      };
      try {
        const transcriber = await BarkTranscriber.fromUrls(
          {
            encoderUrl: MODEL_FILES.encoder,
            decoderUrl: MODEL_FILES.decoder,
            tokenizerUrl: pathToFileURL(MODEL_FILES.vocab).href,
          },
          { wasmPaths },
        );
        try {
          const result = await transcriber.transcribe(pcm, 16000);
          expect(result.text.toLowerCase()).toContain('barcelona');
        } finally {
          transcriber.dispose();
        }
      } finally {
        globalThis.fetch = originalFetch;
      }
    },
    300000,
  );
});

if (!modelFilesPresent || !mp3Present) {
  test('real inference (skipped: model files or MP3 fixture absent)', () => {
    console.info(
      'SKIP: real inference test needs tests/fixtures/models/{encoder-model.int8.onnx, ' +
        'decoder_joint-model.onnx, vocab.txt, nemo128.onnx} and tests/fixtures/bcn_weather.mp3.',
    );
  });
}
