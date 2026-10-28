import type { BarkModelUrls } from './types.js';

/** Static integration facts. Importing this module never loads ONNX Runtime. */
export const BARK_RUNTIME_STATUS = Object.freeze({
  bundledWeights: false,
  estimatedInt8EncoderBytes: 838_000_000,
  browserWebGpuVerified: false,
  language: 'en',
  sampleRate: 16000,
  workingSmallerAlternative: 'https://huggingface.co/onnx-community/whisper-tiny.en',
});

/** A configured URL is not a loaded or verified model. */
export function barkReadiness(urls?: Partial<BarkModelUrls>) {
  const missing = (['encoderUrl', 'decoderUrl', 'tokenizerUrl'] as const).filter(key => !urls?.[key]?.trim());
  return {
    ready: false as const,
    canAttemptLoad: missing.length === 0,
    missing,
    reason: missing.length ? `Supply your own converted model assets: ${missing.join(', ')}.` : 'Assets are configured, but readiness requires awaiting BarkTranscriber.fromUrls(). No weights have been loaded by this check.',
    ...BARK_RUNTIME_STATUS,
  };
}

/** Optional, small HEAD-only preflight. It never downloads model bodies.
 * Host model files with CORS and HEAD support. A successful preflight confirms
 * availability, not ONNX validity or inference quality.
 */
export async function preflightBarkAssets(urls: BarkModelUrls, options: { signal?: AbortSignal; fetcher?: typeof fetch } = {}) {
  const readiness = barkReadiness(urls);
  if (!readiness.canAttemptLoad) throw new Error(readiness.reason);
  const fetcher = options.fetcher ?? fetch;
  const assets = [];
  for (const key of ['encoderUrl', 'decoderUrl', 'tokenizerUrl'] as const) {
    const url = new URL(urls[key]);
    if (!['https:', 'http:'].includes(url.protocol)) throw new Error(`Preflight requires an http(s) URL for ${key}. Use fromModelDir() for local files.`);
    const response = await fetcher(url.href, { method: 'HEAD', signal: options.signal });
    if (!response.ok) throw new Error(`${key} is unavailable (HTTP ${response.status}). Supply your own hosted, converted Bark asset; example.com URLs in the README are placeholders.`);
    const size = Number(response.headers.get('content-length'));
    assets.push({ key, url: url.href, bytes: Number.isSafeInteger(size) && size > 0 ? size : null });
  }
  return { ready: false as const, reachable: true, assets, totalBytes: assets.every(asset => asset.bytes !== null) ? assets.reduce((sum, asset) => sum + asset.bytes!, 0) : null };
}
