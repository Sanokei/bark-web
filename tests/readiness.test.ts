import { describe, expect, test } from 'vitest';
import { barkReadiness, BARK_RUNTIME_STATUS, preflightBarkAssets } from '../src/readiness.js';

const urls = { encoderUrl: 'https://assets.example/encoder.onnx', decoderUrl: 'https://assets.example/decoder.onnx', tokenizerUrl: 'https://assets.example/vocab.txt' };
describe('configuration and readiness', () => {
  test('an install or a configured URL is never a ready model', () => {
    expect(barkReadiness().ready).toBe(false);
    expect(barkReadiness().canAttemptLoad).toBe(false);
    expect(barkReadiness().missing).toEqual(['encoderUrl', 'decoderUrl', 'tokenizerUrl']);
    expect(barkReadiness(urls).canAttemptLoad).toBe(true);
    expect(barkReadiness(urls).ready).toBe(false);
    expect(BARK_RUNTIME_STATUS.bundledWeights).toBe(false);
    expect(BARK_RUNTIME_STATUS.browserWebGpuVerified).toBe(false);
  });
  test('HEAD-only preflight reports sizes without requesting model bodies', async () => {
    const requests: RequestInit[] = [];
    const result = await preflightBarkAssets(urls, { fetcher: (async (_input, init) => { requests.push(init!); return new Response(null, { headers: { 'content-length': '10' } }); }) as typeof fetch });
    expect(requests).toHaveLength(3);
    expect(requests.every(request => request.method === 'HEAD')).toBe(true);
    expect(result.totalBytes).toBe(30);
    expect(result.ready).toBe(false);
  });
  test('missing assets fail before runtime initialization', async () => {
    await expect(preflightBarkAssets(urls, { fetcher: (async () => new Response(null, { status: 404 })) as typeof fetch })).rejects.toThrow(/unavailable.*404/);
  });
  test('unknown content length is not a zero-byte success', async () => {
    const result = await preflightBarkAssets(urls, { fetcher: (async () => new Response(null)) as typeof fetch });
    expect(result.totalBytes).toBeNull();
    expect(result.assets.every(asset => asset.bytes === null)).toBe(true);
  });
  test('preflight rejects local URLs instead of claiming HTTP reachability', async () => {
    await expect(preflightBarkAssets({ ...urls, encoderUrl: 'file:///models/encoder.onnx' })).rejects.toThrow(/http/);
  });
});
