/**
 * Public entry point for bark-web.
 *
 * Re-exports the BarkTranscriber wrapper and its public types, plus the
 * static Bark 3.0 model configuration.
 */

import type { BarkModelConfig } from './types.js';

export { BarkTranscriber } from './bark.js';
export type {
  BarkExecutionProvider,
  BarkLoadOptions,
  BarkModelConfig,
  BarkModelUrls,
  BarkProgressCallback,
  BarkTranscriptionResult,
} from './types.js';

/**
 * Static configuration for Bark 3.0 (Conway/Underdog), an English-only
 * Parakeet-TDT speech-to-text model.
 *
 * - Vocab 8193 (blank token 8192), decoder hidden 640, 2 decoder layers.
 * - Encoder 24 layers / 1024 hidden, 128 mel bins, subsampling factor 8.
 * - TDT durations [0, 1, 2, 3, 4], max 10 symbols per step.
 * - Licensed CC-BY 4.0: attribution is required when redistributing.
 *
 * Note: `repoId` follows the Conway/Underdog Bark 3.0 distribution named in
 * the project brief; update it if the canonical hub id differs.
 */
export const BARK_MODEL_CONFIG: BarkModelConfig = {
  repoId: 'ConwayResearch/Underdog-Bark-3.0-Pro',
  displayName: 'Bark 3.0',
  vocabSize: 8193,
  featuresSize: 128,
  subsampling: 8,
  predHidden: 640,
  predLayers: 2,
};
