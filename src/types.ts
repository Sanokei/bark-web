/**
 * Public types for the Bark 3.0 web transcriber.
 *
 * Bark 3.0 (Conway/Underdog) is an English-only speech-to-text model built on
 * NVIDIA Parakeet TDT. It is distributed under CC-BY 4.0, so attribution is
 * required when redistributing the model or its outputs in a product.
 */

/** Static configuration identifying a Bark model build. */
export interface BarkModelConfig {
  /** Hugging Face repo id (or local identifier) the weights come from. */
  repoId: string;
  /** Human-readable name shown in UIs. */
  displayName: string;
  /** Tokenizer vocabulary size, including the blank token (8193 for Bark 3.0). */
  vocabSize: number;
  /** Number of mel filterbank bins produced by the preprocessor (128). */
  featuresSize: number;
  /** Encoder subsampling factor in frames (8). */
  subsampling: number;
  /** Decoder (predictor) LSTM hidden size (640). */
  predHidden: number;
  /** Decoder (predictor) LSTM layer count (2). */
  predLayers: number;
}

/** URLs of the individual model assets, for remote loading. */
export interface BarkModelUrls {
  /** URL to encoder-model.onnx. */
  encoderUrl: string;
  /** URL to decoder_joint-model.onnx. */
  decoderUrl: string;
  /** URL to the tokenizer vocab (vocab.txt / tokens.txt). */
  tokenizerUrl: string;
  /**
   * URL to the nemo preprocessor ONNX file. Optional: when omitted the pure-JS
   * mel preprocessor is used, which is the default.
   */
  preprocessorUrl?: string;
}

/** ONNX Runtime execution providers the caller is willing to use. */
export type BarkExecutionProvider = 'webgpu' | 'wasm';

/** Progress callback invoked while model assets are loading. */
export type BarkProgressCallback = (loaded: number, total: number, label: string) => void;

/**
 * Options for loading a Bark model. Exactly one of `modelUrls` (remote files)
 * or `modelDir` (local directory, Node) must be provided.
 */
export interface BarkLoadOptions {
  /** Remote asset URLs; use with `BarkTranscriber.fromUrls`. */
  modelUrls?: BarkModelUrls;
  /** Local directory containing the model files; use with `fromModelDir`. */
  modelDir?: string;
  /**
   * Preferred execution providers, in priority order. Mapped onto
   * parakeet.js backends: `webgpu` prefers the WebGPU-hybrid backend,
   * `wasm` forces the WASM backend.
   */
  executionProviders?: BarkExecutionProvider[];
  /** Optional progress callback fired during loading. */
  onProgress?: BarkProgressCallback;
  /**
   * Local path or file:// URL prefix for the ONNX Runtime Web WASM
   * binaries. Required in Node because parakeet.js's default CDN URL
   * cannot be ESM-imported by Node. In browsers it can be left unset
   * (the CDN default works).
   */
  wasmPaths?: string;
}

/** Result of a single transcription. */
export interface BarkTranscriptionResult {
  /** The transcribed utterance text. */
  text: string;
  /** Decoded token ids, when requested/available from the decoder. */
  tokens?: number[];
  /**
   * Overall log-probability of the transcription, when the decoder reports
   * confidence scores. Absent when no confidence information is available.
   */
  confidence?: number;
}
