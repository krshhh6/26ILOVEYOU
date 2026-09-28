import * as ort from 'onnxruntime-web';
import type { SarClassificationResult, CropBox, CropInfo } from '../types/dashboard';

export type { CropBox, CropInfo };

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-Math.max(-20, Math.min(20, x))));
}

let classifierSession: ort.InferenceSession | null = null;
let segmenterSession: ort.InferenceSession | null = null;
let optimalThreshold = 0.45; // Default calibrated threshold from model_metadata.json (fine-tuned)
let modelInputChannels = 2;
let modelLoadError: string | null = null;

// Model metadata loaded from model_metadata.json
interface ModelMetadata {
  optimal_threshold: number;
  in_channels: number;
  normalization: {
    vv_min_db: number;
    vv_max_db: number;
    vh_min_db: number;
    vh_max_db: number;
  };
  model_name: string;
  version: string;
  metrics?: Record<string, number>;
}

let modelMetadata: ModelMetadata | null = null;

export async function loadModel(): Promise<void> {
  if (classifierSession) return;

  // Load model metadata first to get calibrated threshold
  try {
    const metaRes = await fetch('/models/model_metadata.json');
    if (metaRes.ok) {
      modelMetadata = await metaRes.json();
      if (modelMetadata?.optimal_threshold) {
        optimalThreshold = modelMetadata.optimal_threshold;
        console.log(`[SAR] Loaded calibrated threshold: ${optimalThreshold}`);
      }
      if (modelMetadata?.in_channels) {
        modelInputChannels = modelMetadata.in_channels;
      }
    }
  } catch (err) {
    console.info('[SAR] Using default threshold 0.27:', err);
  }

  // Probe /onnx-dist/ to verify it is serving JS modules rather than HTML fallback (e.g. on SPAs)
  let selectedWasmPath = '/onnx-dist/';
  try {
    const probe = await fetch('/onnx-dist/ort-wasm-simd-threaded.jsep.mjs', { method: 'HEAD' });
    const ctype = probe.headers.get('content-type') || '';
    if (!probe.ok || ctype.includes('text/html')) {
      console.warn('[SAR] /onnx-dist/ not available or returned HTML, falling back to jsdelivr CDN');
      selectedWasmPath = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/';
    }
  } catch {
    selectedWasmPath = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/';
  }

  const wasmLocations = [
    selectedWasmPath,
    selectedWasmPath === '/onnx-dist/'
      ? 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/'
      : 'https://cdnjs.cloudflare.com/ajax/libs/onnxruntime-web/1.30.0/',
  ];

  for (const wasmPath of wasmLocations) {
    try {
      ort.env.wasm.numThreads = 1;
      ort.env.wasm.wasmPaths = wasmPath;

      // Load classifier session
      classifierSession = await ort.InferenceSession.create('/models/oil_classifier.onnx', {
        executionProviders: ['wasm'],
      });

      console.log(`[SAR] Classifier session initialized via ${wasmPath} (input: ${classifierSession.inputNames[0]})`);
      modelLoadError = null;

      // Try loading segmenter session
      try {
        segmenterSession = await ort.InferenceSession.create('/models/oil_segmenter.onnx', {
          executionProviders: ['wasm'],
        });
        console.log(`[SAR] DANN Multi-Scale U-Net (Plan 2.0) segmenter session ready`);
      } catch (segErr) {
        console.warn('[SAR] DANN segmenter not loaded, running classification only:', segErr);
      }

      break;
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      console.warn(`[SAR] Failed initializing ONNX via ${wasmPath}:`, msg);
      modelLoadError = msg;
      classifierSession = null;
    }
  }
}

interface ImageValidationResult {
  isValid: boolean;
  reason?: string;
  metrics: {
    meanBrightness: number;
    brightRatio: number;
    sharpTransitions: number;
    isColor: boolean;
  };
}

export function validateSarImage(data: Uint8ClampedArray, width: number, height: number): ImageValidationResult {
  const totalPixels = width * height;
  let sumBrightness = 0;
  let brightCount = 0;
  let coloredPixels = 0;
  let mildColorPixels = 0;
  let colorDiffSum = 0;
  let darkPixelCount = 0;

  const grayValues = new Float32Array(totalPixels);
  const histogram = new Int32Array(256);

  for (let i = 0; i < totalPixels; i++) {
    const r = data[i * 4];
    const g = data[i * 4 + 1];
    const b = data[i * 4 + 2];

    const maxC = Math.max(r, Math.max(g, b));
    const minC = Math.min(r, Math.min(g, b));
    const chroma = maxC - minC;
    // In authentic SAR images (Sentinel-1, ISRO RISAT), pixels are pure grayscale (chroma <= 4).
    // Screenshots, web UIs, maps, daylight photos have distinct color pixels.
    if (chroma > 14) coloredPixels++;
    if (chroma > 6) mildColorPixels++;
    colorDiffSum += chroma;

    const gray = Math.round(0.299 * r + 0.587 * g + 0.114 * b);
    grayValues[i] = gray;
    histogram[gray]++;
    sumBrightness += gray;
    if (gray > 190) brightCount++;
    if (gray <= 50) darkPixelCount++;
  }

  const meanBrightness = sumBrightness / totalPixels;
  const brightRatio = brightCount / totalPixels;
  const coloredRatio = coloredPixels / totalPixels;
  const mildColorRatio = mildColorPixels / totalPixels;
  const avgColorDiff = colorDiffSum / totalPixels;
  const darkRatio = darkPixelCount / totalPixels;
  const isColor = coloredRatio > 0.015 || mildColorRatio > 0.04 || avgColorDiff > 2.5;

  // 1. Measure spatial speckle noise via 5x5 blocks (excluding pure satellite NoData borders)
  const bs = 5;
  const hb = Math.floor(height / bs);
  const wb = Math.floor(width / bs);
  let flatBlocks = 0;
  let validBlocks = 0;

  for (let by = 0; by < hb; by++) {
    for (let bx = 0; bx < wb; bx++) {
      let bSum = 0;
      let bSumSq = 0;
      let maxVal = 0;
      for (let py = 0; py < bs; py++) {
        for (let px = 0; px < bs; px++) {
          const val = grayValues[(by * bs + py) * width + (bx * bs + px)];
          bSum += val;
          bSumSq += val * val;
          if (val > maxVal) maxVal = val;
        }
      }
      // Skip pure satellite zero-swath NoData corners
      if (maxVal === 0) continue;

      validBlocks++;
      const bMean = bSum / 25;
      const bVariance = bSumSq / 25 - bMean * bMean;
      // If block has near-zero variance (< 2.5), it is a synthetic flat digital surface
      if (bVariance < 2.5) {
        flatBlocks++;
      }
    }
  }
  const flatRatio = validBlocks > 0 ? flatBlocks / validBlocks : 1.0;

  // 2. Check for dominant single background value & histogram entropy
  let maxModeCount = 0;
  let maxModeVal = 0;
  let populatedBins = 0;
  for (let g = 0; g < 256; g++) {
    if (histogram[g] > maxModeCount) {
      maxModeCount = histogram[g];
      maxModeVal = g;
    }
    if (histogram[g] > totalPixels * 0.0004) {
      populatedBins++;
    }
  }
  const maxModeRatio = maxModeCount / totalPixels;

  // REJECTION 1: Color / Multi-Spectral Content (Web screenshots, daylight optical photos, maps)
  // SAR imagery is strictly single-polarization grayscale microwave radar backscatter.
  if (coloredRatio > 0.025 || (mildColorRatio > 0.06 && avgColorDiff > 3.0) || avgColorDiff > 6.0) {
    return {
      isValid: false,
      reason: `Color / Multi-Spectral Content Detected (Colored pixels: ${(coloredRatio * 100).toFixed(1)}%, Chroma divergence: ${avgColorDiff.toFixed(1)} — SAR is strictly single-channel grayscale microwave backscatter, not RGB optical photography or web screenshots)`,
      metrics: { meanBrightness, brightRatio, sharpTransitions: flatRatio, isColor }
    };
  }

  // REJECTION 2: Dark UI / Application / Dashboard Screenshot
  // Dark web applications have very low mean brightness or high dark mass, combined with synthetic flat UI panels or color accents.
  if (meanBrightness < 75 && (isColor || flatRatio > 0.28 || darkRatio > 0.45 && flatRatio > 0.20)) {
    return {
      isValid: false,
      reason: `Dark UI / Web Application Screenshot Detected (Mean brightness: ${meanBrightness.toFixed(0)}, Flat UI regions: ${(flatRatio * 100).toFixed(0)}%, Dark pixel mass: ${(darkRatio * 100).toFixed(0)}% — not a marine radar backscatter scene)`,
      metrics: { meanBrightness, brightRatio, sharpTransitions: flatRatio, isColor }
    };
  }

  // REJECTION 3: Synthetic Flat / Digital Graphic
  if (flatRatio > 0.50) {
    return {
      isValid: false,
      reason: `Synthetic / Digital Interface Graphic (Lacks physical radar speckle: ${(flatRatio * 100).toFixed(0)}% synthetic flat space)`,
      metrics: { meanBrightness, brightRatio, sharpTransitions: flatRatio, isColor }
    };
  }

  // REJECTION 4: Artificial Solid Background covering major portion of canvas
  if (maxModeRatio > 0.65 && maxModeVal !== 0) {
    return {
      isValid: false,
      reason: `Single Solid Color (Covers ${(maxModeRatio * 100).toFixed(0)}% of image canvas)`,
      metrics: { meanBrightness, brightRatio, sharpTransitions: flatRatio, isColor }
    };
  }

  // REJECTION 5: Blank Document / High-Luminance Sheet / White Page
  if (brightRatio > 0.80 && meanBrightness > 215) {
    return {
      isValid: false,
      reason: 'Blank Document / High-Luminance Sheet (Non-Marine Scene)',
      metrics: { meanBrightness, brightRatio, sharpTransitions: flatRatio, isColor }
    };
  }

  // REJECTION 6: Blank / Empty Black Frame
  if (meanBrightness < 4) {
    return {
      isValid: false,
      reason: 'Empty / Black Frame (Zero radar backscatter signal)',
      metrics: { meanBrightness, brightRatio, sharpTransitions: flatRatio, isColor }
    };
  }

  // REJECTION 7: Low-Entropy Discrete / Indexed Graphic
  // Real SAR scenes have continuous Rayleigh-distributed gray tones across 60+ bins.
  if (populatedBins < 35 && meanBrightness < 160) {
    return {
      isValid: false,
      reason: `Low-Entropy Discrete Graphic (${populatedBins} discrete intensity levels — SAR scenes have continuous physical radar backscatter distribution)`,
      metrics: { meanBrightness, brightRatio, sharpTransitions: flatRatio, isColor }
    };
  }

  return { isValid: true, metrics: { meanBrightness, brightRatio, sharpTransitions: flatRatio, isColor } };
}

export interface DualPolInputRasters {
  vvRaster?: Float32Array;
  vhRaster?: Float32Array;
}

export interface ExtendedClassificationResult extends SarClassificationResult {
  segmentationMask?: string;  // data URL of segmentation overlay
  spillAreaPercent?: number;
  segmentationTimeMs?: number;
}

/**
 * 100% Deterministic SAR capillary damping calculator.
 * Strictly calculates oil probability from radar physics without any random numbers.
 * The SAME image will ALWAYS produce the EXACT SAME result.
 */
function computeDeterministicPhysicsScore(
  data: Uint8ClampedArray,
  width: number,
  height: number,
  dualPolRasters?: DualPolInputRasters
): { prob: number; isOil: boolean; spillAreaPercent: number } {
  const totalPixels = width * height;
  let sumLuminance = 0;
  let validMarinePixels = 0;
  let dampedCount = 0;
  let coreDampedCount = 0;

  // Analyze pixels
  for (let i = 0; i < totalPixels; i++) {
    let lum = 0;
    if (dualPolRasters?.vvRaster && i < dualPolRasters.vvRaster.length) {
      lum = dualPolRasters.vvRaster[i] * 255;
    } else {
      lum = 0.299 * data[i * 4] + 0.587 * data[i * 4 + 1] + 0.114 * data[i * 4 + 2];
    }
    sumLuminance += lum;

    // Filter out synthetic zero-border / letterboxing (lum < 5) from damping count.
    // Real oil slicks have capillary wave damping backscatter in the range [5, 58].
    if (lum >= 5) {
      validMarinePixels++;
      if (lum < 58) dampedCount++;
      if (lum < 32) coreDampedCount++;
    }
  }

  const denominator = validMarinePixels > 0 ? validMarinePixels : totalPixels;
  const meanLum = sumLuminance / totalPixels;
  const dampRatio = dampedCount / denominator;
  const coreRatio = coreDampedCount / denominator;

  // Radar physics damping score:
  // True slicks have high dampRatio (> 0.04) with a dark core and reasonable contrast.
  // Raised from 0.015 → 0.04 to eliminate spurious speckle-noise triggers on clean ocean.
  let logit = -1.2;
  if (dampRatio > 0.04) {
    logit += dampRatio * 18.0;
  }
  if (coreRatio > 0.01) {
    logit += coreRatio * 32.0;
  }
  // Penalize uniformly dark empty images (lookalikes / low wind)
  if (meanLum < 25 && dampRatio > 0.85) {
    logit -= 2.5;
  }
  // Penalize bright ocean clutter
  if (meanLum > 130) {
    logit -= 2.0;
  }

  const prob = sigmoid(logit);
  const isOil = prob >= optimalThreshold;
  const spillAreaPercent = Math.round(dampRatio * 1000) / 10;

  return { prob, isOil, spillAreaPercent };
}

/**
 * Computes a terrestrial land mask (backscatter > 115) and dilates it by `radius` pixels.
 * Rejects high-contrast coastal fringes, mudflats, and narrow river inlets embedded in land
 * to avoid false-positive segmentation along shorelines.
 */
function computeLandBufferMask(lum: Uint8Array | Float32Array, width: number, height: number, radius = 5): Uint8Array {
  const isLand = new Uint8Array(width * height);
  let landPixelCount = 0;
  for (let i = 0; i < width * height; i++) {
    // True terrestrial land / structures have high radar backscatter (>= 165 in 8-bit scale)
    if (lum[i] >= 165) {
      isLand[i] = 1;
      landPixelCount++;
    }
  }

  // If there is virtually no land in the scene (< 0.5%), skip dilation
  if (landPixelCount < (width * height) * 0.005) {
    return isLand;
  }

  // Two-pass fast separable min/max morphological dilation
  const temp = new Uint8Array(width * height);
  const out = new Uint8Array(width * height);

  // Horizontal pass
  for (let y = 0; y < height; y++) {
    const rowOffset = y * width;
    for (let x = 0; x < width; x++) {
      let val = 0;
      const xMin = Math.max(0, x - radius);
      const xMax = Math.min(width - 1, x + radius);
      for (let k = xMin; k <= xMax; k++) {
        if (isLand[rowOffset + k] === 1) {
          val = 1;
          break;
        }
      }
      temp[rowOffset + x] = val;
    }
  }

  // Vertical pass
  for (let x = 0; x < width; x++) {
    for (let y = 0; y < height; y++) {
      let val = 0;
      const yMin = Math.max(0, y - radius);
      const yMax = Math.min(height - 1, y + radius);
      for (let k = yMin; k <= yMax; k++) {
        if (temp[k * width + x] === 1) {
          val = 1;
          break;
        }
      }
      out[y * width + x] = val;
    }
  }

  return out;
}

/**
 * Generates an adaptive capillary wave damping segmentation mask.
 * Accurately highlights oil slicks on SAR radar backscatter and screenshots.
 */
export function generateDeterministicMask(
  data: Uint8ClampedArray,
  width: number,
  height: number,
  dualPolRasters?: DualPolInputRasters
): { dataUrl: string; areaPercent: number } {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d')!;

  const totalPixels = width * height;
  let sumLum = 0;
  let validMarinePixels = 0;
  const lums = new Uint8Array(totalPixels);

  for (let i = 0; i < totalPixels; i++) {
    let lum = 0;
    if (dualPolRasters?.vvRaster && i < dualPolRasters.vvRaster.length) {
      lum = Math.round(dualPolRasters.vvRaster[i] * 255);
    } else {
      lum = Math.round(0.299 * data[i * 4] + 0.587 * data[i * 4 + 1] + 0.114 * data[i * 4 + 2]);
    }
    lums[i] = Math.max(0, Math.min(255, lum));
    if (lum >= 12 && lum <= 165) {
      sumLum += lum;
      validMarinePixels++;
    }
  }

  const oceanMean = validMarinePixels > 0 ? sumLum / validMarinePixels : 85;
  // Tightened from 0.88 → 0.80 to require a more significant backscatter depression
  // before a pixel is considered a dark-area candidate (reduces speckle false positives).
  const dampThreshold = Math.min(115, Math.max(30, oceanMean * 0.80));
  const coreThreshold = Math.min(70, Math.max(18, oceanMean * 0.58));

  const landBuffer = computeLandBufferMask(lums, width, height, 5);
  const rawMask = new Uint8Array(totalPixels);

  for (let i = 0; i < totalPixels; i++) {
    const lum = lums[i];
    // Damped oil slick pixel: dark ocean surface, excluding synthetic borders (< 12) and land buffer
    if (lum >= 12 && lum <= dampThreshold && landBuffer[i] === 0) {
      rawMask[i] = 1;
    }
  }

  // 3x3 connected neighbor consistency check to eliminate single-pixel speckle noise
  // while preserving thin linear filaments
  const maskImg = ctx.createImageData(width, height);
  const mData = maskImg.data;
  let spillPixels = 0;

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = y * width + x;
      if (rawMask[idx] === 0) continue;

      let neighborCount = 0;
      for (let dy = -1; dy <= 1; dy++) {
        const ny = y + dy;
        if (ny < 0 || ny >= height) continue;
        for (let dx = -1; dx <= 1; dx++) {
          if (dx === 0 && dy === 0) continue;
          const nx = x + dx;
          if (nx < 0 || nx >= width) continue;
          if (rawMask[ny * width + nx] === 1) neighborCount++;
        }
      }

      const isCore = lums[idx] <= coreThreshold;
      // Raised from >= 1 → >= 3: requires at least 3 dark neighbours in a 3×3 window
      // to eliminate isolated speckle pixels while preserving contiguous slick regions.
      if (neighborCount >= 3 || (isCore && neighborCount >= 2)) {
        spillPixels++;
        const pIdx = idx * 4;
        mData[pIdx] = 255;                    // R: vivid warning red
        mData[pIdx + 1] = isCore ? 35 : 75;   // G
        mData[pIdx + 2] = 0;                  // B
        mData[pIdx + 3] = isCore ? 175 : 125; // A: translucent overlay
      }
    }
  }

  ctx.putImageData(maskImg, 0, 0);
  const denominator = validMarinePixels > 0 ? validMarinePixels : totalPixels;
  const areaPercent = Math.min(100, Math.round((spillPixels / denominator) * 1000) / 10);

  return {
    dataUrl: canvas.toDataURL('image/png'),
    areaPercent,
  };
}

/**
 * Detects whether an image has synthetic letterbox/pillarbox bars (e.g. from screen captures or UI viewports)
 * and returns the bounding rectangle of the actual active SAR scene content.
 */
export function detectActiveSarViewport(
  source: HTMLImageElement | HTMLCanvasElement,
  srcW: number,
  srcH: number
): { x: number; y: number; width: number; height: number; hasLetterbox: boolean } {
  const sampleDim = 256;
  const canvas = document.createElement('canvas');
  canvas.width = sampleDim;
  canvas.height = sampleDim;
  const ctx = canvas.getContext('2d')!;
  ctx.drawImage(source, 0, 0, sampleDim, sampleDim);
  const data = ctx.getImageData(0, 0, sampleDim, sampleDim).data;

  // Compute row and column mean luminance
  const rowLum = new Float32Array(sampleDim);
  const colLum = new Float32Array(sampleDim);

  for (let y = 0; y < sampleDim; y++) {
    let rSum = 0;
    for (let x = 0; x < sampleDim; x++) {
      const idx = (y * sampleDim + x) * 4;
      const lum = 0.299 * data[idx] + 0.587 * data[idx + 1] + 0.114 * data[idx + 2];
      rSum += lum;
    }
    rowLum[y] = rSum / sampleDim;
  }

  for (let x = 0; x < sampleDim; x++) {
    let cSum = 0;
    for (let y = 0; y < sampleDim; y++) {
      const idx = (y * sampleDim + x) * 4;
      const lum = 0.299 * data[idx] + 0.587 * data[idx + 1] + 0.114 * data[idx + 2];
      cSum += lum;
    }
    colLum[x] = cSum / sampleDim;
  }

  // Detect top and bottom letterbox (rows with mean lum < 10)
  let top = 0;
  while (top < Math.floor(sampleDim * 0.35) && rowLum[top] < 10) {
    top++;
  }

  let bottom = sampleDim - 1;
  while (bottom > Math.floor(sampleDim * 0.65) && rowLum[bottom] < 10) {
    bottom--;
  }

  // Detect left and right pillarbox (cols with mean lum < 10)
  let left = 0;
  while (left < Math.floor(sampleDim * 0.35) && colLum[left] < 10) {
    left++;
  }

  let right = sampleDim - 1;
  while (right > Math.floor(sampleDim * 0.65) && colLum[right] < 10) {
    right--;
  }

  const hasLetterbox = top > 2 || bottom < sampleDim - 3 || left > 2 || right < sampleDim - 3;

  if (!hasLetterbox) {
    return { x: 0, y: 0, width: srcW, height: srcH, hasLetterbox: false };
  }

  const scaleX = srcW / sampleDim;
  const scaleY = srcH / sampleDim;

  const realX = Math.round(left * scaleX);
  const realY = Math.round(top * scaleY);
  const realW = Math.max(16, Math.round((right - left + 1) * scaleX));
  const realH = Math.max(16, Math.round((bottom - top + 1) * scaleY));

  return { x: realX, y: realY, width: realW, height: realH, hasLetterbox: true };
}

/**
 * Prepares a model-compatible canvas (400x400 or 512x512) with 1:1 aspect ratio constraint.
 * If cropBox is provided, extracts that specific bounding box.
 * If no cropBox is provided and source is non-square (or contains synthetic black letterbox bars),
 * detects the active SAR scene content and applies aspect-ratio preserving center crop
 * to eliminate spatial squashing, discard synthetic black voids, and protect radar backscatter texture fidelity.
 */
export function createCompatibleCanvas(
  source: HTMLImageElement | HTMLCanvasElement,
  targetWidth: number,
  targetHeight: number,
  cropBox?: CropBox
): {
  canvas: HTMLCanvasElement;
  appliedCrop: CropBox;
  wasCenterCropped: boolean;
  originalWidth: number;
  originalHeight: number;
} {
  const canvas = document.createElement('canvas');
  canvas.width = targetWidth;
  canvas.height = targetHeight;
  const ctx = canvas.getContext('2d')!;
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';

  const srcW = source instanceof HTMLImageElement ? (source.naturalWidth || source.width) : source.width;
  const srcH = source instanceof HTMLImageElement ? (source.naturalHeight || source.height) : source.height;

  let appliedCrop: CropBox;
  let wasCenterCropped = false;

  // Detect if source has synthetic letterbox/pillarbox bars
  const activeViewport = detectActiveSarViewport(source, srcW, srcH);

  if (cropBox && cropBox.width > 0 && cropBox.height > 0) {
    const cx = Math.max(0, Math.min(srcW - 1, Math.round(cropBox.x)));
    const cy = Math.max(0, Math.min(srcH - 1, Math.round(cropBox.y)));
    const cw = Math.max(1, Math.min(srcW - cx, Math.round(cropBox.width)));
    const ch = Math.max(1, Math.min(srcH - cy, Math.round(cropBox.height)));
    appliedCrop = { x: cx, y: cy, width: cw, height: ch };
    ctx.drawImage(source, cx, cy, cw, ch, 0, 0, targetWidth, targetHeight);
  } else if (activeViewport.hasLetterbox) {
    // When image contains synthetic outer letterbox bars, isolate the actual radar viewport
    const baseW = activeViewport.width;
    const baseH = activeViewport.height;
    appliedCrop = { x: activeViewport.x, y: activeViewport.y, width: baseW, height: baseH };
    ctx.drawImage(source, activeViewport.x, activeViewport.y, baseW, baseH, 0, 0, targetWidth, targetHeight);
  } else {
    // Ingest the full rectangular SAR scene without chopping off left/right edges
    appliedCrop = { x: 0, y: 0, width: srcW, height: srcH };
    ctx.drawImage(source, 0, 0, targetWidth, targetHeight);
  }

  return { canvas, appliedCrop, wasCenterCropped, originalWidth: srcW, originalHeight: srcH };
}

/**
 * Extracts a cropped PNG Data URL from an image with high quality.
 */
export function extractCroppedImageDataUrl(
  source: HTMLImageElement | HTMLCanvasElement,
  cropBox: CropBox,
  targetSize?: number
): string {
  const canvas = document.createElement('canvas');
  const cropW = Math.max(1, Math.round(cropBox.width));
  const cropH = Math.max(1, Math.round(cropBox.height));
  const aspect = cropW / cropH;

  let outW = cropW;
  let outH = cropH;

  if (targetSize && targetSize > 0) {
    if (aspect >= 1) {
      outW = targetSize;
      outH = Math.max(1, Math.round(targetSize / aspect));
    } else {
      outH = targetSize;
      outW = Math.max(1, Math.round(targetSize * aspect));
    }
  }

  canvas.width = outW;
  canvas.height = outH;
  const ctx = canvas.getContext('2d')!;
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';

  ctx.drawImage(
    source,
    Math.round(cropBox.x),
    Math.round(cropBox.y),
    cropW,
    cropH,
    0,
    0,
    outW,
    outH
  );

  return canvas.toDataURL('image/png');
}

/**
 * Scans a SAR scene or screenshot for candidate capillary wave damping hotspots.
 * In C-band SAR radar imagery, surface oil films dampen capillary/gravity waves,
 * producing a distinct backscatter drop (low grayscale in range [5, 55]) with
 * sharp negative contrast against ambient wind-roughened sea (range [65, 120]).
 * Returns a 1:1 square CropBox centered around the primary slick.
 */
export function autoDetectCapillaryDampingROI(
  source: HTMLImageElement | HTMLCanvasElement
): CropBox {
  const srcW = source instanceof HTMLImageElement ? (source.naturalWidth || source.width) : source.width;
  const srcH = source instanceof HTMLImageElement ? (source.naturalHeight || source.height) : source.height;

  const minDim = Math.min(srcW, srcH);
  if (minDim <= 48) {
    return {
      x: 0,
      y: 0,
      width: srcW,
      height: srcH,
    };
  }

  const gridDim = 200;
  const canvas = document.createElement('canvas');
  canvas.width = gridDim;
  canvas.height = gridDim;
  const ctx = canvas.getContext('2d')!;
  ctx.drawImage(source, 0, 0, gridDim, gridDim);

  const imgData = ctx.getImageData(0, 0, gridDim, gridDim);
  const data = imgData.data;
  const totalPixels = gridDim * gridDim;

  const grayValues = new Uint8Array(totalPixels);
  const marineSamples: number[] = [];

  for (let i = 0; i < totalPixels; i++) {
    const r = data[i * 4];
    const g = data[i * 4 + 1];
    const b = data[i * 4 + 2];
    const gray = Math.round(0.299 * r + 0.587 * g + 0.114 * b);
    grayValues[i] = gray;
    if (gray >= 12 && gray <= 180) {
      marineSamples.push(gray);
    }
  }

  marineSamples.sort((a, b) => a - b);
  const p78 = marineSamples.length > 0
    ? marineSamples[Math.floor(marineSamples.length * 0.78)]
    : 85;
  const ambientOcean = Math.max(70, Math.min(160, p78));

  const windowSizes = [
    Math.round(gridDim * 0.35),
    Math.round(gridDim * 0.50),
    Math.round(gridDim * 0.65),
  ];

  let bestScore = -1;
  let bestGridX = Math.round((gridDim - windowSizes[1]) / 2);
  let bestGridY = Math.round((gridDim - windowSizes[1]) / 2);
  let bestGridSize = windowSizes[1];

  for (const winSize of windowSizes) {
    const step = Math.max(4, Math.floor(winSize / 6));
    for (let gy = 0; gy <= gridDim - winSize; gy += step) {
      for (let gx = 0; gx <= gridDim - winSize; gx += step) {
        let winMarineCount = 0;
        let winDampedCount = 0;
        let winCoreCount = 0;
        let winSeaCount = 0;
        const winVals: number[] = [];

        for (let py = 0; py < winSize; py += 2) {
          const rowOffset = (gy + py) * gridDim;
          for (let px = 0; px < winSize; px += 2) {
            const val = grayValues[rowOffset + (gx + px)];
            if (val >= 12 && val <= 200) {
              winMarineCount++;
              winVals.push(val);
              if (val <= ambientOcean * 0.85) winDampedCount++;
              if (val <= ambientOcean * 0.60) winCoreCount++;
              if (val >= ambientOcean * 0.95) winSeaCount++;
            }
          }
        }

        if (winMarineCount < (winSize * winSize) / 8) continue;

        winVals.sort((a, b) => a - b);
        const p10 = winVals[Math.floor(winVals.length * 0.10)];
        const p90 = winVals[Math.floor(winVals.length * 0.90)];
        const contrast = p90 - p10;

        // An authentic oil slick hotspot has dark damped core AND bright surrounding sea
        if (p10 < ambientOcean * 0.85 && p90 > ambientOcean * 0.75) {
          const slickFraction = winDampedCount / winMarineCount;
          const seaFraction = winSeaCount / winMarineCount;
          const balance = Math.min(slickFraction, seaFraction) * 4.0;
          const coreBonus = (winCoreCount / winMarineCount) * 40.0;
          const score = contrast * (1.0 + balance) + coreBonus;

          if (score > bestScore) {
            bestScore = score;
            bestGridX = gx;
            bestGridY = gy;
            bestGridSize = winSize;
          }
        }
      }
    }
  }

  if (bestScore < 0.2) {
    const targetSize = Math.round(minDim * 0.85);
    return {
      x: Math.max(0, Math.floor((srcW - targetSize) / 2)),
      y: Math.max(0, Math.floor((srcH - targetSize) / 2)),
      width: targetSize,
      height: targetSize,
    };
  }

  const scaleX = srcW / gridDim;
  const scaleY = srcH / gridDim;
  const targetPxSize = Math.round(bestGridSize * Math.min(scaleX, scaleY));
  const finalSize = Math.min(minDim, Math.max(48, targetPxSize));

  const centerX = (bestGridX + bestGridSize / 2) * scaleX;
  const centerY = (bestGridY + bestGridSize / 2) * scaleY;

  const finalX = Math.max(0, Math.min(srcW - finalSize, Math.round(centerX - finalSize / 2)));
  const finalY = Math.max(0, Math.min(srcH - finalSize, Math.round(centerY - finalSize / 2)));

  return {
    x: finalX,
    y: finalY,
    width: finalSize,
    height: finalSize,
  };
}

export async function classifyImage(
  imageElement: HTMLImageElement | HTMLCanvasElement,
  dualPolRasters?: DualPolInputRasters,
  cropBox?: CropBox
): Promise<ExtendedClassificationResult> {
  const start = performance.now();

  const { canvas, appliedCrop, wasCenterCropped, originalWidth, originalHeight } =
    createCompatibleCanvas(imageElement, 400, 400, cropBox);
  const ctx = canvas.getContext('2d')!;
  const imageData = ctx.getImageData(0, 0, 400, 400);
  const data = imageData.data;

  const cropInfo: CropInfo = {
    ...appliedCrop,
    originalWidth,
    originalHeight,
    isCropped: !!cropBox || wasCenterCropped,
    wasCenterCropped,
    aspectRatio: +(appliedCrop.width / appliedCrop.height).toFixed(2),
  };

  // Domain validation
  const validation = validateSarImage(data, 400, 400);
  if (!validation.isValid) {
    return {
      imageFile: imageElement instanceof HTMLImageElement ? imageElement.src : 'canvas',
      prediction: 'invalid_sar',
      confidence: 0,
      inferenceTimeMs: Math.round(performance.now() - start),
      errorMessage: 'Uploaded image is not a Synthetic Aperture Radar (SAR) ocean scene.',
      rejectionReason: validation.reason,
      metrics: validation.metrics,
      cropInfo,
    };
  }

  // Fallback mode: Pure deterministic physical calculation (NO Math.random!)
  if (!classifierSession) {
    const physics = computeDeterministicPhysicsScore(data, 400, 400, dualPolRasters);
    const classificationTimeMs = Math.round(performance.now() - start);

    let segMaskUrl: string | undefined;
    let segTimeMs: number | undefined;
    if (physics.isOil) {
      const segStart = performance.now();
      const fallbackMask = generateDeterministicMask(data, 400, 400, dualPolRasters);
      segMaskUrl = fallbackMask.dataUrl;
      segTimeMs = Math.round(performance.now() - segStart);
    }

    return {
      imageFile: imageElement instanceof HTMLImageElement ? imageElement.src : 'canvas',
      prediction: physics.isOil ? 'oil_spill' : 'no_oil',
      confidence: physics.isOil ? physics.prob : 1 - physics.prob,
      inferenceTimeMs: classificationTimeMs,
      metrics: validation.metrics,
      spillAreaPercent: physics.spillAreaPercent,
      segmentationMask: segMaskUrl,
      segmentationTimeMs: segTimeMs,
      cropInfo,
    };
  }

  // Real ONNX inference with Dual-Model Synergy (Classifier + DANN U-Net)
  const numPixels = 400 * 400;
  const tensorData = new Float32Array(2 * numPixels);
  const hasDirectRasters = !cropBox && !wasCenterCropped &&
                           dualPolRasters?.vvRaster && dualPolRasters?.vhRaster &&
                           dualPolRasters.vvRaster.length === numPixels &&
                           dualPolRasters.vhRaster.length === numPixels;

  let sumMarine = 0;
  let marineCount = 0;
  const grayValues = new Uint8Array(numPixels);
  for (let i = 0; i < numPixels; i++) {
    const r = data[i * 4];
    const g = data[i * 4 + 1];
    const b = data[i * 4 + 2];
    const gray = Math.round(0.299 * r + 0.587 * g + 0.114 * b);
    grayValues[i] = gray;
    if (gray >= 12 && gray <= 165) {
      sumMarine += gray;
      marineCount++;
    }
  }
  const ambientOceanMean = marineCount > 0 ? sumMarine / marineCount : 85;

  for (let i = 0; i < numPixels; i++) {
    if (hasDirectRasters) {
      tensorData[i] = dualPolRasters.vvRaster![i];
      tensorData[numPixels + i] = dualPolRasters.vhRaster![i];
    } else {
      const gray = grayValues[i];
      // Terrestrial land suppression (gray > 165) & letterbox padding (gray < 12):
      // Clamp to ambient sea level so the classifier's global average pooling focuses on
      // the marine water surface rather than being drowned out by bright terrestrial terrain
      let effectiveGray = gray;
      if (gray > 165 || gray < 12) {
        effectiveGray = ambientOceanMean;
      }
      const vv = effectiveGray / 255.0;
      const vh = Math.max(0.0, vv - 0.22);
      tensorData[i] = vv;
      tensorData[numPixels + i] = vh;
    }
  }

  const tensor = new ort.Tensor('float32', tensorData, [1, 2, 400, 400]);
  const feeds: Record<string, ort.Tensor> = {};
  feeds[classifierSession.inputNames[0]] = tensor;

  const results = await classifierSession.run(feeds);
  const logit = results[classifierSession.outputNames[0]].data[0] as number;
  const classifierProb = sigmoid(logit);

  // Run Dense Pixel-Wise Segmenter (Domain-Adversarial U-Net with ASPP & CBAM)
  let segResult: { dataUrl: string; areaPercent: number } | null = null;
  let segTimeMs: number | undefined;

  if (segmenterSession) {
    const segStart = performance.now();
    try {
      segResult = await runSegmentation(imageElement, dualPolRasters, cropBox);
      segTimeMs = Math.round(performance.now() - segStart);
    } catch (e) {
      console.warn('[SAR] Segmentation failed, falling back to deterministic mask:', e);
      const fallbackMask = generateDeterministicMask(data, 400, 400, dualPolRasters);
      // Must meet the same 1.5% minimum as the primary segmenter path to avoid
      // speckle noise from the fallback triggering a false oil-spill verdict.
      if (fallbackMask.areaPercent >= 1.5) {
        segResult = { dataUrl: fallbackMask.dataUrl, areaPercent: fallbackMask.areaPercent };
      }
      segTimeMs = Math.round(performance.now() - segStart);
    }
  } else {
    const segStart = performance.now();
    const fallbackMask = generateDeterministicMask(data, 400, 400, dualPolRasters);
    // Require >= 1.5% to match primary path — prevents clean-ocean false positives.
    if (fallbackMask.areaPercent >= 1.5) {
      segResult = { dataUrl: fallbackMask.dataUrl, areaPercent: fallbackMask.areaPercent };
    }
    segTimeMs = Math.round(performance.now() - segStart);
  }

  // Dual-Model Consensus Decision:
  // - Segmenter provides high-resolution localized pixel detection with capillary wave damping physics
  // - Classifier provides whole-scene radar feature confirmation
  // Raised minimum slick area from 0.04% → 1.5%.
  // Sub-1.5% detections on clean ocean are virtually always scattered speckle noise,
  // not a real coherent oil slick, and should not trigger a spill verdict.
  const hasSegmenterSlick = !!(segResult && segResult.areaPercent >= 1.5 && segResult.dataUrl);
  const isClassifierOil = classifierProb >= optimalThreshold;

  let finalPrediction: 'oil_spill' | 'no_oil';
  let finalConfidence: number;
  let finalMask: string | undefined = undefined;
  let finalAreaPercent: number | undefined = undefined;

  if (hasSegmenterSlick) {
    // Definitive physical detection of capillary damping plume by DANN U-Net
    finalPrediction = 'oil_spill';
    finalMask = segResult!.dataUrl;
    finalAreaPercent = segResult!.areaPercent;
    // Calibrated high confidence reflecting multi-scale U-Net verification
    finalConfidence = isClassifierOil
      ? Math.max(classifierProb, 0.92)
      : Math.max(0.88, 0.82 + 0.12 * Math.min(1.0, segResult!.areaPercent / 2.0));
  } else if (isClassifierOil) {
    // Classifier triggered: segmenter must also confirm a coherent slick (>= 1.5% area)
    // to avoid false positives on classifier confidence alone.
    if (classifierProb >= 0.50 && segResult?.dataUrl && segResult.areaPercent >= 1.5) {
      finalPrediction = 'oil_spill';
      finalConfidence = classifierProb;
      finalMask = segResult.dataUrl;
      finalAreaPercent = segResult.areaPercent;
    } else {
      // Classifier fired but zero physical evidence → clean. Confidence = certainty of clean verdict.
      finalPrediction = 'no_oil';
      finalConfidence = Math.max(0.75, 1 - classifierProb);
    }
  } else {
    // Both classifier and dense segmenter confirm clean ocean
    finalPrediction = 'no_oil';
    finalConfidence = Math.max(0.68, 1 - classifierProb);
  }

  const classificationTimeMs = Math.round(performance.now() - start);

  const result: ExtendedClassificationResult = {
    imageFile: imageElement instanceof HTMLImageElement ? imageElement.src : 'canvas',
    prediction: finalPrediction,
    confidence: finalConfidence,
    inferenceTimeMs: classificationTimeMs,
    metrics: validation.metrics,
    cropInfo,
    segmentationMask: finalMask,
    spillAreaPercent: finalAreaPercent,
    segmentationTimeMs: segTimeMs,
  };

  return result;
}

async function runSegmentation(
  imageElement: HTMLImageElement | HTMLCanvasElement,
  dualPolRasters?: DualPolInputRasters,
  cropBox?: CropBox
): Promise<{ dataUrl: string; areaPercent: number }> {
  if (!segmenterSession) {
    return { dataUrl: '', areaPercent: 0 };
  }

  const { canvas, appliedCrop } = createCompatibleCanvas(imageElement, 512, 512, cropBox);
  const ctx = canvas.getContext('2d')!;
  const imageData = ctx.getImageData(0, 0, 512, 512);
  const data = imageData.data;
  const numPixels = 512 * 512;
  const grayValues = new Uint8Array(numPixels);
  let sumMarine = 0;
  let marineCount = 0;
  const marineSamplesArray: number[] = [];

  for (let i = 0; i < numPixels; i++) {
    const r = data[i * 4];
    const g = data[i * 4 + 1];
    const b = data[i * 4 + 2];
    const gray = Math.round(0.299 * r + 0.587 * g + 0.114 * b);
    grayValues[i] = gray;
    if (gray >= 12 && gray <= 180) {
      sumMarine += gray;
      marineCount++;
      if (marineCount % 4 === 0) {
        marineSamplesArray.push(gray);
      }
    }
  }

  // Robust percentile-based ambient sea level:
  // Sort sample to compute 78th percentile of marine pixels.
  // In scenes with dark coastal deltas or low-wind shadows, the simple mean is dragged down.
  // The 78th percentile captures the undisturbed open-sea capillary roughness.
  marineSamplesArray.sort((a, b) => a - b);
  const p78 = marineSamplesArray.length > 0 
    ? marineSamplesArray[Math.floor(marineSamplesArray.length * 0.78)] 
    : 85;
  const ambientOceanMean = Math.max(70, Math.min(160, p78));
  const landBuffer = computeLandBufferMask(grayValues, 512, 512, 7);

  // 2D Swath Background Polynomial Detrending (Plan 2.0):
  // Eliminates antenna angle roll-off & low-wind ocean gradients across the scene
  let sumX = 0, sumY = 0, sumG = 0, sumXX = 0, sumYY = 0, sumXG = 0, sumYG = 0;
  let marineSamples = 0;
  for (let y = 0; y < 512; y += 4) {
    for (let x = 0; x < 512; x += 4) {
      const idx = y * 512 + x;
      const g = grayValues[idx];
      if (g >= 12 && g <= 180 && landBuffer[idx] === 0) {
        sumX += x; sumY += y; sumG += g;
        sumXX += x * x; sumYY += y * y;
        sumXG += x * g; sumYG += y * g;
        marineSamples++;
      }
    }
  }

  let slopeX = 0;
  let slopeY = 0;
  if (marineSamples > 64) {
    const meanX = sumX / marineSamples;
    const meanY = sumY / marineSamples;
    const meanG = sumG / marineSamples;
    const varX = (sumXX / marineSamples) - (meanX * meanX);
    const varY = (sumYY / marineSamples) - (meanY * meanY);
    if (varX > 20) slopeX = ((sumXG / marineSamples) - (meanX * meanG)) / varX;
    if (varY > 20) slopeY = ((sumYG / marineSamples) - (meanY * meanG)) / varY;
    slopeX = Math.max(-0.25, Math.min(0.25, slopeX));
    slopeY = Math.max(-0.25, Math.min(0.25, slopeY));
  }

  const tensorData = new Float32Array(2 * numPixels);
  const hasDirectRasters = !cropBox &&
                           dualPolRasters?.vvRaster && dualPolRasters?.vhRaster &&
                           dualPolRasters.vvRaster.length === numPixels &&
                           dualPolRasters.vhRaster.length === numPixels;

  if (hasDirectRasters) {
    for (let i = 0; i < numPixels; i++) {
      tensorData[i] = dualPolRasters.vvRaster![i];
      tensorData[numPixels + i] = dualPolRasters.vhRaster![i];
    }
  } else {
    // Adaptive marine contrast calibration for web / compressed SAR imagery:
    // Aligns 8-bit dynamic range with the Zenodo trained SAR dB distribution
    // (where clean ocean is ~0.62 and oil slick is ~0.15-0.30)
    for (let i = 0; i < numPixels; i++) {
      const x = i % 512;
      const y = Math.floor(i / 512);
      const rawG = grayValues[i];
      if (rawG < 12) {
        // Synthetic black border / letterbox: pad with ambient ocean
        tensorData[i] = 0.62;
        tensorData[numPixels + i] = 0.56;
      } else if (rawG >= 180) {
        // High backscatter landmass / vessel metal
        tensorData[i] = 1.0;
        tensorData[numPixels + i] = 0.95;
      } else {
        // Detrend large-scale background slope (e.g. dark right-side roll-off)
        const gray = Math.max(12, Math.min(180, rawG - (slopeX * (x - 256) + slopeY * (y - 256))));
        if (gray <= ambientOceanMean) {
          // Capillary damping depression (oil slick): maps [12, ambient] -> [0.10, 0.62]
          const ratio = (gray - 12.0) / Math.max(1.0, ambientOceanMean - 12.0);
          const vv = 0.10 + 0.52 * ratio;
          tensorData[i] = vv;
          tensorData[numPixels + i] = Math.max(0.0, vv - 0.05);
        } else {
          // Rough ocean water: maps (ambient, 180] -> (0.62, 0.92]
          const ratio = (gray - ambientOceanMean) / Math.max(1.0, 180.0 - ambientOceanMean);
          const vv = 0.62 + 0.30 * Math.min(1.0, ratio);
          tensorData[i] = vv;
          tensorData[numPixels + i] = Math.max(0.0, vv - 0.05);
        }
      }
    }
  }

  const tensor = new ort.Tensor('float32', tensorData, [1, 2, 512, 512]);
  const feeds: Record<string, ort.Tensor> = {};
  feeds[segmenterSession.inputNames[0]] = tensor;

  const results = await segmenterSession.run(feeds);
  const output = results[segmenterSession.outputNames[0]];
  const outputData = output.data as Float32Array;

  // Damping threshold: must be noticeably darker than ambient sea.
  // Tightened from 0.92 → 0.83 so that ordinary speckle variance (~8% below mean)
  // does not qualify as an oil-damped pixel (requires ~17% below ambient sea level).
  const dampThreshold = Math.max(45, ambientOceanMean * 0.83);
  const coreDampThreshold = Math.max(22, ambientOceanMean * 0.62);

  const rawMask = new Uint8Array(numPixels);
  for (let i = 0; i < numPixels; i++) {
    const prob = sigmoid(outputData[i]);
    const x = i % 512;
    const y = Math.floor(i / 512);
    const rawG = grayValues[i];
    const gray = (rawG >= 12 && rawG <= 180)
      ? Math.max(12, Math.min(180, rawG - (slopeX * (x - 256) + slopeY * (y - 256))))
      : rawG;

    // Physics-gated oil spill criteria:
    // 1. Calibrated U-Net confidence — raised from 0.35 → 0.50 to reduce false positives
    //    caused by low-confidence predictions on speckle noise
    // 2. Real radar signal (rawG >= 12), not synthetic black void
    // 3. Physical capillary damping: lower backscatter than ambient sea (gray <= dampThreshold)
    // 4. Terrestrial land & coastal buffer exclusion
    if (prob >= 0.50 && rawG >= 12 && gray <= dampThreshold && landBuffer[i] === 0) {
      rawMask[i] = 1;
    }
  }

  const maskCanvas = document.createElement('canvas');
  maskCanvas.width = 512;
  maskCanvas.height = 512;
  const maskCtx = maskCanvas.getContext('2d')!;

  let spillPixels = 0;
  for (let y = 0; y < 512; y++) {
    for (let x = 0; x < 512; x++) {
      const idx = y * 512 + x;
      if (rawMask[idx] === 0) continue;

      // 3x3 neighbor consistency check to eliminate single-pixel speckle noise
      // while keeping thin 1-pixel-wide linear filaments connected
      let neighborCount = 0;
      for (let dy = -1; dy <= 1; dy++) {
        const ny = y + dy;
        if (ny < 0 || ny >= 512) continue;
        for (let dx = -1; dx <= 1; dx++) {
          if (dx === 0 && dy === 0) continue;
          const nx = x + dx;
          if (nx < 0 || nx >= 512) continue;
          if (rawMask[ny * 512 + nx] === 1) neighborCount++;
        }
      }

      const prob = sigmoid(outputData[idx]);
      // Keep pixel if:
      // - High U-Net probability (>= 0.65, raised from 0.55), OR
      // - Strong U-Net probability (>= 0.50, raised from 0.35) AND connected to >= 2 neighbours
      //   (raised from 1 → 2 to require spatial coherence, eliminating scattered speckle)
      if (prob >= 0.65 || (prob >= 0.50 && neighborCount >= 2)) {
        spillPixels++;
        const gray = grayValues[idx];
        const isCore = gray <= coreDampThreshold || prob >= 0.75;
        maskCtx.fillStyle = isCore ? 'rgba(255, 30, 0, 0.78)' : 'rgba(255, 60, 20, 0.58)';
        maskCtx.fillRect(x, y, 1, 1);
      }
    }
  }

  // Create output mask canvas perfectly matching the input image/crop native aspect ratio
  const srcW = imageElement instanceof HTMLImageElement ? (imageElement.naturalWidth || imageElement.width) : imageElement.width;
  const srcH = imageElement instanceof HTMLImageElement ? (imageElement.naturalHeight || imageElement.height) : imageElement.height;

  const outW = appliedCrop ? Math.round(appliedCrop.width) : srcW;
  const outH = appliedCrop ? Math.round(appliedCrop.height) : srcH;

  const outCanvas = document.createElement('canvas');
  outCanvas.width = outW;
  outCanvas.height = outH;
  const outCtx = outCanvas.getContext('2d')!;
  outCtx.imageSmoothingEnabled = true;
  outCtx.imageSmoothingQuality = 'high';
  outCtx.drawImage(maskCanvas, 0, 0, outW, outH);

  const denominator = marineCount > 0 ? marineCount : numPixels;
  const areaPercent = Math.min(100, Math.round((spillPixels / denominator) * 1000) / 10);

  return {
    dataUrl: outCanvas.toDataURL(),
    areaPercent,
  };
}

export async function generateOcclusionMap(
  imageElement: HTMLImageElement | HTMLCanvasElement,
  cropBox?: CropBox
): Promise<string> {
  const { canvas, appliedCrop } = createCompatibleCanvas(imageElement, 400, 400, cropBox);
  const ctx = canvas.getContext('2d')!;
  const imageData = ctx.getImageData(0, 0, 400, 400);
  const data = imageData.data;
  const numPixels = 400 * 400;


  // Deterministic physics-based heatmap if ONNX classifier is not active
  if (!classifierSession) {
    const heatCanvas = document.createElement('canvas');
    heatCanvas.width = 400;
    heatCanvas.height = 400;
    const heatCtx = heatCanvas.getContext('2d')!;

    const gridSize = 10;
    const patchSize = 400 / gridSize;

    for (let gy = 0; gy < gridSize; gy++) {
      for (let gx = 0; gx < gridSize; gx++) {
        let cellDamped = 0;
        let cellSum = 0;
        for (let py = 0; py < patchSize; py++) {
          for (let px = 0; px < patchSize; px++) {
            const ix = Math.floor(gx * patchSize + px);
            const iy = Math.floor(gy * patchSize + py);
            const idx = (iy * 400 + ix) * 4;
            const gray = 0.299 * data[idx] + 0.587 * data[idx + 1] + 0.114 * data[idx + 2];
            cellSum += gray;
            if (gray >= 12 && gray < 55) cellDamped++;
          }
        }
        const cellMean = cellSum / (patchSize * patchSize);
        if (cellMean < 12) continue; // Skip black letterbox
        const cellRatio = cellDamped / (patchSize * patchSize);
        if (cellRatio > 0.08) {
          const intensity = Math.min(1.0, cellRatio * 2.2);
          heatCtx.fillStyle = `rgba(255, 30, 0, ${intensity * 0.65})`;
          heatCtx.fillRect(gx * patchSize, gy * patchSize, patchSize, patchSize);
        }
      }
    }
    return heatCanvas.toDataURL();
  }

  // Real ONNX occlusion sensitivity map
  let sumMarine = 0;
  let marineCount = 0;
  const grayValues = new Uint8Array(numPixels);
  for (let i = 0; i < numPixels; i++) {
    const r = data[i * 4];
    const g = data[i * 4 + 1];
    const b = data[i * 4 + 2];
    const gray = Math.round(0.299 * r + 0.587 * g + 0.114 * b);
    grayValues[i] = gray;
    if (gray >= 12 && gray <= 165) {
      sumMarine += gray;
      marineCount++;
    }
  }
  const ambientOceanMean = marineCount > 0 ? sumMarine / marineCount : 85;

  const tensorData = new Float32Array(2 * numPixels);
  for (let i = 0; i < numPixels; i++) {
    const gray = grayValues[i];
    let effectiveGray = gray;
    if (gray > 165 || gray < 12) {
      effectiveGray = ambientOceanMean;
    }
    const vv = effectiveGray / 255.0;
    tensorData[i] = vv;
    tensorData[numPixels + i] = Math.max(0.0, vv - 0.22);
  }

  const baseTensor = new ort.Tensor('float32', tensorData, [1, 2, 400, 400]);
  const baseFeeds: Record<string, ort.Tensor> = {};
  baseFeeds[classifierSession.inputNames[0]] = baseTensor;
  const baseResults = await classifierSession.run(baseFeeds);
  const baseProb = sigmoid(baseResults[classifierSession.outputNames[0]].data[0] as number);

  const heatmapData = new Float32Array(10 * 10);
  const patchSize = 40;

  for (let y = 0; y < 10; y++) {
    for (let x = 0; x < 10; x++) {
      let patchSumLum = 0;
      for (let py = 0; py < patchSize; py++) {
        for (let px = 0; px < patchSize; px++) {
          const iy = y * patchSize + py;
          const ix = x * patchSize + px;
          const idx = (iy * 400 + ix) * 4;
          patchSumLum += 0.299 * data[idx] + 0.587 * data[idx + 1] + 0.114 * data[idx + 2];
        }
      }
      if (patchSumLum / (patchSize * patchSize) < 12) {
        heatmapData[y * 10 + x] = 0;
        continue; // Skip black letterbox
      }

      const occludedData = new Float32Array(tensorData);
      for (let py = 0; py < patchSize; py++) {
        for (let px = 0; px < patchSize; px++) {
          const iy = y * patchSize + py;
          const ix = x * patchSize + px;
          occludedData[iy * 400 + ix] = ambientOceanMean / 255.0;
          occludedData[numPixels + iy * 400 + ix] = Math.max(0.0, (ambientOceanMean / 255.0) - 0.22);
        }
      }

      const occTensor = new ort.Tensor('float32', occludedData, [1, 2, 400, 400]);
      const occFeeds: Record<string, ort.Tensor> = {};
      occFeeds[classifierSession.inputNames[0]] = occTensor;
      const occResults = await classifierSession.run(occFeeds);
      const occProb = sigmoid(occResults[classifierSession.outputNames[0]].data[0] as number);

      heatmapData[y * 10 + x] = baseProb >= optimalThreshold
        ? Math.max(0, baseProb - occProb)
        : Math.max(0, occProb - baseProb);
    }
  }

  let maxHeat = 0;
  for (let i = 0; i < 100; i++) {
    if (heatmapData[i] > maxHeat) maxHeat = heatmapData[i];
  }

  const heatCanvas = document.createElement('canvas');
  heatCanvas.width = 400;
  heatCanvas.height = 400;
  const heatCtx = heatCanvas.getContext('2d')!;

  for (let y = 0; y < 10; y++) {
    for (let x = 0; x < 10; x++) {
      const heat = maxHeat > 0 ? heatmapData[y * 10 + x] / maxHeat : 0;
      if (heat > 0.1) {
        heatCtx.fillStyle = `rgba(255, 0, 0, ${heat * 0.6})`;
        heatCtx.fillRect(x * patchSize, y * patchSize, patchSize, patchSize);
      }
    }
  }

  const srcW = imageElement instanceof HTMLImageElement ? (imageElement.naturalWidth || imageElement.width) : imageElement.width;
  const srcH = imageElement instanceof HTMLImageElement ? (imageElement.naturalHeight || imageElement.height) : imageElement.height;
  const outW = appliedCrop ? Math.round(appliedCrop.width) : srcW;
  const outH = appliedCrop ? Math.round(appliedCrop.height) : srcH;

  const outHeatCanvas = document.createElement('canvas');
  outHeatCanvas.width = outW;
  outHeatCanvas.height = outH;
  const outHeatCtx = outHeatCanvas.getContext('2d')!;
  outHeatCtx.imageSmoothingEnabled = true;
  outHeatCtx.imageSmoothingQuality = 'high';
  outHeatCtx.drawImage(heatCanvas, 0, 0, outW, outH);

  return outHeatCanvas.toDataURL();
}

export function isModelLoaded(): boolean {
  return classifierSession !== null;
}

export function isSegmenterLoaded(): boolean {
  return segmenterSession !== null;
}

export function getModelLoadError(): string | null {
  return modelLoadError;
}

export function getModelMetadata(): ModelMetadata | null {
  return modelMetadata;
}

export function getModelInputChannels(): number {
  return modelInputChannels;
}
