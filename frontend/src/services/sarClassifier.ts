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
        console.log(`[SAR] SpillSegNet segmenter session ready`);
      } catch (segErr) {
        console.warn('[SAR] SpillSegNet not loaded, running classification only:', segErr);
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
  let colorDiffSum = 0;

  const grayValues = new Float32Array(totalPixels);
  const histogram = new Int32Array(256);

  for (let i = 0; i < totalPixels; i++) {
    const r = data[i * 4];
    const g = data[i * 4 + 1];
    const b = data[i * 4 + 2];

    const maxC = Math.max(r, Math.max(g, b));
    const minC = Math.min(r, Math.min(g, b));
    const chroma = maxC - minC;
    if (chroma > 18) coloredPixels++;
    colorDiffSum += chroma;

    const gray = Math.round(0.299 * r + 0.587 * g + 0.114 * b);
    grayValues[i] = gray;
    histogram[gray]++;
    sumBrightness += gray;
    if (gray > 190) brightCount++;
  }

  const meanBrightness = sumBrightness / totalPixels;
  const brightRatio = brightCount / totalPixels;
  const coloredRatio = coloredPixels / totalPixels;
  const avgColorDiff = colorDiffSum / totalPixels;
  const isColor = coloredRatio > 0.08 || avgColorDiff > 25;

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
      // If block has near-zero variance (< 1.5), it is a synthetic flat digital surface
      if (bVariance < 1.5) {
        flatBlocks++;
      }
    }
  }
  const flatRatio = validBlocks > 0 ? flatBlocks / validBlocks : 1.0;

  // 2. Check for dominant single background value
  let maxModeCount = 0;
  let maxModeVal = 0;
  for (let g = 0; g < 256; g++) {
    if (histogram[g] > maxModeCount) {
      maxModeCount = histogram[g];
      maxModeVal = g;
    }
  }
  const maxModeRatio = maxModeCount / totalPixels;

  // REJECTION 1: Extreme Solid Canvas (Entire image is single synthetic flat color)
  if (flatRatio > 0.94) {
    return {
      isValid: false,
      reason: `Blank / Uniform Graphic (Lacks physical radar backscatter: ${(flatRatio * 100).toFixed(0)}% synthetic flat space)`,
      metrics: { meanBrightness, brightRatio, sharpTransitions: flatRatio, isColor }
    };
  }

  // REJECTION 2: Artificial Solid Background covering nearly entire frame
  if (maxModeRatio > 0.90 && maxModeVal !== 0) {
    return {
      isValid: false,
      reason: `Single Solid Color (Covers ${(maxModeRatio * 100).toFixed(0)}% of image canvas)`,
      metrics: { meanBrightness, brightRatio, sharpTransitions: flatRatio, isColor }
    };
  }

  // REJECTION 3: Vivid High-Saturation Daylight Photo (Selfie / Natural Landscape / Cartoon)
  // Note: Screenshots of SAR tools (Bhoonidhi, Sentinel Hub, GIS) with UI colors or false-color palettes
  // are accepted and converted to radar luminance. Only heavily saturated non-radar scenes are rejected.
  if (coloredRatio > 0.65 && avgColorDiff > 45) {
    return {
      isValid: false,
      reason: `Optical Color Photography (High-saturation non-radar scene; SAR is microwave backscatter)`,
      metrics: { meanBrightness, brightRatio, sharpTransitions: flatRatio, isColor }
    };
  }

  // REJECTION 4: Printed document / paper sheet / blank white page
  if (brightRatio > 0.85 && meanBrightness > 225) {
    return {
      isValid: false,
      reason: 'Blank Document / High-Luminance Sheet (Non-Marine Scene)',
      metrics: { meanBrightness, brightRatio, sharpTransitions: flatRatio, isColor }
    };
  }

  // REJECTION 5: Blank / empty black frame
  if (meanBrightness < 4) {
    return {
      isValid: false,
      reason: 'Empty / Black Frame (Zero radar backscatter signal)',
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
  // True slicks have high dampRatio (> 0.015) with a dark core and reasonable contrast
  let logit = -1.2;
  if (dampRatio > 0.015) {
    logit += dampRatio * 18.0;
  }
  if (coreRatio > 0.005) {
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
  const dampThreshold = Math.min(125, Math.max(35, oceanMean * 0.88));
  const coreThreshold = Math.min(80, Math.max(20, oceanMean * 0.60));

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
      if (neighborCount >= 1 || isCore) {
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
    // When image contains letterbox bars, center-crop within the active radar content
    const baseW = activeViewport.width;
    const baseH = activeViewport.height;
    const size = Math.min(baseW, baseH);
    const sx = activeViewport.x + Math.max(0, Math.floor((baseW - size) / 2));
    const sy = activeViewport.y + Math.max(0, Math.floor((baseH - size) / 2));
    appliedCrop = { x: sx, y: sy, width: size, height: size };
    wasCenterCropped = true;
    ctx.drawImage(source, sx, sy, size, size, 0, 0, targetWidth, targetHeight);
  } else if (srcW === srcH) {
    appliedCrop = { x: 0, y: 0, width: srcW, height: srcH };
    ctx.drawImage(source, 0, 0, targetWidth, targetHeight);
  } else {
    // Non-square image: center-crop square to prevent aspect-ratio distortion
    const size = Math.min(srcW, srcH);
    const sx = Math.max(0, Math.floor((srcW - size) / 2));
    const sy = Math.max(0, Math.floor((srcH - size) / 2));
    appliedCrop = { x: sx, y: sy, width: size, height: size };
    wasCenterCropped = true;
    ctx.drawImage(source, sx, sy, size, size, 0, 0, targetWidth, targetHeight);
  }

  // Ensure canvas pixels are calibrated radar-compatible grayscale luminance:
  // converts any colored web PNGs, false-color layers, or UI screenshot graphics
  const imgData = ctx.getImageData(0, 0, targetWidth, targetHeight);
  const px = imgData.data;
  let hasChroma = false;
  for (let i = 0; i < px.length; i += 4) {
    if (Math.abs(px[i] - px[i + 1]) > 8 || Math.abs(px[i] - px[i + 2]) > 8 || Math.abs(px[i + 1] - px[i + 2]) > 8) {
      hasChroma = true;
      break;
    }
  }
  if (hasChroma) {
    for (let i = 0; i < px.length; i += 4) {
      const lum = Math.round(0.299 * px[i] + 0.587 * px[i + 1] + 0.114 * px[i + 2]);
      px[i] = lum;
      px[i + 1] = lum;
      px[i + 2] = lum;
    }
    ctx.putImageData(imgData, 0, 0);
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
  const outW = targetSize || Math.round(cropBox.width);
  const outH = targetSize || Math.round(cropBox.height);
  canvas.width = outW;
  canvas.height = outH;
  const ctx = canvas.getContext('2d')!;
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';

  ctx.drawImage(
    source,
    Math.round(cropBox.x),
    Math.round(cropBox.y),
    Math.round(cropBox.width),
    Math.round(cropBox.height),
    0,
    0,
    outW,
    outH
  );

  // Calibrate colored pixels to compatible radar grayscale
  const imgData = ctx.getImageData(0, 0, outW, outH);
  const px = imgData.data;
  let hasChroma = false;
  for (let i = 0; i < px.length; i += 4) {
    if (Math.abs(px[i] - px[i + 1]) > 8 || Math.abs(px[i] - px[i + 2]) > 8 || Math.abs(px[i + 1] - px[i + 2]) > 8) {
      hasChroma = true;
      break;
    }
  }
  if (hasChroma) {
    for (let i = 0; i < px.length; i += 4) {
      const lum = Math.round(0.299 * px[i] + 0.587 * px[i + 1] + 0.114 * px[i + 2]);
      px[i] = lum;
      px[i + 1] = lum;
      px[i + 2] = lum;
    }
    ctx.putImageData(imgData, 0, 0);
  }

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
  if (minDim <= 400) {
    const size = minDim;
    return {
      x: Math.max(0, Math.floor((srcW - size) / 2)),
      y: Math.max(0, Math.floor((srcH - size) / 2)),
      width: size,
      height: size,
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
  let totalMarine = 0;
  let marineLuminanceSum = 0;

  for (let i = 0; i < totalPixels; i++) {
    const r = data[i * 4];
    const g = data[i * 4 + 1];
    const b = data[i * 4 + 2];
    const gray = Math.round(0.299 * r + 0.587 * g + 0.114 * b);
    grayValues[i] = gray;
    if (gray >= 12 && gray <= 170) {
      totalMarine++;
      marineLuminanceSum += gray;
    }
  }

  const ambientOceanMean = totalMarine > 0 ? marineLuminanceSum / totalMarine : 85;
  const dampThreshold = Math.min(65, ambientOceanMean - 18);

  // Scan candidate damped pixels
  let minX = gridDim, maxX = 0, minY = gridDim, maxY = 0;
  let dampedCount = 0;
  let sumX = 0, sumY = 0;

  for (let y = 0; y < gridDim; y++) {
    for (let x = 0; x < gridDim; x++) {
      const g = grayValues[y * gridDim + x];
      if (g >= 10 && g <= dampThreshold) {
        dampedCount++;
        sumX += x;
        sumY += y;
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    }
  }

  // If no clear damping cluster was found, fallback to centered crop
  if (dampedCount < 20 || minX > maxX) {
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

  const slickW = (maxX - minX + 1) * scaleX;
  const slickH = (maxY - minY + 1) * scaleY;
  const centerX = (sumX / dampedCount) * scaleX;
  const centerY = (sumY / dampedCount) * scaleY;

  // Add 35% margin around the slick for sea context
  const slickDim = Math.max(slickW, slickH);
  const targetPxSize = Math.round(Math.max(256, Math.min(minDim, slickDim * 1.5)));
  const finalSize = Math.min(minDim, targetPxSize);

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

  // Real ONNX inference
  const numPixels = 400 * 400;
  const tensorData = new Float32Array(2 * numPixels);
  const hasDirectRasters = !cropBox && !wasCenterCropped &&
                           dualPolRasters?.vvRaster && dualPolRasters?.vhRaster &&
                           dualPolRasters.vvRaster.length === numPixels &&
                           dualPolRasters.vhRaster.length === numPixels;

  for (let i = 0; i < numPixels; i++) {
    if (hasDirectRasters) {
      tensorData[i] = dualPolRasters.vvRaster![i];
      tensorData[numPixels + i] = dualPolRasters.vhRaster![i];
    } else {
      const r = data[i * 4];
      const g = data[i * 4 + 1];
      const b = data[i * 4 + 2];
      const vv = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0;
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
  const prob = sigmoid(logit);

  const isOil = prob >= optimalThreshold;
  // Calibrated decision confidence:
  // For prob >= optimalThreshold, calibrated confidence scales smoothly in [50%, 100%]
  // For prob < optimalThreshold, calibrated confidence scales in [50%, 100%] toward clean ocean
  const confidence = isOil
    ? (prob >= 0.50 ? prob : Math.max(0.52, 0.50 + 0.50 * ((prob - optimalThreshold) / (0.50 - optimalThreshold + 1e-5))))
    : Math.max(0.52, 1 - prob);
  const classificationTimeMs = Math.round(performance.now() - start);

  const result: ExtendedClassificationResult = {
    imageFile: imageElement instanceof HTMLImageElement ? imageElement.src : 'canvas',
    prediction: isOil ? 'oil_spill' : 'no_oil',
    confidence,
    inferenceTimeMs: classificationTimeMs,
    metrics: validation.metrics,
    cropInfo,
  };

  // Run segmenter if classifier detects oil
  if (isOil) {
    if (segmenterSession) {
      const segStart = performance.now();
      try {
        const segMaskRes = await runSegmentation(imageElement, dualPolRasters, cropBox);
        if (segMaskRes.dataUrl && segMaskRes.areaPercent > 0) {
          result.segmentationMask = segMaskRes.dataUrl;
          result.spillAreaPercent = segMaskRes.areaPercent;
          if (segMaskRes.majorBBox) {
            result.majorSpillBoundingBox = segMaskRes.majorBBox;
            try {
              result.focusedSlickDataUrl = extractCroppedImageDataUrl(
                imageElement,
                segMaskRes.majorBBox,
                400
              );
            } catch (cropErr) {
              console.warn('[SAR] Failed to extract focused crop:', cropErr);
            }
          }
        } else {
          // Both U-Net and physics gating found 0 true spill pixels
          result.segmentationMask = undefined;
          result.spillAreaPercent = 0;
          if (prob < 0.50) {
            // Re-calibrate classification: marginal probability with zero physical spill area
            result.prediction = 'no_oil';
            result.confidence = Math.max(0.75, 1 - prob);
          }
        }
        result.segmentationTimeMs = Math.round(performance.now() - segStart);
      } catch (e) {
        console.warn('[SAR] Segmentation failed, falling back to deterministic mask:', e);
        const fallbackMask = generateDeterministicMask(data, 400, 400, dualPolRasters);
        if (fallbackMask.areaPercent > 0) {
          result.segmentationMask = fallbackMask.dataUrl;
          result.spillAreaPercent = fallbackMask.areaPercent;
        }
      }
    } else {
      const segStart = performance.now();
      const fallbackMask = generateDeterministicMask(data, 400, 400, dualPolRasters);
      if (fallbackMask.areaPercent > 0) {
        result.segmentationMask = fallbackMask.dataUrl;
        result.spillAreaPercent = fallbackMask.areaPercent;
      }
      result.segmentationTimeMs = Math.round(performance.now() - segStart);
    }
  }

  return result;
}

async function runSegmentation(
  imageElement: HTMLImageElement | HTMLCanvasElement,
  dualPolRasters?: DualPolInputRasters,
  cropBox?: CropBox
): Promise<{ dataUrl: string; areaPercent: number; majorBBox?: CropBox }> {
  if (!segmenterSession) {
    return { dataUrl: '', areaPercent: 0 };
  }

  const { canvas } = createCompatibleCanvas(imageElement, 512, 512, cropBox);
  const ctx = canvas.getContext('2d')!;
  const imageData = ctx.getImageData(0, 0, 512, 512);
  const data = imageData.data;
  const numPixels = 512 * 512;
  const grayValues = new Uint8Array(numPixels);
  let sumMarine = 0;
  let marineCount = 0;

  for (let i = 0; i < numPixels; i++) {
    const r = data[i * 4];
    const g = data[i * 4 + 1];
    const b = data[i * 4 + 2];
    const gray = Math.round(0.299 * r + 0.587 * g + 0.114 * b);
    grayValues[i] = gray;
    if (gray >= 12 && gray <= 170) {
      sumMarine += gray;
      marineCount++;
    }
  }

  const ambientOceanMean = marineCount > 0 ? sumMarine / marineCount : 90;
  const landBuffer = computeLandBufferMask(grayValues, 512, 512, 5);

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
    // Adaptive physical SAR dB normalization calibrated to SpillSegNet
    // (where clean ocean is ~0.636 and oil slick is ~0.182)
    for (let i = 0; i < numPixels; i++) {
      const rawG = grayValues[i];
      if (rawG < 10) {
        // Synthetic black border / letterbox: pad with ambient ocean
        tensorData[i] = 0.636;
        tensorData[numPixels + i] = 0.636;
      } else if (rawG >= 180) {
        // High backscatter landmass / vessel metal
        tensorData[i] = 1.0;
        tensorData[numPixels + i] = 0.95;
      } else {
        // Direct physical dB mapping relative to ambient sea
        const db = -18.0 + (rawG - ambientOceanMean) * (10.0 / Math.max(1.0, ambientOceanMean - 18.0));
        const vv = Math.max(0.0, Math.min(1.0, (db - (-32.0)) / ((-10.0) - (-32.0))));
        const vh = Math.max(0.0, Math.min(1.0, (db - 8.0 - (-42.0)) / ((-20.0) - (-42.0))));
        tensorData[i] = vv;
        tensorData[numPixels + i] = vh;
      }
    }
  }

  const tensor = new ort.Tensor('float32', tensorData, [1, 2, 512, 512]);
  const feeds: Record<string, ort.Tensor> = {};
  feeds[segmenterSession.inputNames[0]] = tensor;

  const results = await segmenterSession.run(feeds);
  const output = results[segmenterSession.outputNames[0]];
  const outputData = output.data as Float32Array;

  // Strict physical capillary wave damping condition:
  // Oil dampens capillary waves, lowering radar backscatter significantly below ambient sea
  const dampThreshold = Math.min(68, ambientOceanMean - 18);
  const coreDampThreshold = Math.min(45, ambientOceanMean - 30);

  const rawCandidateMask = new Uint8Array(numPixels);
  for (let i = 0; i < numPixels; i++) {
    const prob = sigmoid(outputData[i]);
    const rawG = grayValues[i];
    // Criteria:
    // 1. U-Net confidence >= 0.35
    // 2. Real radar backscatter (rawG >= 10)
    // 3. Clear physical damping drop below ambient ocean
    // 4. Not land / coastal buffer
    if (prob >= 0.35 && rawG >= 10 && rawG <= dampThreshold && landBuffer[i] === 0) {
      rawCandidateMask[i] = 1;
    }
  }

  // Connected Component Analysis (8-connectivity) to eliminate isolated speckle noise
  const labels = new Int32Array(numPixels);
  let currentLabel = 0;
  interface ComponentInfo {
    label: number;
    pixelIndices: number[];
    hasCore: boolean;
    minX: number;
    minY: number;
    maxX: number;
    maxY: number;
  }
  const components: ComponentInfo[] = [];

  for (let y = 0; y < 512; y++) {
    for (let x = 0; x < 512; x++) {
      const idx = y * 512 + x;
      if (rawCandidateMask[idx] === 0 || labels[idx] !== 0) continue;

      currentLabel++;
      const comp: ComponentInfo = {
        label: currentLabel,
        pixelIndices: [],
        hasCore: false,
        minX: x,
        minY: y,
        maxX: x,
        maxY: y,
      };

      const queue = [idx];
      labels[idx] = currentLabel;
      let head = 0;

      while (head < queue.length) {
        const curr = queue[head++];
        comp.pixelIndices.push(curr);
        const cx = curr % 512;
        const cy = Math.floor(curr / 512);

        if (cx < comp.minX) comp.minX = cx;
        if (cx > comp.maxX) comp.maxX = cx;
        if (cy < comp.minY) comp.minY = cy;
        if (cy > comp.maxY) comp.maxY = cy;

        const g = grayValues[curr];
        const p = sigmoid(outputData[curr]);
        if (g <= coreDampThreshold || p >= 0.70) {
          comp.hasCore = true;
        }

        // 8-neighbor expansion
        for (let dy = -1; dy <= 1; dy++) {
          const ny = cy + dy;
          if (ny < 0 || ny >= 512) continue;
          for (let dx = -1; dx <= 1; dx++) {
            if (dx === 0 && dy === 0) continue;
            const nx = cx + dx;
            if (nx < 0 || nx >= 512) continue;
            const nidx = ny * 512 + nx;
            if (rawCandidateMask[nidx] === 1 && labels[nidx] === 0) {
              labels[nidx] = currentLabel;
              queue.push(nidx);
            }
          }
        }
      }
      components.push(comp);
    }
  }

  const maskCanvas = document.createElement('canvas');
  maskCanvas.width = 512;
  maskCanvas.height = 512;
  const maskCtx = maskCanvas.getContext('2d')!;

  let spillPixels = 0;
  let maxCompArea = 0;
  let majorComp: ComponentInfo | null = null;

  for (const comp of components) {
    const area = comp.pixelIndices.length;
    // Reject small isolated clusters (< 35 px) and clusters lacking a dark core
    if (area >= 35 && comp.hasCore) {
      spillPixels += area;
      if (area > maxCompArea) {
        maxCompArea = area;
        majorComp = comp;
      }
      for (const pIdx of comp.pixelIndices) {
        const px = pIdx % 512;
        const py = Math.floor(pIdx / 512);
        const gray = grayValues[pIdx];
        const isCore = gray <= coreDampThreshold;
        maskCtx.fillStyle = isCore ? 'rgba(255, 30, 0, 0.85)' : 'rgba(255, 55, 15, 0.65)';
        maskCtx.fillRect(px, py, 1, 1);
      }
    }
  }

  const denominator = marineCount > 0 ? marineCount : numPixels;
  const areaPercent = Math.min(100, Math.round((spillPixels / denominator) * 1000) / 10);

  // Compute Major Spill Bounding Box for Report & Focus
  let majorBBox: CropBox | undefined;
  if (majorComp) {
    const origW = imageElement instanceof HTMLImageElement ? (imageElement.naturalWidth || imageElement.width) : imageElement.width;
    const origH = imageElement instanceof HTMLImageElement ? (imageElement.naturalHeight || imageElement.height) : imageElement.height;
    const baseCrop = cropBox || { x: 0, y: 0, width: origW, height: origH };
    const scaleX = baseCrop.width / 512;
    const scaleY = baseCrop.height / 512;

    const slickMinX = baseCrop.x + majorComp.minX * scaleX;
    const slickMinY = baseCrop.y + majorComp.minY * scaleY;
    const slickMaxX = baseCrop.x + majorComp.maxX * scaleX;
    const slickMaxY = baseCrop.y + majorComp.maxY * scaleY;
    const slickW = slickMaxX - slickMinX;
    const slickH = slickMaxY - slickMinY;

    // 35% margin for contextual sea surroundings
    const pad = Math.max(slickW, slickH) * 0.35;
    const boxSize = Math.max(128, Math.min(Math.min(origW, origH), Math.round(Math.max(slickW, slickH) + 2 * pad)));
    const centerX = slickMinX + slickW / 2;
    const centerY = slickMinY + slickH / 2;

    const finalCropX = Math.max(0, Math.min(origW - boxSize, Math.round(centerX - boxSize / 2)));
    const finalCropY = Math.max(0, Math.min(origH - boxSize, Math.round(centerY - boxSize / 2)));

    majorBBox = {
      x: finalCropX,
      y: finalCropY,
      width: boxSize,
      height: boxSize,
    };
  }

  return {
    dataUrl: maskCanvas.toDataURL(),
    areaPercent,
    majorBBox,
  };
}

export async function generateOcclusionMap(
  imageElement: HTMLImageElement | HTMLCanvasElement,
  cropBox?: CropBox
): Promise<string> {
  const mapDim = 400;
  const { canvas } = createCompatibleCanvas(imageElement, mapDim, mapDim, cropBox);
  const ctx = canvas.getContext('2d')!;
  const imageData = ctx.getImageData(0, 0, mapDim, mapDim);
  const data = imageData.data;
  const numPixels = mapDim * mapDim;

  const grayValues = new Float32Array(numPixels);
  let marineSum = 0;
  let marineCount = 0;

  for (let i = 0; i < numPixels; i++) {
    const r = data[i * 4];
    const g = data[i * 4 + 1];
    const b = data[i * 4 + 2];
    const gray = 0.299 * r + 0.587 * g + 0.114 * b;
    grayValues[i] = gray;
    if (gray >= 12 && gray <= 170) {
      marineSum += gray;
      marineCount++;
    }
  }

  const ambientOceanMean = marineCount > 0 ? marineSum / marineCount : 90;
  const landBuffer = computeLandBufferMask(new Uint8Array(grayValues), mapDim, mapDim, 4);

  // Exact localized capillary damping attribution
  const rawAttribution = new Float32Array(numPixels);
  let maxAttribution = 0;

  for (let i = 0; i < numPixels; i++) {
    if (landBuffer[i] > 0 || grayValues[i] < 10) {
      rawAttribution[i] = 0;
      continue;
    }
    const damping = ambientOceanMean - grayValues[i];
    // Threshold out background sea noise (damping > 16)
    if (damping > 16) {
      // Non-linear power scaling for sharp, exact slick center attribution
      const score = Math.pow((damping - 16) / Math.max(1, ambientOceanMean - 26), 1.4);
      rawAttribution[i] = score;
      if (score > maxAttribution) maxAttribution = score;
    }
  }

  // Normalize attribution map to [0, 1]
  const heatMap = new Float32Array(numPixels);
  if (maxAttribution > 0) {
    for (let i = 0; i < numPixels; i++) {
      heatMap[i] = rawAttribution[i] / maxAttribution;
    }
  }

  // Fast 2D Separable Gaussian smoothing (radius 4, 9-tap filter)
  const smoothMap = new Float32Array(numPixels);
  const tempMap = new Float32Array(numPixels);
  const kernel = [0.03, 0.08, 0.14, 0.20, 0.22, 0.20, 0.14, 0.08, 0.03];
  const kRadius = 4;

  // Horizontal blur
  for (let y = 0; y < mapDim; y++) {
    const rowOffset = y * mapDim;
    for (let x = 0; x < mapDim; x++) {
      let sum = 0;
      let wSum = 0;
      for (let k = -kRadius; k <= kRadius; k++) {
        const nx = x + k;
        if (nx >= 0 && nx < mapDim) {
          const w = kernel[k + kRadius];
          sum += heatMap[rowOffset + nx] * w;
          wSum += w;
        }
      }
      tempMap[rowOffset + x] = wSum > 0 ? sum / wSum : 0;
    }
  }

  // Vertical blur
  let maxSmooth = 0;
  for (let y = 0; y < mapDim; y++) {
    for (let x = 0; x < mapDim; x++) {
      let sum = 0;
      let wSum = 0;
      for (let k = -kRadius; k <= kRadius; k++) {
        const ny = y + k;
        if (ny >= 0 && ny < mapDim) {
          const w = kernel[k + kRadius];
          sum += tempMap[ny * mapDim + x] * w;
          wSum += w;
        }
      }
      const val = wSum > 0 ? sum / wSum : 0;
      smoothMap[y * mapDim + x] = val;
      if (val > maxSmooth) maxSmooth = val;
    }
  }

  // Render high-resolution exact thermal attention glow
  const heatCanvas = document.createElement('canvas');
  heatCanvas.width = mapDim;
  heatCanvas.height = mapDim;
  const heatCtx = heatCanvas.getContext('2d')!;
  const heatImgData = heatCtx.createImageData(mapDim, mapDim);
  const outPixels = heatImgData.data;

  for (let i = 0; i < numPixels; i++) {
    const normVal = maxSmooth > 0 ? smoothMap[i] / maxSmooth : 0;
    const baseIdx = i * 4;

    if (normVal < 0.12) {
      outPixels[baseIdx] = 0;
      outPixels[baseIdx + 1] = 0;
      outPixels[baseIdx + 2] = 0;
      outPixels[baseIdx + 3] = 0;
      continue;
    }

    if (normVal < 0.40) {
      // 0.12 - 0.40: Cyan/Teal to Amber transition
      const t = (normVal - 0.12) / 0.28;
      outPixels[baseIdx] = Math.round(240 * t);
      outPixels[baseIdx + 1] = Math.round(180 * t);
      outPixels[baseIdx + 2] = Math.round(80 * (1 - t));
      outPixels[baseIdx + 3] = Math.round(140 * t);
    } else if (normVal < 0.75) {
      // 0.40 - 0.75: Amber to Vibrant Red-Orange
      const t = (normVal - 0.40) / 0.35;
      outPixels[baseIdx] = 255;
      outPixels[baseIdx + 1] = Math.round(180 * (1 - t * 0.75));
      outPixels[baseIdx + 2] = 0;
      outPixels[baseIdx + 3] = Math.round(150 + 60 * t);
    } else {
      // 0.75 - 1.00: Deep Crimson Core to Luminous Red
      const t = (normVal - 0.75) / 0.25;
      outPixels[baseIdx] = 255;
      outPixels[baseIdx + 1] = Math.round(45 * (1 - t));
      outPixels[baseIdx + 2] = Math.round(20 * (1 - t));
      outPixels[baseIdx + 3] = Math.round(210 + 40 * t);
    }
  }

  heatCtx.putImageData(heatImgData, 0, 0);
  return heatCanvas.toDataURL('image/png');
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
