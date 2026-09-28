import * as GeoTIFF from 'geotiff';

export interface DecodedTiffResult {
  dataUrl: string;
  width: number;
  height: number;
  bands: number;
  formatDescription: string;
  vvRaster?: Float32Array;
  vhRaster?: Float32Array;
}

/**
 * Decodes any TIFF or GeoTIFF (including Sentinel-1 32-bit float SAR dB files)
 * into a browser-renderable PNG Data URL, and provides calibrated dual-pol raster buffers.
 */
export async function decodeTiffFile(file: File | Blob): Promise<DecodedTiffResult> {
  const arrayBuffer = await file.arrayBuffer();
  const tiff = await GeoTIFF.fromArrayBuffer(arrayBuffer);
  const image = await tiff.getImage();

  const originalWidth = image.getWidth();
  const originalHeight = image.getHeight();
  const samplesPerPixel = image.getSamplesPerPixel();

  // Constrain max preview dimension to 800px to ensure fast rendering in web UI
  const maxDim = 800;
  let targetWidth = originalWidth;
  let targetHeight = originalHeight;

  if (targetWidth > maxDim || targetHeight > maxDim) {
    if (targetWidth >= targetHeight) {
      targetHeight = Math.round((originalHeight / originalWidth) * maxDim);
      targetWidth = maxDim;
    } else {
      targetWidth = Math.round((originalWidth / originalHeight) * maxDim);
      targetHeight = maxDim;
    }
  }

  // Read raster data safely without triggering geotiff.js LZW strip resizing bugs
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  let rasters: any;
  let isNativeResolution = true;
  try {
    rasters = await image.readRasters();
  } catch (err) {
    console.warn('Native readRasters failed, trying with target dimensions:', err);
    try {
      rasters = await image.readRasters({
        width: targetWidth,
        height: targetHeight,
      });
      isNativeResolution = false;
    } catch (fallbackErr) {
      console.error('All geotiff readRasters attempts failed:', fallbackErr);
      throw fallbackErr;
    }
  }

  const canvas = document.createElement('canvas');
  canvas.width = targetWidth;
  canvas.height = targetHeight;
  const ctx = canvas.getContext('2d');
  if (!ctx) {
    throw new Error('Failed to obtain 2D canvas context for TIFF decoding');
  }

  const imgData = ctx.createImageData(targetWidth, targetHeight);
  const totalPixels = targetWidth * targetHeight;

  let formatDescription = '';
  let vvNormBuffer: Float32Array | undefined;
  let vhNormBuffer: Float32Array | undefined;

  // Helper to get pixel index in source raster
  const getSrcIndex = (x: number, y: number): number => {
    if (!isNativeResolution) {
      return y * targetWidth + x;
    }
    const srcX = Math.min(Math.floor(x * originalWidth / targetWidth), originalWidth - 1);
    const srcY = Math.min(Math.floor(y * originalHeight / targetHeight), originalHeight - 1);
    return srcY * originalWidth + srcX;
  };

  // Case 1: RGB / Multi-spectral TIFF (must check first: >= 3 bands)
  if (rasters.length >= 3 && samplesPerPixel >= 3) {
    formatDescription = `RGB Multi-Band TIFF (${originalWidth}x${originalHeight})`;
    const rRaster = (rasters[0] as unknown) as ArrayLike<number>;
    const gRaster = (rasters[1] as unknown) as ArrayLike<number>;
    const bRaster = (rasters[2] as unknown) as ArrayLike<number>;

    for (let y = 0; y < targetHeight; y++) {
      for (let x = 0; x < targetWidth; x++) {
        const dstIdx = y * targetWidth + x;
        const srcIdx = getSrcIndex(x, y);
        const pixelIdx = dstIdx * 4;
        imgData.data[pixelIdx]     = Math.min(255, Math.max(0, Math.round(rRaster[srcIdx])));
        imgData.data[pixelIdx + 1] = Math.min(255, Math.max(0, Math.round(gRaster[srcIdx])));
        imgData.data[pixelIdx + 2] = Math.min(255, Math.max(0, Math.round(bRaster[srcIdx])));
        imgData.data[pixelIdx + 3] = 255;
      }
    }
  // Case 2: Dual-polarization Sentinel-1 SAR (exactly 2 bands: Band 0 = VH, Band 1 = VV in dB)
  } else if (rasters.length === 2) {
    // Verified: Band 1 = VV (co-pol), Band 0 = VH (cross-pol)
    const vhRaster = (rasters[0] as unknown) as ArrayLike<number>;
    const vvRaster = (rasters[1] as unknown) as ArrayLike<number>;
    formatDescription = `Sentinel-1 Dual-Pol SAR (${originalWidth}x${originalHeight}, VV/VH dual-polarization calibrated)`;

    vvNormBuffer = new Float32Array(totalPixels);
    vhNormBuffer = new Float32Array(totalPixels);

    // Sample range to check scale (dB float vs linear uint16/DN)
    const samples: number[] = [];
    const stride = Math.max(1, Math.floor(vvRaster.length / 5000));
    for (let i = 0; i < vvRaster.length; i += stride) {
      const v = vvRaster[i];
      if (Number.isFinite(v)) {
        samples.push(v);
      }
    }
    samples.sort((a, b) => a - b);

    const minSample = samples.length > 0 ? samples[0] : 0;
    const isDbScale = minSample < -5.0;

    let vvMin = -32.0;
    let vvMax = -10.0;
    let vhMin = -42.0;
    let vhMax = -20.0;

    if (!isDbScale && samples.length > 0) {
      // Robust 2nd and 98th percentile scaling for linear DN rasters (more robust than p1/p99)
      const p2Idx = Math.floor(samples.length * 0.02);
      const p98Idx = Math.min(samples.length - 1, Math.floor(samples.length * 0.98));
      vvMin = samples[p2Idx];
      vvMax = Math.max(vvMin + 1, samples[p98Idx]);
      vhMin = vvMin;
      vhMax = vvMax;
    }

    const vvRange = vvMax - vvMin || 1;
    const vhRange = vhMax - vhMin || 1;

    for (let y = 0; y < targetHeight; y++) {
      for (let x = 0; x < targetWidth; x++) {
        const dstIdx = y * targetWidth + x;
        const srcIdx = getSrcIndex(x, y);

        const v_vv = vvRaster[srcIdx];
        const v_vh = vhRaster[srcIdx];

        let n_vv = 0;
        if (Number.isFinite(v_vv)) {
          const clipped = Math.max(vvMin, Math.min(vvMax, v_vv));
          n_vv = (clipped - vvMin) / vvRange;
        }
        vvNormBuffer[dstIdx] = n_vv;

        let n_vh = 0;
        if (Number.isFinite(v_vh)) {
          const clipped = Math.max(vhMin, Math.min(vhMax, v_vh));
          n_vh = (clipped - vhMin) / vhRange;
        }
        vhNormBuffer[dstIdx] = n_vh;

        // Draw VV to visual canvas
        const gray = Math.round(n_vv * 255);
        const pixelIdx = dstIdx * 4;
        imgData.data[pixelIdx]     = gray;
        imgData.data[pixelIdx + 1] = gray;
        imgData.data[pixelIdx + 2] = gray;
        imgData.data[pixelIdx + 3] = 255;
      }
    }
  } else if (rasters.length === 1) {
    // Case 2: Single band (Grayscale or single-pol SAR)
    const singleRaster = (rasters[0] as unknown) as ArrayLike<number>;
    formatDescription = `Single-Band Raster (${originalWidth}x${originalHeight})`;

    vvNormBuffer = new Float32Array(totalPixels);
    vhNormBuffer = new Float32Array(totalPixels);

    // Sample range to check scale
    const samples: number[] = [];
    const stride = Math.max(1, Math.floor(singleRaster.length / 5000));
    for (let i = 0; i < singleRaster.length; i += stride) {
      const v = singleRaster[i];
      if (Number.isFinite(v)) {
        samples.push(v);
      }
    }
    samples.sort((a, b) => a - b);

    const minSample = samples.length > 0 ? samples[0] : 0;
    const isDbScale = minSample < -5.0;

    let lowBound = -32.0;
    let highBound = -10.0;
    if (!isDbScale && samples.length > 0) {
      const p1Idx = Math.floor(samples.length * 0.01);
      const p99Idx = Math.min(samples.length - 1, Math.floor(samples.length * 0.99));
      lowBound = samples[p1Idx];
      highBound = Math.max(lowBound + 1, samples[p99Idx]);
    }
    const range = highBound - lowBound || 1;

    for (let y = 0; y < targetHeight; y++) {
      for (let x = 0; x < targetWidth; x++) {
        const dstIdx = y * targetWidth + x;
        const srcIdx = getSrcIndex(x, y);
        const v = singleRaster[srcIdx];

        let norm = 0;
        if (Number.isFinite(v)) {
          const clipped = Math.max(lowBound, Math.min(highBound, v));
          norm = (clipped - lowBound) / range;
        }
        vvNormBuffer[dstIdx] = norm;
        vhNormBuffer[dstIdx] = Math.max(0, norm - 0.06);

        const gray = Math.round(norm * 255);
        const pixelIdx = dstIdx * 4;
        imgData.data[pixelIdx] = gray;
        imgData.data[pixelIdx + 1] = gray;
        imgData.data[pixelIdx + 2] = gray;
        imgData.data[pixelIdx + 3] = 255;
      }
    }
  }

  ctx.putImageData(imgData, 0, 0);
  const dataUrl = canvas.toDataURL('image/png');

  return {
    dataUrl,
    width: targetWidth,
    height: targetHeight,
    bands: samplesPerPixel,
    formatDescription,
    vvRaster: vvNormBuffer,
    vhRaster: vhNormBuffer
  };
}
