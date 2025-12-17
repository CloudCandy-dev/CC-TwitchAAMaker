/**
 * CC-TwitchAAMaker v1.0
 * 画像をTwitchチャット対応の点字アスキーアート（Braille AA）に変換するWebアプリケーション
 * 
 * @author CloudCandy
 * @license MIT
 * @version 1.0.0
 * 
 */

// ========================================
// Constants
// ========================================
const TENJI_COLS = 2; // 点字の横ドット数
const TENJI_ROWS = 4; // 点字の縦ドット数
const BRAILLE_BASE = 0x2800;
const STORAGE_KEY_THEME = 'ccaa-theme';
const STORAGE_KEY_PRESET = 'ccaa-custom-preset';
const STORAGE_KEY_TOS_ACCEPTED = 'ccaa-tos-accepted';
const TOS_VERSION = '1.0'; // 利用規約のバージョン（変更時にインクリメント）
const DEBOUNCE_DELAY = 150; // ms

// ========================================
// Buffer Pool - メモリ再利用でGC圧力を軽減
// ========================================
class BufferPool {
  constructor() {
    this.float32Pools = new Map(); // size -> array of buffers
    this.uint8Pools = new Map();
    this.maxPoolSize = 4; // 各サイズで保持する最大バッファ数
  }

  /**
   * Float32Arrayを取得（プールから再利用または新規作成）
   */
  getFloat32(size) {
    const pool = this.float32Pools.get(size);
    if (pool && pool.length > 0) {
      const buffer = pool.pop();
      buffer.fill(0); // 初期化
      return buffer;
    }
    return new Float32Array(size);
  }

  /**
   * Uint8ClampedArrayを取得
   */
  getUint8Clamped(size) {
    const pool = this.uint8Pools.get(size);
    if (pool && pool.length > 0) {
      const buffer = pool.pop();
      buffer.fill(0);
      return buffer;
    }
    return new Uint8ClampedArray(size);
  }

  /**
   * Float32Arrayをプールに返却
   */
  releaseFloat32(buffer) {
    if (!buffer || !(buffer instanceof Float32Array)) return;
    const size = buffer.length;
    if (!this.float32Pools.has(size)) {
      this.float32Pools.set(size, []);
    }
    const pool = this.float32Pools.get(size);
    if (pool.length < this.maxPoolSize) {
      pool.push(buffer);
    }
  }

  /**
   * Uint8ClampedArrayをプールに返却
   */
  releaseUint8Clamped(buffer) {
    if (!buffer || !(buffer instanceof Uint8ClampedArray)) return;
    const size = buffer.length;
    if (!this.uint8Pools.has(size)) {
      this.uint8Pools.set(size, []);
    }
    const pool = this.uint8Pools.get(size);
    if (pool.length < this.maxPoolSize) {
      pool.push(buffer);
    }
  }

  /**
   * プールをクリア（メモリ解放）
   */
  clear() {
    this.float32Pools.clear();
    this.uint8Pools.clear();
  }
}

// グローバルバッファプール
const bufferPool = new BufferPool();

// Built-in Presets
const PRESETS = {
  default: {
    name: 'デフォルト',
    width: 30,
    threshold: 128,
    autoThreshold: true,
    contrast: 0,
    sharpen: 0.5,
    reverse: true,
    dither: false,
    gamma: 1.0,
    noiseReduction: 'none',
    noiseStrength: 1,
    adaptiveThreshold: false,
    adaptiveBlockSize: 11,
    adaptiveC: 2,
    edgeFilter: 'none',
    useDotForBlank: true,
    contourEnhance: false
  },
  photo: {
    name: '写真向け',
    width: 30,
    threshold: 128,
    autoThreshold: true,
    contrast: 20,
    sharpen: 0.3,
    reverse: true,
    dither: true,
    gamma: 1.2,
    noiseReduction: 'gaussian',
    noiseStrength: 0.8,
    adaptiveThreshold: false,
    adaptiveBlockSize: 11,
    adaptiveC: 2,
    edgeFilter: 'none',
    useDotForBlank: true,
    contourEnhance: false
  },
  lineart: {
    name: '線画・ロゴ向け',
    width: 30,
    threshold: 128,
    autoThreshold: false,
    contrast: 30,
    sharpen: 0.8,
    reverse: true,
    dither: false,
    gamma: 1.0,
    noiseReduction: 'none',
    noiseStrength: 1,
    adaptiveThreshold: true,
    adaptiveBlockSize: 15,
    adaptiveC: 5,
    edgeFilter: 'none',
    useDotForBlank: true,
    contourEnhance: false
  },
  pixel: {
    name: 'ドット絵向け',
    width: 30,
    threshold: 128,
    autoThreshold: false,
    contrast: 0,
    sharpen: 0,
    reverse: true,
    dither: false,
    gamma: 1.0,
    noiseReduction: 'none',
    noiseStrength: 1,
    adaptiveThreshold: false,
    adaptiveBlockSize: 11,
    adaptiveC: 2,
    edgeFilter: 'none',
    useDotForBlank: true,
    contourEnhance: false
  },
  illustration: {
    name: 'イラスト向け',
    width: 30,
    threshold: 128,
    autoThreshold: true,
    contrast: 15,
    sharpen: 0.4,
    reverse: true,
    dither: false,
    gamma: 1.1,
    noiseReduction: 'median',
    noiseStrength: 1,
    adaptiveThreshold: false,
    adaptiveBlockSize: 11,
    adaptiveC: 2,
    edgeFilter: 'none',
    useDotForBlank: true,
    contourEnhance: true
  },
  edge: {
    name: 'エッジ抽出',
    width: 30,
    threshold: 50,
    autoThreshold: false,
    contrast: 0,
    sharpen: 0,
    reverse: false,
    dither: false,
    gamma: 1.0,
    noiseReduction: 'none',
    noiseStrength: 1.0,
    adaptiveThreshold: false,
    adaptiveBlockSize: 11,
    adaptiveC: 2,
    edgeFilter: 'canny',
    useDotForBlank: true,
    contourEnhance: false
  }
};

// ========================================
// Utility Functions
// ========================================

/**
 * 値を0-255の範囲にクランプ
 */
const clampToByte = (v) => Math.max(0, Math.min(255, Math.round(v)));

/**
 * デバウンス関数
 */
const debounce = (fn, delay) => {
  let timeoutId;
  return (...args) => {
    clearTimeout(timeoutId);
    timeoutId = setTimeout(() => fn(...args), delay);
  };
};

/**
 * アルファ合成を考慮したグレースケール値を計算（白背景）
 */
const computeGrayOnWhite = (data, idx) => {
  const alpha = data[idx + 3] / 255;
  const grayBase = (data[idx] * 0.299 + data[idx + 1] * 0.587 + data[idx + 2] * 0.114);
  return grayBase * alpha + 255 * (1 - alpha);
};

/**
 * 8ビットパターンをUnicode点字文字に変換
 * @param {number} bits - 8ビットパターン
 * @param {boolean} useDotForBlank - 空白に1ドット文字を使用するか
 */
const numberToTenji = (bits, useDotForBlank = true) => {
  // ビット配置の変換: 入力ビット → Unicode Braille パターン
  // 入力: [0,1,2,3,4,5,6,7] → 出力: [0,1,2,6,3,4,5,7]
  let flags = 0;
  flags += (bits & 0b00001000) << 3; // bit3 → bit6
  flags += (bits & 0b01110000) >> 1; // bit4-6 → bit3-5
  flags += bits & 0b10000111;        // bit0-2,7 そのまま
  
  // 空の場合の文字を選択
  if (flags === 0) {
    // useDotForBlank: true → 「⡀」(U+2840) 幅統一用
    // useDotForBlank: false → 「⠀」(U+2800) 完全な空白
    return useDotForBlank ? '⡀' : '⠀';
  }
  return String.fromCharCode(flags + 0x2800);
};

// ========================================
// Image Processing Functions
// ========================================

/**
 * Otsu法による最適閾値の自動計算
 */
const calculateOtsuThreshold = (data, width, height) => {
  // ヒストグラムを作成
  const histogram = new Array(256).fill(0);
  const totalPixels = width * height;
  
  for (let i = 0; i < data.length; i += 4) {
    const gray = Math.round(computeGrayOnWhite(data, i));
    histogram[clampToByte(gray)]++;
  }
  
  // 総和を計算
  let sum = 0;
  for (let i = 0; i < 256; i++) {
    sum += i * histogram[i];
  }
  
  let sumB = 0;
  let wB = 0;
  let wF = 0;
  let maxVariance = 0;
  let threshold = 128;
  
  for (let t = 0; t < 256; t++) {
    wB += histogram[t];
    if (wB === 0) continue;
    
    wF = totalPixels - wB;
    if (wF === 0) break;
    
    sumB += t * histogram[t];
    
    const mB = sumB / wB;
    const mF = (sum - sumB) / wF;
    
    const variance = wB * wF * (mB - mF) * (mB - mF);
    
    if (variance > maxVariance) {
      maxVariance = variance;
      threshold = t;
    }
  }
  
  return threshold;
};

/**
 * コントラスト調整 (-100 ~ 100)
 */
const applyContrast = (data, value) => {
  const v = Math.max(-100, Math.min(100, value));
  if (v === 0) return;
  
  const factor = (259 * (v + 255)) / (255 * (259 - v));
  
  for (let i = 0; i < data.length; i += 4) {
    data[i]     = clampToByte(factor * (data[i] - 128) + 128);
    data[i + 1] = clampToByte(factor * (data[i + 1] - 128) + 128);
    data[i + 2] = clampToByte(factor * (data[i + 2] - 128) + 128);
  }
};

/**
 * ガンマ補正を適用
 * @param {Uint8ClampedArray} data - 画像データ
 * @param {number} gamma - ガンマ値 (0.1〜3.0)
 */
const applyGamma = (data, gamma) => {
  if (gamma === 1.0) return;
  
  // ルックアップテーブルを作成（高速化）
  const gammaLUT = new Uint8Array(256);
  const invGamma = 1.0 / gamma;
  
  for (let i = 0; i < 256; i++) {
    gammaLUT[i] = clampToByte(Math.pow(i / 255, invGamma) * 255);
  }
  
  for (let i = 0; i < data.length; i += 4) {
    data[i] = gammaLUT[data[i]];
    data[i + 1] = gammaLUT[data[i + 1]];
    data[i + 2] = gammaLUT[data[i + 2]];
  }
};

/**
 * メディアンフィルタを適用（ノイズ除去）
 * ヒストグラムベースの最適化アルゴリズム - O(W*H)に近い計算量
 * @param {Uint8ClampedArray} data - 画像データ
 * @param {number} width - 画像幅
 * @param {number} height - 画像高さ
 * @param {number} radius - フィルタ半径 (1〜3)
 */
const applyMedianFilter = (data, width, height, radius = 1) => {
  // radiusを整数に変換（小数値対応）
  radius = Math.max(1, Math.round(radius));
  if (radius > 3) radius = 3; // 最大値制限
  
  const output = bufferPool.getUint8Clamped(data.length);
  const size = radius * 2 + 1;
  const windowSize = size * size;
  const medianPos = Math.floor(windowSize / 2);
  
  // 各チャンネル用ヒストグラム（256階調）
  const histR = new Uint16Array(256);
  const histG = new Uint16Array(256);
  const histB = new Uint16Array(256);
  
  /**
   * ヒストグラムから中央値を取得
   */
  const getMedian = (hist, count) => {
    const target = count >> 1; // Math.floor(count / 2)
    let sum = 0;
    for (let i = 0; i < 256; i++) {
      sum += hist[i];
      if (sum > target) return i;
    }
    return 255;
  };
  
  // 各行を処理
  for (let y = 0; y < height; y++) {
    // ヒストグラムをリセット
    histR.fill(0);
    histG.fill(0);
    histB.fill(0);
    
    // 最初のウィンドウを構築
    let pixelCount = 0;
    for (let wy = Math.max(0, y - radius); wy <= Math.min(height - 1, y + radius); wy++) {
      for (let wx = 0; wx <= Math.min(width - 1, radius); wx++) {
        const idx = (wy * width + wx) * 4;
        histR[data[idx]]++;
        histG[data[idx + 1]]++;
        histB[data[idx + 2]]++;
        pixelCount++;
      }
    }
    
    // 最初のピクセルを処理
    const idx0 = y * width * 4;
    output[idx0] = getMedian(histR, pixelCount);
    output[idx0 + 1] = getMedian(histG, pixelCount);
    output[idx0 + 2] = getMedian(histB, pixelCount);
    output[idx0 + 3] = data[idx0 + 3];
    
    // 右にスライドしながら処理
    for (let x = 1; x < width; x++) {
      const leftX = x - radius - 1;
      const rightX = x + radius;
      
      // 左端の列を削除
      if (leftX >= 0) {
        for (let wy = Math.max(0, y - radius); wy <= Math.min(height - 1, y + radius); wy++) {
          const removeIdx = (wy * width + leftX) * 4;
          histR[data[removeIdx]]--;
          histG[data[removeIdx + 1]]--;
          histB[data[removeIdx + 2]]--;
          pixelCount--;
        }
      }
      
      // 右端の列を追加
      if (rightX < width) {
        for (let wy = Math.max(0, y - radius); wy <= Math.min(height - 1, y + radius); wy++) {
          const addIdx = (wy * width + rightX) * 4;
          histR[data[addIdx]]++;
          histG[data[addIdx + 1]]++;
          histB[data[addIdx + 2]]++;
          pixelCount++;
        }
      }
      
      // 中央値を取得
      const idx = (y * width + x) * 4;
      output[idx] = getMedian(histR, pixelCount);
      output[idx + 1] = getMedian(histG, pixelCount);
      output[idx + 2] = getMedian(histB, pixelCount);
      output[idx + 3] = data[idx + 3];
    }
  }
  
  // 元のデータに書き戻し
  data.set(output);
  bufferPool.releaseUint8Clamped(output);
};

/**
 * ガウシアンブラーを適用（ノイズ除去・平滑化）
 * 分離可能フィルタで O(W*H*(2*radius)) に最適化
 * @param {Uint8ClampedArray} data - 画像データ
 * @param {number} width - 画像幅
 * @param {number} height - 画像高さ
 * @param {number} sigma - ブラーの強さ
 */
const applyGaussianBlur = (data, width, height, sigma = 1.0) => {
  if (sigma <= 0) return;
  
  // カーネルサイズを計算
  const radius = Math.ceil(sigma * 3);
  const size = radius * 2 + 1;
  
  // ガウシアンカーネルを生成（固定小数点演算用にスケーリング）
  const kernel = new Float32Array(size);
  let sum = 0;
  
  for (let i = 0; i < size; i++) {
    const x = i - radius;
    const g = Math.exp(-(x * x) / (2 * sigma * sigma));
    kernel[i] = g;
    sum += g;
  }
  
  // 正規化
  const invSum = 1 / sum;
  for (let i = 0; i < size; i++) {
    kernel[i] *= invSum;
  }
  
  const temp = bufferPool.getUint8Clamped(data.length);
  
  // 水平方向のブラー
  for (let y = 0; y < height; y++) {
    const yOffset = y * width * 4;
    for (let x = 0; x < width; x++) {
      let r = 0, g = 0, b = 0;
      
      for (let k = 0; k < size; k++) {
        const nx = Math.min(Math.max(x + k - radius, 0), width - 1);
        const idx = yOffset + nx * 4;
        const w = kernel[k];
        
        r += data[idx] * w;
        g += data[idx + 1] * w;
        b += data[idx + 2] * w;
      }
      
      const idx = yOffset + x * 4;
      temp[idx] = clampToByte(r);
      temp[idx + 1] = clampToByte(g);
      temp[idx + 2] = clampToByte(b);
      temp[idx + 3] = data[idx + 3];
    }
  }
  
  // 垂直方向のブラー（直接dataに書き戻し）
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let r = 0, g = 0, b = 0;
      
      for (let k = 0; k < size; k++) {
        const ny = Math.min(Math.max(y + k - radius, 0), height - 1);
        const idx = (ny * width + x) * 4;
        const w = kernel[k];
        
        r += temp[idx] * w;
        g += temp[idx + 1] * w;
        b += temp[idx + 2] * w;
      }
      
      const idx = (y * width + x) * 4;
      data[idx] = clampToByte(r);
      data[idx + 1] = clampToByte(g);
      data[idx + 2] = clampToByte(b);
      // アルファはtempから復元
      data[idx + 3] = temp[idx + 3];
    }
  }
  
  bufferPool.releaseUint8Clamped(temp);
};

/**
 * 適応的閾値を適用（局所的な閾値計算）
 * @param {Uint8ClampedArray} data - 画像データ
 * @param {number} width - 画像幅
 * @param {number} height - 画像高さ
 * @param {number} blockSize - ブロックサイズ（奇数）
 * @param {number} C - 閾値から引く定数
 */
const applyAdaptiveThreshold = (data, width, height, blockSize = 11, C = 2) => {
  // ブロックサイズは奇数に
  blockSize = blockSize % 2 === 0 ? blockSize + 1 : blockSize;
  const radius = Math.floor(blockSize / 2);
  
  // まずグレースケールに変換
  const gray = new Float32Array(width * height);
  for (let i = 0; i < width * height; i++) {
    const idx = i * 4;
    gray[i] = 0.299 * data[idx] + 0.587 * data[idx + 1] + 0.114 * data[idx + 2];
  }
  
  // 積分画像を計算（高速化のため）
  const integral = new Float64Array((width + 1) * (height + 1));
  
  for (let y = 0; y < height; y++) {
    let rowSum = 0;
    for (let x = 0; x < width; x++) {
      rowSum += gray[y * width + x];
      integral[(y + 1) * (width + 1) + (x + 1)] = 
        integral[y * (width + 1) + (x + 1)] + rowSum;
    }
  }
  
  // 適応的閾値を適用
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const x1 = Math.max(0, x - radius);
      const x2 = Math.min(width, x + radius + 1);
      const y1 = Math.max(0, y - radius);
      const y2 = Math.min(height, y + radius + 1);
      
      const count = (x2 - x1) * (y2 - y1);
      
      // 積分画像から平均を計算
      const sum = integral[y2 * (width + 1) + x2]
                - integral[y1 * (width + 1) + x2]
                - integral[y2 * (width + 1) + x1]
                + integral[y1 * (width + 1) + x1];
      
      const mean = sum / count;
      const threshold = mean - C;
      
      const idx = (y * width + x) * 4;
      const value = gray[y * width + x] > threshold ? 255 : 0;
      
      data[idx] = value;
      data[idx + 1] = value;
      data[idx + 2] = value;
    }
  }
};

/**
 * ガウシアンブラーを適用（エッジ検出の前処理用）
 * 分離可能フィルタで最適化、バッファプール使用
 * @param {Float32Array} gray - グレースケール配列
 * @param {number} width - 画像幅
 * @param {number} height - 画像高さ
 * @param {number} sigma - ガウシアンのσ値
 * @returns {Float32Array} ブラー適用後の配列
 */
const gaussianBlurGray = (gray, width, height, sigma = 1.4) => {
  const kernelSize = Math.ceil(sigma * 3) * 2 + 1;
  const kernel = new Float32Array(kernelSize);
  const half = Math.floor(kernelSize / 2);
  let sum = 0;
  
  for (let i = 0; i < kernelSize; i++) {
    const x = i - half;
    const g = Math.exp(-(x * x) / (2 * sigma * sigma));
    kernel[i] = g;
    sum += g;
  }
  const invSum = 1 / sum;
  for (let i = 0; i < kernelSize; i++) {
    kernel[i] *= invSum;
  }
  
  const size = width * height;
  const temp = bufferPool.getFloat32(size);
  const result = bufferPool.getFloat32(size);
  
  // 水平方向
  for (let y = 0; y < height; y++) {
    const yOffset = y * width;
    for (let x = 0; x < width; x++) {
      let acc = 0;
      for (let k = 0; k < kernelSize; k++) {
        const sx = Math.min(Math.max(x + k - half, 0), width - 1);
        acc += gray[yOffset + sx] * kernel[k];
      }
      temp[yOffset + x] = acc;
    }
  }
  
  // 垂直方向
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let acc = 0;
      for (let k = 0; k < kernelSize; k++) {
        const sy = Math.min(Math.max(y + k - half, 0), height - 1);
        acc += temp[sy * width + x] * kernel[k];
      }
      result[y * width + x] = acc;
    }
  }
  
  bufferPool.releaseFloat32(temp);
  // resultは呼び出し元で管理
  return result;
};

/**
 * Scharr オペレータによるエッジ検出（Sobelより高精度）
 * 展開ループで高速化
 */
const applyScharr = (gray, width, height) => {
  const size = width * height;
  const magnitude = bufferPool.getFloat32(size);
  const direction = bufferPool.getFloat32(size);
  
  // Scharr カーネル係数（展開済み）
  // kernelX = [[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]];
  // kernelY = [[-3, -10, -3], [0, 0, 0], [3, 10, 3]];
  
  for (let y = 1; y < height - 1; y++) {
    const yOffset = y * width;
    for (let x = 1; x < width - 1; x++) {
      // 9近傍のインデックス（キャッシュフレンドリーに）
      const idx00 = (y - 1) * width + (x - 1);
      const idx01 = (y - 1) * width + x;
      const idx02 = (y - 1) * width + (x + 1);
      const idx10 = yOffset + (x - 1);
      const idx12 = yOffset + (x + 1);
      const idx20 = (y + 1) * width + (x - 1);
      const idx21 = (y + 1) * width + x;
      const idx22 = (y + 1) * width + (x + 1);
      
      // Scharrカーネル適用（展開）
      const sumX = -3 * gray[idx00] + 3 * gray[idx02]
                 - 10 * gray[idx10] + 10 * gray[idx12]
                 - 3 * gray[idx20] + 3 * gray[idx22];
      
      const sumY = -3 * gray[idx00] - 10 * gray[idx01] - 3 * gray[idx02]
                 + 3 * gray[idx20] + 10 * gray[idx21] + 3 * gray[idx22];
      
      const idx = yOffset + x;
      magnitude[idx] = Math.sqrt(sumX * sumX + sumY * sumY);
      direction[idx] = Math.atan2(sumY, sumX);
    }
  }
  
  return { magnitude, direction };
};

/**
 * 非最大値抑制 (Cannyエッジ検出用)
 * バッファプール使用で最適化
 */
const nonMaxSuppression = (magnitude, direction, width, height) => {
  const size = width * height;
  const result = bufferPool.getFloat32(size);
  
  // ラジアンから度への変換係数
  const RAD_TO_DEG = 180 / Math.PI;
  
  for (let y = 1; y < height - 1; y++) {
    const yOffset = y * width;
    for (let x = 1; x < width - 1; x++) {
      const idx = yOffset + x;
      const angle = direction[idx] * RAD_TO_DEG;
      const mag = magnitude[idx];
      
      let neighbor1, neighbor2;
      
      // 角度に基づいて隣接ピクセルを選択（ブランチ予測最適化）
      if ((angle >= -22.5 && angle < 22.5) || (angle >= 157.5 || angle < -157.5)) {
        neighbor1 = magnitude[yOffset + (x - 1)];
        neighbor2 = magnitude[yOffset + (x + 1)];
      } else if ((angle >= 22.5 && angle < 67.5) || (angle >= -157.5 && angle < -112.5)) {
        neighbor1 = magnitude[(y - 1) * width + (x + 1)];
        neighbor2 = magnitude[(y + 1) * width + (x - 1)];
      } else if ((angle >= 67.5 && angle < 112.5) || (angle >= -112.5 && angle < -67.5)) {
        neighbor1 = magnitude[(y - 1) * width + x];
        neighbor2 = magnitude[(y + 1) * width + x];
      } else {
        neighbor1 = magnitude[(y - 1) * width + (x - 1)];
        neighbor2 = magnitude[(y + 1) * width + (x + 1)];
      }
      
      result[idx] = (mag >= neighbor1 && mag >= neighbor2) ? mag : 0;
    }
  }
  
  return result;
};

/**
 * ヒステリシス閾値処理 (Cannyエッジ検出用)
 * スタックベースの連結成分探索で高速化
 */
const hysteresisThreshold = (nms, width, height, lowThreshold, highThreshold) => {
  const size = width * height;
  const result = bufferPool.getFloat32(size);
  const visited = new Uint8Array(size); // 訪問フラグ
  const strong = 255;
  
  // 強いエッジのリストを作成
  const strongEdges = [];
  
  // 初期分類
  for (let i = 0; i < size; i++) {
    if (nms[i] >= highThreshold) {
      result[i] = strong;
      strongEdges.push(i);
      visited[i] = 1;
    } else if (nms[i] >= lowThreshold) {
      result[i] = 0; // 後で判定
    } else {
      result[i] = 0;
      visited[i] = 1; // 処理不要
    }
  }
  
  // 8近傍オフセット（事前計算）
  const offsets = [
    -width - 1, -width, -width + 1,
    -1,                  1,
    width - 1,  width,   width + 1
  ];
  
  // スタックベースの連結成分探索（DFS）
  const stack = [...strongEdges];
  
  while (stack.length > 0) {
    const current = stack.pop();
    const x = current % width;
    const y = (current / width) | 0; // Math.floor
    
    // 境界チェック用
    const atLeft = x === 0;
    const atRight = x === width - 1;
    const atTop = y === 0;
    const atBottom = y === height - 1;
    
    // 8近傍を探索
    for (let d = 0; d < 8; d++) {
      // 境界チェック
      if (atLeft && (d === 0 || d === 3 || d === 5)) continue;
      if (atRight && (d === 2 || d === 4 || d === 7)) continue;
      if (atTop && (d === 0 || d === 1 || d === 2)) continue;
      if (atBottom && (d === 5 || d === 6 || d === 7)) continue;
      
      const neighbor = current + offsets[d];
      
      if (!visited[neighbor] && nms[neighbor] >= lowThreshold) {
        result[neighbor] = strong;
        visited[neighbor] = 1;
        stack.push(neighbor);
      }
    }
  }
  
  return result;
};

/**
 * Cannyエッジ検出
 * バッファプール使用でメモリ効率化
 * @param {Float32Array} gray - グレースケール配列
 * @param {number} width - 画像幅
 * @param {number} height - 画像高さ
 * @param {number} sigma - ガウシアンブラーのσ
 * @param {number} lowThreshold - 低閾値
 * @param {number} highThreshold - 高閾値
 */
const applyCanny = (gray, width, height, sigma = 1.4, lowThreshold = 20, highThreshold = 50) => {
  // 1. ガウシアンブラーでノイズ除去
  const blurred = gaussianBlurGray(gray, width, height, sigma);
  
  // 2. Scharrオペレータで勾配計算
  const { magnitude, direction } = applyScharr(blurred, width, height);
  
  // blurredは不要になったので解放
  bufferPool.releaseFloat32(blurred);
  
  // 3. 非最大値抑制
  const nms = nonMaxSuppression(magnitude, direction, width, height);
  
  // magnitude, directionは不要になったので解放
  bufferPool.releaseFloat32(magnitude);
  bufferPool.releaseFloat32(direction);
  
  // 4. ヒステリシス閾値処理
  const result = hysteresisThreshold(nms, width, height, lowThreshold, highThreshold);
  
  // nmsは不要になったので解放
  bufferPool.releaseFloat32(nms);
  
  return result;
};

/**
 * Difference of Gaussians (DoG) - 輪郭強調に効果的
 * バッファプール使用で最適化
 * @param {Float32Array} gray - グレースケール配列
 * @param {number} width - 画像幅
 * @param {number} height - 画像高さ
 * @param {number} sigma1 - 小さいσ
 * @param {number} sigma2 - 大きいσ
 */
const applyDoG = (gray, width, height, sigma1 = 1.0, sigma2 = 2.0) => {
  const blur1 = gaussianBlurGray(gray, width, height, sigma1);
  const blur2 = gaussianBlurGray(gray, width, height, sigma2);
  
  const size = width * height;
  const result = bufferPool.getFloat32(size);
  
  // DoG計算と最大値検出を同時に行う
  let maxVal = 0;
  for (let i = 0; i < size; i++) {
    const diff = blur1[i] - blur2[i];
    const absVal = Math.abs(diff) * 2;
    result[i] = absVal;
    if (absVal > maxVal) maxVal = absVal;
  }
  
  // blur1, blur2を解放
  bufferPool.releaseFloat32(blur1);
  bufferPool.releaseFloat32(blur2);
  
  // 正規化
  if (maxVal > 0) {
    const scale = 255 / maxVal;
    for (let i = 0; i < size; i++) {
      result[i] *= scale;
    }
  }
  
  return result;
};

/**
 * エッジブレンド - 元画像とエッジを合成して輪郭を強調
 * バッファプール使用
 * @param {Float32Array} gray - 元のグレースケール
 * @param {Float32Array} edge - エッジ検出結果
 * @param {number} strength - ブレンド強度 (0-1)
 * @param {string} mode - ブレンドモード ('add', 'multiply', 'overlay')
 */
const blendEdge = (gray, edge, width, height, strength = 0.5, mode = 'add') => {
  const size = width * height;
  const result = bufferPool.getFloat32(size);
  
  for (let i = 0; i < size; i++) {
    let blended;
    const g = gray[i];
    const e = edge[i];
    
    switch (mode) {
      case 'multiply':
        // エッジ部分を暗くする
        blended = g * (1 - (e / 255) * strength);
        break;
      case 'overlay':
        // オーバーレイ合成
        if (g < 128) {
          blended = (2 * g * (255 - e * strength)) / 255;
        } else {
          blended = 255 - (2 * (255 - g) * (255 - (255 - e) * strength)) / 255;
        }
        break;
      case 'subtract':
        // 減算（エッジ部分を黒く）
        blended = Math.max(0, g - e * strength);
        break;
      case 'add':
      default:
        // 加算（エッジ部分を白く強調）
        blended = Math.min(255, g + e * strength);
        break;
    }
    
    result[i] = blended;
  }
  
  return result;
};

/**
 * エッジ検出フィルタを適用（強化版）
 * バッファプール使用、グレースケール再利用で最適化
 * @param {Uint8ClampedArray} data - 画像データ
 * @param {number} width - 画像幅
 * @param {number} height - 画像高さ
 * @param {string} type - フィルタタイプ ('sobel', 'laplacian', 'prewitt', 'canny', 'scharr', 'dog', 'edge-enhance')
 */
const applyEdgeDetection = (data, width, height, type = 'sobel') => {
  const size = width * height;
  
  // グレースケールに変換（バッファプール使用）
  const gray = bufferPool.getFloat32(size);
  for (let i = 0; i < size; i++) {
    const idx = i * 4;
    gray[i] = 0.299 * data[idx] + 0.587 * data[idx + 1] + 0.114 * data[idx + 2];
  }
  
  let output;
  
  switch (type) {
    case 'canny':
      // Cannyエッジ検出（最も高品質）
      output = applyCanny(gray, width, height, 1.4, 15, 40);
      break;
      
    case 'scharr': {
      // Scharr（Sobelより高精度）
      const scharrResult = applyScharr(gray, width, height);
      output = bufferPool.getFloat32(size);
      let maxMag = 0;
      for (let i = 0; i < size; i++) {
        if (scharrResult.magnitude[i] > maxMag) maxMag = scharrResult.magnitude[i];
      }
      const scale = maxMag > 0 ? 255 / maxMag : 1;
      for (let i = 0; i < size; i++) {
        output[i] = scharrResult.magnitude[i] * scale;
      }
      // scharrResultのバッファを解放
      bufferPool.releaseFloat32(scharrResult.magnitude);
      bufferPool.releaseFloat32(scharrResult.direction);
      break;
    }
      
    case 'dog':
      // Difference of Gaussians（輪郭抽出）
      output = applyDoG(gray, width, height, 0.8, 1.6);
      break;
      
    case 'edge-enhance': {
      // エッジ強調（元画像 + エッジ）
      const edgeForBlend = applyDoG(gray, width, height, 1.0, 2.0);
      output = blendEdge(gray, edgeForBlend, width, height, 0.7, 'subtract');
      bufferPool.releaseFloat32(edgeForBlend);
      break;
    }
      
    case 'sobel':
    case 'prewitt':
    case 'laplacian':
    default: {
      // 既存のフィルター（展開ループで高速化）
      output = bufferPool.getFloat32(size);
      
      if (type === 'laplacian') {
        // Laplacian（1パス）
        for (let y = 1; y < height - 1; y++) {
          const yOffset = y * width;
          for (let x = 1; x < width - 1; x++) {
            const center = gray[yOffset + x];
            const laplacian = 
              -gray[(y - 1) * width + x] 
              - gray[yOffset + (x - 1)]
              + 4 * center
              - gray[yOffset + (x + 1)]
              - gray[(y + 1) * width + x];
            output[yOffset + x] = Math.min(255, Math.abs(laplacian));
          }
        }
      } else {
        // Sobel/Prewitt（展開ループ）
        const isSobel = type === 'sobel';
        const w1 = isSobel ? 2 : 1; // 中心の重み
        
        for (let y = 1; y < height - 1; y++) {
          const yOffset = y * width;
          for (let x = 1; x < width - 1; x++) {
            const idx00 = (y - 1) * width + (x - 1);
            const idx01 = (y - 1) * width + x;
            const idx02 = (y - 1) * width + (x + 1);
            const idx10 = yOffset + (x - 1);
            const idx12 = yOffset + (x + 1);
            const idx20 = (y + 1) * width + (x - 1);
            const idx21 = (y + 1) * width + x;
            const idx22 = (y + 1) * width + (x + 1);
            
            const gx = -gray[idx00] + gray[idx02]
                     - w1 * gray[idx10] + w1 * gray[idx12]
                     - gray[idx20] + gray[idx22];
            
            const gy = -gray[idx00] - w1 * gray[idx01] - gray[idx02]
                     + gray[idx20] + w1 * gray[idx21] + gray[idx22];
            
            output[yOffset + x] = Math.min(255, Math.sqrt(gx * gx + gy * gy));
          }
        }
      }
      break;
    }
  }
  
  // グレースケールバッファを解放
  bufferPool.releaseFloat32(gray);
  
  // 結果を画像データに書き戻し
  for (let i = 0; i < size; i++) {
    const idx = i * 4;
    const value = clampToByte(output[i]);
    data[idx] = value;
    data[idx + 1] = value;
    data[idx + 2] = value;
    data[idx + 3] = 255;
  }
  
  // outputバッファを解放
  bufferPool.releaseFloat32(output);
};

/**
 * シャープ化フィルタ (3x3カーネル)
 * バッファプール使用、ループ展開で最適化
 */
const applySharpen = (data, width, height, strength = 0.5) => {
  const s = Math.max(0, Math.min(2, strength));
  if (s === 0) return;
  
  const center = 1 + 4 * s;
  const side = -s;
  const src = bufferPool.getUint8Clamped(data.length);
  src.set(data);
  
  const rowStride = width * 4;
  
  for (let y = 0; y < height; y++) {
    const yOffset = y * rowStride;
    const atTop = y === 0;
    const atBottom = y === height - 1;
    
    for (let x = 0; x < width; x++) {
      const idx = yOffset + x * 4;
      const atLeft = x === 0;
      const atRight = x === width - 1;
      
      // RGB各チャンネルを処理
      for (let c = 0; c < 3; c++) {
        let acc = src[idx + c] * center;
        
        if (!atLeft)   acc += src[idx - 4 + c] * side;
        if (!atRight)  acc += src[idx + 4 + c] * side;
        if (!atTop)    acc += src[idx - rowStride + c] * side;
        if (!atBottom) acc += src[idx + rowStride + c] * side;
        
        data[idx + c] = clampToByte(acc);
      }
      // アルファは維持
    }
  }
  
  bufferPool.releaseUint8Clamped(src);
};

/**
 * アンシャープマスク (Unsharp Mask) - 輪郭強調に最も効果的
 * LAZE SOFTWAREの「輪郭を強調する」と同等の処理
 * バッファプール使用で最適化
 * @param {Uint8ClampedArray} data - 画像データ
 * @param {number} width - 画像幅
 * @param {number} height - 画像高さ
 * @param {number} amount - 強度 (0.5-3.0推奨)
 * @param {number} radius - ぼかし半径 (1-3推奨)
 * @param {number} threshold - 閾値 (0-50推奨、ノイズ除去)
 */
const applyUnsharpMask = (data, width, height, amount = 1.5, radius = 1, threshold = 0) => {
  const size = width * height;
  
  // グレースケール配列を作成（バッファプール使用）
  const gray = bufferPool.getFloat32(size);
  for (let i = 0; i < size; i++) {
    const idx = i * 4;
    gray[i] = 0.299 * data[idx] + 0.587 * data[idx + 1] + 0.114 * data[idx + 2];
  }
  
  // ガウシアンブラーを適用
  const blurred = gaussianBlurGray(gray, width, height, radius);
  
  // アンシャープマスクを適用（インプレースで高速化）
  for (let i = 0; i < size; i++) {
    const diff = gray[i] - blurred[i];
    
    // 閾値チェック（小さな変化はスキップしてノイズを防ぐ）
    if (Math.abs(diff) < threshold) continue;
    
    const idx = i * 4;
    const enhancement = diff * amount;
    data[idx] = clampToByte(data[idx] + enhancement);
    data[idx + 1] = clampToByte(data[idx + 1] + enhancement);
    data[idx + 2] = clampToByte(data[idx + 2] + enhancement);
  }
  
  // バッファを解放
  bufferPool.releaseFloat32(gray);
  bufferPool.releaseFloat32(blurred);
};

/**
 * 輪郭強調フィルター（LAZE SOFTWARE互換）
 * 元画像とエッジを合成して輪郭を際立たせる
 * バッファプール使用で最適化
 * @param {Uint8ClampedArray} data - 画像データ
 * @param {number} width - 画像幅
 * @param {number} height - 画像高さ
 */
const applyContourEnhancement = (data, width, height) => {
  const size = width * height;
  
  // グレースケール配列を作成（バッファプール使用）
  const gray = bufferPool.getFloat32(size);
  for (let i = 0; i < size; i++) {
    const idx = i * 4;
    gray[i] = 0.299 * data[idx] + 0.587 * data[idx + 1] + 0.114 * data[idx + 2];
  }
  
  // DoGでエッジを検出
  const edge = applyDoG(gray, width, height, 0.8, 1.6);
  
  // グレースケールバッファを解放
  bufferPool.releaseFloat32(gray);
  
  // エッジを元画像に合成（減算モードで輪郭を黒く強調）
  for (let i = 0; i < size; i++) {
    const idx = i * 4;
    const factor = 1 - (edge[i] / 255) * 0.6;
    
    // 全チャンネルを一度に処理
    data[idx] = clampToByte(data[idx] * factor);
    data[idx + 1] = clampToByte(data[idx + 1] * factor);
    data[idx + 2] = clampToByte(data[idx + 2] * factor);
  }
  
  // エッジバッファを解放
  bufferPool.releaseFloat32(edge);
};

/**
 * Floyd-Steinberg ディザリング（モノクロ、境界クリップ付き）
 */
const applyMonochromeFSDithering = (data, width, height, threshold = 128) => {
  const distribute = (x, y, factor, err) => {
    if (x < 0 || x >= width || y < 0 || y >= height) return;
    const i = (y * width + x) * 4;
    for (let k = 0; k < 3; k++) {
      data[i + k] = clampToByte(data[i + k] + err * factor);
    }
  };
  
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = (y * width + x) * 4;
      const gray = computeGrayOnWhite(data, idx);
      const next = gray < threshold ? 0 : 255;
      const error = gray - next;
      
      data[idx] = data[idx + 1] = data[idx + 2] = next;
      data[idx + 3] = 255;
      
      // 誤差分配 (Floyd-Steinberg係数)
      distribute(x + 1, y,     7 / 16, error);
      distribute(x - 1, y + 1, 3 / 16, error);
      distribute(x,     y + 1, 5 / 16, error);
      distribute(x + 1, y + 1, 1 / 16, error);
    }
  }
};

/**
 * 画像を点字AAに変換
 */
const convertImage = (image, params) => {
  const { 
    cols, 
    threshold, 
    autoThreshold, 
    reverse, 
    dither, 
    contrast, 
    sharpen,
    gamma,
    noiseReduction,
    noiseStrength,
    adaptiveThreshold,
    adaptiveBlockSize,
    adaptiveC,
    edgeFilter,
    useDotForBlank,
    contourEnhance
  } = params;
  
  const targetCols = parseInt(cols, 10);
  let thresholdValue = parseFloat(threshold);
  const contrastValue = parseFloat(contrast);
  const sharpenValue = parseFloat(sharpen);
  const gammaValue = parseFloat(gamma) || 1.0;
  const noiseStrengthValue = parseFloat(noiseStrength) || 1;
  const blockSizeValue = parseInt(adaptiveBlockSize) || 11;
  const cValue = parseFloat(adaptiveC) || 2;
  
  // Canvas サイズを計算
  const canvasWidth = TENJI_COLS * targetCols;
  const canvasHeight = Math.round(image.height * (canvasWidth / image.width));
  
  // Canvas を作成して画像を描画
  const canvas = document.createElement('canvas');
  canvas.width = canvasWidth;
  canvas.height = canvasHeight;
  
  const ctx = canvas.getContext('2d');
  ctx.drawImage(image, 0, 0, canvasWidth, canvasHeight);
  
  // ピクセルデータを取得
  const imageData = ctx.getImageData(0, 0, canvasWidth, canvasHeight);
  const data = imageData.data;
  
  // 前処理1: ノイズ除去（最初に適用）
  if (noiseReduction && noiseReduction !== 'none') {
    if (noiseReduction === 'median') {
      applyMedianFilter(data, canvasWidth, canvasHeight, noiseStrengthValue);
    } else if (noiseReduction === 'gaussian') {
      applyGaussianBlur(data, canvasWidth, canvasHeight, noiseStrengthValue);
    }
  }
  
  // 前処理2: ガンマ補正
  applyGamma(data, gammaValue);
  
  // 前処理3: コントラスト調整
  applyContrast(data, contrastValue);
  
  // 前処理4: シャープ化
  applySharpen(data, canvasWidth, canvasHeight, sharpenValue);
  
  // 前処理5: 輪郭強調（LAZE SOFTWARE互換）
  if (contourEnhance) {
    applyContourEnhancement(data, canvasWidth, canvasHeight);
  }
  
  // 前処理6: エッジ検出フィルタ（オプション）
  if (edgeFilter && edgeFilter !== 'none') {
    applyEdgeDetection(data, canvasWidth, canvasHeight, edgeFilter);
  }
  
  // 二値化処理
  if (adaptiveThreshold) {
    // 適応的閾値を使用
    applyAdaptiveThreshold(data, canvasWidth, canvasHeight, blockSizeValue, cValue);
    // 適応的閾値使用時は中間値を閾値として扱う
    thresholdValue = 128;
  } else {
    // Otsu法で自動閾値を計算（有効な場合）
    if (autoThreshold) {
      thresholdValue = calculateOtsuThreshold(data, canvasWidth, canvasHeight);
    }
    
    // ディザリング（オプション）
    if (dither) {
      applyMonochromeFSDithering(data, canvasWidth, canvasHeight, thresholdValue);
    }
  }
  
  // 変更をCanvasに反映
  ctx.putImageData(imageData, 0, 0);
  
  // 点字AAを生成
  const lines = [];
  
  for (let blockY = 0; blockY < canvasHeight; blockY += TENJI_ROWS) {
    const lineChars = [];
    
    for (let blockX = 0; blockX < canvasWidth; blockX += TENJI_COLS) {
      let brailleBits = 0;
      
      for (let dy = 0; dy < TENJI_ROWS; dy++) {
        const y = blockY + dy;
        if (y >= canvasHeight) break;
        
        for (let dx = 0; dx < TENJI_COLS; dx++) {
          const x = blockX + dx;
          if (x >= canvasWidth) break;
          
          const idx = (y * canvasWidth + x) * 4;
          const gray = computeGrayOnWhite(data, idx);
          const below = gray < thresholdValue;
          const bitOn = reverse ? !below : below;
          
          if (bitOn) {
            let bit = 1 << dy;
            if (dx === 1) bit <<= 4;
            brailleBits += bit;
          }
        }
      }
      
      lineChars.push(numberToTenji(brailleBits, useDotForBlank));
    }
    
    lines.push(lineChars.join(''));
  }
  
  return { 
    canvas, 
    aa: lines.join('\n'),
    threshold: thresholdValue,
    cols: targetCols,
    rows: lines.length
  };
};

// ========================================
// Application State & Controller
// ========================================

class AAMakerApp {
  constructor() {
    this.currentImage = null;
    this.isProcessing = false;
    this.elements = {};
    this.debouncedProcess = debounce(() => this.processImage(), DEBOUNCE_DELAY);
  }
  
  /**
   * アプリケーション初期化
   */
  init() {
    this.cacheElements();
    this.bindEvents();
    this.initTheme();
    this.updateSliderLabels();
    this.checkTermsOfService();
  }
  
  /**
   * 利用規約の同意状態を確認
   */
  checkTermsOfService() {
    const accepted = localStorage.getItem(STORAGE_KEY_TOS_ACCEPTED);
    
    if (accepted !== TOS_VERSION) {
      // 未同意の場合はモーダルを表示
      this.showTermsOfServiceModal();
    }
  }
  
  /**
   * 利用規約モーダルを表示
   */
  showTermsOfServiceModal() {
    const modal = new bootstrap.Modal(this.elements.tosModal);
    modal.show();
  }
  
  /**
   * 利用規約への同意を記録
   */
  acceptTermsOfService() {
    localStorage.setItem(STORAGE_KEY_TOS_ACCEPTED, TOS_VERSION);
    const modal = bootstrap.Modal.getInstance(this.elements.tosModal);
    modal.hide();
    this.showToast('利用規約に同意しました。ご利用ありがとうございます！', 'success');
  }
  
  /**
   * 利用規約への同意を拒否（ページを閉じる案内）
   */
  declineTermsOfService() {
    // ページをブロック状態にする
    document.body.classList.add('app-blocked');
    
    // モーダルを閉じてメッセージを表示
    const modal = bootstrap.Modal.getInstance(this.elements.tosModal);
    modal.hide();
    
    // 警告メッセージ
    setTimeout(() => {
      alert('利用規約に同意いただけない場合、本サービスはご利用いただけません。\nページを閉じてください。');
      // 再度モーダルを表示
      document.body.classList.remove('app-blocked');
      this.showTermsOfServiceModal();
    }, 100);
  }
  
  /**
   * DOM要素をキャッシュ
   */
  cacheElements() {
    this.elements = {
      // Input
      imageInput: document.getElementById('imageInput'),
      dropZone: document.getElementById('dropZone'),
      
      // Preview
      imagePreview: document.getElementById('imagePreview'),
      ditherImagePreview: document.getElementById('ditherImagePreview'),
      originalPlaceholder: document.getElementById('originalPlaceholder'),
      processedPlaceholder: document.getElementById('processedPlaceholder'),
      
      // Parameters
      width: document.getElementById('aa-width'),
      threshold: document.getElementById('aa-threshold'),
      autoThreshold: document.getElementById('autoThreshold'),
      thresholdType: document.getElementById('aa-threshold-type'),
      contrast: document.getElementById('aa-contrast'),
      sharpen: document.getElementById('aa-sharpen'),
      gamma: document.getElementById('aa-gamma'),
      noiseReduction: document.getElementById('aa-noise'),
      noiseStrength: document.getElementById('aa-noise-strength'),
      edgeFilter: document.getElementById('aa-edge'),
      adaptiveBlockSize: document.getElementById('aa-adaptive-blocksize'),
      adaptiveC: document.getElementById('aa-adaptive-c'),
      reverse: document.getElementById('aa-reverse'),
      dither: document.getElementById('aa-dither'),
      useDotForBlank: document.getElementById('aa-dot-blank'),
      contourEnhance: document.getElementById('aa-contour'),
      
      // Conditional Groups
      noiseStrengthGroup: document.getElementById('noiseStrengthGroup'),
      binaryThresholdGroup: document.getElementById('binaryThresholdGroup'),
      adaptiveBlockSizeGroup: document.getElementById('adaptiveBlockSizeGroup'),
      adaptiveCGroup: document.getElementById('adaptiveCGroup'),
      
      // Advanced toggle
      toggleAdvanced: document.getElementById('toggleAdvanced'),
      advancedParams: document.getElementById('advancedParams'),
      
      // Labels
      widthValue: document.getElementById('widthValue'),
      thresholdValue: document.getElementById('thresholdValue'),
      contrastValue: document.getElementById('contrastValue'),
      sharpenValue: document.getElementById('sharpenValue'),
      gammaValue: document.getElementById('gammaValue'),
      noiseStrengthValue: document.getElementById('noiseStrengthValue'),
      adaptiveBlockSizeValue: document.getElementById('adaptiveBlockSizeValue'),
      adaptiveCValue: document.getElementById('adaptiveCValue'),
      
      // Output
      aaResult: document.getElementById('aa-result'),
      statCols: document.getElementById('statCols'),
      statRows: document.getElementById('statRows'),
      statTotal: document.getElementById('statTotal'),
      
      // Actions
      copyAA: document.getElementById('copyAA'),
      themeToggle: document.getElementById('themeToggle'),
      
      // Toast
      toast: document.getElementById('toast'),
      toastMessage: document.getElementById('toastMessage'),
      
      // Crop Modal
      cropModal: document.getElementById('cropModal'),
      cropContainer: document.getElementById('cropContainer'),
      cropImage: document.getElementById('cropImage'),
      cropSizeInfo: document.getElementById('cropSizeInfo'),
      cropOriginalSize: document.getElementById('cropOriginalSize'),
      cropReset: document.getElementById('cropReset'),
      cropRotateLeft: document.getElementById('cropRotateLeft'),
      cropRotateRight: document.getElementById('cropRotateRight'),
      cropFlipH: document.getElementById('cropFlipH'),
      cropFlipV: document.getElementById('cropFlipV'),
      cropConfirm: document.getElementById('cropConfirm'),
      cropSkip: document.getElementById('cropSkip'),
      
      // Terms of Service Modal
      tosModal: document.getElementById('tosModal'),
      tosAgreeCheck: document.getElementById('tosAgreeCheck'),
      tosAgree: document.getElementById('tosAgree'),
      tosDecline: document.getElementById('tosDecline')
    };
    
    // Cropper.js instance
    this.cropper = null;
    this.cropperOriginalImage = null;
  }
  
  /**
   * イベントリスナーを設定
   */
  bindEvents() {
    // File input
    this.elements.imageInput.addEventListener('change', (e) => this.handleFileSelect(e));
    
    // Drag & Drop
    this.elements.dropZone.addEventListener('click', () => this.elements.imageInput.click());
    this.elements.dropZone.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        this.elements.imageInput.click();
      }
    });
    this.elements.dropZone.addEventListener('dragover', (e) => this.handleDragOver(e));
    this.elements.dropZone.addEventListener('dragleave', (e) => this.handleDragLeave(e));
    this.elements.dropZone.addEventListener('drop', (e) => this.handleDrop(e));
    
    // Parameters - Range inputs (常にリアルタイムプレビュー)
    const rangeInputs = [
      this.elements.width,
      this.elements.threshold,
      this.elements.contrast,
      this.elements.sharpen,
      this.elements.gamma,
      this.elements.noiseStrength,
      this.elements.adaptiveBlockSize,
      this.elements.adaptiveC
    ];
    
    rangeInputs.forEach(input => {
      input.addEventListener('input', () => {
        this.updateSliderLabels();
        this.debouncedProcess();
      });
      input.addEventListener('change', () => this.processImage());
    });
    
    // Parameters - Select inputs
    const selectInputs = [
      this.elements.noiseReduction,
      this.elements.edgeFilter,
      this.elements.thresholdType
    ];
    
    selectInputs.forEach(input => {
      input.addEventListener('change', () => {
        this.updateConditionalUI();
        this.processImage();
      });
    });
    
    // Parameters - Checkboxes
    const checkboxInputs = [
      this.elements.autoThreshold,
      this.elements.reverse,
      this.elements.dither,
      this.elements.useDotForBlank,
      this.elements.contourEnhance
    ];
    
    checkboxInputs.forEach(input => {
      input.addEventListener('change', () => {
        this.updateThresholdState();
        this.processImage();
      });
    });
    
    // Copy button
    this.elements.copyAA.addEventListener('click', () => this.copyToClipboard());
    
    // Theme toggle
    this.elements.themeToggle.addEventListener('click', () => this.toggleTheme());
    
    // Advanced toggle
    this.elements.toggleAdvanced.addEventListener('click', () => this.toggleAdvancedParams());
    
    // Presets
    document.querySelectorAll('[data-preset]').forEach(btn => {
      btn.addEventListener('click', (e) => {
        const presetName = e.target.dataset.preset;
        this.applyPreset(presetName);
      });
    });
    
    document.getElementById('savePreset')?.addEventListener('click', () => this.saveCustomPreset());
    document.getElementById('loadPreset')?.addEventListener('click', () => this.loadCustomPreset());
    
    // Crop Modal Events
    this.bindCropEvents();
    
    // Terms of Service Events
    this.bindTosEvents();
    
    // 初期状態の条件付きUI更新
    this.updateConditionalUI();
  }
  
  /**
   * 利用規約モーダルのイベントを設定
   */
  bindTosEvents() {
    // チェックボックスの状態変更
    this.elements.tosAgreeCheck.addEventListener('change', (e) => {
      const isChecked = e.target.checked;
      this.elements.tosAgree.disabled = !isChecked;
      this.elements.tosDecline.disabled = isChecked;
    });
    
    // 同意ボタン
    this.elements.tosAgree.addEventListener('click', () => this.acceptTermsOfService());
    
    // 同意しないボタン
    this.elements.tosDecline.addEventListener('click', () => this.declineTermsOfService());
  }
  
  /**
   * クリッピングモーダルのイベントを設定 (Cropper.js版)
   */
  bindCropEvents() {
    // 回転ボタン
    this.elements.cropRotateLeft.addEventListener('click', () => {
      if (this.cropper) this.cropper.rotate(-90);
    });
    
    this.elements.cropRotateRight.addEventListener('click', () => {
      if (this.cropper) this.cropper.rotate(90);
    });
    
    // 反転ボタン
    this.elements.cropFlipH.addEventListener('click', () => {
      if (this.cropper) {
        const scaleX = this.cropper.getData().scaleX || 1;
        this.cropper.scaleX(-scaleX);
      }
    });
    
    this.elements.cropFlipV.addEventListener('click', () => {
      if (this.cropper) {
        const scaleY = this.cropper.getData().scaleY || 1;
        this.cropper.scaleY(-scaleY);
      }
    });
    
    // リセットボタン
    this.elements.cropReset.addEventListener('click', () => {
      if (this.cropper) this.cropper.reset();
    });
    
    // 確定ボタン（クリッピング適用）
    this.elements.cropConfirm.addEventListener('click', () => this.confirmCrop());
    
    // スキップボタン（クリッピングせずに使用）
    this.elements.cropSkip.addEventListener('click', () => this.skipCrop());
    
    // モーダルが閉じられたときの処理
    this.elements.cropModal.addEventListener('hidden.bs.modal', () => {
      this.destroyCropper();
    });
  }
  
  /**
   * クリッピングモーダルを表示 (Cropper.js版)
   */
  showCropModal(image) {
    this.cropperOriginalImage = image;
    
    // 画像をセット
    this.elements.cropImage.src = image.src;
    
    // 元画像サイズを表示
    this.elements.cropOriginalSize.textContent = `元画像: ${image.width} × ${image.height}px`;
    this.elements.cropSizeInfo.textContent = '選択範囲: --';
    
    // モーダルを表示
    const modal = new bootstrap.Modal(this.elements.cropModal);
    modal.show();
    
    // モーダルが表示された後にCropperを初期化
    this.elements.cropModal.addEventListener('shown.bs.modal', () => {
      this.initCropper();
    }, { once: true });
  }
  
  /**
   * Cropper.jsを初期化
   */
  initCropper() {
    // 既存のCropperがあれば破棄
    this.destroyCropper();
    
    const image = this.elements.cropImage;
    
    this.cropper = new Cropper(image, {
      viewMode: 1,
      dragMode: 'crop',
      aspectRatio: NaN, // 自由なアスペクト比
      autoCropArea: 0.8,
      restore: false,
      guides: true,
      center: true,
      highlight: true,
      cropBoxMovable: true,
      cropBoxResizable: true,
      toggleDragModeOnDblclick: true,
      
      // クリッピング範囲変更時のコールバック
      crop: (event) => {
        const data = event.detail;
        const width = Math.round(data.width);
        const height = Math.round(data.height);
        this.elements.cropSizeInfo.textContent = `選択範囲: ${width} × ${height}px`;
      }
    });
  }
  
  /**
   * Cropperを破棄
   */
  destroyCropper() {
    if (this.cropper) {
      this.cropper.destroy();
      this.cropper = null;
    }
  }
  
  /**
   * クリッピングを確定 (Cropper.js版)
   */
  confirmCrop() {
    if (!this.cropper) return;
    
    // クリッピングされた画像をCanvasとして取得
    const croppedCanvas = this.cropper.getCroppedCanvas({
      imageSmoothingEnabled: true,
      imageSmoothingQuality: 'high'
    });
    
    if (!croppedCanvas) {
      this.showToast('クリッピングに失敗しました', 'danger');
      return;
    }
    
    // CanvasからImageを作成
    const croppedImage = new Image();
    croppedImage.onload = () => {
      // モーダルを閉じる
      const modal = bootstrap.Modal.getInstance(this.elements.cropModal);
      modal.hide();
      
      // 画像を適用
      this.applyImage(croppedImage);
    };
    croppedImage.src = croppedCanvas.toDataURL();
  }
  
  /**
   * クリッピングせずに画像を使用
   */
  skipCrop() {
    // モーダルを閉じる
    const modal = bootstrap.Modal.getInstance(this.elements.cropModal);
    modal.hide();
    
    // 元の画像をそのまま適用
    if (this.cropperOriginalImage) {
      this.applyImage(this.cropperOriginalImage);
    }
  }
  
  /**
   * 画像を適用（プレビューと処理）
   */
  applyImage(image) {
    this.currentImage = image;
    
    // 元画像プレビューを表示
    const previewCanvas = document.createElement('canvas');
    previewCanvas.width = image.width;
    previewCanvas.height = image.height;
    const ctx = previewCanvas.getContext('2d');
    ctx.drawImage(image, 0, 0);
    
    this.elements.imagePreview.src = previewCanvas.toDataURL();
    this.elements.imagePreview.classList.add('visible');
    this.elements.originalPlaceholder.classList.add('hidden');
    
    // 処理を実行
    this.processImage();
  }
  
  /**
   * 詳細設定の表示/非表示を切り替え
   */
  toggleAdvancedParams() {
    const btn = this.elements.toggleAdvanced;
    const panel = this.elements.advancedParams;
    const isExpanded = btn.dataset.expanded === 'true';
    
    if (isExpanded) {
      panel.style.display = 'none';
      btn.innerHTML = '<i class="bi bi-chevron-down"></i> 詳細設定を表示';
      btn.dataset.expanded = 'false';
    } else {
      panel.style.display = '';
      btn.innerHTML = '<i class="bi bi-chevron-up"></i> 詳細設定を非表示';
      btn.dataset.expanded = 'true';
    }
  }
  
  /**
   * 条件付きUIの表示/非表示を更新
   */
  updateConditionalUI() {
    // ノイズ除去強度の表示制御
    const noiseType = this.elements.noiseReduction.value;
    this.elements.noiseStrengthGroup.style.display = noiseType !== 'none' ? '' : 'none';
    
    // 閾値タイプによる表示制御
    const thresholdType = this.elements.thresholdType.value;
    const isBinary = thresholdType === 'binary';
    
    this.elements.binaryThresholdGroup.style.display = isBinary ? '' : 'none';
    this.elements.adaptiveBlockSizeGroup.style.display = isBinary ? 'none' : '';
    this.elements.adaptiveCGroup.style.display = isBinary ? 'none' : '';
  }
  
  /**
   * ファイル選択ハンドラ
   */
  handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
      this.loadImage(file);
    }
    // 同じファイルを再選択できるようにリセット
    e.target.value = '';
  }
  
  /**
   * ドラッグオーバーハンドラ
   */
  handleDragOver(e) {
    e.preventDefault();
    e.stopPropagation();
    this.elements.dropZone.classList.add('drag-over');
  }
  
  /**
   * ドラッグリーブハンドラ
   */
  handleDragLeave(e) {
    e.preventDefault();
    e.stopPropagation();
    this.elements.dropZone.classList.remove('drag-over');
  }
  
  /**
   * ドロップハンドラ
   */
  handleDrop(e) {
    e.preventDefault();
    e.stopPropagation();
    this.elements.dropZone.classList.remove('drag-over');
    
    const files = e.dataTransfer.files;
    if (files.length > 0 && files[0].type.startsWith('image/')) {
      this.loadImage(files[0]);
    }
  }
  
  /**
   * 画像を読み込み
   */
  loadImage(file) {
    if (!file || !file.type.startsWith('image/')) {
      this.showToast('画像ファイルを選択してください', 'warning');
      return;
    }
    
    const reader = new FileReader();
    
    reader.onload = (e) => {
      const img = new Image();
      img.onload = () => {
        // クリッピングモーダルを表示
        this.showCropModal(img);
      };
      img.src = e.target.result;
    };
    
    reader.readAsDataURL(file);
  }
  
  /**
   * 画像を処理してAAを生成
   */
  processImage() {
    if (!this.currentImage || this.isProcessing) return;
    
    this.isProcessing = true;
    
    const params = this.getParams();
    
    try {
      const result = convertImage(this.currentImage, params);
      
      // 処理後プレビューを更新
      this.elements.ditherImagePreview.src = result.canvas.toDataURL();
      this.elements.ditherImagePreview.classList.add('visible');
      this.elements.processedPlaceholder.classList.add('hidden');
      
      // AA結果を更新
      this.elements.aaResult.value = result.aa;
      
      // 統計を更新
      this.updateStats(result);
      
      // 自動閾値の場合、UIに反映
      if (params.autoThreshold) {
        this.elements.threshold.value = result.threshold;
        this.elements.thresholdValue.textContent = Math.round(result.threshold);
      }
      
    } catch (error) {
      console.error('Processing error:', error);
      this.showToast('処理中にエラーが発生しました', 'danger');
    }
    
    this.isProcessing = false;
  }
  
  /**
   * 現在のパラメータを取得
   */
  getParams() {
    const thresholdType = this.elements.thresholdType.value;
    return {
      cols: this.elements.width.value,
      threshold: this.elements.threshold.value,
      autoThreshold: this.elements.autoThreshold.checked,
      contrast: this.elements.contrast.value,
      sharpen: this.elements.sharpen.value,
      gamma: this.elements.gamma.value,
      noiseReduction: this.elements.noiseReduction.value,
      noiseStrength: this.elements.noiseStrength.value,
      adaptiveThreshold: thresholdType === 'adaptive',
      adaptiveBlockSize: this.elements.adaptiveBlockSize.value,
      adaptiveC: this.elements.adaptiveC.value,
      edgeFilter: this.elements.edgeFilter.value,
      reverse: this.elements.reverse.checked,
      dither: this.elements.dither.checked,
      useDotForBlank: this.elements.useDotForBlank.checked,
      contourEnhance: this.elements.contourEnhance.checked
    };
  }
  
  /**
   * スライダーラベルを更新
   */
  updateSliderLabels() {
    this.elements.widthValue.textContent = this.elements.width.value;
    this.elements.thresholdValue.textContent = this.elements.threshold.value;
    this.elements.contrastValue.textContent = this.elements.contrast.value;
    this.elements.sharpenValue.textContent = this.elements.sharpen.value;
    this.elements.gammaValue.textContent = this.elements.gamma.value;
    this.elements.noiseStrengthValue.textContent = this.elements.noiseStrength.value;
    this.elements.adaptiveBlockSizeValue.textContent = this.elements.adaptiveBlockSize.value;
    this.elements.adaptiveCValue.textContent = this.elements.adaptiveC.value;
  }
  
  /**
   * 閾値スライダーの有効/無効を更新
   */
  updateThresholdState() {
    const isAuto = this.elements.autoThreshold.checked;
    this.elements.threshold.disabled = isAuto;
    this.elements.threshold.style.opacity = isAuto ? '0.5' : '1';
  }
  
  /**
   * 統計情報を更新
   */
  updateStats(result) {
    this.elements.statCols.textContent = result.cols;
    this.elements.statRows.textContent = result.rows;
    this.elements.statTotal.textContent = result.cols * result.rows;
  }
  
  /**
   * クリップボードにコピー
   */
  async copyToClipboard() {
    const text = this.elements.aaResult.value;
    if (!text) {
      this.showToast('コピーするAAがありません', 'warning');
      return;
    }
    
    try {
      if (navigator.clipboard) {
        await navigator.clipboard.writeText(text);
      } else {
        // フォールバック
        this.elements.aaResult.select();
        document.execCommand('copy');
      }
      this.showToast('クリップボードにコピーしました！', 'success');
    } catch (error) {
      console.error('Copy error:', error);
      this.showToast('コピーに失敗しました', 'danger');
    }
  }
  
  /**
   * トースト通知を表示
   */
  showToast(message, type = 'success') {
    const toast = this.elements.toast;
    const toastMessage = this.elements.toastMessage;
    
    // スタイルを設定
    toast.classList.remove('text-bg-success', 'text-bg-warning', 'text-bg-danger');
    toast.classList.add(`text-bg-${type}`);
    
    toastMessage.textContent = message;
    
    const bsToast = new bootstrap.Toast(toast, { delay: 2000 });
    bsToast.show();
  }
  
  // ========================================
  // Theme Management
  // ========================================
  
  /**
   * テーマを初期化
   */
  initTheme() {
    const saved = localStorage.getItem(STORAGE_KEY_THEME);
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    const theme = saved || (prefersDark ? 'dark' : 'light');
    
    this.setTheme(theme);
  }
  
  /**
   * テーマを設定
   */
  setTheme(theme) {
    document.body.classList.remove('theme-light', 'theme-dark');
    document.body.classList.add(`theme-${theme}`);
    
    const icon = this.elements.themeToggle.querySelector('i');
    if (icon) {
      icon.className = theme === 'dark' ? 'bi bi-moon-fill' : 'bi bi-sun-fill';
    }
    
    localStorage.setItem(STORAGE_KEY_THEME, theme);
  }
  
  /**
   * テーマを切り替え
   */
  toggleTheme() {
    const isDark = document.body.classList.contains('theme-dark');
    this.setTheme(isDark ? 'light' : 'dark');
  }
  
  // ========================================
  // Preset Management
  // ========================================
  
  /**
   * プリセットを適用
   */
  applyPreset(presetName) {
    const preset = PRESETS[presetName];
    if (!preset) return;
    
    this.elements.width.value = preset.width;
    this.elements.threshold.value = preset.threshold;
    this.elements.autoThreshold.checked = preset.autoThreshold;
    this.elements.contrast.value = preset.contrast;
    this.elements.sharpen.value = preset.sharpen;
    this.elements.gamma.value = preset.gamma || 1.0;
    this.elements.noiseReduction.value = preset.noiseReduction || 'none';
    this.elements.noiseStrength.value = preset.noiseStrength || 1;
    this.elements.thresholdType.value = preset.adaptiveThreshold ? 'adaptive' : 'binary';
    this.elements.adaptiveBlockSize.value = preset.adaptiveBlockSize || 11;
    this.elements.adaptiveC.value = preset.adaptiveC || 2;
    this.elements.edgeFilter.value = preset.edgeFilter || 'none';
    this.elements.reverse.checked = preset.reverse;
    this.elements.dither.checked = preset.dither;
    this.elements.useDotForBlank.checked = preset.useDotForBlank ?? true;
    this.elements.contourEnhance.checked = preset.contourEnhance ?? false;
    
    this.updateSliderLabels();
    this.updateThresholdState();
    this.updateConditionalUI();
    this.processImage();
    
    this.showToast(`プリセット「${preset.name}」を適用しました`, 'success');
  }
  
  /**
   * カスタムプリセットを保存
   */
  saveCustomPreset() {
    const params = this.getParams();
    localStorage.setItem(STORAGE_KEY_PRESET, JSON.stringify(params));
    this.showToast('現在の設定を保存しました', 'success');
  }
  
  /**
   * カスタムプリセットを読み込み
   */
  loadCustomPreset() {
    const saved = localStorage.getItem(STORAGE_KEY_PRESET);
    if (!saved) {
      this.showToast('保存された設定がありません', 'warning');
      return;
    }
    
    try {
      const params = JSON.parse(saved);
      
      this.elements.width.value = params.cols || 30;
      this.elements.threshold.value = params.threshold || 128;
      this.elements.autoThreshold.checked = params.autoThreshold ?? true;
      this.elements.contrast.value = params.contrast || 0;
      this.elements.sharpen.value = params.sharpen || 0.5;
      this.elements.gamma.value = params.gamma || 1.0;
      this.elements.noiseReduction.value = params.noiseReduction || 'none';
      this.elements.noiseStrength.value = params.noiseStrength || 1;
      this.elements.thresholdType.value = params.adaptiveThreshold ? 'adaptive' : 'binary';
      this.elements.adaptiveBlockSize.value = params.adaptiveBlockSize || 11;
      this.elements.adaptiveC.value = params.adaptiveC || 2;
      this.elements.edgeFilter.value = params.edgeFilter || 'none';
      this.elements.reverse.checked = params.reverse ?? true;
      this.elements.dither.checked = params.dither ?? false;
      this.elements.useDotForBlank.checked = params.useDotForBlank ?? true;
      this.elements.contourEnhance.checked = params.contourEnhance ?? false;
      
      this.updateSliderLabels();
      this.updateThresholdState();
      this.updateConditionalUI();
      this.processImage();
      
      this.showToast('保存した設定を読み込みました', 'success');
    } catch (error) {
      console.error('Load preset error:', error);
      this.showToast('設定の読み込みに失敗しました', 'danger');
    }
  }
}

// ========================================
// Initialize Application
// ========================================
document.addEventListener('DOMContentLoaded', () => {
  const app = new AAMakerApp();
  app.init();
});
