import path from 'path';
import sharp from 'sharp';

// Use WASM build to avoid native @tensorflow/tfjs-node dependency
// eslint-disable-next-line @typescript-eslint/no-require-imports
const faceapi = require('@vladmandic/face-api/dist/face-api.node-wasm') as typeof import('@vladmandic/face-api');

const MODEL_PATH = path.join(
  path.dirname(require.resolve('@vladmandic/face-api/package.json')),
  'model',
);

let modelsLoaded = false;
let loadPromise: Promise<void> | null = null;

async function ensureModels(): Promise<void> {
  if (modelsLoaded) return;
  if (loadPromise) { await loadPromise; return; }

  loadPromise = (async () => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const tf = faceapi.tf as any;
    await tf.setBackend('wasm');
    await tf.ready();
    await faceapi.nets.tinyFaceDetector.loadFromDisk(MODEL_PATH);
    modelsLoaded = true;
  })();

  await loadPromise;
}

export interface FaceBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export async function detectLargestFace(imageBuffer: Buffer): Promise<FaceBox | null> {
  await ensureModels();

  const { data, info } = await sharp(imageBuffer)
    .removeAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });

  // Create a 3D tensor [height, width, channels] from raw pixel data
  const tensor = faceapi.tf.tensor3d(
    new Uint8Array(data),
    [info.height, info.width, info.channels],
  );

  try {
    const detections = await faceapi.detectAllFaces(
      tensor as Parameters<typeof faceapi.detectAllFaces>[0],
      new faceapi.TinyFaceDetectorOptions({ inputSize: 608, scoreThreshold: 0.1 }),
    );

    if (!detections.length) return null;

    const largest = detections.reduce((a, b) =>
      b.box.width * b.box.height > a.box.width * a.box.height ? b : a,
    );

    return {
      x: largest.box.x,
      y: largest.box.y,
      width: largest.box.width,
      height: largest.box.height,
    };
  } finally {
    tensor.dispose();
  }
}
