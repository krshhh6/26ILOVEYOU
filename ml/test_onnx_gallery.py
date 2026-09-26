"""Test ONNX model inference on all demo gallery images."""
import onnxruntime as ort
import numpy as np
from PIL import Image
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
model_path = ROOT_DIR / 'frontend' / 'public' / 'models' / 'oil_classifier.onnx'
demo_dir = ROOT_DIR / 'frontend' / 'public' / 'demo-sar'

sess = ort.InferenceSession(str(model_path))
print(f'ONNX model loaded: {sess.get_inputs()[0].name} shape={sess.get_inputs()[0].shape}')

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))

threshold = 0.27

print(f'\nThreshold: {threshold}')
print('=' * 60)

correct = 0
total = 0

print('\n--- CLASS 1 (Oil Spill) ---')
for n in range(1, 11):
    p = demo_dir / f'class_1_{n}.jpg'
    if not p.exists():
        continue
    img = Image.open(p).convert('L').resize((400, 400))
    arr = np.array(img, dtype=np.float32) / 255.0
    inp = np.stack([arr, arr * 0.85], axis=0)[np.newaxis, ...]
    logit = sess.run(None, {'input': inp.astype(np.float32)})[0][0][0]
    prob = float(sigmoid(logit))
    decision = 'OIL' if prob >= threshold else 'CLEAN'
    ok = decision == 'OIL'
    correct += int(ok)
    total += 1
    status = 'OK' if ok else 'WRONG'
    print(f'  class_1_{n:02d}.jpg: P={prob:.4f} -> {decision} [{status}]')

print('\n--- CLASS 0 (Clean Ocean) ---')
for n in range(1, 11):
    p = demo_dir / f'class_0_{n}.jpg'
    if not p.exists():
        continue
    img = Image.open(p).convert('L').resize((400, 400))
    arr = np.array(img, dtype=np.float32) / 255.0
    inp = np.stack([arr, arr * 0.85], axis=0)[np.newaxis, ...]
    logit = sess.run(None, {'input': inp.astype(np.float32)})[0][0][0]
    prob = float(sigmoid(logit))
    decision = 'OIL' if prob >= threshold else 'CLEAN'
    ok = decision == 'CLEAN'
    correct += int(ok)
    total += 1
    status = 'OK' if ok else 'WRONG'
    print(f'  class_0_{n:02d}.jpg: P={prob:.4f} -> {decision} [{status}]')

print(f'\n{"="*60}')
print(f'ONNX Accuracy: {correct}/{total} ({100*correct/total:.0f}%)')
print(f'{"="*60}')
