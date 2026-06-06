import numpy as np
import onnxruntime as ort
import tensorflow as tf

dummy = np.random.randn(1, 5).astype(np.float32)

# onnx
sess = ort.InferenceSession("./scripts/conversion/outputs/onnx/gaussian_pole.onnx")
onnx_out = sess.run(None, {"obs": dummy})[0]

# tflite
interp = tf.lite.Interpreter(
    model_path="./scripts/conversion/outputs/policy_tflite/gaussian_pole/gaussian_pole_float32.tflite"
)
interp.allocate_tensors()
inp = interp.get_input_details()
out = interp.get_output_details()
interp.set_tensor(inp[0]["index"], dummy)
interp.invoke()
tflite_out = interp.get_tensor(out[0]["index"])

print(f"ONNX  : {onnx_out}")
print(f"TFLite: {tflite_out}")
print(f"Max diff: {np.abs(onnx_out - tflite_out).max():.2e}")
