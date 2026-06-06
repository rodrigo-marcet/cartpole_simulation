# Policy Conversion Pipeline

Gets a trained SKRL policy from Isaac Lab onto the ESP32.

> **WARNING:** Both the conversion scripts and this document were created using AI. They do work, but just to keep you informed.

```
best_agent.pt  →  policy.onnx  →  policy_float32.tflite  →  policy.h
```

## Step 1 — PT → ONNX

Uses the Isaac Lab venv (needs PyTorch with the full checkpoint format).

```bash
# activate Isaac Lab venv first
python scripts\conversion\pt_to_onnx.py \
  -i <path\to\best_agent.pt> \
  -o <path\to\.onnx\file>
```

---

## Step 2 — ONNX → TFLite

Uses the conversion venv.

```bash
venv_conversion\Scripts\activate

onnx2tf \
  -i <path\to\.onnx\file> \
  -o <path\to\tflite\directory>
```

Output will be at `scripts\conversion\outputs\policy_tflite\<name>\<name>_float32.tflite`.

---

## Step 3 — Verify the conversion

Runs both models on the same inputs and checks they agree.

```bash
# still in conversion venv
python scripts\conversion\test_conversion.py \
  -i <path\to\best_agent.pt> \
  -c <path\to\.tflite\file>
```

Expects max diff < 1e-3 across 1000 random inputs. If it fails, the conversion went wrong — don't flash this to the ESP32.

---

## Step 4 — TFLite → C Header

Embeds the model weights and the obs normalizer (running mean/variance) into a single `.h` file. Switch back to the Isaac Lab venv for this since it reads the checkpoint.

```bash
# activate Isaac Lab venv
python scripts\conversion\tflite_to_header.py \
  -i <path\to\.tflite\file> \
  -o <path\to\.h\file> \
  -c <path\to\best_agent.pt>
```

Drop the resulting `.h` into the ESP32 firmware and update the `#include`.

---

## Full example

```bash
set PT=outputs\eridani\weights\skrl_pendulum_v0_054\...\checkpoints\best_agent.pt
set NAME=smooth_action

# Step 1 (Isaac Lab venv)
python scripts\conversion\pt_to_onnx.py -i %PT% -o scripts\conversion\outputs\onnx\%NAME%.onnx

# Steps 2-3 (conversion venv)
venv_conversion\Scripts\activate
onnx2tf -i scripts\conversion\outputs\onnx\%NAME%.onnx -o scripts\conversion\outputs\policy_tflite\%NAME%
python scripts\conversion\test_conversion.py -i %PT% -c scripts\conversion\outputs\policy_tflite\%NAME%\%NAME%_float32.tflite

# Step 4 (Isaac Lab venv)
..\venv\Scripts\activate
python scripts\conversion\tflite_to_header.py ^
  -i scripts\conversion\outputs\policy_tflite\%NAME%\%NAME%_float32.tflite ^
  -o scripts\conversion\outputs\headers\%NAME%.h ^
  -c %PT%
```

---

## Notes

- **Two venvs are required.** `onnx2tf` and Isaac Lab have conflicting TensorFlow/PyTorch dependencies — they cannot share an environment.
- **The `.h` file includes the obs normalizer.** The ESP32 firmware must apply `(obs - mean) / sqrt(var + 1e-8)` before running inference. The arrays `input_mean` and `input_var` are in the header.
- **Architecture is hardcoded** to `5 → 32 → 32 → 1` with ELU activations and a Tanh output. If you change the network, update `pt_to_onnx.py` and `test_conversion.py` together.
