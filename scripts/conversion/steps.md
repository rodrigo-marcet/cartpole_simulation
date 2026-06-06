1. train
2. (with isaaclab venv) python
.\scripts\conversion\pt_to_onnx.py -i D:\omniverse\pendulum\outputs\eridani\weights\skrl_pendulum_v0_054\skrl_pendulum_v0_054\skrl\pendulum_direct\2026-06-02_16-57-58_ppo_torch\checkpoints\best_agent.pt -o .\scripts\conversion\outputs\onnx\smooth_action.onnx
3. new venv
    2.1 pip install numpy==1.26.4
    2.2 pip install torch --index-url https://download.pytorch.org/whl/cpu
    2.3 pip install onnxruntime==1.18.0

(from now on we use new venv)
4. venv/Scripts/activate
6. pip install onnx2tf tensorflow tf_keras onnx onnx_graphsurgeon psutil ai_edge_litert sng4onnx
7. onnx2tf -i .\scripts\conversion\outputs\onnx\smooth_action.onnx -o .\scripts\conversion\outputs\policy_tflite\smooth_action
8. python .\scripts\conversion\test_tflite.py
9. ..\venv\Scripts\activate
10. python .\scripts\conversion\tflite_to_header.py -i .\scripts\conversion\outputs\policy_tflite\smooth_action\smooth_action_float32.tflite -o .\scripts\conversion\outputs\headers\smooth_action.h -c .\outputs\eridani\weights\skrl_pendulum_v0_054\skrl_pendulum_v0_054\skrl\pendulum_direct\2026-06-02_16-57-58_ppo_torch\checkpoints\best_agent.pt
