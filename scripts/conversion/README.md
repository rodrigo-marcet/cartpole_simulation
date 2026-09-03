python .\scripts\conversion\pt_to_onnx.py -i D:\omniverse\pendulum\logs\skrl\double_pendulum_direct\finally\checkpoints\best_agent.pt -o .\scripts\conversion\outputs\double\onnx/finally.onnx --type double

.\venv\Scripts\activate

python -m onnx2tf -i  .\scripts\conversion\outputs\double\onnx\finally.onnx -o .\scripts\conversion\outputs\double\policy_tflite\finally

python .\scripts\conversion\test_conversion.py -c .\scripts\conversion\outputs\double\policy_tflite\finally/finally_float32.tflite -i .\logs\skrl\double_pendulum_direct\finally\checkpoints\best_agent.pt --type double

python .\scripts\conversion\tflite_to_header.py -i .\scripts\conversion\outputs\double\policy_tflite\finally/finally_float32.tflite -c .\logs\skrl\double_pendulum_direct\finally\checkpoints\best_agent.pt -o .\scripts\conversion\outputs\double\headers\finally.h





python .\scripts\conversion\pt_to_onnx.py -i D:\omniverse\pendulum\logs\skrl\double_pendulum_direct\delayed_action_st_30\checkpoints\best_agent.pt -o .\scripts\conversion\outputs\double\onnx/delayed_action_st_30.onnx --type double

.\venv\Scripts\activate

python -m onnx2tf -i  .\scripts\conversion\outputs\double\onnx\delayed_action_st_30.onnx -o .\scripts\conversion\outputs\double\policy_tflite\delayed_action_st_30

python .\scripts\conversion\test_conversion.py -c .\scripts\conversion\outputs\double\policy_tflite\delayed_action_st_30/delayed_action_st_30_float32.tflite -i .\logs\skrl\double_pendulum_direct\delayed_action_st_30\checkpoints\best_agent.pt --type double

python scripts/conversion/tflite_to_header.py -i .\scripts\conversion\outputs\double\policy_tflite\delayed_action_st_30/delayed_action_st_30_float32.tflite -o .\scripts\conversion\outputs\double\headers\delayed_action_st_30.h -c .\logs\skrl\double_pendulum_direct\delayed_action_st_30\checkpoints\best_agent.pt --obs-scales "2.5,0.4,1,1,1/15,1,1,1/15"


python scripts/simulations/double_cartpole/basin.py --task Swingup-DoubleCartpole-v0 --mode volume --num_envs 4096 --samples 4096 --checkpoint "D:\omniverse\pendulum\logs\skrl\double_pendulum_direct\balance_transfer_30\checkpoints\best_agent.pt" --headless
