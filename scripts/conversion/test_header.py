import tflite.Model as tflite_model

with open("fixed_input_float32.tflite", "rb") as f:
    buf = f.read()

model = tflite_model.Model.GetRootAsModel(buf, 0)
subgraph = model.Subgraphs(0)

print("Inputs:")
for i in range(subgraph.InputsLength()):
    print(" ", subgraph.Tensors(subgraph.Inputs(i)).Name())

print("Outputs:")
for i in range(subgraph.OutputsLength()):
    print(" ", subgraph.Outputs(i))
