import onnx
from max.graph import Graph

MODEL_PATH = "models/clip/model_static.onnx"

print("Reading CLIP ONNX model...")

onnx_model = onnx.load(MODEL_PATH)

print("ONNX model loaded!")
print("Inputs:", [x.name for x in onnx_model.graph.input])
print("Outputs:", [x.name for x in onnx_model.graph.output])
print("Nodes:", len(onnx_model.graph.node))