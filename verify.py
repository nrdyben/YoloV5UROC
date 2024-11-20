import torch
import torch.onnx
import numpy as np
import tensorflow as tf
import os
import sys
from pathlib import Path
from models.experimental import attempt_load

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

# Path to the PyTorch model and TFLite model
torch_model_path = ROOT / 'best.pt'
tflite_model_path = ROOT / 'best-fp16_arl.tflite'

# Load your PyTorch model
model = attempt_load(torch_model_path, device=None, inplace=True, fuse=True)


# Prepare input data for PyTorch
input_data_pytorch = torch.rand(1, 3, 640, 640).float()  # Adjust the shape based on your model's input requirements

# Perform inference with PyTorch
with torch.no_grad():
    output_pytorch = model(input_data_pytorch)

# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path=str(tflite_model_path))  # Convert PosixPath to string
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Prepare input data for TensorFlow Lite
# input_data_tflite = input_data_pytorch

input_data_tflite = np.random.rand(1, 640, 640, 3).astype(np.float32)  # Adjust the shape based on your model's input requirements

# Set input tensor for TensorFlow Lite
interpreter.set_tensor(input_details[0]['index'], input_data_tflite)

# Run inference with TensorFlow Lite
interpreter.invoke()

# Get the output tensor for TensorFlow Lite
output_tflite = interpreter.get_tensor(output_details[0]['index'])

# If there's only one output tensor, you may need to access it as output_tflite[0]
# if len(output_tflite.shape) == 4 and output_tflite.shape[0] == 1:
#     output_tflite = output_tflite[0]

# print(output_tflite)
# print(output_pytorch)
# Compare outputs
tolerance = 1e-5
print(output_pytorch)
# print( output_tflite[0])
is_close = np.allclose(np.array(output_pytorch), output_tflite[0], rtol=tolerance, atol=tolerance)

if is_close:
    print("The outputs match within the specified tolerance.")
else:
    print("The outputs do not match.")