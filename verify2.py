import numpy as np
import tensorflow as tf
import torch
from models.experimental import attempt_load

np_input = np.random.randint(256, size=(1, 3, 640, 640))
np_input_tflite = np_input.reshape((1, 640, 640, 3))

print(np_input.dtype)

TFLITE_FILE_PATH = 'best-fp16_arl.tflite'
TORCH_FILE_PATH = 'best.pt'

# Load the TFLite model in TFLite Interpreter
interpreter = tf.lite.Interpreter(TFLITE_FILE_PATH)

#allocate the tensors
interpreter.allocate_tensors()

#get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


tflite_input_type = input_details[0]['dtype']

tflite_input = np_input_tflite.astype(tflite_input_type)

# Create input tensor out of raw features
interpreter.set_tensor(input_details[0]['index'], tflite_input)

# Run inference
interpreter.invoke()

tflite_output = []
# output_details[i]['index'] = the index which provides the input
tflite_output.append(interpreter.get_tensor(output_details[0]['index']))

print("Inference output 1:", tflite_output[0])
print(tflite_output[0].shape)
print(type(tflite_output[0]))


# ============= Load your PyTorch model
model = attempt_load(TORCH_FILE_PATH, device=None, inplace=True, fuse=True)


# Perform inference with PyTorch
torch_input = torch.from_numpy(np_input)

# Prepare input data for PyTorch
input_data_pytorch = torch_input.float()  # Adjust the shape based on your model's input requirements

# Perform inference with PyTorch
with torch.no_grad():
    torch_output = model(input_data_pytorch)


torch_output_0 = torch_output[0].detach().numpy()
print("Torch Inference output 1:", torch_output_0)
print(torch_output_0.shape)
print(type(torch_output_0))


output_dif_0 = torch_output_0 - tflite_output[0]
print('Maximum difference 0:',max(output_dif_0[0]))
print('Maximum difference ratio 0:',max(output_dif_0[0])/np.mean(torch_output_0[0]))
