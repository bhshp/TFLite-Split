import numpy as np
import tensorflow as tf
from PIL import Image

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="../build/mobilenet_v2/mobilenet_v2_0_0.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']

# Load and preprocess the image
img = Image.open('../data/coffee.png').convert('RGB')
img = img.resize((input_shape[1], input_shape[2]))
img_array = np.array(img, dtype=np.float32) / 255.0
img_tensor = np.expand_dims(img_array, axis=0)

# Set the input tensor for the model
interpreter.set_tensor(input_details[0]['index'], img_tensor)

# Run inference on the model
interpreter.invoke()
model_output = interpreter.get_tensor(output_details[0]['index'])

# Print the output
print(model_output)
