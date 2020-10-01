
import tensorflow as tf
import numpy as np
import os
import keras

from .image_utils import load_images

class LiteClassifier:
    def __init__(self):
        self.interpreter = tf.lite.Interpreter(model_path="converted_model.tflite")
        self.interpreter.allocate_tensors()

    def classify(self, image_paths, size=(256,256)):
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()

        loaded_images, _ = load_images(image_paths, size, image_paths)

        result = []
        for img in loaded_images:
            img = np.expand_dims(img, axis=0)
            input_data = np.array(img, dtype=np.float32)
            self.interpreter.set_tensor(input_details[0]['index'], input_data)

            self.interpreter.invoke()

            # The function `get_tensor()` returns a copy of the tensor data.
            # Use `tensor()` in order to get a pointer to the tensor.
            output_data = self.interpreter.get_tensor(output_details[0]['index'])
            result.append({"unsafe": output_data[0][0], "safe": output_data[0][1]})
        return result
