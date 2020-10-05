import os
import pydload
import numpy as np
import tensorflow as tf

from .image_utils import load_images

class LiteClassifier:
    def __init__(self):
        url = "https://github.com/notAI-tech/NudeNet/releases/download/v0/classifier.tflite"
        home = os.path.expanduser("~")
        model_folder = os.path.join(home, ".NudeNet/")
        if not os.path.exists(model_folder):
            os.mkdir(model_folder)

        model_path = os.path.join(model_folder, "lite_classifier")

        if not os.path.exists(model_path):
            print("Downloading the checkpoint to", model_path)
            pydload.dload(url, save_to_path=model_path, max_time=None)

        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

    def classify(self, image_paths, size=(256,256)):
        if isinstance(image_paths, str):
            image_paths = [image_paths]

        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()

        loaded_images, _ = load_images(image_paths, size, image_paths)

        result = {}
        for image_path, img in zip(image_paths, loaded_images):
            img = np.expand_dims(img, axis=0)
            input_data = np.array(img, dtype=np.float32)
            self.interpreter.set_tensor(input_details[0]['index'], input_data)

            self.interpreter.invoke()

            # The function `get_tensor()` returns a copy of the tensor data.
            # Use `tensor()` in order to get a pointer to the tensor.
            output_data = self.interpreter.get_tensor(output_details[0]['index'])
            result[image_path] = {"unsafe": output_data[0][0], "safe": output_data[0][1]}
            
        return result
