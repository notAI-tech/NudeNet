import os
import cv2
import pydload
import numpy as np

from .image_utils import load_images


class LiteClassifier:
    def __init__(self):
        url = "https://github.com/notAI-tech/NudeNet/releases/download/v0/classifier_lite.onnx"
        home = os.path.expanduser("~")
        model_folder = os.path.join(home, ".NudeNet/")
        if not os.path.exists(model_folder):
            os.mkdir(model_folder)

        model_path = os.path.join(model_folder, os.path.basename(url))

        if not os.path.exists(model_path):
            print("Downloading the checkpoint to", model_path)
            pydload.dload(url, save_to_path=model_path, max_time=None)

        self.lite_model = cv2.dnn.readNet(model_path)

    def classify(self, image_paths, size=(256, 256)):
        if isinstance(image_paths, str):
            image_paths = [image_paths]

        result = {}
        for image_path in image_paths:
            loaded_images, _ = load_images([image_path], size, image_names=[image_path])
            loaded_images = np.rollaxis(loaded_images, 3, 1)

            self.lite_model.setInput(loaded_images)
            pred = self.lite_model.forward()

            result[image_path] = {
                "unsafe": pred[0][0],
                "safe": pred[0][1],
            }

        return result
