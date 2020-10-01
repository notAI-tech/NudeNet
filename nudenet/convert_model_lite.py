import os
import pydload
import tensorflow as tf

url = "https://github.com/bedapudi6788/NudeNet/releases/download/v0/classifier_model"
home = os.path.expanduser("~")
model_folder = os.path.join(home, ".NudeNet/")
if not os.path.exists(model_folder):
    os.mkdir(model_folder)

model_path = os.path.join(model_folder, "classifier")

if not os.path.exists(model_path):
                print("Downloading the checkpoint to", model_path)
                pydload.dload(url, save_to_path=model_path, max_time=None)

converter = tf.lite.TFLiteConverter.from_keras_model_file(model_path)
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)
