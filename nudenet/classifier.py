import os
import cv2
import keras
import pydload
import logging
import numpy as np
from .video_utils import get_interest_frames_from_video

from PIL import Image as pil_image

if pil_image is not None:
    _PIL_INTERPOLATION_METHODS = {
        "nearest": pil_image.NEAREST,
        "bilinear": pil_image.BILINEAR,
        "bicubic": pil_image.BICUBIC,
    }
    # These methods were only introduced in version 3.4.0 (2016).
    if hasattr(pil_image, "HAMMING"):
        _PIL_INTERPOLATION_METHODS["hamming"] = pil_image.HAMMING
    if hasattr(pil_image, "BOX"):
        _PIL_INTERPOLATION_METHODS["box"] = pil_image.BOX
    # This method is new in version 1.1.3 (2013).
    if hasattr(pil_image, "LANCZOS"):
        _PIL_INTERPOLATION_METHODS["lanczos"] = pil_image.LANCZOS


def load_img(
    path, grayscale=False, color_mode="rgb", target_size=None, interpolation="nearest"
):
    """Loads an image into PIL format.
    
    :param path: Path to image file.
    :param grayscale: DEPRECATED use `color_mode="grayscale"`.
    :param color_mode: One of "grayscale", "rgb", "rgba". Default: "rgb".
        The desired image format.
    :param target_size: Either `None` (default to original size)
        or tuple of ints `(img_height, img_width)`.
    :param interpolation: Interpolation method used to resample the image if the
        target size is different from that of the loaded image.
        Supported methods are "nearest", "bilinear", and "bicubic".
        If PIL version 1.1.3 or newer is installed, "lanczos" is also
        supported. If PIL version 3.4.0 or newer is installed, "box" and
        "hamming" are also supported. By default, "nearest" is used.
    
    :return: A PIL Image instance.
    """
    if grayscale is True:
        logging.warn("grayscale is deprecated. Please use " 'color_mode = "grayscale"')
        color_mode = "grayscale"
    if pil_image is None:
        raise ImportError(
            "Could not import PIL.Image. " "The use of `load_img` requires PIL."
        )

    if isinstance(path, type("")):
        img = pil_image.open(path)
    else:
        path = cv2.cvtColor(path, cv2.COLOR_BGR2RGB)
        img = pil_image.fromarray(path)

    if color_mode == "grayscale":
        if img.mode != "L":
            img = img.convert("L")
    elif color_mode == "rgba":
        if img.mode != "RGBA":
            img = img.convert("RGBA")
    elif color_mode == "rgb":
        if img.mode != "RGB":
            img = img.convert("RGB")
    else:
        raise ValueError('color_mode must be "grayscale", "rgb", or "rgba"')
    if target_size is not None:
        width_height_tuple = (target_size[1], target_size[0])
        if img.size != width_height_tuple:
            if interpolation not in _PIL_INTERPOLATION_METHODS:
                raise ValueError(
                    "Invalid interpolation method {} specified. Supported "
                    "methods are {}".format(
                        interpolation, ", ".join(_PIL_INTERPOLATION_METHODS.keys())
                    )
                )
            resample = _PIL_INTERPOLATION_METHODS[interpolation]
            img = img.resize(width_height_tuple, resample)
    return img


def load_images(image_paths, image_size, image_names):
    """
    Function for loading images into numpy arrays for passing to model.predict
    inputs:
        image_paths: list of image paths to load
        image_size: size into which images should be resized
    
    outputs:
        loaded_images: loaded images on which keras model can run predictions
        loaded_image_indexes: paths of images which the function is able to process
    
    """
    loaded_images = []
    loaded_image_paths = []

    for i, img_path in enumerate(image_paths):
        try:
            image = load_img(img_path, target_size=image_size)
            image = keras.preprocessing.image.img_to_array(image)
            image /= 255
            loaded_images.append(image)
            loaded_image_paths.append(image_names[i])
        except Exception as ex:
            logging.exception(f"Error reading {img_path} {ex}", exc_info=True)

    return np.asarray(loaded_images), loaded_image_paths


class Classifier:
    """
        Class for loading model and running predictions.
        For example on how to use take a look the if __name__ == '__main__' part.
    """

    nsfw_model = None

    def __init__(self):
        """
            model = Classifier()
        """
        url = "https://github.com/bedapudi6788/NudeNet/releases/download/v0/classifier_model"
        home = os.path.expanduser("~")
        model_folder = os.path.join(home, ".NudeNet/")
        if not os.path.exists(model_folder):
            os.mkdir(model_folder)

        model_path = os.path.join(model_folder, "classifier")

        if not os.path.exists(model_path):
            print("Downloading the checkpoint to", model_path)
            pydload.dload(url, save_to_path=model_path, max_time=None)

        self.nsfw_model = keras.models.load_model(model_path)

    def classify_video(
        self,
        video_path,
        batch_size=4,
        image_size=(256, 256),
        categories=["unsafe", "safe"],
    ):
        frame_indices = None
        frame_indices, frames, fps, video_length = get_interest_frames_from_video(
            video_path
        )
        logging.debug(
            f"VIDEO_PATH: {video_path}, FPS: {fps}, Important frame indices: {frame_indices}, Video length: {video_length}"
        )

        frames, frame_names = load_images(frames, image_size, image_names=frame_indices)

        if not frame_names:
            return {}

        model_preds = self.nsfw_model.predict(frames, batch_size=batch_size)
        preds = np.argsort(model_preds, axis=1).tolist()

        probs = []
        for i, single_preds in enumerate(preds):
            single_probs = []
            for j, pred in enumerate(single_preds):
                single_probs.append(model_preds[i][pred])
                preds[i][j] = categories[pred]

            probs.append(single_probs)

        return_preds = {
            "metadata": {
                "fps": fps,
                "video_length": video_length,
                "video_path": video_path,
            },
            "preds": {},
        }

        for i, frame_name in enumerate(frame_names):
            return_preds["preds"][frame_name] = {}
            for _ in range(len(preds[i])):
                return_preds["preds"][frame_name][preds[i][_]] = probs[i][_]

        return return_preds

    def classify(
        self,
        image_paths=[],
        batch_size=4,
        image_size=(256, 256),
        categories=["unsafe", "safe"],
    ):
        """
            inputs:
                image_paths: list of image paths or can be a string too (for single image)
                batch_size: batch_size for running predictions
                image_size: size to which the image needs to be resized
                categories: since the model predicts numbers, categories is the list of actual names of categories
        """
        if isinstance(image_paths, str):
            image_paths = [image_paths]

        loaded_images, loaded_image_paths = load_images(
            image_paths, image_size, image_names=image_paths
        )

        if not loaded_image_paths:
            return {}

        model_preds = self.nsfw_model.predict(
            loaded_images, batch_size=batch_size
        )

        preds = np.argsort(model_preds, axis=1).tolist()

        probs = []
        for i, single_preds in enumerate(preds):
            single_probs = []
            for j, pred in enumerate(single_preds):
                single_probs.append(model_preds[i][pred])
                preds[i][j] = categories[pred]

            probs.append(single_probs)

        images_preds = {}

        for i, loaded_image_path in enumerate(loaded_image_paths):
            if not isinstance(loaded_image_path, str):
                loaded_image_path = i

            images_preds[loaded_image_path] = {}
            for _ in range(len(preds[i])):
                images_preds[loaded_image_path][preds[i][_]] = probs[i][_]

        return images_preds


if __name__ == "__main__":
    print(
        '\n Enter path for the keras weights, leave empty to use "./nsfw.299x299.h5" \n'
    )
    weights_path = input().strip()
    if not weights_path:
        weights_path = "../nsfw.299x299.h5"

    m = Classifier()

    while 1:
        print(
            "\n Enter single image path or multiple images seperated by || (2 pipes) \n"
        )
        images = input().split("||")
        images = [image.strip() for image in images]
        print(m.predict(images), "\n")
