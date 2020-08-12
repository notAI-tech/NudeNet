import os
import keras
import pydload
from keras_retinanet import models
from keras_retinanet.utils.image import preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

from .video_utils import get_interest_frames_from_video

import cv2
import numpy as np

import logging

from PIL import Image as pil_image

from progressbar import progressbar


def read_image_bgr(path):
    """ Read an image in BGR format.
    Args
        path: Path to the image.
    """
    if isinstance(path, str):
        image = np.ascontiguousarray(pil_image.open(path).convert("RGB"))
    else:
        path = cv2.cvtColor(path, cv2.COLOR_BGR2RGB)
        image = np.ascontiguousarray(pil_image.fromarray(path))

    return image[:, :, ::-1]


def dummy(x):
    return x


FILE_URLS = {
    "default": {
        "checkpoint": "https://github.com/notAI-tech/NudeNet/releases/download/v0/detector_v2_default_checkpoint",
        "classes": "https://github.com/notAI-tech/NudeNet/releases/download/v0/detector_v2_default_classes",
    },
    "base": {
        "checkpoint": "https://github.com/notAI-tech/NudeNet/releases/download/v0/detector_v2_base_checkpoint",
        "classes": "https://github.com/notAI-tech/NudeNet/releases/download/v0/detector_v2_base_classes",
    },
}


class Detector:
    detection_model = None
    classes = None

    def __init__(self, model_name="default"):
        """
            model = Detector()
        """
        checkpoint_url = FILE_URLS[model_name]["checkpoint"]
        classes_url = FILE_URLS[model_name]["classes"]

        home = os.path.expanduser("~")
        model_folder = os.path.join(home, f".NudeNet/{model_name}")
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)

        checkpoint_path = os.path.join(model_folder, "checkpoint")
        classes_path = os.path.join(model_folder, "classes")

        if not os.path.exists(checkpoint_path):
            print("Downloading the checkpoint to", checkpoint_path)
            pydload.dload(checkpoint_url, save_to_path=checkpoint_path, max_time=None)

        if not os.path.exists(classes_path):
            print("Downloading the classes list to", classes_path)
            pydload.dload(classes_url, save_to_path=classes_path, max_time=None)

        self.detection_model = models.load_model(
            checkpoint_path, backbone_name="resnet50"
        )
        self.classes = [
            c.strip() for c in open(classes_path).readlines() if c.strip()
        ]

    def detect_video(self, video_path, min_prob=0.6, batch_size=2, show_progress=True):
        frame_indices, frames, fps, video_length = get_interest_frames_from_video(
            video_path
        )
        logging.debug(
            f"VIDEO_PATH: {video_path}, FPS: {fps}, Important frame indices: {frame_indices}, Video length: {video_length}"
        )
        frames = [read_image_bgr(frame) for frame in frames]
        frames = [preprocess_image(frame) for frame in frames]
        frames = [resize_image(frame) for frame in frames]
        scale = frames[0][1]
        frames = [frame[0] for frame in frames]
        all_results = {
            "metadata": {
                "fps": fps,
                "video_length": video_length,
                "video_path": video_path,
            },
            "preds": {},
        }

        progress_func = progressbar

        if not show_progress:
            progress_func = dummy

        for _ in progress_func(range(int(len(frames) / batch_size) + 1)):
            batch = frames[:batch_size]
            batch_indices = frame_indices[:batch_size]
            frames = frames[batch_size:]
            frame_indices = frame_indices[batch_size:]
            if batch_indices:
                boxes, scores, labels = self.detection_model.predict_on_batch(
                    np.asarray(batch)
                )
                boxes /= scale
                for frame_index, frame_boxes, frame_scores, frame_labels in zip(
                    frame_indices, boxes, scores, labels
                ):
                    if frame_index not in all_results["preds"]:
                        all_results["preds"][frame_index] = []

                    for box, score, label in zip(
                        frame_boxes, frame_scores, frame_labels
                    ):
                        if score < min_prob:
                            continue
                        box = box.astype(int).tolist()
                        label = self.classes[label]

                        all_results["preds"][frame_index].append(
                            {"box": box, "score": score, "label": label}
                        )

        return all_results

    def detect(self, img_path, min_prob=0.6):
        image = read_image_bgr(img_path)
        image = preprocess_image(image)
        image, scale = resize_image(image)
        boxes, scores, labels = self.detection_model.predict_on_batch(
            np.expand_dims(image, axis=0)
        )
        boxes /= scale
        processed_boxes = []
        for box, score, label in zip(boxes[0], scores[0], labels[0]):
            if score < min_prob:
                continue
            box = box.astype(int).tolist()
            label = self.classes[label]
            processed_boxes.append({"box": box, "score": score, "label": label})

        return processed_boxes

    def censor(self, img_path, out_path=None, visualize=False, parts_to_blur=[]):
        if not out_path and not visualize:
            print(
                "No out_path passed and visualize is set to false. There is no point in running this function then."
            )
            return

        image = cv2.imread(img_path)
        boxes = self.detect(img_path)

        if parts_to_blur:
            boxes = [i["box"] for i in boxes if i["label"] in parts_to_blur]
        else:
            boxes = [i["box"] for i in boxes]

        for box in boxes:
            part = image[box[1] : box[3], box[0] : box[2]]
            image = cv2.rectangle(
                image, (box[0], box[1]), (box[2], box[3]), (0, 0, 0), cv2.FILLED
            )

        if visualize:
            cv2.imshow("Blurred image", image)
            cv2.waitKey(0)

        if out_path:
            cv2.imwrite(out_path, image)


if __name__ == "__main__":
    m = Detector()
    print(m.detect("/Users/bedapudi/Desktop/n2.jpg"))
