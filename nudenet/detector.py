import os
import keras
import pydload
from keras_retinanet import models
from keras_retinanet.utils.image import preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

from video_utils import get_interest_frames_from_video

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
        image = np.ascontiguousarray(pil_image.open(path).convert('RGB'))
    else:
        path = cv2.cvtColor(path, cv2.COLOR_BGR2RGB)
        image = np.ascontiguousarray(pil_image.fromarray(path))

    return image[:, :, ::-1]

class Detector():
    detection_model = None
    classes = [
        'BELLY',
        'BUTTOCKS',
        'F_BREAST',
        'F_GENITALIA',
        'M_GENITALIA',
        'M_BREAST',
    ]
    
    def __init__(self):
        '''
            model = Detector()
        '''
        url = 'https://github.com/bedapudi6788/NudeNet/releases/download/v0/detector_model'
        home = os.path.expanduser("~")
        model_folder = os.path.join(home, '.NudeNet/')
        if not os.path.exists(model_folder):
            os.mkdir(model_folder)
        
        model_path = os.path.join(model_folder, 'detector')

        if not os.path.exists(model_path):
            print('Downloading the checkpoint to', model_path)
            pydload.dload(url, save_to_path=model_path, max_time=None)

        Detector.detection_model = models.load_model(model_path, backbone_name='resnet101')
    
    def detect_video(self, video_path=None, min_prob=0.6, batch_size=2):
        frame_indices, frames, fps, video_length = get_interest_frames_from_video(video_path)
        logging.debug(f'VIDEO_PATH: {video_path}, FPS: {fps}, Important frame indices: {frame_indices}, Video length: {video_length}')
        frames = [read_image_bgr(frame) for frame in frames]
        frames = [preprocess_image(frame) for frame in frames]
        frames = [resize_image(frame) for frame in frames]
        scale = frames[0][1]
        frames = [frame[0] for frame in frames]
        all_results = {}

        for _ in progressbar(range(int(len(frames)/batch_size) + 1)):
            batch = frames[:batch_size]
            batch_indices = frame_indices[:batch_size]
            frames = frames[batch_size:]
            frame_indices = frame_indices[batch_size:]
            if len(batch):
                boxes, scores, labels = Detector.detection_model.predict_on_batch(np.asarray(batch))
                boxes /= scale
                for frame_index, frame_boxes, frame_scores, frame_labels in zip(frame_indices, boxes, scores, labels):
                    if frame_index not in all_results:
                        all_results[frame_index] = []
                    
                    for box, score, label in zip(frame_boxes, frame_scores, frame_labels):
                        if score < min_prob:
                            continue
                        box = box.astype(int).tolist()

    def detect(self, img_path, min_prob=0.6):
        image = read_image_bgr(img_path)
        image = preprocess_image(image)
        image, scale = resize_image(image)
        boxes, scores, labels = Detector.detection_model.predict_on_batch(np.expand_dims(image, axis=0))
        boxes /= scale
        processed_boxes = []
        for box, score, label in zip(boxes[0], scores[0], labels[0]):
            if score < min_prob:
                continue
            box = box.astype(int).tolist()
            label = Detector.classes[label]
            processed_boxes.append({'box': box, 'score': score, 'label': label})
            
        return processed_boxes
    
    def censor(self, img_path, out_path=None, visualize=True, parts_to_blur=['BELLY', 'BUTTOCKS', 'F_BREAST', 'F_GENITALIA', 'M_GENETALIA', 'M_BREAST']):
        if not out_path and not visualize:
            print('No out_path passed and visualize is set to false. There is no point in running this function then.')

        image = cv2.imread(img_path)
        boxes = Detector.detect(self, img_path)
        boxes = [i['box'] for i in boxes if i['label'] in parts_to_blur]

        for box in boxes:
            part = image[box[1]:box[3], box[0]:box[2]]
            image = cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 0, 0), cv2.FILLED)
        
        if visualize:
            cv2.imshow("Blurred image", image)
            cv2.waitKey(0)
        
        if out_path:
            cv2.imwrite(out_path, image)


if __name__ == '__main__':
    m = Detector()
    print(m.censor('/Users/bedapudi/Desktop/n2.jpg', out_path='a.jpg'))
