import keras
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

import cv2
import numpy as np

class Detector():
    detection_model = None
    classes = [
        'BELLY',
        'BUTTOCKS',
        'F_BREAST',
        'F_GENITALIA',
        'M_GENETALIA',
        'M_BREAST',
    ]
    
    def __init__(self, model_path):
        '''
            model = Classifier('path_to_weights')
        '''
        Detector.detection_model = models.load_model(model_path, backbone_name='resnet101')
    
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
            # image = cv2.GaussianBlur(part,(23, 23), 30)
            # image[box[1]:box[3], box[0]:box[2]] = part
        
        if visualize:
            cv2.imshow("Blurred image", image)
            cv2.waitKey(0)
        
        if out_path:
            cv2.imwrite(out_path, image)


if __name__ == '__main__':
    m = Detector('/Users/bedapudi/Desktop/inference_resnet50_csv_14.h5')
    print(m.censor('/Users/bedapudi/Desktop/n2.jpg', out_path='a.jpg'))
