# NudeNet: An ensemble of Neural Nets for Nudity Detection and Censoring

![](https://i.imgur.com/Fs6exOx.png)


# Classification Classes

nude -> image contains nudity

safe -> image doesn't contain nudity

# Detection Classes
BELLY -> exposed belly (both male and female)

BUTTOCKS -> exposed buttocks (both male and female)

F_BREAST -> exposed female breast

F_GENITALIA -> exposed female genitalia

M_GENETALIA -> exposed male genetalia

M_BREAST -> exposed male breast

# Insallation
```
pip install nudenet
```

# Classifier Usage
```
from NudeNet import NudeClassifier
classifier = NudeClassifier('classifier_checkpoint_path')
classifier.classify('path_to_nude_image')
# {'path_to_nude_image': {'safe': 5.8822202e-08, 'nude': 1.0}}
```

# Detector Usage
```
from NudeNet import NudeDetector
detector = NudeDetector('detector_checkpoint_path')

# Performing detection
detector.detect('path_to_nude_image')
# [{'box': [352, 688, 550, 858], 'score': 0.9603578, 'label': 'BELLY'}, {'box': [507, 896, 586, 1055], 'score': 0.94103414, 'label': 'F_GENITALIA'}, {'box': [221, 467, 552, 650], 'score': 0.8011624, 'label': 'F_BREAST'}, {'box': [359, 464, 543, 626], 'score': 0.6324697, 'label': 'F_BREAST'}]

# Censoring an image
detector.censor('path_to_nude_image', out_path='censored_image_path', visualize=False)

```


#To Do:
1. Improve Documentation for the functions. (Right now user has to see the function definition to understand all the params)
2. Convert these models into tflite, tfjs and create another repo that used tfjs to perform in browser detection and censor.

Note: Entire credit for collecting the object recognition dataset goes to Jae Jin and his team https://github.com/Kadantte
