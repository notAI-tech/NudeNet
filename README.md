# NudeNet: Neural Nets for Nudity Detection and Censoring

**NudeClassifier is available as a free to use API (https://fastdeploy.notai.tech/free_apis) and as a self-hostable service via https://github.com/notAI-tech/fastDeploy**

[![DOI](https://zenodo.org/badge/173154449.svg)](https://zenodo.org/badge/latestdoi/173154449)


Uncensored version of the following image can be found at https://i.imgur.com/rga6845.jpg (NSFW)

![](https://i.imgur.com/2mhyqnt.jpg)

Classification scores on the data available at https://dataturks.com/projects/Mohan/NSFW(Nudity%20Detection)%20Image%20Moderation%20Datatset
![](https://i.imgur.com/lXvvsdN.jpg)

# Classification Classes

unsafe -> image contains nudity

safe -> image doesn't contain nudity

# Detection Classes
BELLY -> exposed belly (both male and female)

BUTTOCKS -> exposed buttocks (both male and female)

F_BREAST -> exposed female breast

F_GENITALIA -> exposed female genitalia

M_GENITALIA -> exposed male genitalia

M_BREAST -> exposed male breast

# Installation
```bash
pip install nudenet
# or
pip install git+https://github.com/bedapudi6788/NudeNet
```

# Classifier Usage
```python
from nudenet import NudeClassifier
classifier = NudeClassifier()
classifier.classify('path_to_nude_image')
# {'path_to_nude_image': {'safe': 5.8822202e-08, 'unsafe': 1.0}}
```

# Classifier now available with tfserving docker image
```bash
# Get the docker image
docker pull bedapudi6788/nudeclassifier:v1
docker run -d -p 8500:8500 bedapudi6788/nudeclassifier:v1

# Installing python client
pip install nudeclient

```

```python
import nudeclient
# Single image prediction
nudeclient.predict('path_to_nude_image')
{'path_to_nude_image': {'safe': 5.8822202e-08, 'unsafe': 1.0}}

# Batch predictions
nudeclient.predict(['path_to_image_1', 'path_to_image2])
{'path_to_image_1': {'safe': 5.8822202e-08, 'unsafe': 1.0}, 'path_to_image_2': {'safe': 5.8822202e-08, 'unsafe': 1.0}}

```

# Detector Usage
```python
from nudenet import NudeDetector
detector = NudeDetector()

# Performing detection
detector.detect('path_to_nude_image')
# [{'box': [352, 688, 550, 858], 'score': 0.9603578, 'label': 'BELLY'}, {'box': [507, 896, 586, 1055], 'score': 0.94103414, 'label': 'F_GENITALIA'}, {'box': [221, 467, 552, 650], 'score': 0.8011624, 'label': 'F_BREAST'}, {'box': [359, 464, 543, 626], 'score': 0.6324697, 'label': 'F_BREAST'}]

# Censoring an image
detector.censor('path_to_nude_image', out_path='censored_image_path', visualize=False)

```

# Classifier data available at https://archive.org/details/NudeNet_classifier_dataset_v1

# To Do:
1. Improve Documentation for the functions. (Right now user has to see the function definition to understand all the params)
2. Convert these models into tflite, tfjs and create another repo that used tfjs to perform in browser detection and censor.

**Note: Entire credit for collecting the object recognition dataset goes to http://www.cti-community.net/ (NSFW). The link for their api and the discord are as follows API here: http://pury.fi/ Discord: https://discord.gg/k4qM4Jh**
 
