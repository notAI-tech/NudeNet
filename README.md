# NudeNet: Neural Nets for Nudity Classification, Detection and selective censoring

[![DOI](https://zenodo.org/badge/173154449.svg)](https://zenodo.org/badge/latestdoi/173154449)

Uncensored version of the following image can be found at https://i.imgur.com/rga6845.jpg (NSFW)

![](https://i.imgur.com/0KPJbl9.jpg)

**Classifier classes:**
|class name   |  Description    |
|--------|:--------------:
|safe | Image/Video is not sexually explicit     |
|unsafe | Image/Video is sexually explicit|

**Default Detector classes:**
|class name   |  Description                    |
|--------|:-------------------------------------:
|EXPOSED_ANUS | Exposed Anus; Any gender |
|EXPOSED_ARMPITS | Exposed Armpits; Any gender |
|COVERED_BELLY | Provocative, but covered Belly; Any gender |
|EXPOSED_BELLY | Exposed Belly; Any gender |
|COVERED_BUTTOCKS | Provocative, but covered Buttocks; Any gender |
|EXPOSED_BUTTOCKS | Exposed Buttocks; Any gender |
|FACE_F | Female Face|
|FACE_M | Male Face|
|COVERED_FEET |Covered Feet; Any gender |
|EXPOSED_FEET | Exposed Feet; Any gender|
|COVERED_BREAST_F | Provocative, but covered Breast; Female |
|EXPOSED_BREAST_F | Exposed Breast; Female |
|COVERED_GENITALIA_F |Provocative, but covered Genitalia; Female|
|EXPOSED_GENITALIA_F |Exposed Genitalia; Female |
|EXPOSED_BREAST_M |Exposed Breast; Male |
|EXPOSED_GENITALIA_M |Exposed Genitalia; Male |

**Base Detector classes:**
|class name   |  Description    |
|--------|:--------------:
|EXPOSED_BELLY | Exposed Belly; Any gender |
|EXPOSED_BUTTOCKS | Exposed Buttocks; Any gender |
|EXPOSED_BREAST_F | Exposed Breast; Female |
|EXPOSED_GENITALIA_F |Exposed Genitalia; Female |
|EXPOSED_GENITALIA_M |Exposed Genitalia; Male |
|EXPOSED_BREAST_M |Exposed Breast; Male |

# As self-hostable API service
```bash
# Classifier
docker run -it -p8080:8080 notaitech/nudenet:classifier

# Detector
docker run -it -p8080:8080 notaitech/nudenet:detector

# See fastDeploy-file_client.py for running predictions via fastDeploy's REST endpoints 
wget https://raw.githubusercontent.com/notAI-tech/fastDeploy/master/cli/fastDeploy-file_client.py
# Single input
python fastDeploy-file_client.py --file PATH_TO_YOUR_IMAGE

# Client side batching
python fastDeploy-file_client.py --dir PATH_TO_FOLDER --ext jpg
```

**Note: golang example https://github.com/notAI-tech/NudeNet/issues/63#issuecomment-729555360**, thanks to [Preetham Kamidi](https://github.com/preetham)


# As Python module
**Installation**:
```bash
pip install --upgrade nudenet
```

**Classifier Usage**:
```python
# Import module
from nudenet import NudeClassifier

# initialize classifier (downloads the checkpoint file automatically the first time)
classifier = NudeClassifier()

# Classify single image
classifier.classify('path_to_image_1')
# Returns {'path_to_image_1': {'safe': PROBABILITY, 'unsafe': PROBABILITY}}
# Classify multiple images (batch prediction)
# batch_size is optional; defaults to 4
classifier.classify(['path_to_image_1', 'path_to_image_2'], batch_size=BATCH_SIZE)
# Returns {'path_to_image_1': {'safe': PROBABILITY, 'unsafe': PROBABILITY},
#          'path_to_image_2': {'safe': PROBABILITY, 'unsafe': PROBABILITY}}

# Classify video
# batch_size is optional; defaults to 4
classifier.classify_video('path_to_video', batch_size=BATCH_SIZE)
# Returns {"metadata": {"fps": FPS, "video_length": TOTAL_N_FRAMES, "video_path": 'path_to_video'},
#          "preds": {frame_i: {'safe': PROBABILITY, 'unsafe': PROBABILITY}, ....}}

```

Thanks to [Johnny Urosevic](https://github.com/JohnnyUrosevic), NudeClassifier is also available in tflite.

**TFLite Classifier Usage**:
```python
# Import module
from nudenet import NudeClassifierLite

# initialize classifier (downloads the checkpoint file automatically the first time)
classifier_lite = NudeClassifierLite()

# Classify single image
classifier_lite.classify('path_to_image_1')
# Returns {'path_to_image_1': {'safe': PROBABILITY, 'unsafe': PROBABILITY}}
# Classify multiple images (batch prediction)
# batch_size is optional; defaults to 4
classifier_lite.classify(['path_to_image_1', 'path_to_image_2'])
# Returns {'path_to_image_1': {'safe': PROBABILITY, 'unsafe': PROBABILITY},
#          'path_to_image_2': {'safe': PROBABILITY, 'unsafe': PROBABILITY}}

```

Using the tflite classifier from flutter: **https://github.com/ndaysinaiK/nude-test** 

**Detector Usage**:
```python
# Import module
from nudenet import NudeDetector

# initialize detector (downloads the checkpoint file automatically the first time)
detector = NudeDetector() # detector = NudeDetector('base') for the "base" version of detector.

# Detect single image
detector.detect('path_to_image')
# fast mode is ~3x faster compared to default mode with slightly lower accuracy.
detector.detect('path_to_image', mode='fast')
# Returns [{'box': LIST_OF_COORDINATES, 'score': PROBABILITY, 'label': LABEL}, ...]

# Detect video
# batch_size is optional; defaults to 2
# show_progress is optional; defaults to True
detector.detect_video('path_to_video', batch_size=BATCH_SIZE, show_progress=BOOLEAN)
# fast mode is ~3x faster compared to default mode with slightly lower accuracy.
detector.detect_video('path_to_video', batch_size=BATCH_SIZE, show_progress=BOOLEAN, mode='fast')
# Returns {"metadata": {"fps": FPS, "video_length": TOTAL_N_FRAMES, "video_path": 'path_to_video'},
#          "preds": {frame_i: {'box': LIST_OF_COORDINATES, 'score': PROBABILITY, 'label': LABEL}, ...], ....}}



```

# Notes:
- detect_video and classify_video first identify the "unique" frames in a video and run predictions on them for significant performance improvement.
- V1 of NudeDetector (available in master branch of this repo) was trained on 12000 images labelled by the good folks at cti-community.
- V2 (current version) of NudeDetector is trained on 160,000 entirely auto-labelled (using classification heat maps and various other hybrid techniques) images. 
- The entire data for the classifier is available at https://archive.org/details/NudeNet_classifier_dataset_v1
- A part of the auto-labelled data (Images are from the classifier dataset above) used to train the base Detector is available at https://github.com/notAI-tech/NudeNet/releases/download/v0/DETECTOR_AUTO_GENERATED_DATA.zip
