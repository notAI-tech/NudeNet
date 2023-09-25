# NudeNet: lightweight Nudity detection

https://nudenet.notai.tech/ in-browser demo (the detector is run client side, i.e: in your browser, images are not sent to a server)

```bash
pip install --upgrade nudenet
```

```python
from nudenet import NudeDetector
nude_detector = NudeDetector()
nude_detector.detect('image.jpg') # Returns list of detections
```

```python
detection_example = [
 {'class': 'BELLY_EXPOSED',
  'score': 0.799403190612793,
  'box': [64, 182, 49, 51]},
 {'class': 'FACE_FEMALE',
  'score': 0.7881264686584473,
  'box': [82, 66, 36, 43]},
 ]
```

```python
all_labels = [
    "FEMALE_GENITALIA_COVERED",
    "FACE_FEMALE",
    "BUTTOCKS_EXPOSED",
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_BREAST_EXPOSED",
    "ANUS_EXPOSED",
    "FEET_EXPOSED",
    "BELLY_COVERED",
    "FEET_COVERED",
    "ARMPITS_COVERED",
    "ARMPITS_EXPOSED",
    "FACE_MALE",
    "BELLY_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "ANUS_COVERED",
    "FEMALE_BREAST_COVERED",
    "BUTTOCKS_COVERED",
]
```


### Docker

```bash
docker run -it -p8080:8080 ghcr.io/notai-tech/nudenet:latest
```

```bash
curl -F f1=@"images.jpeg" "http://localhost:8080/infer"

{"prediction": [[{"class": "BELLY_EXPOSED", "score": 0.8511635065078735, "box": [71, 182, 31, 50]}, {"class": "FACE_FEMALE", "score": 0.8033977150917053, "box": [83, 69, 21, 37]}, {"class": "FEMALE_BREAST_EXPOSED", "score": 0.7963727712631226, "box": [85, 137, 24, 38]}, {"class": "FEMALE_BREAST_EXPOSED", "score": 0.7709134817123413, "box": [63, 136, 20, 37]}, {"class": "ARMPITS_EXPOSED", "score": 0.7005534172058105, "box": [60, 127, 10, 20]}, {"class": "FEMALE_GENITALIA_EXPOSED", "score": 0.6804671287536621, "box": [81, 241, 14, 24]}]], "success": true}⏎
```
