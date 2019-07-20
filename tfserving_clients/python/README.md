# Classifier now available with tfserving docker image
```
# Get the docker image
docker pull bedapudi6788/nudeclassifier:v1
docker run -d -p 8500:8500 bedapudi6788/nudeclassifier:v1

# Installing python client
pip install nudeclient

import nudeclient
# Single image prediction
nudeclient.predict('path_to_nude_image')
{'path_to_nude_image': {'safe': 5.8822202e-08, 'unsafe': 1.0}}

# Batch predictions
nudeclient.predict(['path_to_image_1', 'path_to_image2])
{'path_to_image_1': {'safe': 5.8822202e-08, 'unsafe': 1.0}, 'path_to_image_2': {'safe': 5.8822202e-08, 'unsafe': 1.0}}

```
