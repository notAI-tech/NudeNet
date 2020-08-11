import os
import base64
from nudenet import NudeDetector

detector = NudeDetector(os.getenv('MODEL_NAME', 'default'))


"""

Your function should take list of items as input
This makes batching possible

"""


def predictor(in_images=[], batch_size=32):
    if not in_images:
        return []
    preds = []

    for in_img in in_images:
        try:
            preds.append(detector.detect(in_img))
        except:
            preds.append(None)

    return preds


if __name__ == "__main__":
    import json
    import pickle
    import base64

    example = ["example.jpg"]

    print(json.dumps(predictor(example)))

    example = {
        file_name: base64.b64encode(open(file_name, "rb").read()).decode("utf-8")
        for file_name in example
    }

    pickle.dump(example, open("example.pkl", "wb"), protocol=2)
