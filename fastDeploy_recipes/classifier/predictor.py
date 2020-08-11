import base64
from nudenet import NudeClassifier

classifier = NudeClassifier()


"""

Your function should take list of items as input
This makes batching possible

"""


def predictor(in_images=[], batch_size=32):
    if not in_images:
        return []
    preds = classifier.classify(in_images, batch_size=batch_size)

    preds = [preds.get(in_image) for in_image in in_images]

    preds = [{k: float(v) for k, v in pred.items()} if pred else pred for pred in preds]

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
