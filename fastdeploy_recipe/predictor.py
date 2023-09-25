from nudenet import NudeDetector

detector = NudeDetector()

def predictor(image_paths, batch_size=1):
    results = []
    for image_path in image_paths:
        result = detector.detect(image_path)
        results.append(result)
    return results
