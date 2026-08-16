import os
import _io
import math
import cv2
import numpy as np
import onnxruntime
from onnxruntime.capi import _pybind_state as C
import argparse
from pprint import pprint

__labels = [
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


def _read_image(image_path, target_size=320):
    # add gif functionality
    if isinstance(image_path, str):
        anim = cv2.Animation()
        success, anim = cv2.imreadanimation(image_path)
        frames = list(anim.frames)
        # list(frame[h,w,c],frame[h,w,c],...)
    elif isinstance(image_path, np.ndarray):
        if len(image_path.shape)==4:
            frames = list(np.unstack(image_path))
            # list(frame[h,w,c],frame[h,w,c],...)
        elif len(image_path.shape)==3:
            frames = [image_path,]
            # list(frame[h,w,c],frame[h,w,c],...)
    elif isinstance(image_path, bytes):
        success, anim = cv2.imdecodeanimation(np.frombuffer(image_path, np.uint8), -1)
        frames = list(anim.frames)
        # list(frame[h,w,c],frame[h,w,c],...)
    elif isinstance(image_path, _io.BufferedReader):
        success, anim = cv2.imdecodeanimation(np.frombuffer(image_path.read(), np.uint8), -1)
        frames = list(anim.frames)
    else:
        raise ValueError(
            "please make sure the image_path is str or np.ndarray or bytes"
        )
    frames_c3 = [cv2.cvtColor(mat, cv2.COLOR_RGBA2BGR) for mat in frames]

    sample_mat = frames_c3[0]
    image_original_width, image_original_height = sample_mat.shape[1], sample_mat.shape[0]
    
    max_size = max(sample_mat.shape[:2])  # get max size from width and height
    x_pad = max_size - sample_mat.shape[1]  # set xPadding
    x_ratio = max_size / sample_mat.shape[1]  # set xRatio
    y_pad = max_size - sample_mat.shape[0]  # set yPadding
    y_ratio = max_size / sample_mat.shape[0]  # set yRatio

    frames_pad = [cv2.copyMakeBorder(mat_c3, 0, y_pad, 0, x_pad, cv2.BORDER_CONSTANT) for mat_c3 in frames_c3]

    frames_input_blob = []
    for frame_pad in frames_pad:
        input_blob = cv2.dnn.blobFromImage(
            frame_pad,
            1 / 255.0,  # normalize
            (target_size, target_size),  # resize to model input size
            (0, 0, 0),  # mean subtraction
            swapRB=True,  # swap red and blue channels
            crop=False,  # don't crop
        )
        frames_input_blob.append(input_blob)

    return (
        frames_input_blob,
        x_ratio,
        y_ratio,
        x_pad,
        y_pad,
        image_original_width,
        image_original_height,
    )


def _postprocess(
    output_frames,
    x_pad,
    y_pad,
    x_ratio,
    y_ratio,
    image_original_width,
    image_original_height,
    model_width,
    model_height,
):
    # single image (as length 1 batch) or single animation output (as batch of frames)
    # extra nested list of length 1
    output = output_frames[0]
    # shape nframes,22 scores, rows
    # previously needed squeeze because of only single frame image
    # outputs = np.transpose(np.squeeze(output))
    outputs = np.transpose(output,axes=(0,2,1))
    # shape nframes,rows,22 scores
    frames = len(outputs)
    rows = outputs.shape[1]
    boxes = [[] for _ in range(frames)]
    scores = [[] for _ in range(frames)]
    class_ids = [[] for _ in range(frames)]
    for f in range(frames):
        for i in range(rows):
            classes_scores = outputs[f][i][4:]
            max_score = np.amax(classes_scores)

            if max_score >= 0.2:
                class_id = np.argmax(classes_scores)
                x, y, w, h = outputs[f][i][0:4]

                # Convert from center coordinates to top-left corner coordinates
                x = x - w / 2
                y = y - h / 2

                # Scale coordinates to original image size
                x = x * (image_original_width + x_pad) / model_width
                y = y * (image_original_height + y_pad) / model_height
                w = w * (image_original_width + x_pad) / model_width
                h = h * (image_original_height + y_pad) / model_height

                # Remove padding
                x = x
                y = y

                # Clip coordinates to image boundaries
                x = max(0, min(x, image_original_width))
                y = max(0, min(y, image_original_height))
                w = min(w, image_original_width - x)
                h = min(h, image_original_height - y)

                class_ids[f].append(class_id)
                scores[f].append(max_score)
                boxes[f].append([x, y, w, h])
    frame_indices = [cv2.dnn.NMSBoxes(boxes[f], scores[f], 0.25, 0.45) for f in range(frames)]
    detections = {'num_frames':frames,'r':[{'frame':i_f+1, 'fr':[]} for i_f in range(frames)]}

    for f, indices in enumerate(frame_indices):
        for i in indices:
            box = boxes[f][i]
            score = scores[f][i]
            class_id = class_ids[f][i]

            x, y, w, h = box
            det = {
                    "class": __labels[class_id],
                    "score": float(score),
                    "box": [int(x), int(y), int(w), int(h)],
                }
            detections['r'][f]['fr'].append(det)

    return detections


class NudeDetector:
    def __init__(self, model_path=None, providers=None, inference_resolution=320):
        self.onnx_session = onnxruntime.InferenceSession(
            os.path.join(os.path.dirname(__file__), "320n.onnx")
            if not model_path
            else model_path,
            # providers=C.get_available_providers() if not providers else providers,
        )
        model_inputs = self.onnx_session.get_inputs()

        self.input_width = inference_resolution
        self.input_height = inference_resolution
        self.input_name = model_inputs[0].name

    def detect(self, image_path):
        (
            preprocessed_frames,
            x_ratio,
            y_ratio,
            x_pad,
            y_pad,
            image_original_width,
            image_original_height,
        ) = _read_image(image_path, self.input_width)
        input_4d = np.vstack(preprocessed_frames)
        outputs_frames = self.onnx_session.run(None, {self.input_name: input_4d})
        detections = _postprocess(
            outputs_frames,
            x_pad,
            y_pad,
            x_ratio,
            y_ratio,
            image_original_width,
            image_original_height,
            self.input_width,
            self.input_height,
        )

        return detections

    def detect_batch(self, image_paths, batch_size=4):
        """
        Perform batch detection on a list of images.

        Args:
            image_paths (List[Union[str, np.ndarray]]): List of image paths or numpy arrays.
            batch_size (int): Number of images to process in each batch.

        Returns:
            List of detection results for each image.
        """
        all_detections = []
        for i in range(0, len(image_paths), batch_size):
            batch = image_paths[i : i + batch_size]
            batch_inputs = []
            batch_metadata = []
            image_ind = 0
            frame_ind = 0
            # todo: make sure frame vs image info is preserved
            batch_img_ind_to_frame_start_end = {}
            for image_path in batch:
                (
                    preprocessed_frames,
                    x_ratio,
                    y_ratio,
                    x_pad,
                    y_pad,
                    image_original_width,
                    image_original_height,
                ) = _read_image(image_path, self.input_width)
                batch_inputs.extend(preprocessed_frames)
                # todo: make sure frame vs image info is preserved
                # image index to frame range:
                batch_img_ind_to_frame_start_end[image_ind]={
                    'start_frame':frame_ind,
                    'end_frame':frame_ind+len(preprocessed_frames)-1
                    }
                frame_ind+=len(preprocessed_frames)
                batch_metadata.append(
                    (
                        x_ratio,
                        y_ratio,
                        x_pad,
                        y_pad,
                        image_original_width,
                        image_original_height,
                    )
                )
                image_ind+=1
            # Stack the preprocessed images into a single numpy array
            batch_input = np.vstack(batch_inputs)
            # Run inference on the batch
            output = self.onnx_session.run(None, {self.input_name: batch_input})
            outputs = np.unstack(output[0])
            # Process the outputs for each image in the batch
            for j_img, metadata in enumerate(batch_metadata):
                (
                    x_ratio,
                    y_ratio,
                    x_pad,
                    y_pad,
                    image_original_width,
                    image_original_height,
                ) = metadata
                start_frame = batch_img_ind_to_frame_start_end[j_img]['start_frame']
                end_frame = batch_img_ind_to_frame_start_end[j_img]['end_frame']
                if start_frame == end_frame:
                    outf = outputs[start_frame]
                    image_outputs = [np.reshape(outf,[-1,outf.shape[0],outf.shape[1]])]
                else:
                    output_by_frame = outputs[start_frame:end_frame]
                    image_outputs = [np.stack(output_by_frame)]
                detections = _postprocess(
                    image_outputs,
                    x_pad,
                    y_pad,
                    x_ratio,
                    y_ratio,
                    image_original_width,
                    image_original_height,
                    self.input_width,
                    self.input_height,
                )
                all_detections.append(detections)

        return all_detections

    def censor(self, image_path, classes=[], output_path=None):
        detections = self.detect(image_path)
        if classes:
            detections = [
                [detection for detection in frame if detection["class"] in classes]
                for frame in detections
                ]

        # add gif functionality
        anim = cv2.Animation()
        success, anim = cv2.imreadanimation(image_path)
        for frame_ind, frame in enumerate(detections):
            for detection in frame:
                box = detection["box"]
                x, y, w, h = box[0], box[1], box[2], box[3]
                # change these pixels to pure black
                anim.frames[frame_ind][y : y + h, x : x + w] = (0, 0, 0)

        image_name, ext = os.path.splitext(image_path)
        if not output_path:
            output_path = f"{image_path}_censored{ext}"
        
        if ext in ['.gif', '.avif', '.apng', '.webp']:
            cv2.imwriteanimation(output_path, anim)
        else:
            img = anim.frames[0]
            cv2.imwrite(output_path, img)

        return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, default=None)
    parser.add_argument("-m", "--model_path", type=str, default=None)
    parser.add_argument("-r", "--inference_resolution", type=int, default=320)
    parser.add_argument("-b", "--batch_size", type=int, default=4)
    args = parser.parse_args()
    detector = NudeDetector(
        model_path=args.model_path,
        inference_resolution=args.inference_resolution,
        )
    if pathlib.Path(args.input).is_dir():
        input_paths = [str(pd.resolve().absolute()) for pd in pathlib.Path(args.input).iterdir()]
    else:
        input_paths = [str(pathlib.Path(p).resolve().absolute()) for p in glob.iglob(args.input)]

    detections = nude_detector.detect_batch(
        input_paths,
        batch_size=args.batch_size)

    for index,det in enumerate(detections):
        pprint(str_paths[index])
        pprint(det)
        print("\n")
