# NOTE: to prevent user trauma from accidental exposure, .zip is renamed to .bin and images have their file extensions changed to _nsfw (so thumbnails won't render automatically & opening won't default to an image application)

''' 
Trains a new NSFW YOLO model based on an existing one (in order to upgrade to newer YOLO base model versions and/or adjust for alternate image dimensions)

# Script Execution Instructions

# Updated torch with cuda install command can be found from https://pytorch.org/get-started/locally/
# install CUDA Toolkit from https://developer.nvidia.com/cuda-toolkit
# install cuDNN from https://developer.nvidia.com/cudnn

# linux
python -m venv yoloNSFWTraining
source yoloNSFWTraining/bin/activate
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install Pillow ultralytics requests tqdm
python train.py

# windows
python -m venv yoloNSFWTraining
yoloNSFWTraining\Scripts\activate
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install Pillow ultralytics requests tqdm
python train.py
'''

import subprocess
import logging
from pathlib import Path
import zipfile
import requests
import shutil
import time
from tqdm import tqdm
import torch
from ultralytics import YOLO
from ultralytics.data.utils import IMG_FORMATS

nsfw_pseudo_file_extensions = ["jpg_nsfw", "jpeg_nsfw", "png_nsfw"]

missing_formats = [fmt for fmt in nsfw_pseudo_file_extensions if fmt not in IMG_FORMATS]
if missing_formats:
    raise ValueError(
        f'Please add {", ".join(nsfw_pseudo_file_extensions)} to IMG_FORMATS in yoloNSFWTraining virtual environment file '
        r'Lib\site-packages\ultralytics\data\utils.py'
    )

# === CONFIGURATION ===
DATASET_URL = "https://archive.org/download/NudeNet_classifier_dataset_v1/NudeNet_Classifier_train_data_x320.zip"
OLD_NSFW_MODEL_URL = "https://github.com/notAI-tech/NudeNet/releases/download/v3.4-weights/640m.pt"
YOLOV11_MODEL = "yolo11n.pt" # Available_Models: "yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt"
CONFIDENCE_THRESHOLD = 0.25
NMS_THRESHOLD = 0.7
PATIENCE = 5
BATCH_SIZE = 128
IMG_SIZE = 320
WORKERS = 16
DEGREES = 180
SHEAR = 20
FLIPUD = .05
FLIPLR = .05
BGR = .05
# === CONFIGURATION ===

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DATASET_PATH = Path("dataset.bin")
EXTRACTED_FOLDER = Path("NSFW_dataset_extracted")
OLD_NSFW_MODEL_PATH = Path(Path(OLD_NSFW_MODEL_URL).name)

LABELS = [
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
    "BUTTOCKS_COVERED"
]

IGNORED_LABELS = [
    "FACE_FEMALE",
    "FEET_EXPOSED",
    "BELLY_COVERED",
    "FEET_COVERED",
    "ARMPITS_COVERED",
    "FACE_MALE"
]

def download_dataset():
    if not DATASET_PATH.exists():
        logging.info(f"Downloading dataset from {DATASET_URL}...")
        response = requests.get(DATASET_URL, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        chunk_size = 8192
        num_bars = total_size // chunk_size if total_size else None
        with DATASET_PATH.open("wb") as f, tqdm(
            desc="Downloading dataset",
            total=num_bars,
            unit='KB',
            unit_scale=True,
            leave=False
        ) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(1)
        logging.info(f"Dataset saved as {DATASET_PATH}.")
    else:
        logging.info(f"Dataset already exists: {DATASET_PATH}")

def extract_dataset():
    download_dataset()
    if not EXTRACTED_FOLDER.exists():
        EXTRACTED_FOLDER.mkdir(parents=True, exist_ok=True)

        counter = 0
        allowed_folders = {"training", "validation", "testing"}
        with zipfile.ZipFile(DATASET_PATH, 'r') as zip_ref:
            for file in tqdm(zip_ref.namelist(), desc="Extracting dataset", unit="file"):
                file_path = Path(file)

                if file.endswith("/"):
                    continue

                if file_path.parts[0] == "nude_sexy_safe_v1_x320":
                    file_path = Path(*file_path.parts[1:])

                if len(file_path.parts) < 2:
                    continue

                if file_path.parts[0] not in allowed_folders:
                    continue

                if file_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    continue

                dest_dir = EXTRACTED_FOLDER / file_path.parent               
                dest = dest_dir / f"{counter}{file_path.suffix}_nsfw"
                counter += 1
                if dest.exists():
                    continue

                dest_dir.mkdir(parents=True, exist_ok=True)

                with zip_ref.open(file) as source, open(dest.resolve(), 'wb') as target:
                    shutil.copyfileobj(source, target)

        logging.info(f"Dataset extracted to {EXTRACTED_FOLDER}")
    else:
        logging.info(f"Dataset already extracted to {EXTRACTED_FOLDER}")

def download_old_nsfw_model():
    if not OLD_NSFW_MODEL_PATH.exists():
        logging.info(f"Model download from GitHub requires authentication. Please download the model through your web browser manually by visiting: {OLD_NSFW_MODEL_URL}")
        logging.info(f"After downloading, save it as {OLD_NSFW_MODEL_PATH.resolve()}")
        while not OLD_NSFW_MODEL_PATH.exists():
            time.sleep(1)
    else:
        logging.info(f"Model already exists: {str(OLD_NSFW_MODEL_PATH)}")

def get_next_multiple_of_32(image_path):
    from PIL import Image
    with Image.open(image_path) as img:
        width, height = img.size
    max_dim = max(width, height)
    return ((max_dim + 31) // 32) * 32
        
def process_images():
    download_old_nsfw_model()
    model = YOLO(str(OLD_NSFW_MODEL_PATH))
    
    images_to_process = [file for file in EXTRACTED_FOLDER.rglob("*")
                         if file.is_file() and file.suffix != ".txt"]
    
    for image_path in tqdm(images_to_process, desc="Processing images", unit="file"):
        txt_output_path = image_path.with_suffix(".txt")
        if txt_output_path.exists():
            continue
            
        try:
            results = model.predict(str(image_path), conf=CONFIDENCE_THRESHOLD, iou=NMS_THRESHOLD, agnostic_nms=True, imgsz=get_next_multiple_of_32(image_path))
            filtered_boxes = []
            for result in results:
                for box in result.boxes:
                    try:
                        x,y,w,h = box.xywhn[0].tolist()
                        class_id = box.cls[0].tolist()
                        label = LABELS[int(class_id)]
                        if label not in IGNORED_LABELS:
                            filtered_boxes.append((label, x, y, w, h))
                    except Exception as e:
                        pass
            if not filtered_boxes:
                if "safe" in txt_output_path.parts:
                    txt_output_path.touch()
                else:
                    logging.error(f"Skipping nude image file with no detections {image_path}")
                    image_path.unlink()
            else:
                if "safe" in txt_output_path.parts:
                    logging.error(f"Skipping safe image file with detections {image_path}")
                    image_path.unlink()
                else:
                    with txt_output_path.open("w") as f:
                        for label, centerX, centerY, width, height in filtered_boxes:                            
                            f.write(f"{LABELS.index(label)} {centerX:.6f} {centerY:.6f} {width:.6f} {height:.6f}\n")
                            
        except Exception as e:
            logging.error(f"Skipping invalid image file {image_path}: {e}")
            image_path.unlink()
            continue
    
def download_yolo11_model(model_name):
    url = f"https://github.com/ultralytics/assets/releases/download/v8.3.0/{model_name}"
    model_path = Path(model_name)
    if not model_path.exists():
        logging.info(f"Downloading {model_name} from {url}...")
        response = requests.get(url, stream=True)
        response.raise_for_status() 
        with model_path.open("wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        logging.info(f"Model {model_name} downloaded successfully.")
    else:
        logging.info(f"Model already exists: {model_path}")

def create_data_yaml_file():
    names_yaml = ""
    for index, label in enumerate(LABELS):
        names_yaml += f"  {index}: {label}\n"
    data_yaml = f"""path: ./{str(EXTRACTED_FOLDER)}
train: {Path('NSFW_dataset_extracted/training').resolve()}
val: {Path('NSFW_dataset_extracted/validation').resolve()}
test: {Path('NSFW_dataset_extracted/testing').resolve()}

# Classes
names:
{names_yaml}"""
    data_yaml_path = Path("data.yaml")
    with data_yaml_path.open("w") as f:
        f.write(data_yaml)
    logging.info("data.yaml created.")

def train_yolo11():
    extract_dataset()
    process_images()
    download_yolo11_model(YOLOV11_MODEL)
    create_data_yaml_file()
    valid_class_indices = [i for i, label in enumerate(LABELS) if label not in IGNORED_LABELS]
    classes_str = "[" + ", ".join(map(str, valid_class_indices)) + "]"
    logging.info("Starting YOLOv11 training...")

    result = subprocess.run([
        "yolo",
        "task=detect",
        "mode=train",
        f"model={YOLOV11_MODEL}",
        f"data={Path('data.yaml').resolve()}",
        f"patience={PATIENCE}",
        f"batch={BATCH_SIZE}",
        f"imgsz={IMG_SIZE}",
        f"workers={WORKERS}",
        f"resume={'True' if Path('runs').exists() else 'False'}",
        f"classes={classes_str}",
        f"degrees={DEGREES}",
        f"shear={SHEAR}",
        f"flipud={FLIPUD}",
        f"fliplr={FLIPLR}",
        f"bgr={BGR}",
    ])
    if result.returncode != 0:
        logging.error("YOLO training failed.")

if __name__ == "__main__":
    train_yolo11()
