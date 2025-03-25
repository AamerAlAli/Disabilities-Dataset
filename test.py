# test.py

import argparse
import os
from PIL import Image
import numpy as np
import cv2

try:
    from ultralytics import YOLO
except ImportError:
    print("Ultralytics YOLO not installed. Install with: pip install ultralytics")

try:
    import tensorflow as tf
except ImportError:
    print("TensorFlow not available.")

try:
    import detectron2
    from detectron2.engine import DefaultPredictor
    from detectron2.config import get_cfg
    from detectron2.model_zoo import get_config_file, get_checkpoint_url
except ImportError:
    print("Detectron2 not available. Please install it to test RetinaNet from Detectron2.")


def test_yolo11(model_path, image_path):
    model = YOLO(model_path)
    results = model(image_path)
    for r in results:
        print(r.names)
        r.show()  # optional visualization


def load_image(image_path, target_size=None):
    image = Image.open(image_path).convert("RGB")
    if target_size:
        image = image.resize(target_size)
    return np.array(image)


def test_tensorflow_model(model_dir, image_path):
    print(f"Loading TensorFlow SavedModel from {model_dir}")
    detect_fn = tf.saved_model.load(model_dir)

    image_np = load_image(image_path)
    input_tensor = tf.convert_to_tensor(image_np)[tf.newaxis, ...]

    detections = detect_fn(input_tensor)
    print("Detections:", detections)


def test_detectron_retinanet(image_path):
    cfg = get_cfg()
    config_file = "COCO-Detection/retinanet_R_50_FPN_1x.yaml"
    cfg.merge_from_file(get_config_file(config_file))
    cfg.MODEL.WEIGHTS = get_checkpoint_url(config_file)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    predictor = DefaultPredictor(cfg)

    image = cv2.imread(image_path)
    outputs = predictor(image)
    print("Detections:", outputs["instances"].to("cpu"))


def main():
    parser = argparse.ArgumentParser(description="Test trained detection models on a single image")
    parser.add_argument("--model", type=str, required=True,
                        choices=["retinanet", "tensorflow", "yolo11"], help="Model type to use")
    parser.add_argument("--model_path", type=str, help="Path to model weights or SavedModel directory")
    parser.add_argument("--image_path", type=str, required=True, help="Path to input image")
    args = parser.parse_args()

    if args.model == "retinanet":
        test_detectron_retinanet(args.image_path)
    elif args.model == "tensorflow":
        if not args.model_path:
            raise ValueError("--model_path is required for TensorFlow models.")
        test_tensorflow_model(args.model_path, args.image_path)
    elif args.model == "yolo11":
        if not args.model_path:
            raise ValueError("--model_path is required for YOLOv11.")
        test_yolo11(args.model_path, args.image_path)


if __name__ == "__main__":
    main()
