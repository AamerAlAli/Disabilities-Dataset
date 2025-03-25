# train.py

import os
import argparse
from PIL import Image
import tensorflow as tf

try:
    from ultralytics import YOLO
except ImportError:
    print("Ultralytics YOLO not installed. Install with: pip install ultralytics")

try:
    from object_detection.utils import config_util
    from object_detection.builders import model_builder
except ImportError:
    print("Please install the TensorFlow Object Detection API.")

try:
    import detectron2
    from detectron2.engine import DefaultTrainer
    from detectron2.config import get_cfg
    from detectron2.model_zoo import get_config_file, get_checkpoint_url
except ImportError:
    print("Detectron2 not available. Please install it to use RetinaNet from Detectron2.")


def resize_with_aspect_ratio(image, target_size):
    width, height = image.size
    if width < height:
        new_width = target_size
        new_height = int(target_size * height / width)
    else:
        new_height = target_size
        new_width = int(target_size * width / height)
    return image.resize((new_width, new_height), Image.ANTIALIAS)


def train_yolo11(model_path="yolov8n.yaml", data_path="data.yaml", epochs=100):
    print("Training YOLOv11 model with Ultralytics...")
    model = YOLO(model_path)
    model.train(data=data_path, epochs=epochs)


def load_tensorflow_model(model_name):
    print(f"Loading TensorFlow model: {model_name}")

    if model_name.lower() == "efficientdet":
        model_dir = "models/efficientdet"
    elif model_name.lower() == "fasterrcnn":
        model_dir = "models/faster_rcnn"
    elif model_name.lower() == "ssd":
        model_dir = "models/ssd"
    else:
        raise ValueError(f"Unsupported TensorFlow model: {model_name}")

    pipeline_config = os.path.join(model_dir, 'pipeline.config')
    model_config = config_util.get_configs_from_pipeline_file(pipeline_config)
    detection_model = model_builder.build(model_config['model'], is_training=True)
    return detection_model


def train_tensorflow_model(model_name):
    model = load_tensorflow_model(model_name)
    print(f"Training {model_name} using TensorFlow... (simulated)")
    # Placeholder: Insert training loop or TF2 model training logic here


def train_detectron_retinanet():
    print("Training RetinaNet using Detectron2...")
    cfg = get_cfg()
    config_file = "COCO-Detection/retinanet_R_50_FPN_1x.yaml"
    cfg.merge_from_file(get_config_file(config_file))
    cfg.MODEL.WEIGHTS = get_checkpoint_url(config_file)
    cfg.OUTPUT_DIR = "./output_retinanet"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


def main():
    parser = argparse.ArgumentParser(description="Train detection models")
    parser.add_argument("--model", type=str, required=True,
                        choices=["yolo11", "efficientdet", "fasterrcnn", "ssd", "retinanet"],
                        help="Model to train")
    parser.add_argument("--yolo_config", type=str, default="yolov8n.yaml", help="YOLOv11 model config")
    parser.add_argument("--yolo_data", type=str, default="data.yaml", help="YOLOv11 data.yaml path")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs for YOLOv11")
    args = parser.parse_args()

    if args.model == "retinanet":
        train_detectron_retinanet()
    elif args.model == "yolo11":
        train_yolo11(args.yolo_config, args.yolo_data, args.epochs)
    else:
        train_tensorflow_model(args.model)


if __name__ == "__main__":
    main()
