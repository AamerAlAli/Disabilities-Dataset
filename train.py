# train.py

import os
import argparse
import tensorflow as tf
from PIL import Image

# Optional imports for TF object detection models
try:
    from object_detection.utils import config_util
    from object_detection.builders import model_builder
except ImportError:
    print("Please install the TensorFlow Object Detection API.")

# Detectron2 support
try:
    import detectron2
    from detectron2.engine import DefaultTrainer
    from detectron2.config import get_cfg
    from detectron2.model_zoo import get_config_file, get_checkpoint_url
except ImportError:
    print("Detectron2 not available. Please install it to use RetinaNet from Detectron2.")


def resize_with_aspect_ratio(image, target_size):
    """
    Resize an image while maintaining aspect ratio.

    Args:
        image (PIL.Image): Input image.
        target_size (int): Target size for the shorter side.

    Returns:
        PIL.Image: Resized image.
    """
    width, height = image.size
    if width < height:
        new_width = target_size
        new_height = int(target_size * height / width)
    else:
        new_height = target_size
        new_width = int(target_size * width / height)
    return image.resize((new_width, new_height), Image.ANTIALIAS)


def load_tensorflow_model(model_name):
    print(f"Loading TensorFlow model: {model_name}")

    if model_name.lower() == "centernet":
        model_dir = "models/centernet"
    elif model_name.lower() == "efficientdet":
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
                        choices=["centernet", "efficientdet", "fasterrcnn", "ssd", "retinanet"],
                        help="Model to train")
    args = parser.parse_args()

    if args.model == "retinanet":
        train_detectron_retinanet()
    else:
        train_tensorflow_model(args.model)


if __name__ == "__main__":
    main()
