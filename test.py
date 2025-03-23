# test.py

import os
import argparse
from PIL import Image


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


def process_folder(input_folder, output_folder, target_size):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(input_folder, filename)
            image = Image.open(img_path).convert("RGB")
            resized_image = resize_with_aspect_ratio(image, target_size)
            output_path = os.path.join(output_folder, filename)
            resized_image.save(output_path)
            print(f"Processed: {filename}")


def main():
    parser = argparse.ArgumentParser(description="Test preprocessing by resizing images with preserved aspect ratio")
    parser.add_argument("--input_folder", type=str, required=True, help="Folder with original images")
    parser.add_argument("--output_folder", type=str, required=True, help="Folder to save resized images")
    parser.add_argument("--target_size", type=int, default=512, help="Target size for the shorter side")
    args = parser.parse_args()

    process_folder(args.input_folder, args.output_folder, args.target_size)


if __name__ == "__main__":
    main()
