# crop_image.py

import os
import argparse
from PIL import Image

def center_crop(image, crop_size):
    """
    Center crop the image to the specified size.

    Args:
        image (PIL.Image): The image to crop.
        crop_size (int or tuple): Size to crop (height, width).

    Returns:
        PIL.Image: Cropped image.
    """
    width, height = image.size
    if isinstance(crop_size, int):
        crop_width, crop_height = crop_size, crop_size
    else:
        crop_width, crop_height = crop_size

    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    right = left + crop_width
    bottom = top + crop_height

    return image.crop((left, top, right, bottom))

def crop_folder(input_dir, output_dir, crop_size):
    os.makedirs(output_dir, exist_ok=True)

    for file in os.listdir(input_dir):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(input_dir, file)
            image = Image.open(image_path).convert("RGB")
            cropped = center_crop(image, crop_size)
            save_path = os.path.join(output_dir, file)
            cropped.save(save_path)
            print(f"Cropped and saved: {file}")

def main():
    parser = argparse.ArgumentParser(description="Center crop images in a folder")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory with input images")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save cropped images")
    parser.add_argument("--crop_size", type=int, nargs='+', required=True, help="Crop size (one value or two: width height)")
    args = parser.parse_args()

    crop_size = tuple(args.crop_size) if len(args.crop_size) > 1 else args.crop_size[0]
    crop_folder(args.input_dir, args.output_dir, crop_size)

if __name__ == "__main__":
    main()