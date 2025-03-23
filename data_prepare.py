# data_prepare.py

import os
import argparse
from PIL import Image
from sklearn.model_selection import StratifiedKFold
import shutil


def resize_with_aspect_ratio(image, target_size):
    width, height = image.size
    if width < height:
        new_width = target_size
        new_height = int(target_size * height / width)
    else:
        new_height = target_size
        new_width = int(target_size * width / height)
    return image.resize((new_width, new_height), Image.ANTIALIAS)


def save_image(image, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    image.save(save_path)


def prepare_kfold_dataset(data_dir, output_dir, target_size=512, n_splits=10):
    class_names = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    print(f"Found classes: {class_names}")

    all_images = []
    all_labels = []

    for class_name in class_names:
        class_path = os.path.join(data_dir, class_name)
        images = [os.path.join(class_path, img) for img in os.listdir(class_path)
                  if img.lower().endswith((".jpg", ".jpeg", ".png"))]
        all_images.extend(images)
        all_labels.extend([class_name] * len(images))

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for fold, (train_idx, test_idx) in enumerate(skf.split(all_images, all_labels)):
        print(f"Processing fold {fold}")
        for split_name, indices in zip(["train", "test"], [train_idx, test_idx]):
            for idx in indices:
                img_path = all_images[idx]
                class_name = all_labels[idx]
                image = Image.open(img_path).convert("RGB")
                resized = resize_with_aspect_ratio(image, target_size)
                filename = os.path.basename(img_path)
                save_path = os.path.join(output_dir, f"fold_{fold}", split_name, class_name, filename)
                save_image(resized, save_path)

        print(f"Fold {fold} completed.")


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset with resized images and 10-fold cross-validation split")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory with class subfolders")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save processed dataset")
    parser.add_argument("--target_size", type=int, default=512, help="Target size for the shorter side")
    parser.add_argument("--n_splits", type=int, default=10, help="Number of folds for cross-validation")
    args = parser.parse_args()

    prepare_kfold_dataset(args.data_dir, args.output_dir, args.target_size, args.n_splits)


if __name__ == "__main__":
    main()