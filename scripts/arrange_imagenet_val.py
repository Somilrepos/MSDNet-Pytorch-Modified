#!/usr/bin/env python3

import argparse
import os
import shutil
import xml.etree.ElementTree as ET


def _find_image_path(images_dir, stem):
    for ext in (".JPEG", ".jpg", ".jpeg", ".png"):
        cand = os.path.join(images_dir, stem + ext)
        if os.path.isfile(cand):
            return cand
    return None


def _get_first_class_name(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    obj = root.find("object")
    if obj is None:
        return None
    name = obj.find("name")
    if name is None:
        return None
    return name.text


def main():
    parser = argparse.ArgumentParser(
        description="Arrange ImageNet val images into class folders using XML annotations."
    )
    parser.add_argument("--images-dir", required=True, help="Path to val images directory.")
    parser.add_argument("--ann-dir", required=True, help="Path to val XML annotations directory.")
    parser.add_argument("--out-dir", required=True, help="Output directory for class folders.")
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move images instead of copying.",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    xml_files = [f for f in os.listdir(args.ann_dir) if f.endswith(".xml")]
    xml_files.sort()

    missing_images = 0
    missing_class = 0

    for xml_file in xml_files:
        xml_path = os.path.join(args.ann_dir, xml_file)
        class_name = _get_first_class_name(xml_path)
        if not class_name:
            missing_class += 1
            continue

        stem = os.path.splitext(xml_file)[0]
        img_path = _find_image_path(args.images_dir, stem)
        if img_path is None:
            missing_images += 1
            continue

        class_dir = os.path.join(args.out_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        dst_path = os.path.join(class_dir, os.path.basename(img_path))

        if args.move:
            shutil.move(img_path, dst_path)
        else:
            shutil.copy2(img_path, dst_path)

    print("Done.")
    print("Missing class in XML:", missing_class)
    print("Missing images:", missing_images)


if __name__ == "__main__":
    main()
