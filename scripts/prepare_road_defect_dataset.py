from pathlib import Path
import shutil
import yaml


# ============================================================
# URBAN-EYE ROAD DEFECT DATASET PREPARATION
# ============================================================

SOURCE_DATASET = Path("datasets/rdd2022_india")
OUTPUT_DATASET = Path("datasets/urban_eye_road_defects")


# Original RDD2022-India class IDs -> URBAN-EYE class IDs
CLASS_MAPPING = {
    0: 0,  # D00 -> Longitudinal Crack
    1: 0,  # D01 -> Longitudinal Crack

    3: 1,  # D10 -> Transverse Crack
    4: 1,  # D11 -> Transverse Crack

    5: 2,  # D20 -> Alligator Crack

    6: 3,  # D40 -> Road Surface Damage
}


CLASS_NAMES = [
    "longitudinal_crack",
    "transverse_crack",
    "alligator_crack",
    "road_surface_damage",
]


def prepare_split(split_name: str):
    """
    Prepare one dataset split:
    train / valid / test
    """

    source_images = SOURCE_DATASET / split_name / "images"
    source_labels = SOURCE_DATASET / split_name / "labels"

    output_images = OUTPUT_DATASET / split_name / "images"
    output_labels = OUTPUT_DATASET / split_name / "labels"

    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)

    total_images = 0
    positive_images = 0
    retained_annotations = 0

    image_files = list(source_images.glob("*"))

    for image_path in image_files:

        if not image_path.is_file():
            continue

        total_images += 1

        label_path = source_labels / f"{image_path.stem}.txt"

        # Copy image
        output_image_path = output_images / image_path.name
        shutil.copy2(image_path, output_image_path)

        new_labels = []

        # Read label file if it exists
        if label_path.exists():

            with open(label_path, "r", encoding="utf-8") as file:

                for line in file:

                    line = line.strip()

                    if not line:
                        continue

                    parts = line.split()

                    original_class_id = int(parts[0])

                    # Keep only useful URBAN-EYE road defect classes
                    if original_class_id in CLASS_MAPPING:

                        new_class_id = CLASS_MAPPING[original_class_id]

                        parts[0] = str(new_class_id)

                        new_labels.append(" ".join(parts))

                        retained_annotations += 1

        # Create output label file
        output_label_path = output_labels / f"{image_path.stem}.txt"

        with open(output_label_path, "w", encoding="utf-8") as file:

            for label in new_labels:
                file.write(label + "\n")

        if new_labels:
            positive_images += 1

    return {
        "split": split_name,
        "total_images": total_images,
        "positive_images": positive_images,
        "negative_images": total_images - positive_images,
        "retained_annotations": retained_annotations,
    }


def create_data_yaml():

    data = {
        "path": str(OUTPUT_DATASET.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(CLASS_NAMES),
        "names": CLASS_NAMES,
    }

    yaml_path = OUTPUT_DATASET / "data.yaml"

    with open(yaml_path, "w", encoding="utf-8") as file:
        yaml.dump(
            data,
            file,
            sort_keys=False,
            default_flow_style=False
        )

    return yaml_path


def main():

    print("\n=== URBAN-EYE ROAD DEFECT DATASET PREPARATION ===\n")

    # Safety check
    if not SOURCE_DATASET.exists():
        raise FileNotFoundError(
            f"Source dataset not found: {SOURCE_DATASET}"
        )

    results = []

    for split in ["train", "valid", "test"]:

        print(f"Processing {split}...")

        result = prepare_split(split)

        results.append(result)

    yaml_path = create_data_yaml()

    print("\n=== DATASET PREPARATION COMPLETE ===\n")

    total_images = 0
    total_positive = 0
    total_annotations = 0

    for result in results:

        print(f"{result['split'].upper()}")

        print(
            f"  Total images         : "
            f"{result['total_images']}"
        )

        print(
            f"  Positive images      : "
            f"{result['positive_images']}"
        )

        print(
            f"  Negative images      : "
            f"{result['negative_images']}"
        )

        print(
            f"  Retained annotations : "
            f"{result['retained_annotations']}"
        )

        print()

        total_images += result["total_images"]
        total_positive += result["positive_images"]
        total_annotations += result["retained_annotations"]

    print("TOTAL")

    print(f"  Total images         : {total_images}")

    print(f"  Positive images      : {total_positive}")

    print(
        f"  Negative images      : "
        f"{total_images - total_positive}"
    )

    print(
        f"  Retained annotations : "
        f"{total_annotations}"
    )

    print(f"\nClasses: {CLASS_NAMES}")

    print(f"\nGenerated config: {yaml_path}")


if __name__ == "__main__":
    main()