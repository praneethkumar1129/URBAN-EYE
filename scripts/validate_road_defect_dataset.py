from pathlib import Path


DATASET_PATH = Path("datasets/urban_eye_road_defects")

VALID_CLASSES = {0, 1, 2, 3}

SPLITS = ["train", "valid", "test"]


def validate_split(split_name: str):

    images_dir = DATASET_PATH / split_name / "images"
    labels_dir = DATASET_PATH / split_name / "labels"

    image_files = {
        file.stem: file
        for file in images_dir.iterdir()
        if file.is_file()
    }

    label_files = {
        file.stem: file
        for file in labels_dir.glob("*.txt")
    }

    missing_labels = []
    missing_images = []
    invalid_labels = []

    class_counts = {
        0: 0,
        1: 0,
        2: 0,
        3: 0,
    }

    total_annotations = 0

    for image_stem in image_files:

        if image_stem not in label_files:
            missing_labels.append(image_stem)

    for label_stem in label_files:

        if label_stem not in image_files:
            missing_images.append(label_stem)

    for label_stem, label_path in label_files.items():

        lines = label_path.read_text(
            encoding="utf-8"
        ).strip().splitlines()

        for line_number, line in enumerate(lines, start=1):

            parts = line.split()

            if len(parts) != 5:

                invalid_labels.append(
                    f"{label_path} "
                    f"(line {line_number}: "
                    f"expected 5 values)"
                )

                continue

            try:

                class_id = int(parts[0])

                coordinates = [
                    float(value)
                    for value in parts[1:]
                ]

            except ValueError:

                invalid_labels.append(
                    f"{label_path} "
                    f"(line {line_number}: "
                    f"non-numeric value)"
                )

                continue

            if class_id not in VALID_CLASSES:

                invalid_labels.append(
                    f"{label_path} "
                    f"(line {line_number}: "
                    f"invalid class {class_id})"
                )

                continue

            if any(
                value < 0 or value > 1
                for value in coordinates
            ):

                invalid_labels.append(
                    f"{label_path} "
                    f"(line {line_number}: "
                    f"coordinates outside 0-1)"
                )

                continue

            class_counts[class_id] += 1

            total_annotations += 1

    return {
        "split": split_name,
        "images": len(image_files),
        "labels": len(label_files),
        "annotations": total_annotations,
        "missing_labels": missing_labels,
        "missing_images": missing_images,
        "invalid_labels": invalid_labels,
        "class_counts": class_counts,
    }


def main():

    print(
        "\n=== URBAN-EYE DATASET VALIDATION ===\n"
    )

    all_valid = True

    total_class_counts = {
        0: 0,
        1: 0,
        2: 0,
        3: 0,
    }

    for split in SPLITS:

        result = validate_split(split)

        print(
            f"--- {split.upper()} ---"
        )

        print(
            f"Images       : "
            f"{result['images']}"
        )

        print(
            f"Labels       : "
            f"{result['labels']}"
        )

        print(
            f"Annotations  : "
            f"{result['annotations']}"
        )

        print("\nClass distribution:")

        for class_id, count in (
            result["class_counts"].items()
        ):

            print(
                f"  Class {class_id}: {count}"
            )

            total_class_counts[class_id] += count

        print()

        if result["missing_labels"]:

            all_valid = False

            print(
                f"Missing labels: "
                f"{len(result['missing_labels'])}"
            )

        else:

            print(
                "Missing labels: 0"
            )

        if result["missing_images"]:

            all_valid = False

            print(
                f"Missing images: "
                f"{len(result['missing_images'])}"
            )

        else:

            print(
                "Missing images: 0"
            )

        if result["invalid_labels"]:

            all_valid = False

            print(
                f"Invalid labels: "
                f"{len(result['invalid_labels'])}"
            )

            print(
                "\nFirst 10 errors:"
            )

            for error in (
                result["invalid_labels"][:10]
            ):

                print(
                    f"  {error}"
                )

        else:

            print(
                "Invalid labels: 0"
            )

        print()

    print(
        "=== TOTAL CLASS DISTRIBUTION ==="
    )

    class_names = {
        0: "longitudinal_crack",
        1: "transverse_crack",
        2: "alligator_crack",
        3: "road_surface_damage",
    }

    for class_id, count in (
        total_class_counts.items()
    ):

        print(
            f"Class {class_id} "
            f"({class_names[class_id]}): "
            f"{count}"
        )

    print()

    if all_valid:

        print(
            "DATASET VALIDATION: PASSED"
        )

    else:

        print(
            "DATASET VALIDATION: FAILED"
        )


if __name__ == "__main__":
    main()