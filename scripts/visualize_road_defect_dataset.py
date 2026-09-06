from pathlib import Path
import random

import cv2


DATASET_PATH = Path("datasets/urban_eye_road_defects")

SPLIT = "train"

OUTPUT_DIR = Path(
    "evidence/dataset_validation"
)

NUM_IMAGES = 12


CLASS_NAMES = {
    0: "longitudinal_crack",
    1: "transverse_crack",
    2: "alligator_crack",
    3: "road_surface_damage",
}


def load_labels(label_path):

    labels = []

    if not label_path.exists():
        return labels

    with open(
        label_path,
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            parts = line.split()

            class_id = int(parts[0])

            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

            labels.append(
                (
                    class_id,
                    x_center,
                    y_center,
                    width,
                    height,
                )
            )

    return labels


def draw_boxes(image, labels):

    height, width = image.shape[:2]

    for (
        class_id,
        x_center,
        y_center,
        box_width,
        box_height,
    ) in labels:

        x_center *= width
        y_center *= height
        box_width *= width
        box_height *= height

        x1 = int(
            x_center - box_width / 2
        )

        y1 = int(
            y_center - box_height / 2
        )

        x2 = int(
            x_center + box_width / 2
        )

        y2 = int(
            y_center + box_height / 2
        )

        class_name = CLASS_NAMES[
            class_id
        ]

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            3,
        )

        cv2.putText(
            image,
            class_name,
            (x1, max(30, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    return image


def main():

    print(
        "\n=== URBAN-EYE VISUAL DATASET VALIDATION ===\n"
    )

    images_dir = (
        DATASET_PATH /
        SPLIT /
        "images"
    )

    labels_dir = (
        DATASET_PATH /
        SPLIT /
        "labels"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    image_files = [
        path
        for path in images_dir.iterdir()
        if path.is_file()
    ]

    positive_images = []

    for image_path in image_files:

        label_path = (
            labels_dir /
            f"{image_path.stem}.txt"
        )

        labels = load_labels(
            label_path
        )

        if labels:
            positive_images.append(
                image_path
            )

    print(
        f"Available images: "
        f"{len(image_files)}"
    )

    print(
        f"Positive images: "
        f"{len(positive_images)}"
    )

    if len(
        positive_images
    ) < NUM_IMAGES:

        raise ValueError(
            "Not enough positive images "
            "for visualization."
        )

    selected_images = random.sample(
        positive_images,
        NUM_IMAGES,
    )

    print()

    for index, image_path in enumerate(
        selected_images,
        start=1,
    ):

        label_path = (
            labels_dir /
            f"{image_path.stem}.txt"
        )

        image = cv2.imread(
            str(image_path)
        )

        if image is None:

            print(
                f"Skipping unreadable image: "
                f"{image_path.name}"
            )

            continue

        labels = load_labels(
            label_path
        )

        image = draw_boxes(
            image,
            labels,
        )

        output_path = (
            OUTPUT_DIR /
            f"validation_{index:02d}.jpg"
        )

        cv2.imwrite(
            str(output_path),
            image,
        )

        print(
            f"{index:02d}. "
            f"{image_path.name} "
            f"→ {len(labels)} annotation(s)"
        )

    print(
        "\n=== VISUAL VALIDATION COMPLETE ==="
    )

    print(
        f"Output directory: "
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()