from pathlib import Path

from ultralytics import YOLO


MODEL_PATH = "yolo11n.pt"
INPUT_DIR = Path("evidence/test_frames")

VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


def test_vehicle_detection() -> None:
    model = YOLO(MODEL_PATH)

    image_files = sorted(INPUT_DIR.glob("*.jpg"))

    total_detections = 0

    print("\n=== URBAN-EYE VEHICLE DETECTION TEST ===")

    for image_path in image_files:

        results = model.predict(
            source=str(image_path),
            conf=0.25,
            device="cpu",
            classes=list(VEHICLE_CLASSES.keys()),
            verbose=False,
        )

        result = results[0]

        frame_detections = 0

        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            class_name = VEHICLE_CLASSES.get(
                class_id,
                "unknown",
            )

            print(
                f"{image_path.name} | "
                f"{class_name} | "
                f"{confidence:.2f}"
            )

            frame_detections += 1
            total_detections += 1

        print(
            f"{image_path.name} → "
            f"{frame_detections} vehicle(s)"
        )

    print("\n=== SUMMARY ===")
    print(f"Frames tested      : {len(image_files)}")
    print(f"Total detections   : {total_detections}")


if __name__ == "__main__":
    test_vehicle_detection()