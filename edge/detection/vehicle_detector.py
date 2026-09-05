from pathlib import Path

from ultralytics import YOLO


MODEL_PATH = "yolo11n.pt"
INPUT_IMAGE = "evidence/test_frames/frame_000000.jpg"
OUTPUT_DIR = Path("evidence/detections")


def detect_vehicles() -> None:
    """Run YOLO object detection on one test frame."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading YOLO model...")

    model = YOLO(MODEL_PATH)

    print("Running inference...")

    results = model.predict(
        source=INPUT_IMAGE,
        conf=0.25,
        device="cpu",
        verbose=False,
    )

    result = results[0]

    output_path = OUTPUT_DIR / "frame_000000_detected.jpg"

    result.save(filename=str(output_path))

    print("\n=== URBAN-EYE VEHICLE DETECTION ===")
    print(f"Input : {INPUT_IMAGE}")
    print(f"Output: {output_path}")
    print(f"Detections: {len(result.boxes)}")

    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        class_name = result.names[class_id]

        print(
            f"  - {class_name}: "
            f"{confidence:.2f}"
        )


if __name__ == "__main__":
    detect_vehicles()