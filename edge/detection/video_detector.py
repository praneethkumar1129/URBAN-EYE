import cv2
from pathlib import Path

from ultralytics import YOLO


MODEL_PATH = "yolo11n.pt"
INPUT_VIDEO = "datasets/sample_video.mp4"
OUTPUT_VIDEO = "evidence/detections/vehicle_detection.mp4"

VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


def detect_video() -> None:
    """Run YOLO vehicle detection on the complete video."""

    output_path = Path(OUTPUT_VIDEO)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(INPUT_VIDEO)

    if not cap.isOpened():
        raise RuntimeError(
            f"Unable to open video: {INPUT_VIDEO}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height),
    )

    frame_index = 0
    total_detections = 0

    print("\n=== URBAN-EYE VIDEO VEHICLE DETECTION ===")
    print(f"Input : {INPUT_VIDEO}")
    print(f"Output: {OUTPUT_VIDEO}")
    print(f"Resolution: {width} x {height}")
    print(f"FPS: {fps:.2f}")

    while True:
        success, frame = cap.read()

        if not success:
            break

        results = model.predict(
            source=frame,
            conf=0.25,
            device="cpu",
            classes=list(VEHICLE_CLASSES.keys()),
            verbose=False,
        )

        result = results[0]

        frame_detections = len(result.boxes)
        total_detections += frame_detections

        annotated_frame = result.plot()

        writer.write(annotated_frame)

        frame_index += 1

        if frame_index % 30 == 0:
            print(
                f"Processed frames: {frame_index} | "
                f"Current detections: {frame_detections}"
            )

    cap.release()
    writer.release()

    print("\n=== DETECTION COMPLETE ===")
    print(f"Frames processed   : {frame_index}")
    print(f"Total detections   : {total_detections}")
    print(f"Output video       : {OUTPUT_VIDEO}")


if __name__ == "__main__":
    detect_video()