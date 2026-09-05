import csv
from pathlib import Path

import cv2
from ultralytics import YOLO


MODEL_PATH = "yolo11n.pt"
INPUT_VIDEO = "datasets/sample_video.mp4"
OUTPUT_CSV = "evidence/tracking/vehicle_tracks.csv"

VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


def extract_track_data() -> None:
    """Extract vehicle tracking information into a CSV file."""

    output_path = Path(OUTPUT_CSV)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(INPUT_VIDEO)

    if not cap.isOpened():
        raise RuntimeError(
            f"Unable to open video: {INPUT_VIDEO}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_index = 0
    rows = []

    print("\n=== URBAN-EYE TRACK DATA EXTRACTION ===")
    print(f"Input : {INPUT_VIDEO}")
    print(f"Output: {OUTPUT_CSV}")
    print("Tracker: ByteTrack")

    while True:
        success, frame = cap.read()

        if not success:
            break

        results = model.track(
            source=frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=0.25,
            device="cpu",
            classes=list(VEHICLE_CLASSES.keys()),
            verbose=False,
        )

        result = results[0]

        if result.boxes.id is not None:

            boxes = result.boxes

            track_ids = boxes.id.int().cpu().tolist()
            class_ids = boxes.cls.int().cpu().tolist()
            confidences = boxes.conf.cpu().tolist()
            coordinates = boxes.xyxy.cpu().tolist()

            for track_id, class_id, confidence, xyxy in zip(
                track_ids,
                class_ids,
                confidences,
                coordinates,
            ):
                x1, y1, x2, y2 = xyxy

                rows.append(
                    {
                        "frame_index": frame_index,
                        "timestamp_seconds": frame_index / fps,
                        "track_id": track_id,
                        "vehicle_class": VEHICLE_CLASSES[class_id],
                        "confidence": round(confidence, 4),
                        "x1": round(x1, 2),
                        "y1": round(y1, 2),
                        "x2": round(x2, 2),
                        "y2": round(y2, 2),
                    }
                )

        frame_index += 1

        if frame_index % 30 == 0:
            print(f"Processed frames: {frame_index}")

    cap.release()

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:

        fieldnames = [
            "frame_index",
            "timestamp_seconds",
            "track_id",
            "vehicle_class",
            "confidence",
            "x1",
            "y1",
            "x2",
            "y2",
        ]

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    unique_tracks = {
        row["track_id"]
        for row in rows
    }

    print("\n=== TRACK DATA COMPLETE ===")
    print(f"Frames processed : {frame_index}")
    print(f"Track records    : {len(rows)}")
    print(f"Unique vehicles  : {len(unique_tracks)}")
    print(f"CSV output       : {OUTPUT_CSV}")


if __name__ == "__main__":
    extract_track_data()