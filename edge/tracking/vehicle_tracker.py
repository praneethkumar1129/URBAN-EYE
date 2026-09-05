import cv2
from pathlib import Path

from ultralytics import YOLO


MODEL_PATH = "yolo11n.pt"
INPUT_VIDEO = "datasets/sample_video.mp4"
OUTPUT_VIDEO = "evidence/tracking/vehicle_tracking.mp4"

VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


def track_vehicles() -> None:
    """Track vehicles across video frames using ByteTrack."""

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
    unique_track_ids = set()

    print("\n=== URBAN-EYE VEHICLE TRACKING ===")
    print(f"Input : {INPUT_VIDEO}")
    print(f"Output: {OUTPUT_VIDEO}")
    print(f"Tracker: ByteTrack")
    print(f"Resolution: {width} x {height}")
    print(f"FPS: {fps:.2f}")

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
            track_ids = result.boxes.id.int().cpu().tolist()

            for track_id in track_ids:
                unique_track_ids.add(track_id)

        annotated_frame = result.plot()

        writer.write(annotated_frame)

        frame_index += 1

        if frame_index % 30 == 0:
            current_tracks = 0

            if result.boxes.id is not None:
                current_tracks = len(result.boxes.id)

            print(
                f"Processed frames: {frame_index} | "
                f"Active tracks: {current_tracks} | "
                f"Unique tracks: {len(unique_track_ids)}"
            )

    cap.release()
    writer.release()

    print("\n=== TRACKING COMPLETE ===")
    print(f"Frames processed : {frame_index}")
    print(f"Unique vehicles  : {len(unique_track_ids)}")
    print(f"Output video     : {OUTPUT_VIDEO}")


if __name__ == "__main__":
    track_vehicles()