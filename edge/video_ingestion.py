import cv2
from pathlib import Path


def inspect_video(video_path: str) -> None:
    """Inspect basic properties of a video file."""

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    duration = frame_count / fps if fps > 0 else 0

    print("\n=== URBAN-EYE VIDEO INSPECTION ===")
    print(f"Video       : {video_path}")
    print(f"Resolution  : {width} x {height}")
    print(f"FPS         : {fps:.2f}")
    print(f"Frames      : {frame_count}")
    print(f"Duration    : {duration:.2f} seconds")

    cap.release()


def extract_frames(
    video_path: str,
    output_dir: str,
    sample_every: int = 30,
) -> int:
    """Extract one frame every N frames."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    frame_index = 0
    saved_count = 0

    while True:
        success, frame = cap.read()

        if not success:
            break

        if frame_index % sample_every == 0:
            filename = output_path / f"frame_{frame_index:06d}.jpg"
            cv2.imwrite(str(filename), frame)
            saved_count += 1

        frame_index += 1

    cap.release()

    return saved_count


if __name__ == "__main__":
    video = "datasets/sample_video.mp4"

    inspect_video(video)

    saved = extract_frames(
        video_path=video,
        output_dir="evidence/test_frames",
        sample_every=30,
    )

    print(f"\nFrames extracted: {saved}")