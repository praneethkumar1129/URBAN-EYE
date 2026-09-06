from pathlib import Path
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_CONFIG = (
    PROJECT_ROOT
    / "datasets"
    / "urban_eye_road_defects"
    / "data.yaml"
)

OUTPUT_DIR = PROJECT_ROOT / "models" / "road_defect"


def main():

    print("\n=== URBAN-EYE ROAD DEFECT MODEL TRAINING ===\n")

    print(f"Dataset config : {DATASET_CONFIG}")
    print(f"Output folder  : {OUTPUT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = YOLO("yolo11n.pt")

    results = model.train(

        data=str(DATASET_CONFIG),

        epochs=50,

        imgsz=640,

        batch=16,

        device=0,

        workers=4,

        project=str(OUTPUT_DIR),

        name="yolo11n_road_defect",

        exist_ok=True,

        pretrained=True,

        patience=10,

        save=True,

        plots=True,
    )

    print("\n=== TRAINING COMPLETE ===")

    return results


if __name__ == "__main__":
    main()