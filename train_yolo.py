import argparse
from ultralytics import YOLO

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="./data/ships.yaml",
                        help="Path to YOLO dataset config YAML")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--output", type=str, default="./results")
    return parser.parse_args()

def main():
    args = get_args()
    model = YOLO("yolov8n.pt")  # Pretrained nano model (fast for hackathon)
    model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        project=args.output,
        name="ship-detection"
    )
    print("✅ Training finished. Results saved in:", args.output)

if __name__ == "__main__":
    main()
