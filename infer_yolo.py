import argparse
from ultralytics import YOLO

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="./results/ship-detection/weights/best.pt",
                        help="Path to trained model weights")
    parser.add_argument("--image", type=str, required=True,
                        help="Path to image for inference")
    return parser.parse_args()

def main():
    args = get_args()
    model = YOLO(args.model)
    results = model.predict(source=args.image, save=True)
    print("✅ Inference complete. Results saved to:", results[0].save_dir)

if __name__ == "__main__":
    main()
