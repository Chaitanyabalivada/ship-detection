import os
import argparse
import torch
import torch.nn as nn
from torchvision import transforms, models, datasets
from PIL import Image
from torch.utils.data import DataLoader

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="../data/train",
                        help="Path to dataset (used to load class labels)")
    parser.add_argument("--model", type=str, default="../results/best_model.pth",
                        help="Path to trained model checkpoint")
    parser.add_argument("--image", type=str,
                        help="Path to input image for inference")
    parser.add_argument("--folder", type=str,
                        help="Path to folder of images for batch inference")
    parser.add_argument("--batch", type=int, default=32,
                        help="Batch size for folder inference")
    return parser.parse_args()

def load_model(model_path, num_classes, device):
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def single_inference(model, transform, image_path, classes, device):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)

    predicted_class = classes[pred.item()]
    confidence = conf.item() * 100
    print(f"🎯 {image_path} → {predicted_class} ({confidence:.2f}% confidence)")

def folder_inference(model, transform, folder_path, classes, device, batch_size):
    dataset = datasets.ImageFolder(folder_path, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    correct, total = 0, 0
    results = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            conf, preds = torch.max(probs, 1)

            for i in range(len(preds)):
                img_path, _ = dataset.samples[total + i]
                pred_class = classes[preds[i].item()]
                confidence = conf[i].item() * 100
                results.append((img_path, pred_class, confidence))

            correct += torch.sum(preds == labels).item()
            total += labels.size(0)

    # Print results
    for img_path, pred_class, confidence in results:
        print(f"🎯 {img_path} → {pred_class} ({confidence:.2f}% confidence)")

    # Print overall accuracy if labels exist
    if total > 0:
        acc = correct / total * 100
        print(f"\n✅ Overall Accuracy on folder '{folder_path}': {acc:.2f}%")

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Using device: {device}")

    # Load class labels
    classes = sorted(os.listdir(args.data))
    print(f"✅ Loaded {len(classes)} classes: {classes}")

    # Transforms (same as training)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Load trained model
    model = load_model(args.model, len(classes), device)

    if args.image:
        single_inference(model, transform, args.image, classes, device)
    elif args.folder:
        folder_inference(model, transform, args.folder, classes, device, args.batch)
    else:
        print("⚠️ Please provide either --image <path> or --folder <path>")

if __name__ == "__main__":
    main()
