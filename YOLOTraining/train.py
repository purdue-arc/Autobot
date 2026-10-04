import torch
from ultralytics import YOLO

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = YOLO("yolo26n.pt")

    results = model.train(
        data="dataset/data.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        device=device
    )

if __name__ == "__main__":
    main()