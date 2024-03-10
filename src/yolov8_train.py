import argparse
from ultralytics import YOLO

def main(yaml_path):
    # Load a model
    model = YOLO('yolov8n-seg.pt')  # load a pretrained model (recommended for training)

    # Train the model
    results = model.train(data=yaml_path, epochs=100, imgsz=256)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train YOLO model.')
    parser.add_argument('yaml_path', type=str, help='Path to YAML file')
    args = parser.parse_args()
    main(args.yaml_path)
