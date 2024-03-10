def main():
    from ultralytics import YOLO

    # Load a model
    model = YOLO('yolov8n-seg.pt')  # load a pretrained model (recommended for training)

    # Train the model
    path_yaml = "/home/mariopasc/Python/Projects/BSC_final/epilepsy-displasia-focal-segmentation/text-info-files/t2flair-study/config.yaml"
    results = model.train(data=path_yaml, epochs=100, imgsz=256)

if __name__ == "__main__":
    main()