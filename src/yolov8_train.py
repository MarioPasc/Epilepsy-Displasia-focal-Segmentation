import argparse
from ultralytics import YOLO


def main(yaml_path):
    # Load a model
    model = YOLO(model='yolov8n-seg.pt', task="segment")
    # Train the model
    # Desabilitar aumentación de datos: https://github.com/ultralytics/ultralytics/issues/4812
    model.train(data=yaml_path, epochs=100, imgsz=256,
                save=True, save_period=10,
                name="Yolov8-Train2", verbose=True,
                seed=42, single_cls=True, plots=True,
                augment=False, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, degrees=0.0, translate=0.0,
                scale=0.0, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.0, mosaic=0.0,
                close_mosaid=0.0, mixup=0.0, copy_paste=0.0, auto_augment="", erasing=0.0)
    # Validate model
    model.val(data=yaml_path, imgsz=256, conf=0.001,
              plots=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train YOLO model.')
    parser.add_argument('yaml_path', type=str, help='Path to YAML file')
    args = parser.parse_args()
    main(args.yaml_path)
