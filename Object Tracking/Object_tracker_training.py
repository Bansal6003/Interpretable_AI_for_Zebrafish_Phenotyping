from ultralytics import YOLO
import os

# TEMPORARY workaround - use only if you're aware of the risks
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def main():
    # Load a model
    # model = YOLO("yolo11n.yaml")  # build a new model from YAML
    model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)
    # model = YOLO("yolo11n.yaml").load("yolo11n.pt")  # build from YAML and transfer weights

    # Train the model
    results = model.train(data=r"C:\Users\Pushkar Bansal\Downloads\Fish_track-2.v1i.yolov11\data.yaml", epochs=200, imgsz=640)


if __name__ == "__main__":
    main()