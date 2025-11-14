from roboflow import Roboflow
from ultralytics import YOLO
import os
from dotenv import load_dotenv
import torch

def main():
    # optional
    torch.manual_seed(42)

    # load API_key from .env
    load_dotenv()
    API_KEY = os.getenv("ROBOFLOW_API_KEY")

    if not API_KEY:
        raise ValueError("none ROBOFLOW_API_KEY in .env")

    # connect Roboflow & download dataset
    print("connecting Roboflow & download dataset...")

    rf = Roboflow(api_key=API_KEY)
    project = rf.workspace("newp").project("hand-gesture-recognition-ktods")
    version = project.version(1)

    # dowload dataset YOLOv8 
    DATASET_DIR = "scripts/dataset"
    os.makedirs(DATASET_DIR, exist_ok=True)
    dataset = version.download("yolov8", location=DATASET_DIR)

    print(f"downloaded to: {dataset.location}")

    print("start training...")

    model = YOLO("yolov8n.pt")  

    model.train(
        data=os.path.join(dataset.location, "data.yaml"),
        epochs=50,
        imgsz=416,
        batch=4,
        device=0,  # GPU
        patience=10,  # early stopping
        name="hand_gesture_model",
        project="runs/train"
    )

    print("done training")

    # export -> ONNX
    best_model_path = os.path.join("runs", "train", "hand_gesture_model", "weights", "best.pt")

    if os.path.exists(best_model_path):
        print("exporting to ONNX...")
        best_model = YOLO(best_model_path)
        best_model.export(format="onnx")
        print("export ONNX")
    else:
        print("none file best.pt — skip export.")

if __name__ == "__main__":
    main()
