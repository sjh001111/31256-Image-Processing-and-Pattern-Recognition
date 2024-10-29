from ultralytics import YOLO
import yaml
import os


def create_dataset_config():
    current_dir = os.getcwd()
    data_yaml = {
        'path': current_dir,
        'train': os.path.join('dataset', 'images', 'train'),
        'val': os.path.join('dataset', 'images', 'val'),
        'test': os.path.join('dataset', 'images', 'test'),
        'names': {
            0: 'license_plate'
        }
    }

    with open('dataset.yaml', 'w') as f:
        yaml.dump(data_yaml, f)


def train():
    model = YOLO('yolo11n.pt')

    results = model.train(
        data='dataset.yaml',
        epochs=50,
        imgsz=640,
        batch=16,
        patience=10,
        device=0,
        workers=8,
        cos_lr=True,
        optimizer='AdamW',
        cache=True
    )


if __name__ == "__main__":
    create_dataset_config()
    train()