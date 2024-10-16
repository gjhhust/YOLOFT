import os
import sys
import datetime
import yaml
from ultralytics.models import YOLOFT
import os
from analy_log import extract_best_results
import traceback

def read_yaml(path):
    """Reads and returns the contents of a YAML file"""
    with open(path, 'r') as file:
        return yaml.safe_load(file)

def get_file_name(path):
    return os.path.basename(path).split(".")[0]

def train_model(repeats, model_config_path, pretrain_model, dataset_config_path, training_config_path, batch_size, epochs, img_size, workers, log_dir='./yoloft/logs'):
    # Ensure that the log directory exists
    os.makedirs(log_dir, exist_ok=True)

    # Get the value of CUDA_VISIBLE_DEVICES in the environment variable
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")

    if cuda_visible_devices is not None:
        cuda_devices = cuda_visible_devices.split(',')
        device = [int(id) for id in cuda_devices]
        print(f"CUDA_VISIBLE_DEVICES: {device}")
    else:
        print("No CUDA_VISIBLE_DEVICES set")

    # Creating log file names
    start_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    log_filename = f"{get_file_name(model_config_path)}_{get_file_name(dataset_config_path)}_{get_file_name(training_config_path)}_batch{batch_size}_epochs{epochs}_img_size{img_size}_{start_time}.log"
    log_path = os.path.join(log_dir, log_filename)

    # Open log file
    sys.stdout = open(log_path, 'w')
    sys.stderr = sys.stdout

    print("model_config_path=", model_config_path)
    print("pretrain_model=", pretrain_model)
    print("dataset_config_path=", dataset_config_path)
    print("training_config_path=", training_config_path)
    print("batch_size=", batch_size)
    print("epochs=", epochs)
    print("img_size=", img_size)
    print("device=", device)
    print("workers=", workers)

    print(f"CUDA_VISIBLE_DEVICES set to {cuda_devices}")
    print(f"Starting experiments with the following parameters:")
    print(f"Model Config Path: {model_config_path}")
    print(f"Dataset Config Path: {dataset_config_path}")
    print(f"Training Config Path: {training_config_path}")
    print(f"Batch Size: {batch_size}, Epochs: {epochs}, Image Size: {img_size}, Devices: {device}, Workers: {workers}")

    # Read and print the contents of the configuration file
    model_config = read_yaml(model_config_path)
    dataset_config = read_yaml(dataset_config_path)
    training_config = read_yaml(training_config_path)
    print("Model Config:")
    for key, value in model_config.items():
        print(f"{key}: {value}")

    print("\nDataset Config:")
    for key, value in dataset_config.items():
        print(f"{key}: {value}")

    print("\nTraining Config:")
    for key, value in training_config.items():
        print(f"{key}: {value}")

    for i in range(repeats):
        print(f"\nStarting training session {i+1}")
        try:
            model = YOLOFT(model_config_path).load(pretrain_model)
            results = model.train(data=dataset_config_path, cfg=training_config_path, batch=batch_size, epochs=epochs, imgsz=img_size, device=device, workers=workers)
            print(f"Training session {i+1} completed.")
        except Exception as e:
            print(f"An error occurred during training session {i+1}: {e}")
            traceback.print_exc()
    # Close log file
    sys.stdout.close()
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    print(f"All experiments completed. Logs saved to {log_path}")
    extract_best_results(log_path)




if __name__ == "__main__":
    # use export CUDA_VISIBLE_DEVICES=0,1,2,3 #使用这个命令来控制运行的设备
    repeats = 2
    model_config_path = "config/yoloft_dev/yoloftv2-L-DCN.yaml"
    pretrain_model = "yolov8l.pt"
    dataset_config_path = "config/dataset_dev/Train_minitrain6_Test_minigaode6.yaml"
    training_config_path = "config/train/orige_stream_noval.yaml"
    batch_size = 12*2
    epochs = 21
    img_size = 896
    workers = 6

    train_model(repeats, model_config_path, pretrain_model, dataset_config_path, training_config_path, batch_size, epochs, img_size, workers)