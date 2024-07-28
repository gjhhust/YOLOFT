import os
import sys
import datetime
import yaml
import pandas as pd
from ultralytics.models import YOLOFT,YOLO
from analy_log import extract_best_results


def process_and_save_results(all_results, final_results_file):
    # move the experiment_name column to the first column and the session column to the second column
    cols = all_results.columns.tolist()
    cols.insert(0, cols.pop(cols.index('experiment_name')))
    cols.insert(1, cols.pop(cols.index('session')))
    all_results = all_results[cols]

    # Group and sort by experiment_name and session
    all_results = all_results.sort_values(by=['experiment_name', 'session', 'AP-ALL', 'AP-0-12', 'AP-12-20'], ascending=[True, True, False, False, False])
    
    # Save the processed results to a CSV file
    all_results.to_csv(final_results_file, index=False)
    print(f"All experiment results processed and saved to {final_results_file}")


def read_yaml(path):
    with open(path, 'r') as file:
        return yaml.safe_load(file)

def get_file_name(path):
    return os.path.basename(path).split(".")[0]

def train_model(model_config_dir, repeats,dataset_config_path, training_config_path, batch_size, epochs, img_size, workers, pretrain_model="yolov8l.pt", log_dir='.runs/logs/'):
    
    #Experiments vary and need to be modified
    experiment_name = os.path.basename(model_config_dir)
    log_dir = os.path.join(log_dir, experiment_name)
    os.makedirs(log_dir, exist_ok=True)
    final_results_file = os.path.join(log_dir, experiment_name+".csv")

    # Get the value of CUDA_VISIBLE_DEVICES in the environment variable
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")

    if cuda_visible_devices is not None:
        cuda_devices = cuda_visible_devices.split(',')
        device = [int(id) for id in cuda_devices]
        print(f"CUDA_VISIBLE_DEVICES: {device}")
    else:
        print("No CUDA_VISIBLE_DEVICES set")

    # Initialize an empty DataFrame for storing all the experiment results
    all_results = pd.DataFrame()

    # Iterate over all YAML files in the model configuration directory
    for model_config_path in sorted(os.listdir(model_config_dir)):
        if model_config_path.endswith(".yaml"):
            print("!!!!!!!pretrain_model: ", pretrain_model)
            model_config_path = os.path.join(model_config_dir, model_config_path)
            
            start_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            log_filename = f"{get_file_name(model_config_path)}_{get_file_name(dataset_config_path)}_{get_file_name(training_config_path)}_batch{batch_size}_epochs{epochs}_img_size{img_size}_{start_time}.log"
            log_path = os.path.join(log_dir, log_filename)

            # Open the log file
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
                    if "flow" in model_config_path:
                        model = YOLOFT(model_config_path).load(pretrain_model)
                    else:
                        model = YOLO(model_config_path).load(pretrain_model)
                    results = model.train(data=dataset_config_path, cfg=training_config_path, batch=batch_size, epochs=epochs, imgsz=img_size, device=device, workers=workers)
                    print(f"Training session {i+1} completed.")
                except Exception as e:
                    print(f"An error occurred during training session {i+1}: {e}")

            # Close the log file
            sys.stdout.close()
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
            print(f"All experiments for {model_config_path} completed. Logs saved to {log_path}")

            # Analyze logs and merge results into all_results
            experiment_results = extract_best_results(log_path, experiment_name=get_file_name(model_config_path), experiment_total=repeats, print_best=True)
            all_results = pd.concat([all_results, experiment_results], ignore_index=True)
            process_and_save_results(all_results, final_results_file)

    # Save all experimental results to a file
    process_and_save_results(all_results, final_results_file)
    print(f"All experiment results saved to {final_results_file}")

if __name__ == "__main__":
    model_config_dir = "config/model_yamls_dir"   # Replace it with model configuration directory

    repeats = 2
    dataset_config_path = "config/visdrone2019VID.yaml"
    training_config_path = "config/train/orige_stream.yaml"
    epochs = 25
    img_size = 1024 
    workers = 6
    pretrain_model="yolov8l.pt"
    batch_size = 12

    train_model(model_config_dir, repeats,dataset_config_path, training_config_path, batch_size, epochs, img_size, workers, pretrain_model)
