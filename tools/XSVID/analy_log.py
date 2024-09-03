import re
import pandas as pd
import argparse

def extract_best_results(log_file, experiment_name="experiment_1", experiment_total=3, print_best=True):
    with open(log_file, 'r') as file:
        lines = file.readlines()

    start_training_pattern = re.compile(r"Starting training session (\d+)")
    end_training_pattern = re.compile(r"Training session \d+ completed.")
    result_marker_pattern = re.compile(r"\*{5}ALL, 0\.75, 0\.5, 0-12, 12-20, 20-32, small, medium, large\*{33}")
    epoch_pattern = re.compile(r"^\s*(\d+)/(\d+)\s+\d+\.\d+G\s+\d+\.\d+\s+\d+\.\d+\s+\d+\.\d+\s+\d+\s+\d+\s+\d+:")
    dir_pattern = re.compile(r"Saving (.*?/)[^/]*$")
    time_pattern = re.compile(r"Speed: (\d+\.\d+)ms preprocess, (\d+\.\d+)ms inference, (\d+\.\d+)ms loss, (\d+\.\d+)ms postprocess per image")

    current_session = None
    results = []
    next_is_results = False
    save_dir = None
    time_info = None
    current_epoch = None
    total_epochs = None
    
    for i, line in enumerate(lines):
        start_match = start_training_pattern.search(line)
        if re.search(f"Training session {experiment_total} completed", line):
            break

        if start_match:
            current_session = int(start_match.group(1))
            continue

        end_match = end_training_pattern.search(line)
        if end_match:
            continue

        epoch_match = epoch_pattern.search(line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))
            total_epochs = int(epoch_match.group(2))

        if dir_pattern.search(line):
            save_dir = dir_pattern.search(line).group(1)

        time_match = time_pattern.search(line)
        if time_match:
            time_info = {
                'preprocess': float(time_match.group(1)),
                'inference': float(time_match.group(2)),
                'loss': float(time_match.group(3)),
                'postprocess': float(time_match.group(4))
            }

        if result_marker_pattern.search(line):
            next_is_results = True
            continue

        if next_is_results:
            next_is_results = False
            current_result = [float(num) for num in line.split(" ")]
            result_dict = {
                'AP-ALL': current_result[0],
                'AP-0.75': current_result[1],
                'AP-0.5': current_result[2],
                'AP-0-12': current_result[3],
                'AP-12-20': current_result[4],
                'AP-20-32': current_result[5],
                'AP-small': current_result[6],
                'AP-medium': current_result[7],
                'AP-large': current_result[8],
                'epoch': current_epoch,
                'total_epochs': total_epochs,
                'line': i + 1,
                'dir': save_dir,
                'session': current_session  # Add current session
            }
            if time_info:
                result_dict.update(time_info)
            result_dict['experiment_name'] = experiment_name  # Add experiment name
            results.append(result_dict)
            save_dir = None
            time_info = None  # Reset time_info after adding to the result

    if print_best:
        # Analyze the best results from each session
        overall_best_result = None
        
        for session in set(result['session'] for result in results):
            session_results = [result for result in results if result['session'] == session]
            print(f"session {session}: ")
            for result in session_results:
                result_ap_print = ", ".join([str(ap) for ap in list(result.values())[:9]])
                print(result_ap_print)

        for session in set(result['session'] for result in results):
            session_results = [result for result in results if result['session'] == session]
            session_best = max(session_results, key=lambda x: (x['AP-ALL'], x['AP-0.75']))

            if session_results[session_results.index(session_best) - 1]['epoch'] == session_best['epoch'] and session_best['epoch'] is not None:
                session_best['epoch'] += 1

            print("\n")
            result_ap_print = ", ".join([str(ap) for ap in list(session_best.values())[:9]])
            print(f"Best result for training session {session}: \n{result_ap_print} \n(Epoch {session_best['epoch']} of {session_best['total_epochs']}, Line {session_best['line']})")
            print(f"save_dir: {session_best['dir']}")

            if overall_best_result is None or (session_best['AP-ALL'] > overall_best_result['AP-ALL']) or (session_best['AP-ALL'] == overall_best_result['AP-ALL'] and session_best['AP-0.75'] > overall_best_result['AP-0.75']):
                overall_best_result = session_best

        if overall_best_result:
            print("\n")
            result_ap_print = ", ".join([str(ap) for ap in list(overall_best_result.values())[:9]])
            print(f"Overall best result across all sessions: \n{result_ap_print} \n(Epoch {overall_best_result['epoch']} of {overall_best_result['total_epochs']}, Line {overall_best_result['line']})")
            print(f"save_dir: {overall_best_result['dir']}")

    # Convert results to DataFrame with correct format
    results_df = pd.DataFrame(results)
    return results_df


if __name__ == "__main__":
    # Example usage
    parser = argparse.ArgumentParser(description='Process COCO JSON and create symlinks for images.')
    parser.add_argument('log_file_path', help='Path to the repeat experiments log.')

    args = parser.parse_args()

    # log_file_path = 'runs/logs/flowS_baseline3_start4__Convnoany_UAVTOD_[8,50]_orige_stream_batch22_epochs30_img_size1024_20240606214904.log'
    extract_best_results(args.log_file_path)
