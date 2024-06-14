import pandas as pd
import os
import argparse

def read_and_process_csv(file_path):
     # Read CSV files
    df = pd.read_csv(file_path)

    # Sort the table in descending order by AP-ALL, or by AP-0-12 if AP-ALL is the same
    sorted_df = df.sort_values(by=['AP-ALL', 'AP-0-12'], ascending=[False, False])

    # Group the maximum values for different sessions within each experiment_name
    grouped = sorted_df.groupby(['experiment_name', 'session']).max().reset_index()

    results = []

    # Iterate over each experiment_name
    for name, group in grouped.groupby('experiment_name'):
        # Calculate the value of each statistic
        max_list = group.groupby('session').max().reset_index()
        median_all = max_list['AP-ALL'].median()
        max_all = max_list['AP-ALL'].max()
        median_0_12 = max_list['AP-0-12'].median()
        max_0_12 = max_list['AP-0-12'].max()
        median_12_20 = max_list['AP-12-20'].median()
        max_12_20 = max_list['AP-12-20'].max()
        median_20_32 = max_list['AP-20-32'].median()
        max_20_32 = max_list['AP-20-32'].max()
        min_inference = group['inference'].min()

        # Additional statistical results
        results.append({
            'experiment_name': name,
            'mAP(max)': max_all,
            'mAP(median)': median_all,
            'AP_es(max)': max_0_12,
            'AP_es(median)': median_0_12,
            'AP_rs(max)': max_12_20,
            'AP_rs(median)': median_12_20,
            'AP_gs(max)': max_20_32,
            'AP_gs(median)': median_20_32,
            
            'inference': min_inference
        })

    # Converted to DataFrame and sorted in descending order as required
    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values(
        by=['mAP(max)', 'AP_es(max)', 'AP_rs(max)', 'AP_gs(max)'],
        ascending=[False, False, False, False]
    )

    return sorted_df, result_df

if __name__ == "__main__":
    # Example usage
    parser = argparse.ArgumentParser(description='Process COCO JSON and create symlinks for images.')
    parser.add_argument('csv_file_path', help='Path to the compare experiments csv.')

    args = parser.parse_args()

    # Examples of use
    file_path = args.csv_file_path
    dir = os.path.dirname(file_path)
    sorted_df, result_df = read_and_process_csv(file_path)

    # Output the sorted table
    sorted_df.to_csv(os.path.join(dir, 'sorted_experiments.csv'), index=False)
    result_df.to_csv(os.path.join(dir, 'experiment_statistics.csv'), index=False)

