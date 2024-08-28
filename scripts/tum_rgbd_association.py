"""
For RGB-D
---------
If the dataset does not have an association.txt, copy this file into the dataset folder, 
and run this command:
python tum_rgbd_assoication.py
"""

import numpy as np

# Load data from files
def load_data(file_path):
    timestamps = []
    file_names = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            timestamp, filename = line.strip().split()
            timestamps.append(float(timestamp))
            file_names.append(filename)
    return np.array(timestamps), file_names

# Load RGB and depth data
rgb_timestamps, rgb_files = load_data('rgb.txt')
depth_timestamps, depth_files = load_data('depth.txt')

# Find the best matches for each RGB timestamp
matched_pairs = []
for i, rgb_time in enumerate(rgb_timestamps):
    # Compute the absolute difference between current RGB timestamp and all depth timestamps
    time_diffs = np.abs(depth_timestamps - rgb_time)
    
    # Find the index of the closest depth timestamp
    best_match_idx = np.argmin(time_diffs)
    
    # Store the matched pair along with their timestamps
    matched_pairs.append((rgb_time, rgb_files[i], depth_timestamps[best_match_idx], depth_files[best_match_idx]))

# Output matched pairs to a file in the required format
with open('association.txt', 'w') as f:
    for rgb_time, rgb_file, depth_time, depth_file in matched_pairs:
        f.write(f"{rgb_time:.6f} {rgb_file} {depth_time:.6f} {depth_file}\n")
