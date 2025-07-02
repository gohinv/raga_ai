import os
import pandas as pd
from pathlib import Path

# Path to the raga directories
base_dir = "top_30_raga_data"

# Create a list to store results
results = []

# Check each directory
for raga_dir in sorted(os.listdir(base_dir)):
    dir_path = os.path.join(base_dir, raga_dir)
    
    # Skip if not a directory
    if not os.path.isdir(dir_path):
        continue
    
    # Count MP3 files
    mp3_files = [f for f in os.listdir(dir_path) if f.lower().endswith('.mp3')]
    track_count = len(mp3_files)
    
    # Add to results
    results.append({
        'raga_directory': raga_dir,
        'track_count': track_count,
        'status': 'NEEDS MORE TRACKS' if track_count < 5 else 'OK'
    })

# Create a DataFrame for better display
df = pd.DataFrame(results)

# Sort by track count (ascending)
df = df.sort_values('track_count')

# Print results
print(f"Total raga directories: {len(df)}")
print("\nDirectories with fewer than 5 tracks:")
print(df[df['track_count'] < 5].to_string(index=False))

print("\nAll directories:")
print(df.to_string(index=False))

# Print summary
print(f"\nTotal directories with fewer than 5 tracks: {sum(df['track_count'] < 5)}")
print(f"Total directories with zero tracks: {sum(df['track_count'] == 0)}")