import os
import pydub
from pydub import AudioSegment
import random

data_dir = "top30raga_data"

clip_length = 5000

for raga in os.listdir(data_dir):
    raga_dir = os.path.join(data_dir, raga)
    output_dir = os.path.join(raga_dir, "clips")
    os.makedirs(output_dir, exist_ok=True)
    for track in os.listdir(raga_dir):
        if track == "clips":
            continue
        track_path = os.path.join(raga_dir, track)
        track_name, _ = os.path.splitext(track)

        audio = AudioSegment.from_mp3(track_path)
        duration_ms = len(audio)
        max_start = duration_ms - clip_length

        clips_per_track = 10 if duration_ms < 720000 else 25

        # determine start positions ensuring disjoint clips
        # divide into equal windows and sample 1 clip per window
        seg_width = max_start // clips_per_track
        starts = [i * seg_width for i in range(clips_per_track)]
        random.shuffle(starts)

        # track-specific output folder

        for idx, start in enumerate(starts):
            clip = audio[start:start + clip_length]
            clip_filename = f"{track_name}_clip{idx+1:02d}.mp3"
            clip_path = os.path.join(output_dir, clip_filename)
            clip.export(clip_path, "mp3")

        print(f"Extracted {clips_per_track} clips for {track} into {output_dir} as mp3")
print("Successfully extracted clips for all tracks.")