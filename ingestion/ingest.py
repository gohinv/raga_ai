import IPython.display as ipd
from pydub import AudioSegment, effects
import essentia
import numpy as np
import librosa
from compiam.melody.tonic_identification.tonic_multipitch import TonicIndianMultiPitch
from compiam.melody.pitch_extraction import FTANetCarnatic
import compiam
import os
import glob
import uuid
import traceback
from tqdm import tqdm
from qdrant_client import QdrantClient, models
import yaml
import gc

from feature_extractor import (
    MuleEmbedder, 
    cents, 
    tdms, 
    gamaka_stats, 
    note_cqt_stats, 
    tempo_stretch, 
    add_noise
)

# ---------- 0.  CONFIG -------------------------------------------------
TARGET_TONIC_HZ   = 130.813      # C3
WIN_LEN_SEC       = 5.0          # slice size
HOP_PCT           = 0.50         # 50 % overlap
AUG_TONIC_SHIFTS  = [-2, -1, 0, +1, +2]   # in semitones
AUG_JITTER_CENTS  = 50           # ± cents
AUG_TEMPO_RANGE   = (0.9, 1.1)
AUG_NOISE_SNR_DB  = 30

# --- QDRANT CONFIG ---
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "raga_clips_v1"
VECTOR_DIM = 1844  # 1728 (mule) + 2 (tdms) + 2 (gamaka) + 112 (cqt)
BATCH_SIZE = 16

# ---------- 1.  UTILS (REMOVED) --------------------------------------

# ---------- 2.  MAIN INGEST ROUTINE -----------------------------------
def ingest_one_clip(mp3_path, tonic_model, ftanet_model, mule_embedder):
    """
    Processes a single mp3 file, generates augmentations, and returns a list 
    of Qdrant PointStructs ready for upserting.
    """
    points = []
    try:
        audio     = AudioSegment.from_mp3(mp3_path).set_channels(1)
        audio     = effects.normalize(audio).high_pass_filter(70).low_pass_filter(4000)
        y         = np.array(audio.get_array_of_samples(), dtype=np.float32)
        sr        = audio.frame_rate
        y, _      = librosa.effects.trim(y, top_db=30)
        if len(y) < sr * 1.0: # skip clips shorter than 1s
            del audio, y
            gc.collect()
            return []

        tonic_est = tonic_model.extract(y, sr)
        base_steps = 12 * np.log2(TARGET_TONIC_HZ / tonic_est)
        # shift pitch to C3
        y_base = librosa.effects.pitch_shift(y, sr, n_steps=base_steps)

        emb_vec = mule_embedder.predict(y_base, sr)
        if emb_vec is None:
            return []
        cqt_mu, cqt_sd = note_cqt_stats(y_base, sr)
        
        # Also extract the base pitch contour once
        pitch_data = ftanet_model.predict(y_base, sr, hop_size=512)
        f0_hz = pitch_data[:, 1]
        base_cents_contour = cents(f0_hz, TARGET_TONIC_HZ)

        raga = mp3_path.split(os.sep)[-3]

        # data-aug loop
        for shift in AUG_TONIC_SHIFTS:
            cents_contour = base_cents_contour + (shift * 100)

            cents_contour += np.random.uniform(-AUG_JITTER_CENTS, AUG_JITTER_CENTS, size=cents_contour.shape)
            tempo_fac = np.random.uniform(*AUG_TEMPO_RANGE)
            cents_aug = tempo_stretch(cents_contour, tempo_fac)
            tdms_mu, tdms_sd= tdms(cents_aug)
            gmk_sd, gmk_mu  = gamaka_stats(cents_aug)

            vector = np.hstack([
                emb_vec.flatten(),
                np.array([tdms_mu]),
                np.array([tdms_sd]),
                np.array([gmk_sd]),
                np.array([gmk_mu]),
                cqt_mu,
                cqt_sd
            ]).tolist()

            # create Qdrant point with vector and metadata payload
            point = models.PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "source_file": mp3_path,
                    "raga": raga,
                    "augmentation_shift_semitones": shift,
                    "augmentation_tempo_factor": round(tempo_fac, 2),
                }
            )
            points.append(point)
    
    except Exception as e:
        print(f"\n--- ERROR processing {mp3_path} ---")
        traceback.print_exc()
        print("--- END ERROR ---")

    gc.collect()
    return points


if __name__ == "__main__":
    client = QdrantClient(url=QDRANT_URL, timeout=60.0)
    try:
        print("Recreating collection with indexing disabled for bulk upload...")
        client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(size=VECTOR_DIM, distance=models.Distance.COSINE),
            # defer indexing until all points are uploaded
            optimizers_config=models.OptimizersConfigDiff(
                deleted_threshold=0.2,
                vacuum_min_vector_number=1000,
                default_segment_number=0,
                flush_interval_sec=5,
                indexing_threshold=0
            )
        )
        print(f"Collection '{COLLECTION_NAME}' created.")
    except Exception as e:
        print(f"Could not create collection: {e}")
        exit()

    print("Loading FTANet model...")
    ftanet = compiam.load_model("melody:ftanet-carnatic")
    print("Model loaded.")

    print("Loading MULE embedding model...")
    mule_config_path = "music-audio-representations/supporting_data/configs/mule_embedding_average.yml"
    mule = MuleEmbedder(mule_config_path)
    print("MULE model loaded.")

    print("Loading Tonic model...")
    tonic_identifier = TonicIndianMultiPitch()
    print("Tonic model loaded.")
    
    # --- Find all mp3 files ---
    files_to_process = glob.glob('top30raga_data/**/clips/*.mp3', recursive=True)
    print(f"Found {len(files_to_process)} MP3 files to process.")

    points_batch = []
    for fpath in tqdm(files_to_process, desc="Processing files"):
        new_points = ingest_one_clip(fpath, tonic_identifier, ftanet, mule)
        points_batch.extend(new_points)

        if len(points_batch) >= BATCH_SIZE:
            print(f"UPSERTING {len(points_batch)} points...")
            client.upsert(collection_name=COLLECTION_NAME, points=points_batch, wait=True)
            print(f"UPSERTED {len(points_batch)} points...")
            points_batch = []
    if points_batch:
        client.upsert(collection_name=COLLECTION_NAME, points=points_batch, wait=True)

    print("\nData upload complete. Re-enabling indexing on the server...")
    try:
        client.update_collection(
            collection_name=COLLECTION_NAME,
            optimizer_config=models.OptimizersConfigDiff(indexing_threshold=1000)
        )
        print("Indexing has been enabled. The server will now build the index in the background.")
    except Exception as e:
        print(f"Could not re-enable indexing: {e}")

    print("\nIngestion complete.")
    print(f"Total points in collection: {client.get_collection(COLLECTION_NAME).points_count}")

        