import numpy as np
import librosa
from pydub import AudioSegment, effects
import yaml
import tempfile
import soundfile as sf
from mule.features import AudioWaveform
from mule.features.transform_features import MelSpectrogram, EmbeddingFeature, TimelineAverage
import gc
import traceback


class MuleEmbedder:
    """
    A wrapper class to load the MULE model once and provide a method
    for extracting embeddings from in-memory audio. This version uses a
    hybrid approach to prevent memory leaks without sacrificing performance.
    """
    def __init__(self, config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        self.audio_waveform_cfg = config['Analysis']['source_feature']

        # Store configs for lightweight transformers
        self.mel_cfg = {'MelSpectrogram': config['Analysis']['feature_transforms'][0]['MelSpectrogram']}
        self.average_cfg = {'TimelineAverage': config['Analysis']['feature_transforms'][2]['TimelineAverage']}

        # Pre-load the heavy EmbeddingFeature transformer once
        embedding_cfg = {'EmbeddingFeature': config['Analysis']['feature_transforms'][1]['EmbeddingFeature']}
        self.embedding_transformer = EmbeddingFeature(cfg=embedding_cfg)

    def predict(self, y, sr):
        """
        Generates embeddings using a hybrid state management:
        - Recreates lightweight transformers on each call.
        - Reuses the pre-loaded heavyweight transformer.
        """
        # Step 1: Resample audio to match model's expected sample rate
        if sr != self.audio_waveform_cfg['AudioWaveform']['sample_rate']:
            y = librosa.resample(y=y, orig_sr=sr, target_sr=self.audio_waveform_cfg['AudioWaveform']['sample_rate'])

        # Step 2: Create a source feature object from the raw audio array.
        waveform_feature = AudioWaveform(cfg=self.audio_waveform_cfg)
        
        with tempfile.NamedTemporaryFile(suffix='.wav') as temp_file:
            sf.write(temp_file.name, y, self.audio_waveform_cfg['AudioWaveform']['sample_rate'])
            waveform_feature.from_file(temp_file.name)

        # Step 3: Create a fresh MelSpectrogram transformer and apply it
        mel_transformer = MelSpectrogram(cfg=self.mel_cfg)
        mel_transformer.from_feature(waveform_feature)

        # Step 4: Reuse the pre-loaded EmbeddingFeature transformer
        self.embedding_transformer.from_feature(mel_transformer)

        # Step 5: Create a fresh TimelineAverage transformer and apply it
        average_transformer = TimelineAverage(cfg=self.average_cfg)
        average_transformer.from_feature(self.embedding_transformer)

        return average_transformer.data


# ---------- UTILS --------------------------------------------------
def cents(f0, tonic_hz):
    return 1200.0 * np.log2(np.maximum(f0, 1e-6) / tonic_hz)

def tempo_stretch(c, factor):
    # Resample contour by factor, then re-interpolate to original length
    idx = np.linspace(0, len(c) - 1, int(len(c) / factor))
    return np.interp(np.arange(len(c)), idx, c[idx.astype(int)])

def tdms(contour_cents, lags=np.arange(4, 40)):          # 4–40 frames (≈20–200 ms)
    if len(contour_cents) <= max(lags):
        return 0.0, 0.0 # Return default values for short contours
    mats = [np.roll(contour_cents, -k) - contour_cents for k in lags]
    mats = np.stack(mats, axis=0)[:, :-max(lags)]        # drop wrap-around tail
    
    # Replace any potential NaNs with 0, just in case
    mean = np.nan_to_num(mats.mean())
    std = np.nan_to_num(mats.std())
    return mean, std

def gamaka_stats(contour_cents):
    if len(contour_cents) < 3:
        return 0.0, 0.0 # Return default values for short contours
    d1 = np.diff(contour_cents)
    d2 = np.diff(d1)
    
    # Replace any potential NaNs with 0
    std_d1 = np.nan_to_num(np.std(d1))
    mean_d2 = np.nan_to_num(np.mean(d2))
    return std_d1, mean_d2

def note_cqt_stats(y, sr, bins=56):
    cqt = np.abs(librosa.cqt(y, sr=sr, bins_per_octave=12, n_bins=bins))
    return cqt.mean(axis=1), cqt.std(axis=1)             # 56-mean + 56-std

def add_noise(y, snr_db):
    sig_pow = np.mean(y**2)
    noise_pow = sig_pow / (10**(snr_db/10))
    noise = np.random.normal(scale=np.sqrt(noise_pow), size=y.shape)
    return y + noise

# ---------- MAIN FEATURE EXTRACTOR FOR QUERYING -----------------------
def extract_features(
    audio_path, 
    tonic_model, 
    ftanet_model, 
    mule_embedder,
    target_tonic_hz=130.813 # C3
):
    """
    Processes a single audio file for querying.
    - No data augmentation.
    - Returns the feature vector and the pitch contour for DTW.
    """
    vector = None
    contour = None
    try:
        # --- 1. Load, clean, and get base audio ---
        audio     = AudioSegment.from_file(audio_path).set_channels(1)
        audio     = effects.normalize(audio).high_pass_filter(70).low_pass_filter(4000)
        y         = np.array(audio.get_array_of_samples(), dtype=np.float32)
        sr        = audio.frame_rate
        y, _      = librosa.effects.trim(y, top_db=30)
        if len(y) < sr * 1.0: # skip clips shorter than 1s
            return None, None

        # --- 2. Tonic detection & base pitch shift to C3 ---
        tonic_est = tonic_model.extract(y, sr)
        base_steps = 12 * np.log2(target_tonic_hz / tonic_est)
        y_base = librosa.effects.pitch_shift(y, sr, n_steps=base_steps)

        # --- 3. Run expensive, non-augmented feature extraction ONCE ---
        emb_vec = mule_embedder.predict(y_base, sr)
        if emb_vec is None:
            return None, None
        
        cqt_mu, cqt_sd = note_cqt_stats(y_base, sr)
        
        # Also extract the base pitch contour once
        pitch_data = ftanet_model.predict(y_base, sr, hop_size=512)
        f0_hz = pitch_data[:, 1]
        contour = cents(f0_hz, target_tonic_hz)

        # --- 4. FEATURE STACK (No Augmentation) ---
        tdms_mu, tdms_sd = tdms(contour)
        gmk_sd, gmk_mu  = gamaka_stats(contour)

        vector = np.hstack([
            emb_vec.flatten(),
            np.array([tdms_mu]),
            np.array([tdms_sd]),
            np.array([gmk_sd]),
            np.array([gmk_mu]),
            cqt_mu,
            cqt_sd
        ]).tolist()

    except Exception:
        print(f"\n--- ERROR processing {audio_path} for feature extraction ---")
        traceback.print_exc()
        print("--- END ERROR ---")

    finally:
        # Explicitly trigger garbage collection
        gc.collect()
        return vector, contour 