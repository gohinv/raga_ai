import IPython.display as ipd
from pydub import AudioSegment, effects
import essentia
import numpy as np
import librosa
from compiam.melody.pitch_extraction import TonicIndianMultiPitch
from compiam.utils.augment import attack_remix, spectral_shape
from compiam.melody.pitch_extraction import FTANetCarnatic
import compiam
import laion_clap
import os
import soundfile as sf
import tempfile

# ---------- 0.  CONFIG -------------------------------------------------
TARGET_TONIC_HZ   = 130.813      # C3
WIN_LEN_SEC       = 5.0          # slice size
HOP_PCT           = 0.50         # 50 % overlap
AUG_TONIC_SHIFTS  = [-2, -1, 0, +1, +2]   # in semitones
AUG_JITTER_CENTS  = 50           # ± cents
AUG_TEMPO_RANGE   = (0.9, 1.1)
AUG_NOISE_SNR_DB  = 30

# ---------- 1.  UTILS --------------------------------------------------
def cents(f0, tonic_hz):
    return 1200.0 * np.log2(np.maximum(f0, 1e-6) / tonic_hz)

def tempo_stretch(c, factor):
    # Resample contour by factor, then re-interpolate to original length
    idx = np.linspace(0, len(c) - 1, int(len(c) / factor))
    return np.interp(np.arange(len(c)), idx, c[idx.astype(int)])

def tdms(contour_cents, lags=np.arange(4, 40)):          # 4–40 frames (≈20–200 ms)
    mats = [np.roll(contour_cents, -k) - contour_cents for k in lags]
    mats = np.stack(mats, axis=0)[:, :-max(lags)]        # drop wrap-around tail
    return mats.mean(), mats.std()                       # tiny (2-float) summary

def gamaka_stats(contour_cents):
    d1 = np.diff(contour_cents)
    d2 = np.diff(d1)
    return np.std(d1), np.mean(d2)

def note_cqt_stats(y, sr, bins=56):
    cqt = np.abs(librosa.cqt(y, sr=sr, bins_per_octave=12, n_bins=bins))
    return cqt.mean(axis=1), cqt.std(axis=1)             # 56-mean + 56-std

def add_noise(y, snr_db=AUG_NOISE_SNR_DB):
    sig_pow = np.mean(y**2)
    noise_pow = sig_pow / (10**(snr_db/10))
    noise = np.random.normal(scale=np.sqrt(noise_pow), size=y.shape)
    return y + noise

def get_embedding_from_audio(y, sr):
    with tempfile.TemporaryDirectory() as tmpdir:
        in_path = os.path.join(tmpdir, "audio.wav")
        out_path = os.path.join(tmpdir, "embedding.npy")
        sf.write(in_path, y, sr)
        
        # Assuming ingest.py is run from the repo root
        config_path = "./supporting_data/configs/mule_embedding_timeline.yml"
        os.system(f"mule analyze --config {config_path} -i {in_path} -o {out_path}")
        
        if os.path.exists(out_path):
            return np.load(out_path)
        else:
            return None


# ---------- 2.  MAIN INGEST ROUTINE -----------------------------------
def ingest_one_clip(mp3_path):
    # --- 2.1 load & basic clean-up (you already had this) ---
    audio     = AudioSegment.from_mp3(mp3_path).set_channels(1)
    audio     = effects.normalize(audio).high_pass_filter(70).low_pass_filter(4000)
    y         = np.array(audio.get_array_of_samples(), dtype=np.float32)
    sr        = audio.frame_rate
    y, _      = librosa.effects.trim(y, top_db=30)

    # --- 2.2 tonic detection & base pitch shift to C3 ---
    tonic_est = TonicIndianMultiPitch().extract(y, sr)
    base_steps = 12 * np.log2(TARGET_TONIC_HZ / tonic_est)
    y_base = librosa.effects.pitch_shift(y, sr, n_steps=base_steps)


    ftanet    = compiam.load_model("melody:ftanet-carnatic")              # keep outside loop if batching

    # ------------ 2.3 DATA-AUG LOOP --------------------
    for shift in AUG_TONIC_SHIFTS:
        y_shift = librosa.effects.pitch_shift(y_base, sr, n_steps=shift)

        # micro-jitter & tempo warp happen in *contour* domain, so extract f0 first
        f0_hz  = ftanet.predict(y_shift, sr, hop_size=512)    # hop ≈11.6 ms
        tonic  = TARGET_TONIC_HZ * 2**(shift/12)
        cents_contour = cents(f0_hz, tonic)

        # augment contour
        cents_contour += np.random.uniform(-AUG_JITTER_CENTS,
                                            AUG_JITTER_CENTS,
                                            size=cents_contour.shape)

        tempo_fac = np.random.uniform(*AUG_TEMPO_RANGE)
        cents_aug = tempo_stretch(cents_contour, tempo_fac)

        # optional audio-domain noise
        y_aug = add_noise(y_shift)

        # ------------ 2.5 FEATURE STACK ----------------
        emb_vec = get_embedding_from_audio(y_aug, sr)
        if emb_vec is None:
            print(f"Warning: Could not generate embedding for a clip from {mp3_path}. Skipping this augmentation.")
            continue

        tdms_mu, tdms_sd= tdms(cents_aug)                         # 2-d
        gmk_sd, gmk_mu  = gamaka_stats(cents_aug)                 # 2-d
        cqt_mu, cqt_sd= note_cqt_stats(y_aug, sr)              # 56+56

        vector = np.hstack([emb_vec.flatten(),      # was (1728, 2), now (3456,)
                            tdms_mu, tdms_sd,
                            gmk_sd, gmk_mu,
                            cqt_mu, cqt_sd])

        # ------------ 2.6 UPSERT ------------------------
        # insert vector into vector db
        # TODO: insert into vector db

        