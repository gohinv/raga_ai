import IPython.display as ipd
from pydub import AudioSegment, effects
import essentia
import numpy as np
import librosa
from compiam.melody.pitch_extraction import TonicIndianMultiPitch
from compiam.utils.augment import attack_remix, spectral_shape
from compiam.melody.pitch_extraction import FTANetCarnatic
import compiam

audio_path = "top30raga_data/kanada/clips/K.V. Narayanaswami - Nera Nammiti_clip09.mp3"


# AUDIO PREPROCESSING  

audio = AudioSegment.from_mp3(audio_path)
ipd.Audio(audio_path)

tonic_identifier = TonicIndianMultiPitch()
audio = audio.set_channels(1)

# bandpass filter
audio = effects.normalize(audio).high_pass_filter(70).low_pass_filter(4000)
y = np.array(audio.get_array_of_samples(), dtype=np.float32)
sr = audio.frame_rate

# trim silence
y, _ = librosa.effects.trim(y, top_db=30)

# detect tonic and shift

tonic_1 = tonic_identifier.extract(y, sr)

target_tonic_hz = 130.813 # C3

n_steps = 12 * np.log2(target_tonic_hz / tonic_1)

# pitch-shifted version of the audio
y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)

# DATA AUGMENTATION

ftanet = compiam.load_model("melody:ftanet-carnatic")
cae = compiam.load_model("melody:cae-carnatic")





