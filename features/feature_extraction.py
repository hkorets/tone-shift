import librosa
import numpy as np
from scipy.ndimage import median_filter


def get_f0(wave):
    f0, _, _ = librosa.pyin(wave,
                            sr=16000,
                            fmin=librosa.note_to_hz('C2'),
                            fmax=librosa.note_to_hz('C7'))
    return f0


def get_loudness(
    wave, sr=16000, n_fft=2048, hop_length=512, 
    smoothing_window_frames=15,
    floor_db=-60.0, ref_percentile=95
):
    y = np.asarray(wave, dtype=np.float32)
    
    peak = np.max(np.abs(y))
    if peak < 1e-6:
        N_frames = librosa.util.frame(y, frame_length=n_fft, hop_length=hop_length).shape[1]
        return np.zeros(N_frames, dtype=np.float32)

    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    P = (S ** 2)

    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    A_db = librosa.A_weighting(freqs)
    A_lin = 10.0 ** (A_db / 20.0) 
    P_aw = P * (A_lin[:, None] ** 2) 
    pow_aw = np.sum(P_aw, axis=0) + 1e-12 
    loud_db = 10.0 * np.log10(pow_aw) 

    if smoothing_window_frames > 1:
        if smoothing_window_frames % 2 == 0:
            smoothing_window_frames += 1
        loud_db_smoothed = median_filter(
            loud_db, 
            size=smoothing_window_frames 
        )
    else:
        loud_db_smoothed = loud_db

    ref_db = np.percentile(loud_db_smoothed, ref_percentile)
    loud_rel = loud_db_smoothed - ref_db 
    loud_rel = np.clip(loud_rel, floor_db, 0.0)
    loud_unit = (loud_rel - floor_db) / (-floor_db) 

    return loud_unit.astype(np.float32)

def get_onset(wave, f0):
    onset_envelope = librosa.onset.onset_strength(y=wave, sr=16000, hop_length=512)
    onset_frames = librosa.util.peak_pick(
        x=onset_envelope,
        pre_max=3,
        post_max=3,
        pre_avg=3,
        post_avg=5,
        delta=0.10,     
        wait=2      
    )
    N_FRAMES_TARGET = len(f0)
    onset_vector = np.zeros(N_FRAMES_TARGET)
    for frame_idx in onset_frames:
        if frame_idx < N_FRAMES_TARGET:
            onset_vector[frame_idx] = 1
    return onset_vector

def extract_features(wav):
    y, _ = librosa.load(wav, sr=None, mono=True)
    f0 = get_f0(y)
    loudness = get_loudness(y)
    onset = get_onset(y, f0)
    result = list(zip(f0, loudness, onset))
    return result
    




    




