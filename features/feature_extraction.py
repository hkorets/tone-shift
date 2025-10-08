import librosa
import torchcrepe
import torch
import numpy as np
import scipy.signal

def get_f0(wave):
    '''
    function to detect note of each moment of time
    '''
    wave_torch = torch.tensor(wave).unsqueeze(0)
    f0 = torchcrepe.predict(
        audio=wave_torch,
        sample_rate=16000,
        hop_length=320,       
        fmin=50.0,
        fmax=2000.0,
        model='full'
    )
    f0 = f0.squeeze().numpy()
    f0[f0 <= 0] = np.nan

    nans = np.isnan(f0)
    if np.any(nans):
        not_nan = np.logical_not(nans)
        indices = np.arange(len(f0))
        f0[nans] = np.interp(indices[nans], indices[not_nan], f0[not_nan])

    f0 = scipy.signal.medfilt(f0, kernel_size=5) 

    f0 = np.clip(f0, 50, 2000)

    return f0


def get_loudness(wave, frame_lenght= 640, hop=320):
    '''
    function to detect level of loudness at each moment
    '''
    loudness = []
    for i in range(0, len(wave) - frame_lenght, hop):
        frame = wave[i:i+frame_lenght]
        rms = np.sqrt(np.mean(frame**2))
        loudness.append(20 * np.log10(rms + 1e-6))
    loudness = np.array(loudness)
    loudness = np.clip(loudness, -60.0, 0.0)
    mean = np.mean(loudness)
    std = np.std(loudness) + 1e-6
    loudness = (loudness - mean) / std
    return loudness


def get_onset(wave):
    '''
    check if there is some increase in energy in signal
    '''
    onset_env = librosa.onset.onset_strength(y=wave, sr=16000, hop_length=320)
    onset_env = onset_env / onset_env.max() 
    return onset_env


def get_features(filepath):
    '''
    vse v kuchu
    '''
    x, _ = librosa.load(filepath, sr=16000, mono=True)
    x = librosa.util.normalize(x)
    result = list(zip(get_f0(x), get_loudness(x), get_onset(x)))
    return result