import librosa
import numpy as np

def preprocess_audio(file_path, max_pad_len=174):
    try:
        audio, sample_rate = librosa.load(file_path, sr=22050)
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        if mfcc.shape[1] < max_pad_len:
            pad_width = max_pad_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :max_pad_len]
        mfcc = mfcc[np.newaxis, ..., np.newaxis]  # (1, 40, 174, 1)
        return mfcc
    except Exception as e:
        print("Error during preprocessing:", e)
        return None
