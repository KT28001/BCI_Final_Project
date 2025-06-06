import numpy as np
from scipy.signal import resample, cheby1, filtfilt
from sklearn.cross_decomposition import CCA

def chebyshev_bandpass(data, lowcut, highcut, fs, order=4, ripple=1):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = cheby1(order, ripple, [low, high], btype='band')
    return filtfilt(b, a, data, axis=-1)

def generate_reference_signals(freq, harmonics, length, fs):
    t = np.arange(0, length) / fs
    ref = []
    for i in range(1, harmonics + 1):
        ref.append(np.sin(2 * np.pi * freq * i * t))
        ref.append(np.cos(2 * np.pi * freq * i * t))
    return np.array(ref)

def cca(X, Y):
    cca = CCA(n_components=1)
    cca.fit(X.T, Y.T)
    X_c, Y_c = cca.transform(X.T, Y.T)
    return np.corrcoef(X_c.T, Y_c.T)[0, 1]

def fbcca_full(trial, freqs, fs, harmonics=3):
    sub_bands = [(6, 10), (8, 12), (10, 14), (12, 16)]
    weights = 1 / np.arange(1, len(sub_bands)+1)**1.25
    feature_vector = []

    for freq in freqs:
        corrs = []
        for i, (low, high) in enumerate(sub_bands):
            filtered =  chebyshev_bandpass(trial, low, high, fs)
            # filtered = trial
            ref = generate_reference_signals(freq, harmonics, trial.shape[1], fs)
            corr = cca(filtered, ref)
            corrs.append(corr * weights[i])
        feature_vector.append(np.sum(corrs))

    return np.array(feature_vector)