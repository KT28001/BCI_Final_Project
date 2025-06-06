import os
import numpy as np
import h5py
import mne
from mne.preprocessing import ICA
from autoreject import AutoReject
import matplotlib.pyplot as plt

from scipy.signal import resample, cheby1, filtfilt
from sklearn.cross_decomposition import CCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

freq_indices = np.arange(7, 14)        # index 7–13 對應 8–14 Hz
freqs = np.arange(8, 15)             
fs_original = 1000
fs_target = 250
selected_channels = [48, 54, 55, 56, 57, 58, 61, 62, 63]  # Pz, PO5, PO3, POz, PO4, PO6, O1, Oz, O2
T = 5.14  

def load_subject_continuous(mat_path, contrast_idx=1):
    mat = h5py.File(mat_path, 'r')
    raw_data = np.array(mat['datas'])  # (12,60,5140,64,2)
    mat.close()
    data = raw_data[1:, :, :, :, contrast_idx]
    #data = raw_data[:, :, :, :, contrast_idx]  # 取高對比度 → (12,60,5140,64)
    n_blocks, n_freqs, n_times, n_ch = data.shape
    all_trials = data.reshape(n_blocks * n_freqs, n_times, n_ch)    # (720,5140,64)
    all_trials_T = np.transpose(all_trials, (0, 2, 1))
    concatenated = np.hstack([trial for trial in all_trials_T])
    return concatenated

def make_rawarray_from_numpy(data_2d, sfreq=1000):
    ch_names = [f'ch{idx+1}' for idx in range(data_2d.shape[0])]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
    raw = mne.io.RawArray(data_2d, info)
    return raw


from pyprep.prep_pipeline import PrepPipeline

def apply_pyprep(raw):
    raw_copy = raw.copy()
    prep_params = {
        "ref_chs": raw_copy.info["ch_names"],
        "reref_chs": raw_copy.info["ch_names"],
        # 50 或 60 Hz 主頻和其倍頻去除
        "line_freqs": [50, 100]
    }
    montage = None 
    prep = PrepPipeline(raw_copy, prep_params, montage=montage)
    prep.fit()
    return prep.raw_eeg

def apply_asr_to_raw(raw, epoch_duration_sec=2.0):
    sfreq = raw.info['sfreq']
    epoch_length = int(epoch_duration_sec * sfreq)
    n_samples = raw.n_times
    n_epochs = n_samples // epoch_length
    if n_epochs < 1:
        print("  [Warning] Raw 太短，無法切分 Epoch → 跳過 ASR。")
        return raw

    # Produce non-repect event
    events = np.zeros((n_epochs, 3), int)
    for i in range(n_epochs):
        events[i, 0] = i * epoch_length
        events[i, 2] = 1  # event_id=1
    event_dict = {'ASR_epoch': 1}

    # Form epochs
    try:
        epochs = mne.Epochs(raw, events, event_id=event_dict, tmin=0.0,
                            tmax=epoch_duration_sec - 1e-8,
                            baseline=None, preload=True, verbose=False)
    except Exception as e:
        print("  [Warning] 切 Epoch 失敗，跳過 ASR。錯誤：", e)
        return raw

    # AutoReject
    try:
        ar = AutoReject(n_interpolate=[1,4,8], consensus=[0.5], verbose=False)
        epochs_clean, _ = ar.fit_transform(epochs, return_log=True)
    except Exception as e:
        print("  [Warning] AutoReject 修正失敗，跳過 ASR。錯誤：", e)
        return raw

    # Epochs to Array
    data_clean = epochs_clean.get_data()                   # (n_epochs, n_ch, epoch_length)
    data_clean = np.transpose(data_clean, (1, 0, 2))       # (n_ch, n_epochs, epoch_length)
    data_clean = data_clean.reshape(data_clean.shape[0], -1)  # (n_ch, n_epochs * epoch_length)

    raw_clean = make_rawarray_from_numpy(data_clean, sfreq=sfreq)
    return raw_clean

def run_ica_find_bads(raw, desc):
    """
    raw: mne.io.RawArray (建議至少先做 1 Hz 高通)
    desc: 字串標籤，例如 'Raw', 'Filtered', 'ASR'
    回傳 {'brain': count, 'non_brain': count}
    """
    raw_copy = raw.copy().set_eeg_reference('average', projection=False)

    ica = ICA(n_components=64, method='fastica')
    #ica = ICA(n_components=0.95, method='fastica', random_state=42, max_iter='auto')
    ica.fit(raw_copy)

    try:
        eog_inds, _ = ica.find_bads_eog(raw_copy, threshold=3.0)
    except (RuntimeError, ValueError):
        eog_inds = []

    try:
        ecg_inds, _ = ica.find_bads_ecg(raw_copy, threshold=0.3)
    except (RuntimeError, ValueError):
        ecg_inds = []

    non_brain = set(eog_inds) | set(ecg_inds)
    all_inds = set(range(ica.n_components_))
    brain = all_inds - non_brain

    brain_count = len(brain)
    non_brain_count = len(non_brain)

    print(f"{desc:<12s} | Brain IC: {brain_count:3d} | Non-Brain IC: {non_brain_count:3d}")
    return {'brain': brain_count, 'non_brain': non_brain_count}

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
    cca_model = CCA(n_components=1, max_iter=1000)
    cca_model.fit(X.T, Y.T)
    X_c, Y_c = cca_model.transform(X.T, Y.T)
    return np.corrcoef(X_c.T, Y_c.T)[0, 1]

def fbcca_full(trial, freqs, fs, harmonics=3):
    sub_bands = [(6, 10), (8, 12), (10, 14), (12, 16)]
    weights = 1 / (np.arange(1, len(sub_bands)+1) ** 1.25)
    feature_vector = []
    for freq in freqs:
        corrs = []
        for i, (low, high) in enumerate(sub_bands):
            filtered = chebyshev_bandpass(trial, low, high, fs)
            ref = generate_reference_signals(freq, harmonics, trial.shape[1], fs)
            corr = cca(filtered, ref)
            corrs.append(corr * weights[i])
        feature_vector.append(np.sum(corrs))
    return np.array(feature_vector)

def compute_itr(acc, N, T):
    from math import log2
    if acc == 0 or acc == 1:
        acc = max(0.0001, min(0.9999, acc))
    return (log2(N) + acc*log2(acc) + (1 - acc)*log2((1 - acc) / (N - 1))) * (60 / T)

def run_ssvep_classification(mat_path):
    mat = h5py.File(mat_path, 'r')
    raw_data = np.array(mat['datas'])
    mat.close()
    data = raw_data[:, :, :, :, 1]  

    X_trials, y_labels = [], []
    for block in range(12):
        for f_idx in freq_indices:
            trial = data[block, f_idx].T   # (64,5140)
            X_trials.append(trial)
            y_labels.append(f_idx + 1)     # 8–14
    X = np.array(X_trials)              # (84,64,5140)
    y = np.array(y_labels)

    samples_target = int(X.shape[2] * fs_target / fs_original)
    X_down = resample(X, samples_target, axis=2)      # (84,64,1285)
    X_down = X_down[:, selected_channels, :]          # (84,9,1285)

    features = []
    for trial in X_down:
        fv = []
        for freq in freqs:
            corrs = []
            for low, high in [(6,10),(8,12),(10,14),(12,16)]:
                filt = chebyshev_bandpass(trial, low, high, fs_target)
                ref = generate_reference_signals(freq, 3, trial.shape[1], fs_target)
                cor = cca(filt, ref)
                corrs.append(cor)
            weights = 1 / (np.arange(1,5) ** 1.25)
            fv.append(np.dot(corrs, weights))
        features.append(fv)
    features = np.array(features)  # (84,7)

    X_train, X_test, y_train, y_test = train_test_split(
        features, y, test_size=0.2, random_state=42, stratify=y
    )
    clf = SVC(kernel='rbf')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=freqs)
    return acc, cm

if __name__ == '__main__':
    DATA_DIR = './SSVEP_dataset/'
    file_list = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('1_64.mat')])

    print("\n=== 開始對各受試者做 ICA 標記（Brain/Non-Brain）===\n")
    for fname in file_list:
        subj = fname.split('.')[0]
        print(f"--- {subj} ---")
        mat_path = os.path.join(DATA_DIR, fname)

        data2d = load_subject_continuous(mat_path, contrast_idx=1)
        raw = make_rawarray_from_numpy(data2d, sfreq=1000)
        raw.load_data()
        raw.filter(l_freq=1.0, h_freq=None, fir_design='firwin', verbose=False) 

        run_ica_find_bads(raw, desc="Raw")

        raw_f = raw.copy().filter(0.5, 100.0, verbose=False)
        run_ica_find_bads(raw_f, desc="Filtered")

        raw_asr = apply_pyprep(raw_f)
        run_ica_find_bads(raw_asr, desc="pyPREP ASR")

    print("\n=== 開始對各受試者做 SSVEP 分類 + 混淆矩陣 ===\n")
    for fname in file_list:
        subj = fname.split('.')[0]
        print(f"--- {subj} ---")
        mat_path = os.path.join(DATA_DIR, fname)

        acc, cm = run_ssvep_classification(mat_path)
        print(f"Accuracy: {acc:.2%}")

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=freqs)
        disp.plot(cmap='Blues')
        plt.title(f"{subj} Confusion Matrix (8–14 Hz)")
        plt.xlabel("Predicted Hz")
        plt.ylabel("True Hz")
        plt.tight_layout()
        plt.show()

    print("\n=== 全部完成 ===")
