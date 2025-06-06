import numpy as np
import h5py
from scipy.signal import resample, cheby1, filtfilt
from sklearn.cross_decomposition import CCA
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pickle
import fbcca
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

file_path = './SSVEP_dataset/data_s2_64.mat'
mat = h5py.File(file_path, 'r')
#print("Top-level keys:", list(mat.keys()))  # ['datas']
raw_data = np.array(mat['datas'])  # shape: (12, 60, 5140, 64, 2)

data = raw_data[:, :, :, :, 1]  # shape: (12, 60, 5140, 64)

freq_indices = np.arange(7, 14)
X = []
y = []
for block in range(12):
    for f_idx in freq_indices:
        trial = data[block, f_idx].T  # (64, 5140)
        X.append(trial)
        y.append(f_idx + 1)
X = np.array(X)
y = np.array(y)
#print("X shape:", X.shape)

fs_original = 1000
fs_target = 250
samples_target = int(X.shape[2] * fs_target / fs_original)
X_down = resample(X, samples_target, axis=2)
print("X downsampled shape:", X_down.shape)

selected_channels = [48, 54, 55, 56, 57, 58, 61, 62, 63]  # Pz, PO5, PO3, POz, PO4, PO6, O1, Oz, O2
X_down = X_down[:, selected_channels, :]

freqs = np.arange(8, 15)
features = np.array([fbcca.fbcca_full(trial, freqs, fs_target) for trial in X_down])

X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.2, random_state=42)
# clf = LinearSVC()
clf = SVC(kernel='rbf')

# clf = RandomForestClassifier()
# clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train, y_train)

# with open('model.pkl','wb') as f:
#     pickle.dump(clf,f)

y_pred = clf.predict(X_test)
print(y_pred)
acc = accuracy_score(y_test, y_pred)
print(f"Classification Accuracy: {acc:.2%}")

cm = confusion_matrix(y_test, y_pred, labels=freqs)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=freqs)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix - SSVEP Classification (FBCCA)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.grid(False)
plt.tight_layout()
plt.show()

def compute_itr(acc, N, T):
    from math import log2
    if acc == 0 or acc == 1:
        acc = max(0.0001, min(0.9999, acc))
    return (log2(N) + acc * log2(acc) + (1 - acc) * log2((1 - acc) / (N - 1))) * (60 / T)

