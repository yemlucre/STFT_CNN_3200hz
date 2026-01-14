from scipy.io import loadmat

# Load a single .mat file containing the vibration time-series
mat_path = "data/Outer Race/Orthogonal/0007/144.mat"
mat = loadmat(mat_path)

print(mat.keys())

# Extract the drive-end accelerometer channel
signal = mat['X144_DE_time']
print(signal.shape)

# Drop singleton dimensions to get a 1D array
import numpy as np
signal = signal.squeeze()
print(signal.shape)

# Quick look at the first 2000 samples of the raw waveform
import matplotlib.pyplot as plt
plt.figure(figsize=(10,4))
plt.plot(signal[:2000])
plt.title("Raw vibration signal (first 2000 samples)")
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()

# Sampling rate provided by the dataset
fs = 12000  # Hz

# Take a fixed-length segment for time-frequency analysis
segment_len = 2048
segment = signal[:segment_len]


# Compute STFT with Hann window to inspect frequency content over time
from scipy.signal import stft

f, t, Zxx = stft(
    segment,
    fs=fs,
    window='hann',
    nperseg=256,
    noverlap=128,
    nfft=256,
    padded=False,
    boundary=None
)


import numpy as np

# 只保留 200–3200 Hz
idx = (f >= 200) & (f <= 6400)
f = f[idx]
Zxx = Zxx[idx, :]


# Magnitude of the complex STFT result
spec = np.abs(Zxx)

# Plot the spectrogram for the selected segment
plt.figure(figsize=(6,4))
plt.pcolormesh(t, f, spec, shading='gouraud')
plt.title("STFT Spectrogram")
plt.ylabel("Frequency (Hz)")
plt.xlabel("Time (s)")
plt.colorbar(label='Magnitude')
plt.tight_layout()
plt.show()


