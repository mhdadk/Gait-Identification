import torch
import torchaudio
import pandas as pd
import librosa.display
import matplotlib.pyplot as plt
import scipy
import numpy as np

# load

x = pd.read_csv('../../../data/projF/train/x_val.csv').to_numpy()
x = torch.from_numpy(x)

# pick a window

window_length = 160
window = torch.transpose(x[:window_length,:],0,1).unsqueeze(dim=0)

# compute the log spectrogram

# f, t, Zxx = scipy.signal.stft(x = window[0],
#                               fs = 40,
#                               nperseg = window_length,
#                               noverlap = window_length//2,
#                               nfft = 512)

# zero mean each channel to avoid outliers in spectrograms

for i in range(len(window)):
    temp = window[:,i,:].transpose(0,1) - torch.mean(window[:,i,:],dim=1)
    window[:,i,:] = temp.transpose(0,1)

spec = torchaudio.transforms.Spectrogram(n_fft = window_length,
                                         win_length = window_length,
                                         hop_length = 2,
                                         pad = 0,
                                         power = 2.0,
                                         normalized = False)(window)
log_spec = torchaudio.transforms.AmplitudeToDB(stype = 'power')(spec)

# plot the log spectrogram

# librosa.display.specshow(log_spec.numpy()[0],
#                          y_axis='linear',
#                          x_axis='time',
#                          sr=40)

plt.imshow(log_spec[0,0].numpy(),cmap='winter')
plt.colorbar(format='%+2.0f dB')
# plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud')
plt.title('STFT Magnitude')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
# plt.show()
