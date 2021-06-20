import numpy as np
import cv2
import soundfile as sf
from tqdm import tqdm
import os



class CFG:
    n_mels = 256
    fmin = 20
    fmax = 16000
    n_fft = 2048
    hop_length = 2048
    sample_rate = 32000
    img_size = 512


def mono_to_color(
    X: np.ndarray, mean=None, std=None,
    norm_max=None, norm_min=None, eps=1e-6
):
    # Stack X as [X,X,X]
    X = np.stack([X,X,X], axis=-1)

    # Standardize
    mean = mean or X.mean()
    X = X - mean
    std = std or X.std()
    Xstd = X / (std + eps)
    _min, _max = Xstd.min(), Xstd.max()
    norm_max = norm_max or _max
    norm_min = norm_min or _min
    if (_max - _min) > eps:
        # Normalize to [0, 255]
        V = Xstd
        V[V < norm_min] = norm_min
        V[V > norm_max] = norm_max
        V = 255 * (V - norm_min) / (norm_max - norm_min)
        V = V.astype(np.uint8)
    else:
        # Just zero
        V = np.zeros_like(Xstd, dtype=np.uint8)
    return V


for file in tqdm(os.listdir('wavs')):
    start = file.split('.')[0].split('_')[-1] == 'start'

    if start:
        period = 200
    else:
        period = 180

    y, sr = sf.read('wavs/' + file)

    len_y = len(y)

    effective_length = sr * period
    if len_y < effective_length:
        new_y = np.zeros(effective_length, dtype=y.dtype)
        new_y[:len_y] = y
        y = new_y.astype(np.float32)
    elif len_y > effective_length:
        y = y[:effective_length].astype(np.float32)
    else:
        y = y.astype(np.float32)

    melspec = librosa.feature.melspectrogram(y, sr=sr, n_fft=CFG.n_fft, n_mels=CFG.n_mels, hop_length=CFG.hop_length,
                                             fmin=CFG.fmin, fmax=CFG.fmax)
    melspec = librosa.power_to_db(melspec).astype(np.float32)


    image = mono_to_color(melspec)

    image = cv2.resize(image, (CFG.img_size, CFG.img_size))

    image = np.moveaxis(image, 2, 0)
    # print(image.shape)
    # image = image.reshape(3, CFG.img_size, CFG.img_size)

    image = (image / 255.0).astype(np.float32)


    # cv2.imwrite('spectrogram/' + file.split('.')[0] + '.jpg', image)
    np.save('spectrogram/' + file.split('.')[0] + '.npy', image)