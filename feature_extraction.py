from keras.utils import to_categorical
import pyeeg as pe
import pickle as pickle
from antropy import perm_entropy
from sklearn.preprocessing import normalize
import os
import time
import numpy as np

def calc_wavelet_energy(data_set):
  """
    Input : 1 * N vector
    Output: Float with the wavelet energy of the input vector,
    rounded to 3 decimal places.
  """
  # p_sqr = [i ** 2 for i in data_set]
  wavelet_energy = np.nansum(np.log2(np.square(data_set)))
  return round(wavelet_energy, 3)


def FFT_Processing (sub, channel, band, window_size, step_size, sample_rate, input_path, output_path):
    '''
    arguments:  string subject
                list channel indice
                list band
                int window size for FFT
                int step size for FFT
                int sample rate for FFT
    return:     void
    '''
    meta = []
    with open(input_path + sub + '.dat', 'rb') as file:

        subject = pickle.load(file, encoding='latin1')

        for i in range (0,40):
            # loop over 0-39 trails
            data = subject["data"][i]
            labels = subject["labels"][i]
            start = 384;

#            while start + window_size < data.shape[1]:
            while start + window_size < data.shape[1]:
                meta_array = []
                meta_data = []
                for j in channel:
                    X = data[j][start : start + window_size]
                    Y = pe.bin_power(X, band, sample_rate)
                    meta_data = meta_data + list(Y[0])

                meta_array.append(np.array(meta_data))
                meta_array.append(labels)

                meta.append(np.array(meta_array))
                start = start + step_size

        meta = np.array(meta)
        print(meta.shape)
        np.save(output_path + sub, meta, allow_pickle=True, fix_imports=True)


def feature_extraction(sub, channel, band, window_size, step_size, sample_rate, input_path, output_path, is_3D=False):
    import os
    meta = []
    with open(os.path.join(input_path, sub + '.dat'), 'rb') as file:
        subject = pickle.load(file, encoding='latin1')
        data_all = subject["data"]
        labels_all = subject["labels"]

        for trial_idx in range(40):
            data = data_all[trial_idx]      # shape (40, 8064)
            labels = labels_all[trial_idx]  # shape (4,)

            # Precompute permutation entropy per channel (once per trial)
            pe_trial = [
                perm_entropy(data[ch], order=3, delay=1, normalize=True)
                for ch in channel
            ]
            pe_trial = np.array(pe_trial)

            start = 384
            while start + window_size <= data.shape[1]:
                band_matrix = []

                for ch in channel:
                    segment = data[ch][start:start + window_size]
                    powers, _ = pe.bin_power(segment, band, sample_rate)
                    band_matrix.append(powers)

                band_matrix = np.array(band_matrix)  # (14, 5)
                total_power = band_matrix.sum(axis=1) + 1e-8
                gamma_ratio = band_matrix[:, -1] / total_power  # (14,)

                sample_feature = []
                for ch in range(len(channel)):
                    for b in range(len(band) - 1):
                        bp = band_matrix[ch, b]
                        gr = gamma_ratio[ch]
                        pe_val = pe_trial[ch]
                        sample_feature.append([bp, gr, pe_val])

                sample_feature = np.array(sample_feature)  # shape: (70, 3)

                if is_3D:
                    # pad with zeros and reshape to (70, 2, 2)
                    sample_feature = np.concatenate([sample_feature, np.zeros((70, 1))], axis=1)
                    sample_feature = sample_feature.reshape(70, 2, 2)

                meta.append((sample_feature, labels))
                start += step_size

    meta = np.array(meta, dtype=object)
    np.save(os.path.join(output_path, sub + ".npy"), meta, allow_pickle=True)

def load_data(
    subject_list,
    data_dir,
    label_index=3,
    samples_per_trial=464
):
    """
    Load DEAP features and one-hot labels for 10-class classification (scores 0–9).
    """

    data_training = []
    label_training = []
    data_testing = []
    label_testing = []

    for subj in subject_list:
        filepath = os.path.join(data_dir, f"s{subj}.npy")
        sub_data = np.load(filepath, allow_pickle=True)

        i = 0
        while i < len(sub_data):
            if i % (5 * samples_per_trial) == 0:
                for _ in range(samples_per_trial):
                    data_testing.append(sub_data[i][0])
                    label_testing.append(sub_data[i][1][label_index])
                    i += 1
            else:
                data_training.append(sub_data[i][0])
                label_training.append(sub_data[i][1][label_index])
                i += 1

    # Convert to numpy arrays
    X_train = np.array(data_training)  # (N, 70, 3)
    X_test = np.array(data_testing)
    y_train_raw = np.array(label_training).astype(int)  # (N,)
    y_test_raw = np.array(label_testing).astype(int)

    # Normalize feature vectors
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    X_train_norm = normalize(X_train_flat)
    X_test_norm = normalize(X_test_flat)
    X_train = X_train_norm.reshape(X_train.shape)
    X_test = X_test_norm.reshape(X_test.shape)

    # One-hot encode 10 classes (0–9)
    y_train = to_categorical(y_train_raw, num_classes=10)
    y_test = to_categorical(y_test_raw, num_classes=10)

    return X_train, y_train, X_test, y_test

