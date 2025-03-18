import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy
import skimage

import data_loader


def visualize_filter(f, r):
    """
    Function visualize the filter response
    :param f: frequency samples
    :param r: filter response
    :return: None
    """
    plt.figure()
    plt.plot(f, np.abs(r))
    plt.xlabel("Frequency in (Hz)")
    plt.ylabel("Magnitude")
    plt.title("Magnitude response")
    plt.grid(True)
    plt.show()


class Preprocessing:

    def __init__(self, sample_frequency):
        self.F = sample_frequency
        self.T = 1 / self.F

        # create filter coefficients
        self.b_1, self.a_1, frequency, response = self.design_bandpass_filter(order=4, lowcut=0.4, highcut=4)
        # visualize_filter(f=frequency, r=response)

    def load_data(self, filename):
        """
        Loading the data
        :param filename: file name of the data to load
        :return: raw PPG and ACC data and the labels as bpm of ECG
        along with a time vector and the used window duration/size
        """

        # bpm_ecg: ECG heart rate
        # bpm_proposed: ECG times
        # rawPPG: PPG signal
        # rawAcc: Accelerometer signal

        try:
            data = scipy.io.loadmat(filename)
            rawPPG = np.array(data["rawPPG"])
            rawACC = np.array(data["rawAcc"])
            bpmECG = np.array(data["bpm_ecg"])

            time = np.arange(0, len(rawPPG) * self.T, self.T)
            window = np.arange(0, 8 * self.F, 1)

            return rawPPG, rawACC, bpmECG, time, window

        except Exception as e:
            print("Error: ", e)

    def design_bandpass_filter(self, order, lowcut, highcut):
        """
        Creating the coefficients (b, a), frequency samples and filter response of a bandpass filter
        :param order: filter order
        :param lowcut: lower cutoff frequency
        :param highcut: higher cutoff frequency
        :param fs: sampling frequency
        :return: coefficients, frequency samples and filter response
        """
        nyq = 0.5 * self.F
        low_n = lowcut / nyq
        high_n = highcut / nyq

        # Design the Butterworth bandpass filter
        b, a = scipy.signal.butter(order, [low_n, high_n], btype='band')

        freq, resp = scipy.signal.freqz(b, a, fs=self.F)

        return b, a, freq, resp

    def create_windows(self, signal, window_size, shift, number_windows):
        """
        Creating the windows for the data
        :param number_windows: number of windows
        :param signal: signal data, here: PPG or ACC
        :param window_size: the window size or duration
        :param shift: the seconds to shift the window along the data
        :return: returns the created windows
        """
        windows = np.empty((number_windows, 400))
        start = 0
        end = window_size
        temp = 0
        while end <= len(signal) and temp < number_windows:
            window = signal[start:end]
            start += shift
            end += shift

            windows[temp] = window
            temp += 1
        return windows

    def filter_windows(self, windows, b, a, num_windows):
        """
        Function filters the windows using the created bandpass filter coefficients
        :param num_windows: number of windows
        :param windows: the window data which should be filtered
        :param b: coefficients of numerator
        :param a: coefficients of denominator
        :return: returns the filtered windows
        """
        w_filtered = np.empty((num_windows, windows.shape[1]))
        for x in range(0, num_windows):
            out = scipy.signal.filtfilt(b, a, windows[x])
            w_filtered[x] = out
        return w_filtered

    def create_power_spectrum(self, signal, number_windows):
        """
        Creates the spectra of each window in the frequency domain
        :param number_windows: number of windows
        :param signal: signal/window which will translated from time to frequency domain
        :return: returns the power spectra
        """
        power_spectrum = np.empty((number_windows, 2048))
        for i in range(0, len(signal)):
            out = np.fft.fft(signal[i])
            power_spectrum[i] = (np.abs(out) ** 2)
        return power_spectrum

    def zero_pad(self, signal, length, number_windows):
        """
        Applies zero padding on the previously created spectra
        :param number_windows: number of windows
        :param signal: spectrum which is then zero padded
        :param length: number of total frequency bins to pad
        :return: returns the padded spectrum
        """
        pad_signal = np.empty((number_windows, length))
        for i in range(0, number_windows):
            out = np.pad(signal[i], (0, length - len(signal[i])), mode="constant")
            pad_signal[i] = out
        return pad_signal

    def normalize_power_spectrum(self, spectrum, number_windows):
        """
        Normalizing the power spectra
        :param number_windows: number of windows
        :param spectrum: spectrum which is then normalized
        :return: returns the normalized spectrum
        """
        norm_spectrum = np.empty((number_windows, 2048))
        out = np.empty((number_windows, 2048))
        for i in range(0, len(spectrum)):
            min_val = np.min(spectrum[i])
            max_val = np.max(spectrum[i])

            norm_spectrum[i] = (spectrum[i] - min_val) / (max_val - min_val)

            # out[i] = np.linalg.norm(spectrum[i], keepdims=True)
            # norm_spectrum[i] = spectrum[i] / out[i]
        return norm_spectrum

    def extract_frequency(self, spectrum, min_freq, max_freq, number_windows):
        """
        Extracting/Filtering frequencies of the spectrum
        :param number_windows: number of windows
        :param spectrum: spectrum which is filtered
        :param min_freq: lower frequency
        :param max_freq: higher frequency
        :return: returns the spectrum with extracted frequencies
        """
        bin_axis = np.arange(0, 2047, 1)
        bin_res = 0.01215
        idx_min = min_freq / bin_res
        idx_max = max_freq / bin_res
        idx = np.where((bin_axis >= idx_min) & (bin_axis <= idx_max))

        extracted_spectrum = np.empty((number_windows, 222))
        for i in range(0, len(spectrum)):
            extracted_spectrum[i] = spectrum[i][idx]
        return extracted_spectrum

    def calc_mean_intensity_acc(self, windows):
        """
        Calculates the mean acc intensity using the absolute Hilbert signal method
        :return: returns the mean acc intensity (envelope signal of the mean acc data)
        """
        mean_window_intensity = np.empty((len(windows), 1))
        for i in range(0, len(windows)):
            mean_window_intensity[i] = np.mean(np.abs(scipy.signal.hilbert(windows[i])))
        return mean_window_intensity

    def stack_windows(self, ppg, acc, acc_int, hr, data_type):
        """
        windows are in shape of a list with numpy arrays, the function stacks the arrays up on each other
        :param ppg: preprocessed ppg windows
        :param acc: preprocessed acc windows
        :param acc_int: preprocessed intensity
        :param hr: ground truth to calculate the gaussian distribution
        :return: returns the three stacked data matrices, ppg and acc stacked together with shape (..., 2, 222),
        intensity with shape (..., 1, 1) and labels (ground truth gaussian distribution) with shape (..., 1, 222)
        """

        if data_type == "ISPC":
            temp_list = [np.array([ppg[i], acc[i]]) for i in range(len(ppg))]
            data = np.array(temp_list).reshape((len(temp_list), 2, 222))

            temp_list = hr
            labels = np.array(temp_list).reshape((len(temp_list), 1, 222))

            temp_list = acc_int
            intensity = np.array(temp_list).reshape((len(temp_list), 1, 1))

            return data, labels, intensity
        else:
            # Combine PPG and ACC data
            temp_list = [np.array([ppg[i][j], acc[i][j]]) for i in range(len(ppg)) for j in range(len(ppg[i]))]
            data = np.array(temp_list).reshape((len(temp_list), 2, 222))

            temp_list = [np.array(hr[i][j]) for i in range(len(hr)) for j in range(len(hr[i]))]
            labels = np.array(temp_list).reshape((len(temp_list), 1, 222))

            # Convert intensity data
            temp_list = [np.array(acc_int[i][j]) for i in range(len(acc_int)) for j in range(len(acc_int[i]))]
            intensity = np.array(temp_list).reshape((len(temp_list), 1, 1))

            return data, labels, intensity

    def split_into_sequence(self, X, y, intensity):
        """
        Splits the data into sequences of the size of 6
        :param X: data with features (ppg and acc windows stacked)
        :param intensity: data with features (acc intensity) -> concatenated with LSTM and previous layer
        :param y: data with labels (stacked gaussian distributions of true heart rate)
        :return: returns the sequences of X and y
        """
        time_steps = 6

        # Reshape the data to include the time step dimension
        num_samples = X.shape[0]
        sequence_size = num_samples // time_steps
        remainder = num_samples % time_steps

        if remainder != 0:
            # If the number of samples is not divisible by the number of timestamps,
            # you can either discard the remaining samples or pad the data to form
            # a complete batch. Here, we choose to discard the remaining samples.
            X = X[:num_samples - remainder]
            y = y[:num_samples - remainder]
            intensity = intensity[:num_samples - remainder]

        X = X.reshape(sequence_size, time_steps, 2, 222, 1)
        y = y.reshape(sequence_size, time_steps, 222)
        intensity = intensity.reshape(sequence_size, time_steps, 1)

        return X, y, intensity

    def gaussian_heart_rate(self, labels, num_samples=222, sigma=3 / 60):
        """
        Creates a Gaussian distribution
        :param labels: heart rate value per window
        :param num_samples: number of samples fitting to number of samples in window
        :param sigma: standard deviation, fixed set to 3
        :return:
        """
        out = []
        gaussian_frequency = np.linspace(0.6, 3.3, num_samples)

        for i in range(0, len(labels)):
            gaussian_samples = np.empty((len(labels[i]), num_samples))
            normalized_values = np.empty((len(labels[i]), num_samples))
            for j in range(0, len(labels[i])):
                gaussian_samples[j] = scipy.stats.norm.pdf(gaussian_frequency, labels[i][j] / 60, sigma)
                normalized_values[j] = gaussian_samples[j] / gaussian_samples[j].max()
            out.append(normalized_values)

        return out

    def single_heart_rate(self, labels, num_samples, data_type):
        frequency = np.linspace(0.6, 3.3, num_samples)
        if data_type == "ISPC":
            samples = np.zeros((len(labels), num_samples))
            for i in range(0, len(labels)):
                idx = np.round((((labels[i] / 60) - np.min(frequency)) /
                                ((np.max(frequency) - np.min(frequency)) / num_samples)))
                samples[i][int(idx)] = 1
            return samples
        else:
            out = []
            for i in range(0, len(labels)):
                samples = np.zeros((len(labels[i]), num_samples))
                for j in range(0, len(labels[i])):
                    idx = np.round((((labels[i][j] / 60) - np.min(frequency)) /
                                    ((np.max(frequency) - np.min(frequency)) / num_samples)))
                    samples[j][int(idx)] = 1
                out.append(samples)
            return out

    def processing_BAMI1(self):

        power_spectra_ppg = []
        power_spectra_acc = []
        intensity_acc = []
        ground_truth = []
        number_windows = []

        for i in range(1, 26):
            # load data
            PPG, ACC, GT, t, w = self.load_data("BAMI-1/BAMI1_{}.mat".format(i))
            number_windows.append(len(GT))

            # save ground truth data
            ground_truth.append(GT)

            # create the windows of all 3 PPGs
            windows_ppg_1 = self.create_windows(PPG[0], len(w), 2 * self.F, number_windows=len(GT))
            windows_ppg_2 = self.create_windows(PPG[1], len(w), 2 * self.F, number_windows=len(GT))
            windows_ppg_3 = self.create_windows(PPG[2], len(w), 2 * self.F, number_windows=len(GT))

            # filter all windows of all 3 PPGs
            filtered_windows_ppg_1 = self.filter_windows(windows_ppg_1, self.b_1, self.a_1, num_windows=len(GT))
            filtered_windows_ppg_2 = self.filter_windows(windows_ppg_2, self.b_1, self.a_1, num_windows=len(GT))
            filtered_windows_ppg_3 = self.filter_windows(windows_ppg_3, self.b_1, self.a_1, num_windows=len(GT))

            # normalize all windows of all 3 PPGs
            norm_filtered_windows_ppg_1 = scipy.stats.zscore(filtered_windows_ppg_1)
            norm_filtered_windows_ppg_2 = scipy.stats.zscore(filtered_windows_ppg_2)
            norm_filtered_windows_ppg_3 = scipy.stats.zscore(filtered_windows_ppg_3)

            # calculate the mean of all windows of all 3 PPGs
            norm_filtered_windows_ppg = np.mean([norm_filtered_windows_ppg_1, norm_filtered_windows_ppg_2,
                                                 norm_filtered_windows_ppg_3], axis=0)

            # resample the mean filtered windows
            ds_norm_filtered_windows_ppg = skimage.transform.resize(image=norm_filtered_windows_ppg,
                                                                    output_shape=(len(norm_filtered_windows_ppg), 200))

            # zero pad the down sampled windows to length of 2048
            pad_windows_ppg = self.zero_pad(ds_norm_filtered_windows_ppg, 2048, number_windows=len(GT))

            # calculate the power spectrum
            power_spectrum_ppg = self.create_power_spectrum(pad_windows_ppg, number_windows=len(GT))

            # calculate the normalized power spectrum
            norm_power_spectrum_ppg = self.normalize_power_spectrum(power_spectrum_ppg, number_windows=len(GT))

            # extract the power spectrum in range between 0.6 and 3.3 Hz
            extract_power_spectrum_ppg = self.extract_frequency(norm_power_spectrum_ppg, 0.6, 3.3,
                                                                number_windows=len(GT))

            # extract_power_spectrum_ppg = np.expand_dims(extract_power_spectrum_ppg, axis=2)
            power_spectra_ppg.append(extract_power_spectrum_ppg)

            ############################################################################################################

            # create the windows of all 3 ACCs
            windows_acc_x = self.create_windows(ACC[0], len(w), 2 * self.F, number_windows=len(GT))
            windows_acc_y = self.create_windows(ACC[1], len(w), 2 * self.F, number_windows=len(GT))
            windows_acc_z = self.create_windows(ACC[2], len(w), 2 * self.F, number_windows=len(GT))

            # filter all windows of all 3 ACCs
            filtered_windows_acc_x = self.filter_windows(windows_acc_x, self.b_1, self.a_1, num_windows=len(GT))
            filtered_windows_acc_y = self.filter_windows(windows_acc_y, self.b_1, self.a_1, num_windows=len(GT))
            filtered_windows_acc_z = self.filter_windows(windows_acc_z, self.b_1, self.a_1, num_windows=len(GT))

            # resample the mean filtered windows
            ds_filtered_windows_acc_x = skimage.transform.resize(filtered_windows_acc_x,
                                                                 (len(filtered_windows_acc_x), 200))
            ds_filtered_windows_acc_y = skimage.transform.resize(filtered_windows_acc_y,
                                                                 (len(filtered_windows_acc_y), 200))
            ds_filtered_windows_acc_z = skimage.transform.resize(filtered_windows_acc_z,
                                                                 (len(filtered_windows_acc_z), 200))

            # zero pad the down sampled windows to length of 2048
            pad_windows_acc_x = self.zero_pad(ds_filtered_windows_acc_x, 2048, number_windows=len(GT))
            pad_windows_acc_y = self.zero_pad(ds_filtered_windows_acc_y, 2048, number_windows=len(GT))
            pad_windows_acc_z = self.zero_pad(ds_filtered_windows_acc_z, 2048, number_windows=len(GT))

            # calculate the power spectrum
            power_spectrum_acc_x = self.create_power_spectrum(pad_windows_acc_x, number_windows=len(GT))
            power_spectrum_acc_y = self.create_power_spectrum(pad_windows_acc_y, number_windows=len(GT))
            power_spectrum_acc_z = self.create_power_spectrum(pad_windows_acc_z, number_windows=len(GT))

            # calculate the normalized power spectrum
            norm_power_spectrum_acc_x = self.normalize_power_spectrum(power_spectrum_acc_x, number_windows=len(GT))
            norm_power_spectrum_acc_y = self.normalize_power_spectrum(power_spectrum_acc_y, number_windows=len(GT))
            norm_power_spectrum_acc_z = self.normalize_power_spectrum(power_spectrum_acc_z, number_windows=len(GT))

            norm_power_spectrum_acc = np.mean([norm_power_spectrum_acc_x,
                                               norm_power_spectrum_acc_y,
                                               norm_power_spectrum_acc_z], axis=0)

            # extract the power spectrum in range between 0.6 and 3.3 Hz
            extract_power_spectrum_acc = self.extract_frequency(norm_power_spectrum_acc, 0.6, 3.3,
                                                                number_windows=len(GT))

            # extract_power_spectrum_acc = np.expand_dims(extract_power_spectrum_acc, axis=2)
            power_spectra_acc.append(extract_power_spectrum_acc)

            ############################################################################################################

            mean_intensity_acc_x = self.calc_mean_intensity_acc(ds_filtered_windows_acc_x)
            mean_intensity_acc_y = self.calc_mean_intensity_acc(ds_filtered_windows_acc_y)
            mean_intensity_acc_z = self.calc_mean_intensity_acc(ds_filtered_windows_acc_z)
            intensity_acc.append(np.mean([mean_intensity_acc_x, mean_intensity_acc_y, mean_intensity_acc_z], axis=0))

        ################################################################################################################

        # Convert Gaussian heart rate labels
        labels_list = self.gaussian_heart_rate(ground_truth)
        labels_list_single = self.single_heart_rate(ground_truth, num_samples=222, data_type=None)

        data_array, labels_array, intensity_acc = \
            self.stack_windows(ppg=power_spectra_ppg, acc=power_spectra_acc, acc_int=intensity_acc,
                               hr=labels_list_single, data_type=None)

        X, y, intensity = self.split_into_sequence(X=data_array, y=labels_array, intensity=intensity_acc)

        return X, y, intensity

    def processing_BAMI2(self):

        power_spectra_ppg = []
        power_spectra_acc = []
        intensity_acc = []
        ground_truth = []
        number_windows = []

        for i in range(1, 24):
            # load data
            PPG, ACC, GT, t, w = self.load_data("BAMI-2/BAMI2_{}.mat".format(i))
            number_windows.append(len(GT))

            # save ground truth data
            ground_truth.append(GT)

            # create the windows of all 3 PPGs
            windows_ppg_1 = self.create_windows(PPG[0], len(w), 2 * self.F, number_windows=len(GT))
            windows_ppg_2 = self.create_windows(PPG[1], len(w), 2 * self.F, number_windows=len(GT))
            windows_ppg_3 = self.create_windows(PPG[2], len(w), 2 * self.F, number_windows=len(GT))

            # filter all windows of all 3 PPGs
            filtered_windows_ppg_1 = self.filter_windows(windows_ppg_1, self.b_1, self.a_1, num_windows=len(GT))
            filtered_windows_ppg_2 = self.filter_windows(windows_ppg_2, self.b_1, self.a_1, num_windows=len(GT))
            filtered_windows_ppg_3 = self.filter_windows(windows_ppg_3, self.b_1, self.a_1, num_windows=len(GT))

            # normalize all windows of all 3 PPGs
            norm_filtered_windows_ppg_1 = scipy.stats.zscore(filtered_windows_ppg_1)
            norm_filtered_windows_ppg_2 = scipy.stats.zscore(filtered_windows_ppg_2)
            norm_filtered_windows_ppg_3 = scipy.stats.zscore(filtered_windows_ppg_3)

            # calculate the mean of all windows of all 3 PPGs
            norm_filtered_windows_ppg = np.mean([norm_filtered_windows_ppg_1, norm_filtered_windows_ppg_2,
                                                 norm_filtered_windows_ppg_3], axis=0)

            # resample the mean filtered windows
            ds_norm_filtered_windows_ppg = skimage.transform.resize(image=norm_filtered_windows_ppg,
                                                                    output_shape=(len(norm_filtered_windows_ppg), 200))

            # zero pad the down sampled windows to length of 2048
            pad_windows_ppg = self.zero_pad(ds_norm_filtered_windows_ppg, 2048, number_windows=len(GT))

            # calculate the power spectrum
            power_spectrum_ppg = self.create_power_spectrum(pad_windows_ppg, number_windows=len(GT))

            # calculate the normalized power spectrum
            norm_power_spectrum_ppg = self.normalize_power_spectrum(power_spectrum_ppg, number_windows=len(GT))

            # extract the power spectrum in range between 0.6 and 3.3 Hz
            extract_power_spectrum_ppg = self.extract_frequency(norm_power_spectrum_ppg, 0.6, 3.3,
                                                                number_windows=len(GT))

            # extract_power_spectrum_ppg = np.expand_dims(extract_power_spectrum_ppg, axis=2)
            power_spectra_ppg.append(extract_power_spectrum_ppg)

            ############################################################################################################

            # create the windows of all 3 ACCs
            windows_acc_x = self.create_windows(ACC[0], len(w), 2 * self.F, number_windows=len(GT))
            windows_acc_y = self.create_windows(ACC[1], len(w), 2 * self.F, number_windows=len(GT))
            windows_acc_z = self.create_windows(ACC[2], len(w), 2 * self.F, number_windows=len(GT))

            # filter all windows of all 3 ACCs
            filtered_windows_acc_x = self.filter_windows(windows_acc_x, self.b_1, self.a_1, num_windows=len(GT))
            filtered_windows_acc_y = self.filter_windows(windows_acc_y, self.b_1, self.a_1, num_windows=len(GT))
            filtered_windows_acc_z = self.filter_windows(windows_acc_z, self.b_1, self.a_1, num_windows=len(GT))

            # resample the mean filtered windows
            ds_filtered_windows_acc_x = skimage.transform.resize(filtered_windows_acc_x,
                                                                 (len(filtered_windows_acc_x), 200))
            ds_filtered_windows_acc_y = skimage.transform.resize(filtered_windows_acc_y,
                                                                 (len(filtered_windows_acc_y), 200))
            ds_filtered_windows_acc_z = skimage.transform.resize(filtered_windows_acc_z,
                                                                 (len(filtered_windows_acc_z), 200))

            # zero pad the down sampled windows to length of 2048
            pad_windows_acc_x = self.zero_pad(ds_filtered_windows_acc_x, 2048, number_windows=len(GT))
            pad_windows_acc_y = self.zero_pad(ds_filtered_windows_acc_y, 2048, number_windows=len(GT))
            pad_windows_acc_z = self.zero_pad(ds_filtered_windows_acc_z, 2048, number_windows=len(GT))

            # calculate the power spectrum
            power_spectrum_acc_x = self.create_power_spectrum(pad_windows_acc_x, number_windows=len(GT))
            power_spectrum_acc_y = self.create_power_spectrum(pad_windows_acc_y, number_windows=len(GT))
            power_spectrum_acc_z = self.create_power_spectrum(pad_windows_acc_z, number_windows=len(GT))

            # calculate the normalized power spectrum
            norm_power_spectrum_acc_x = self.normalize_power_spectrum(power_spectrum_acc_x, number_windows=len(GT))
            norm_power_spectrum_acc_y = self.normalize_power_spectrum(power_spectrum_acc_y, number_windows=len(GT))
            norm_power_spectrum_acc_z = self.normalize_power_spectrum(power_spectrum_acc_z, number_windows=len(GT))

            norm_power_spectrum_acc = np.mean([norm_power_spectrum_acc_x,
                                               norm_power_spectrum_acc_y,
                                               norm_power_spectrum_acc_z], axis=0)

            # extract the power spectrum in range between 0.6 and 3.3 Hz
            extract_power_spectrum_acc = self.extract_frequency(norm_power_spectrum_acc, 0.6, 3.3,
                                                                number_windows=len(GT))

            # extract_power_spectrum_acc = np.expand_dims(extract_power_spectrum_acc, axis=2)
            power_spectra_acc.append(extract_power_spectrum_acc)

            ############################################################################################################

            mean_intensity_acc_x = self.calc_mean_intensity_acc(ds_filtered_windows_acc_x)
            mean_intensity_acc_y = self.calc_mean_intensity_acc(ds_filtered_windows_acc_y)
            mean_intensity_acc_z = self.calc_mean_intensity_acc(ds_filtered_windows_acc_z)
            intensity_acc.append(np.mean([mean_intensity_acc_x, mean_intensity_acc_y, mean_intensity_acc_z], axis=0))

        ################################################################################################################

        # Convert Gaussian heart rate labels
        labels_list = self.gaussian_heart_rate(labels=ground_truth)
        labels_list_single = self.single_heart_rate(labels=ground_truth, num_samples=222, data_type=None)

        data_array, labels_array, intensity_acc = \
            self.stack_windows(ppg=power_spectra_ppg, acc=power_spectra_acc, acc_int=intensity_acc,
                               hr=labels_list_single, data_type=None)

        X, y, intensity = self.split_into_sequence(X=data_array, y=labels_array, intensity=intensity_acc)

        return X, y, intensity

    def preprocessing_ISPC(self):
        X_train, y_train = data_loader.load_from_tsfile_to_dataframe(full_file_path_and_name="ISPC/IEEEPPG_TRAIN.ts",
                                                                     return_separate_X_and_y=True)
        X_test, y_test = data_loader.load_from_tsfile_to_dataframe(full_file_path_and_name="ISPC/IEEEPPG_TEST.ts",
                                                                   return_separate_X_and_y=True)
        raw_X, raw_y = pd.concat([X_train, X_test], axis=0), np.concatenate((y_train, y_test), axis=0)

        windows_ppg_1 = np.empty((raw_X.shape[0], 1000))
        windows_ppg_2 = np.empty((raw_X.shape[0], 1000))
        windows_acc_x = np.empty((raw_X.shape[0], 1000))
        windows_acc_y = np.empty((raw_X.shape[0], 1000))
        windows_acc_z = np.empty((raw_X.shape[0], 1000))

        for i in range(0, raw_X.shape[0]):
            windows_ppg_1[i] = raw_X.iloc[i, 0]
            windows_ppg_2[i] = raw_X.iloc[i, 1]
            windows_acc_x[i] = raw_X.iloc[i, 2]
            windows_acc_y[i] = raw_X.iloc[i, 3]
            windows_acc_z[i] = raw_X.iloc[i, 4]

        filtered_windows_ppg_1 = self.filter_windows(windows=windows_ppg_1, b=self.b_1, a=self.a_1,
                                                     num_windows=len(windows_ppg_1))
        filtered_windows_ppg_2 = self.filter_windows(windows=windows_ppg_2, b=self.b_1, a=self.a_1,
                                                     num_windows=len(windows_ppg_2))

        norm_filtered_windows_ppg_1 = scipy.stats.zscore(filtered_windows_ppg_1)
        norm_filtered_windows_ppg_2 = scipy.stats.zscore(filtered_windows_ppg_2)

        norm_filtered_windows_ppg = np.mean([norm_filtered_windows_ppg_1, norm_filtered_windows_ppg_2], axis=0)

        # resample the mean filtered windows
        ds_norm_filtered_windows_ppg = skimage.transform.resize(image=norm_filtered_windows_ppg,
                                                                output_shape=(len(norm_filtered_windows_ppg), 200))

        # zero pad the down sampled windows to length of 2048
        pad_windows_ppg = self.zero_pad(ds_norm_filtered_windows_ppg, 2048,
                                        number_windows=len(ds_norm_filtered_windows_ppg))

        # calculate the power spectrum
        power_spectrum_ppg = self.create_power_spectrum(pad_windows_ppg,
                                                        number_windows=len(ds_norm_filtered_windows_ppg))

        # calculate the normalized power spectrum
        norm_power_spectrum_ppg = self.normalize_power_spectrum(power_spectrum_ppg,
                                                                number_windows=len(ds_norm_filtered_windows_ppg))

        # extract the power spectrum in range between 0.6 and 3.3 Hz
        extract_power_spectrum_ppg = self.extract_frequency(norm_power_spectrum_ppg, 0.6, 3.3,
                                                            number_windows=len(ds_norm_filtered_windows_ppg))

        power_spectra_ppg = extract_power_spectrum_ppg

        ################################################################################################################

        filtered_windows_acc_x = self.filter_windows(windows=windows_acc_x, b=self.b_1, a=self.a_1,
                                                     num_windows=len(windows_acc_x))
        filtered_windows_acc_y = self.filter_windows(windows=windows_acc_y, b=self.b_1, a=self.a_1,
                                                     num_windows=len(windows_acc_y))
        filtered_windows_acc_z = self.filter_windows(windows=windows_acc_z, b=self.b_1, a=self.a_1,
                                                     num_windows=len(windows_acc_z))

        # resample the mean filtered windows
        ds_filtered_windows_acc_x = skimage.transform.resize(filtered_windows_acc_x,
                                                             (len(filtered_windows_acc_x), 200))
        ds_filtered_windows_acc_y = skimage.transform.resize(filtered_windows_acc_y,
                                                             (len(filtered_windows_acc_y), 200))
        ds_filtered_windows_acc_z = skimage.transform.resize(filtered_windows_acc_z,
                                                             (len(filtered_windows_acc_z), 200))

        # zero pad the down sampled windows to length of 2048
        pad_windows_acc_x = self.zero_pad(ds_filtered_windows_acc_x, 2048, number_windows=len(windows_acc_x))
        pad_windows_acc_y = self.zero_pad(ds_filtered_windows_acc_y, 2048, number_windows=len(windows_acc_y))
        pad_windows_acc_z = self.zero_pad(ds_filtered_windows_acc_z, 2048, number_windows=len(windows_acc_z))

        # calculate the power spectrum
        power_spectrum_acc_x = self.create_power_spectrum(pad_windows_acc_x, number_windows=len(windows_acc_x))
        power_spectrum_acc_y = self.create_power_spectrum(pad_windows_acc_y, number_windows=len(windows_acc_y))
        power_spectrum_acc_z = self.create_power_spectrum(pad_windows_acc_z, number_windows=len(windows_acc_z))

        # calculate the normalized power spectrum
        norm_power_spectrum_acc_x = self.normalize_power_spectrum(power_spectrum_acc_x, number_windows=len(windows_acc_x))
        norm_power_spectrum_acc_y = self.normalize_power_spectrum(power_spectrum_acc_y, number_windows=len(windows_acc_x))
        norm_power_spectrum_acc_z = self.normalize_power_spectrum(power_spectrum_acc_z, number_windows=len(windows_acc_x))

        norm_power_spectrum_acc = np.mean([norm_power_spectrum_acc_x,
                                           norm_power_spectrum_acc_y,
                                           norm_power_spectrum_acc_z], axis=0)

        # extract the power spectrum in range between 0.6 and 3.3 Hz
        extract_power_spectrum_acc = self.extract_frequency(norm_power_spectrum_acc, 0.6, 3.3,
                                                            number_windows=len(norm_power_spectrum_acc))

        # extract_power_spectrum_acc = np.expand_dims(extract_power_spectrum_acc, axis=2)
        power_spectra_acc = extract_power_spectrum_acc

        ################################################################################################################

        mean_intensity_acc_x = self.calc_mean_intensity_acc(ds_filtered_windows_acc_x)
        mean_intensity_acc_y = self.calc_mean_intensity_acc(ds_filtered_windows_acc_y)
        mean_intensity_acc_z = self.calc_mean_intensity_acc(ds_filtered_windows_acc_z)
        intensity_acc = np.mean([mean_intensity_acc_x, mean_intensity_acc_y, mean_intensity_acc_z], axis=0)

        ################################################################################################################

        # Convert Gaussian heart rate labels
        # labels_list = self.gaussian_heart_rate(labels=raw_y)
        labels_list_single = self.single_heart_rate(labels=raw_y, num_samples=222, data_type="ISPC")

        data_array, labels_array, intensity_acc = \
            self.stack_windows(ppg=power_spectra_ppg, acc=power_spectra_acc, acc_int=intensity_acc,
                               hr=labels_list_single, data_type="ISPC")

        X, y, intensity = self.split_into_sequence(X=data_array, y=labels_array, intensity=intensity_acc)

        return X, y, intensity
