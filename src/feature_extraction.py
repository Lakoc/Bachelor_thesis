import numpy as np
import librosa
import params
from scipy.fftpack import dct
from decorators import deprecated
import matplotlib.pyplot as plt


def calculate_energy_over_segments(segments):
    """Iterate over all segments and count energy ( E = sum(x^2))"""
    signal_power2 = segments ** 2
    return np.sum(signal_power2, axis=1)


def calculate_mfcc(segments, sampling_rate, mel_filters=40):
    """Extract mfcc from segments,
    inspired by http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/#eqn1
    and https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html
    used formulas:
        # freq -> mel scale
        # M(f) = 2595 * log10(1 + f/700)

        # mel scale -> freq
        # M^-1(m) = 700 (10^(m/2595) -1)
    26-40 filter banks usually used"""

    frames = segments[:, :, 0]

    # Hermitian-symmetric, i.e. the negative frequency terms are just the complex conjugates -> no need for redundancy
    # shannon theorem -> f/2.
    magnitude = np.absolute(np.fft.rfft(frames, params.frequency_resolution))
    power_spectrum = ((1.0 / params.frequency_resolution) * (magnitude ** 2))

    # Get mel bounds
    low_freq_mel = 0
    # Conversion from hz to mel
    high_freq_mel = (2595 * np.log10(1 + (sampling_rate / 2) / 700))

    # Equally space mel and then convert to Hz
    mel_points = np.linspace(low_freq_mel, high_freq_mel, mel_filters + 2)
    hz_points = (700 * (10 ** (mel_points / 2595) - 1))

    # Round to nearest bins on sampled frequency via params.frequency_resolution
    freq_bins = np.floor((params.frequency_resolution + 1) * hz_points / sampling_rate)

    # Create zeros bank
    mel_bank = np.zeros((mel_filters, int(np.floor(params.frequency_resolution / 2 + 1))))

    """For each filter get left, center, right sampled frequency -
    The first filter bank will start at the first point, reach its peak at the second point, 
    then return to zero at the 3rd point.
    """
    for m in range(1, mel_filters + 1):
        f_left = int(freq_bins[m - 1])
        f_center = int(freq_bins[m])
        f_right = int(freq_bins[m + 1])

        """ Sample each filter by following equation
            0 if k < f_left                                                  -
            (k − f_left) / (f_center − f_left) if k is before center        - -
            (f_right - k) / (f_right − f_center) if k is after center      -   -
            0 if k > f_right.                                             -     -               
        """
        for k in range(f_left, f_center):
            mel_bank[m - 1, k] = (k - freq_bins[m - 1]) / (freq_bins[m] - freq_bins[m - 1])
        for k in range(f_center, f_right):
            mel_bank[m - 1, k] = (freq_bins[m + 1] - k) / (freq_bins[m + 1] - freq_bins[m])

    # Apply banks on power_spectrum
    filter_banks = np.dot(power_spectrum, mel_bank.T)

    # Replace 0 with smallest float for numerical stability and
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    filter_banks = 20 * np.log10(filter_banks)

    # Decorrelate filter banks and compress it
    mfcc = dct(filter_banks, axis=1, norm='ortho')
    mfcc_compressed = mfcc[:, 1: (params.cepstral_coef_count + 1)]

    # Sinusoidal liftering used to de-emphasize higher MFCCs
    cep_lifter = 22
    (n_frames, n_coef) = mfcc_compressed.shape
    n = np.arange(n_coef)
    lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
    mfcc_compressed *= lift

    # Improve SNR by subtracting mean value
    filter_banks -= np.mean(filter_banks, axis=0)
    mfcc_compressed -= np.mean(mfcc_compressed, axis=0)

    return filter_banks, mfcc_compressed


@deprecated
def calculate_sign_changes(segmented_tracks):
    """Iterate over all segments and count sign change"""
    sign_changes = np.zeros((2, len(segmented_tracks[0])))
    for track_index, track in enumerate(segmented_tracks):
        for segment_index, segment in enumerate(track):
            # check for sign by comparing sign bit of two neighbor numbers
            sign_changes[track_index][segment_index] = np.sum(np.diff(np.signbit(segment)))
    return sign_changes


# TODO Fix
def calculate_lpc_over_segments(segmented_signals, lpc_coefficients_count):
    """Calculate lpc for every segment of speech"""

    def calculate_lpc(signal):
        x = librosa.core.lpc(signal, lpc_coefficients_count)
        #
        # """Return linear predictive coefficients for segment of speech using Levinson-Durbin prediction algorithm"""
        # # TODO: Remove unnecessary values
        # # Compute autocorrelation coefficients R_0 which is energy of signal in segment
        # r_vector = [signal @ signal]
        #
        # # if energy of signal is 0 return array like [1,0 x m,-1]
        # if r_vector[0] == 0:
        #     # return np.array(([1] + [0] * (lpc_coefficients_count - 2) + [-1]))
        #     # sum
        #     return np.zeros(lpc_coefficients_count)
        # else:
        #     # shift signal to calculate rest of autocorrelation coefficients
        #     for shift in range(1, lpc_coefficients_count + 1):
        #         # multiply vectors of signal[current index : end] @ signal[0:current shift] and append to r vector
        #         r = signal[shift:] @ signal[:-shift]
        #         r_vector.append(r)
        #     r_vector = np.array(r_vector)
        #
        #     # set initial values
        #     a_matrix = np.ones((lpc_coefficients_count + 1, lpc_coefficients_count + 1))
        #     error = r_vector[0]
        #
        #     # count first step
        #     a_matrix[1, 1] = (r_vector[1]) / error
        #     error *= (1 - a_matrix[1, 1] ** 2)
        #     # for each step add new coefficient and update coefficients by alpha
        #     for m in range(2, lpc_coefficients_count + 1):
        #         # set error to the small float to prevent division by zero
        #         if error == 0:
        #             error = 10e-20
        #         a_matrix[m, m] = (r_vector[m] - a_matrix[1:m, m - 1] @ r_vector[1:m][::-1]) / error
        #         for i in range(1, m):
        #             # calculate new alpha by multiplying existing coefficient with R[1:i+1] flipped, same as a[m,m]
        #             a_matrix[i, m] = a_matrix[i, m - 1] - a_matrix[m, m] * a_matrix[m - 1, m - 1]
        #         error *= (1 - a_matrix[m, m] ** 2)
        #     y = a_matrix[1:, lpc_coefficients_count]
        return x[1:]

    return np.apply_along_axis(calculate_lpc, 2, segmented_signals)
