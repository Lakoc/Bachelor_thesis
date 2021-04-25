import numpy as np
import params
from scipy import fftpack, ndimage
from helpers.decorators import deprecated, timeit


def calculate_energy_over_segments(segments):
    """Iterate over all segments and count energy ( E = sum(x^2))"""
    signal_power2 = segments ** 2
    return np.sum(signal_power2, axis=1)


def calculate_rmse(segments):
    """Iterate over all segments and count root mean square energy( E = (1/N * sum(x^2))) ^ (1/2)"""
    signal_power2 = segments ** 2
    energy = np.sum(signal_power2, axis=1)
    rmse = np.sqrt(energy / segments.shape[1])
    return rmse


def normalize_energy(energy):
    """Normalize energy over segments"""
    normalized_energy = np.copy(energy)
    normalized_energy -= normalized_energy.mean()
    normalized_energy /= normalized_energy.std()
    return normalized_energy


def normalize_energy_to_1(energy):
    """Normalize energy over segments to interval <-1;1>"""
    normalized_energy = np.copy(energy)
    normalized_energy -= normalized_energy.mean()
    normalized_energy /= np.max(normalized_energy)
    return normalized_energy


def calculate_mfcc(segments, sampling_rate, mel_filters):
    """Extract mfcc from segments,
    inspired by http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/#eqn1
    and https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html
    used formulas:
        # freq -> mel scale
        # M(f) = 2595 * log10(1 + f/700)

        # mel scale -> freq
        # M^-1(m) = 700 (10^(m/2595) -1)
    26-40 filter banks usually used"""

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

    # Hermitian-symmetric, i.e. the negative frequency terms are just the complex conjugates, no need for redundancy
    # shannon theorem -> f/2.
    magnitude = np.absolute(np.fft.rfft(segments, params.frequency_resolution, axis=1))
    power_spectrum = ((1.0 / params.frequency_resolution) * (magnitude ** 2))

    # Apply banks on power_spectrum

    filter_banks = np.transpose(np.dot(np.transpose(power_spectrum, [0, 2, 1]), mel_bank.T), [0, 2, 1])

    # fig, ax = plt.subplots()
    # ax.plot(mel_bank.transpose(1,0))
    # fig.show()
    # Replace 0 with smallest float for numerical stability and
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    filter_banks = 20 * np.log10(filter_banks)

    # Decorrelate filter banks and compress it
    mfcc = fftpack.dct(filter_banks, axis=1, norm='ortho')
    mfcc_compressed = mfcc[:, 1: (params.cepstral_coef_count + 1), :]

    # Sinusoidal liftering used to de-emphasize higher MFCCs
    (_, n_coef, _) = mfcc_compressed.shape
    n = np.arange(n_coef)
    lift = 1 + (params.lifter / 2) * np.sin(np.pi * n / params.lifter)
    mfcc_compressed = np.transpose(lift * np.transpose(mfcc_compressed, [0, 2, 1]), [0, 2, 1])

    # # Improve SNR by subtracting mean value
    filter_banks -= np.mean(filter_banks, axis=0)
    mfcc_compressed -= np.mean(mfcc_compressed, axis=0)

    # np.savetxt('features.txt', mfcc_compressed[:, :, 0])
    return filter_banks, mfcc_compressed, power_spectrum


def calculate_delta(mfcc):
    """Calculate delta changes in mfcc features"""
    delta_neighbours = params.delta_neighbours

    # Calculate normalization coefficient
    normalization = 2 * np.sum(np.arange(1, delta_neighbours + 1) ** 2)

    # Create filter of neighbors for convolution
    neighbors = np.flip(np.arange(-delta_neighbours, delta_neighbours + 1)).reshape((1, -1, 1))

    # Transpose and add zeros for exact sum start
    mfcc = mfcc.transpose(1, 0, 2)
    mfcc_padded = np.pad(mfcc, ((0, 0), (delta_neighbours, delta_neighbours), (0, 0)), 'constant', constant_values=0)

    # Process convolution = sum over neighbors
    deltas = ndimage.convolve(mfcc_padded, neighbors)

    # Normalize, transpose and extract padded segments
    deltas = deltas / normalization
    deltas = deltas.transpose(1, 0, 2)
    # TODO: check
    return deltas[delta_neighbours: deltas.shape[0] - delta_neighbours, :, :]


def append_delta(mfcc, delta, d_delta):
    """Append delta and delta - delta coefficients to mfcc features"""
    mfcc_dd = np.append(mfcc, delta, axis=1)
    return np.append(mfcc_dd, d_delta, axis=1)


@deprecated
def calculate_sign_changes(segmented_tracks):
    """Iterate over all segments and count sign change"""
    sign_changes = np.zeros((2, len(segmented_tracks[0])))
    for track_index, track in enumerate(segmented_tracks):
        for segment_index, segment in enumerate(track):
            # check for sign by comparing sign bit of two neighbor numbers
            sign_changes[track_index][segment_index] = np.sum(np.diff(np.signbit(segment)))
    return sign_changes

