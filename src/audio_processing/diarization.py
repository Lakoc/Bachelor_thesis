import numpy as np
import params
from scipy import ndimage
from copy import deepcopy
from helpers.gmm import MyGmm
from helpers.diarization_helpers import extract_features_for_diarization, likelihood_propagation_matrix, \
    single_gmm_update
from helpers.propagation import mean_filter
from scipy.special import logsumexp


def return_diarization_index(likelihoods, active_segments):
    """Find higher likelihood, transform, remove nonactive segments"""
    diarization = np.argmax(likelihoods, axis=1)
    diarization += 1
    diarization *= active_segments
    likelihoods_diff = likelihoods[:, 0] - likelihoods[:, 1]
    return diarization, likelihoods_diff


def diar_thresholding_step(diarization, likelihoods_diff):
    """Process diar post-processing - waiting for edge bigger than threshold to change state"""
    diar_post = np.copy(diarization)
    new_val = -10
    thresholds = [np.abs(np.min(likelihoods_diff)) / params.prob_threshold_normalization, 0,
                  np.max(likelihoods_diff) / params.prob_threshold_normalization]
    for index, segment in enumerate(likelihoods_diff):
        # if speech was not present, nothing changes
        if diarization[index] != 0:
            sign = np.sign(segment).astype(int)
            # in case of finding edge
            if np.abs(segment) > thresholds[sign + 1]:
                new_val = sign
                # replace segments before that was not classified yet
                diar_post[diar_post == -10] = sign
            diar_post[index] = new_val
        else:
            # end of activity null speaker
            new_val = -10
    return diar_post


def energy_based_diarization(energy, vad):
    """Simple energy based diarization, if voice activity was detected, choose higher energy as speaker"""
    higher_energy = np.argmax(energy, axis=1)
    vad_summed = np.sum(vad, axis=1)
    diarized = np.where(vad_summed > 0, higher_energy + 1, 0)
    diarized_active = diarized[diarized > 0]
    diarized_active = ndimage.median_filter(diarized_active,
                                            int(params.diarization_output_mean_filter / params.window_stride))
    diarized[diarized > 0] = diarized_active
    return diarized


def gmm_mfcc_diarization_1channel(mfcc, vad, energy):
    """Train Gaussian mixture models with mfcc features over speech segments"""

    active_segments, active_segments_index, features_ch1, _, energy_difference = extract_features_for_diarization(mfcc,
                                                                                                                  vad,
                                                                                                                  energy)
    energy_difference = mean_filter(energy_difference,
                                    int(params.diarization_init_array_filter / params.window_stride))

    # Train UBM GMM - speaker independent
    features_active = features_ch1[active_segments_index]
    gmm = MyGmm(n_components=params.gmm_components, verbose=params.gmm_verbose, covariance_type='diag',
                max_iter=params.gmm_max_iterations,
                tol=params.gmm_error_rate).fit(features_active)

    # Create model for each speaker
    gmm1 = deepcopy(gmm)
    gmm2 = deepcopy(gmm)

    # Move means in direction of each speaker
    likelihoods = single_gmm_update(gmm1, gmm2, features_ch1, energy_difference, active_segments_index)

    # Normalize likelihoods and process forward backward propagation
    log_evidence = logsumexp(likelihoods, axis=1)
    likelihoods = np.exp(likelihoods - log_evidence[:, np.newaxis])
    likelihoods = likelihood_propagation_matrix(likelihoods)

    # Calculate difference of each model likelihoods
    likelihoods_difference = likelihoods[:, 0] - likelihoods[:, 1]

    # Smooth difference
    likelihoods_difference = mean_filter(likelihoods_difference,
                                         int(params.diarization_init_array_filter / params.window_stride))

    # Get output diarization
    diarization = np.sign(likelihoods_difference) * active_segments

    # Process thresholding
    diarization = diar_thresholding_step(diarization, likelihoods_difference)

    # thresh = (np.max(likelihoods_difference) + np.abs(np.min(likelihoods_difference))) / 15
    # prev = 0
    # diarization = np.sign(likelihoods_difference) * active_segments

    # energy_mean = np.mean(energy_difference)
    # axs[0].legend()
    # axs[1].plot(time, energy_difference, label='Rozdíl energií kanálů')
    # axs[1].plot(time, zeros, label='Práh')
    # axs[1].legend(fontsize=14)
    # axs[1].set_xlabel('Čas [min]', fontsize=18)
    # axs[1].set_ylabel('Velikost', fontsize=18)
    # plt.xticks(fontsize=16)
    # axs[1].tick_params(axis='both', labelsize=14)

    # ax.plot(time, diarization, color='red', label='hyp')
    # ax.plot(time, zeros, label='threshold')
    # plt.xticks(np.arange(0, 601, 10.0), rotation=45)

    # hyp = np.sign(energy_difference)
    # hyp *= active_segments
    #
    # confusion = np.sum((ref != hyp)) / ref.shape[0] * 100
    #
    # hyp = np.sign(likelihoods_difference)
    # hyp *= active_segments

    #
    # # Train UBM GMM - speaker independent
    # mfcc = np.append(mfcc[:, :, 0], mfcc[:, :, 1], axis=1)
    # features_active = mfcc[active_segments_index]
    # gmm = MyGmm(n_components=params.gmm_components, verbose=params.gmm_verbose, covariance_type='diag',
    #             max_iter=params.gmm_max_iterations,
    #             tol=params.gmm_error_rate).fit(features_active)
    #
    # # Create model for each speaker
    # gmm1 = deepcopy(gmm)
    # gmm2 = deepcopy(gmm)
    #
    # # Move means in direction of each speaker
    # likelihoods = single_gmm_update(gmm1, gmm2, mfcc, energy_difference, active_segments_index, ref)
    # likelihoods_smoothed = likelihood_propagation_matrix(likelihoods)
    #
    # diarization, likelihoods_difference = return_diarization_index(likelihoods_smoothed, active_segments)
    #
    # likelihoods_difference = mean_filter(likelihoods_difference,
    #                                      int(params.diarization_init_array_filter / params.window_stride))

    # axs[3].plot(likelihoods_difference, label='2channels_gmm')
    # axs[3].plot(zeros, label='threshold')
    # axs[3].legend()
    #
    # hyp = np.sign(likelihoods_difference)
    # hyp *= active_segments
    #
    # confusion = np.sum((ref != hyp)) / ref.shape[0] * 100
    # print(f'2 channel GMM: {confusion:.2f}%')
    #
    # likelihoods_diff = likelihoods_smoothed[:, 0] - likelihoods_smoothed[:, 1]
    #
    # likelihoods = single_gmm_update(gmm1, gmm2, mfcc, likelihoods_diff, active_segments_index, ref)
    # likelihoods_smoothed = likelihood_propagation_matrix(likelihoods)
    #
    # diarization, likelihoods_difference = return_diarization_index(likelihoods_smoothed, active_segments)
    # likelihoods_difference = mean_filter(likelihoods_difference,
    #                                      int(params.diarization_init_array_filter / params.window_stride))

    # axs[4].plot(likelihoods_difference, label='2channels2iterations_gmm')
    # axs[4].plot(zeros, label='threshold')
    # axs[4].legend()

    # hyp = np.sign(likelihoods_difference)
    # hyp *= active_segments
    #
    # confusion = np.sum((ref != hyp)) / ref.shape[0] * 100
    # print(f'2 channel 2 iterations GMM: {confusion:.2f}%')
    # fig.tight_layout()

    """"Here"""
    # ref[ref == 2] = -1

    # time = np.linspace(0, 10, energy_difference.shape[0])
    # fig, ax = plt.subplots(nrows=1, sharex=True, figsize=(10, 6))
    # ax.plot(time, ref, color='C0', label='Referenční diarizace')
    # ax.plot(time, likelihoods_difference * 10, color='C1', label='Pravděpodobnost mluvy daného řečníka')
    # plt.grid(axis='x')
    #
    # high_bound = np.percentile(energy_difference, 100 - params.likelihood_percentile)
    # low_bound = np.percentile(energy_difference, params.likelihood_percentile)
    #
    # # Extract features
    # features1 = (energy_difference > high_bound) * 1
    # features2 = (energy_difference < low_bound) * -1
    #
    # ax.fill_between(time, np.zeros_like(features1), features1, color='C2', label='Inicializační segmenty GMM terapeuta')
    # ax.fill_between(time, np.zeros_like(features2), features2, color='C3', label='Inicializační segmenty GMM klienta')
    # plt.xticks(fontsize=12)
    # plt.yticks(fontsize=12)
    # ax.legend(fontsize=12)
    # ax.set_xlabel('Čas [min]', fontsize=16)
    # ax.set_ylabel('Velikost', fontsize=16)
    # plt.show()

    # time = np.linspace(0, 10, energy_difference.shape[0])
    # fig, ax = plt.subplots(nrows=1, sharex=True, figsize=(10, 6))
    # ax.plot(time, ref, color='C0', label='Referenční diarizace')
    # ax.plot(time, diarization, color='C1', label='Hypothesis')
    # plt.grid(axis='x')
    #
    # plt.xticks(fontsize=12)
    # plt.yticks(fontsize=12)
    # ax.legend(fontsize=12)
    # ax.set_xlabel('Čas [min]', fontsize=16)
    # ax.set_ylabel('Velikost', fontsize=16)
    # plt.show()
    #
    # confusion = np.sum(ref != diarization) / ref.shape[0] * 100
    #
    # confusion1 = np.sum(np.logical_and(ref == 1, diarization == -1)) / np.sum(ref == 1) * 100
    # confusion2 = np.sum(np.logical_and(ref == -1, diarization == 1)) / np.sum(ref == -1) * 100
    # print(f'1 channel GMM: {confusion:.2f}%')
    # print(f'Therapist miss: {confusion1:.2f}%')
    # print(f'Client miss: {confusion2:.2f}%')

    # Change output to correspond with rttm notation
    diarization[diarization == -1] = 2
    diarization = diarization.astype(int)

    return diarization


def gmm_mfcc_diarization_2channels(mfcc, vad, energy):
    """Train Gaussian mixture models with mfcc features over speech segments"""
    active_segments, active_segments_index, _, _, energy_difference = extract_features_for_diarization(
        mfcc, vad,
        energy)

    # Train UBM GMM - speaker independent
    features = np.append(mfcc[:, :, 0], mfcc[:, :, 1], axis=1)
    features_active = features[active_segments_index]
    gmm = MyGmm(n_components=params.gmm_components, covariance_type='diag', max_iter=params.gmm_max_iterations,
                tol=params.gmm_error_rate).fit(features_active)

    # Create model for each speaker
    gmm1 = deepcopy(gmm)
    gmm2 = deepcopy(gmm)

    # Move means in direction of each speaker
    likelihoods = single_gmm_update(gmm1, gmm2, features, energy_difference, active_segments_index)

    # Normalize likelihoods and process forward backward propagation
    log_evidence = logsumexp(likelihoods, axis=1)
    likelihoods = np.exp(likelihoods - log_evidence[:, np.newaxis])
    likelihoods = likelihood_propagation_matrix(likelihoods)

    # Calculate difference of each model likelihoods
    likelihoods_difference = likelihoods[:, 0] - likelihoods[:, 1]

    # Smooth difference
    likelihoods_difference = mean_filter(likelihoods_difference,
                                         int(params.diarization_init_array_filter / params.window_stride))

    # Get output diarization
    diarization = np.sign(likelihoods_difference) * active_segments

    # Process thresholding
    diarization = diar_thresholding_step(diarization, likelihoods_difference)

    # Change output to correspond with rttm notation
    diarization[diarization == -1] = 2
    diarization = diarization.astype(int)
    return diarization


def gmm_mfcc_diarization_2channels_2iterations(mfcc, vad, energy):
    """Train Gaussian mixture models with mfcc features over speech segments in 2 iterations of adaptation"""
    active_segments, active_segments_index, _, _, energy_difference = extract_features_for_diarization(
        mfcc, vad,
        energy)

    # Train UBM GMM - speaker independent
    features = np.append(mfcc[:, :, 0], mfcc[:, :, 1], axis=1)
    features_active = features[active_segments_index]
    gmm = MyGmm(n_components=params.gmm_components, covariance_type='diag', max_iter=params.gmm_max_iterations,
                tol=params.gmm_error_rate).fit(features_active)

    # Create model for each speaker
    gmm1 = deepcopy(gmm)
    gmm2 = deepcopy(gmm)

    # Move means in direction of each speaker
    likelihoods = single_gmm_update(gmm1, gmm2, features, energy_difference, active_segments_index)

    # Normalize likelihoods and process forward backward propagation
    log_evidence = logsumexp(likelihoods, axis=1)
    likelihoods = np.exp(likelihoods - log_evidence[:, np.newaxis])
    likelihoods = likelihood_propagation_matrix(likelihoods)

    # Calculate difference of each model likelihoods
    likelihoods_difference = likelihoods[:, 0] - likelihoods[:, 1]

    # Smooth difference
    likelihoods_difference = mean_filter(likelihoods_difference,
                                         int(params.diarization_init_array_filter / params.window_stride))

    """Second iteration"""
    # Create model for each speaker
    gmm1 = deepcopy(gmm)
    gmm2 = deepcopy(gmm)

    # Move means in direction of each speaker
    likelihoods = single_gmm_update(gmm1, gmm2, features, likelihoods_difference, active_segments_index)

    # Normalize likelihoods and process forward backward propagation
    log_evidence = logsumexp(likelihoods, axis=1)
    likelihoods = np.exp(likelihoods - log_evidence[:, np.newaxis])
    likelihoods = likelihood_propagation_matrix(likelihoods)

    # Calculate difference of each model likelihoods
    likelihoods_difference = likelihoods[:, 0] - likelihoods[:, 1]

    # Smooth difference
    likelihoods_difference = mean_filter(likelihoods_difference,
                                         int(params.diarization_init_array_filter / params.window_stride))

    # Get output diarization
    diarization = np.sign(likelihoods_difference) * active_segments

    # Process thresholding
    diarization = diar_thresholding_step(diarization, likelihoods_difference)

    # Change output to correspond with rttm notation
    diarization[diarization == -1] = 2
    diarization = diarization.astype(int)
    return diarization
