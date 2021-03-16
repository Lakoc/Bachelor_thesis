@timeit
def gmm_mfcc_diarization_no_interruptions_1channel(mfcc_dd, vad, energy):
    """Train Gaussian mixture models with mfcc features over speech segments"""

    # Active segments over channels
    active_segments = np.logical_or(vad[:, 0], vad[:, 1]).astype("i1")

    # Extract speech mfcc features
    channel1_mfcc = mfcc_dd[:, :, 0][np.argwhere(active_segments).reshape(-1)]
    channel1_energy = energy[:, 0][np.argwhere(active_segments).reshape(-1)]
    channel2_energy = energy[:, 1][np.argwhere(active_segments).reshape(-1)]

    energy_difference = np.log(channel1_energy) - np.log(channel2_energy)

    # Train gmm
    gmm1 = MyGmm(n_components=32, verbose=1, covariance_type='diag').fit(channel1_mfcc)
    gmm2 = deepcopy(gmm1)
    gmm1.update_centers(channel1_mfcc[energy_difference > 0.5], params.model_means_shift)
    gmm2.update_centers(channel1_mfcc[energy_difference < -0.5], params.model_means_shift)

    speaker1 = gmm1.score_samples(mfcc_dd[:, :, 0])
    speaker2 = gmm2.score_samples(mfcc_dd[:, :, 0])
    speaker1 = speaker1.reshape(-1, 1)
    speaker2 = speaker2.reshape(-1, 1)
    likelihoods = np.append(speaker1, speaker2, axis=1)

    return choose_speaker(likelihoods, active_segments)