import numpy as np
import params
from scipy import ndimage
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from decorators import timeit


def energy_based_diarization_no_interruptions(energy, vad):
    """Simple energy based diarization, if voice activity was detected, choose higher energy as speaker"""
    """"""
    higher_energy = np.argmax(energy, axis=1)
    vad_summed = np.sum(vad, axis=1)
    diarized = np.where(vad_summed > 0, higher_energy + 1, 0)
    diarized = ndimage.median_filter(diarized, params.med_filter_diar)
    return diarized


@timeit
def gmm_mfcc_diarization_no_interruptions(mfcc_dd, vad):
    """Train Gaussian mixture models with mfcc features over speech segments"""

    # Active segments over channels
    active_segments = (vad[:, 0] + vad[:, 1] > 0).astype("i1")

    # Extract speech mfcc features
    channel1_features = mfcc_dd[:, :, 0][np.argwhere(vad[:, 0]).reshape(-1)]
    channel2_features = mfcc_dd[:, :, 1][np.argwhere(vad[:, 1]).reshape(-1)]

    # Append to one features vector
    features = np.append(channel1_features, channel2_features, axis=0)

    # Train gmm
    gmm = GaussianMixture(n_components=32).fit(features)

    # Actualize means by moving 5% to the center of speaker cluster
    gmm_k_means = KMeans(n_clusters=2, random_state=0).fit(gmm.means_)
    gmm_clusters = gmm_k_means.labels_
    gmm.means_ = gmm_k_means.cluster_centers_[gmm_k_means.labels_] * params.means_shift + gmm.means_ * (
            1 - params.means_shift)

    # Predict over channels
    channel1_predict = gmm.predict_proba(mfcc_dd[:, :, 0])
    channel2_predict = gmm.predict_proba(mfcc_dd[:, :, 1])

    # Sum predictions
    channel_predictions = channel1_predict + channel2_predict

    # Sum likelihoods of clusters
    speaker1 = np.sum(channel_predictions[:, ~np.argwhere(gmm_clusters)], axis=1)
    speaker2 = np.sum(channel_predictions[:, np.argwhere(gmm_clusters)], axis=1)

    # Choose higher cluster likelihood
    speakers = np.append(speaker1, speaker2, axis=1)
    diarization = np.argmax(speakers, axis=1) + 1
    llhs = np.max(speakers, axis=1) / 2

    # Remove nonactive segments
    diarization *= active_segments
    llhs *= active_segments

    # Apply median filter
    diarization = ndimage.median_filter(diarization, params.med_filter_diar)
    llhs = ndimage.median_filter(llhs, params.med_filter_diar)
    return diarization, llhs
