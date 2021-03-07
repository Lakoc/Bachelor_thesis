import numpy as np
import params
from scipy import ndimage


def energy_based_diarization_no_interruptions(energy, vad):
    """Simple energy based diarization, if voice activity was detected, choose higher energy as speaker"""
    """"""
    higher_energy = np.argmax(energy, axis=1)
    vad_summed = np.sum(vad, axis=1)
    diarized = np.where(vad_summed > 0, higher_energy + 1, 0)
    diarized = ndimage.median_filter(diarized, params.med_filter_diar)
    return diarized


def gmm_mfcc_diarization_no_interruptions(mfcc_dd, vad):
    """Train Gaussian mixture models with mfcc features over speech segments"""
    pass
