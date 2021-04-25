import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.special import logsumexp


class MyGmm(GaussianMixture):
    """Inherit from sklearn class add functionality for means update"""

    def update_centers(self, data, shift):
        """Actualize means by shift in direction estimated by new data"""

        # E-step
        weighted_log_gamma = self._estimate_weighted_log_prob(data)
        log_evidence = logsumexp(weighted_log_gamma, axis=1)
        gamma = np.exp(weighted_log_gamma - log_evidence[:, np.newaxis])
        gamma_sum = gamma.sum(axis=0) + 10 * np.finfo(weighted_log_gamma.dtype).eps  # Marginalize datapoints

        # M-step
        new_centers = np.dot(gamma.T, data)
        new_centers_normalized = new_centers / gamma_sum[:, np.newaxis]
        new_weights = gamma_sum / len(data)

        # Actualize weights to new centers
        shift_center = shift * new_weights
        shift_center = shift_center[:, np.newaxis]
        self.means_ *= (1 - shift_center)
        self.means_ += shift_center * new_centers_normalized
