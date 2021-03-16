from src import gmm
import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
gmm = gmm.MyGmm(n_components=2)

# Create train data
mean = [0, 0, 0]
cov = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
x1, y1, z1 = np.random.multivariate_normal(mean, cov, 100).T
features1 = np.append(np.append(x1.reshape(-1, 1), y1.reshape(-1, 1), axis=1), z1.reshape(-1, 1), axis=1)

mean = [50, 50, 50]
cov = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
x2, y2, z2 = np.random.multivariate_normal(mean, cov, 100).T
features2 = np.append(np.append(x2.reshape(-1, 1), y2.reshape(-1, 1), axis=1), z2.reshape(-1, 1), axis=1)

features = np.append(features1, features2, axis=0)

# Fit gmm
gmm.fit(features)

x3, y3, z3 = np.random.multivariate_normal(gmm.means_[0], gmm.covariances_[0], 100).T
x4, y4, z4 = np.random.multivariate_normal(gmm.means_[1], gmm.covariances_[1], 100).T
ax.scatter(x3, y3, z3, label='c1')
ax.scatter(x4, y4, z4, label='c2')

# Data to update means
mean = [10, 10, 10]
cov = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
x5, y5, z5 = np.random.multivariate_normal(mean, cov, 100).T
data_to_fit = np.append(np.append(x5.reshape(-1, 1), y5.reshape(-1, 1), axis=1), z5.reshape(-1, 1), axis=1)

ax.scatter(x5, y5, z5, label='data_to_fit')

gmm.update_centers(data_to_fit, 0.5)

# Plot updated gmms
x3, y3, z3 = np.random.multivariate_normal(gmm.means_[0], gmm.covariances_[0], 100).T
x4, y4, z4 = np.random.multivariate_normal(gmm.means_[1], gmm.covariances_[1], 100).T
ax.scatter(x3, y3, z3, label='c1_new')
ax.scatter(x4, y4, z4, label='c2_new')

ax.legend()
plt.show()

x = 5
