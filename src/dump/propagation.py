# def likelihood_propagation(likelihood):
#     def log_add(x, y):
#         if x < y:
#             tmp = x
#             x = y
#             y = tmp
#
#         diff = y - x
#
#         return x if -diff > 10 else np.log(1 + np.exp(diff)) + x
#
#     nToken = 2
#     len = likelihood.shape[0]
#     real_info = np.finfo('d')
#     tokenL = -real_info.max * np.ones((nToken, len))
#     tokenBL = -real_info.max * np.ones((nToken, len))
#
#     TM = np.array([[.9, .1], [.1, 9]])
#
#     ac_scale = 0.03
#
#     tokenL[0, 0] = np.log(0.5) + ac_scale * likelihood[0, 0]
#     tokenL[1, 0] = np.log(0.5) + ac_scale * likelihood[1, 0]
#     for i in range(1, len):
#         tL = -real_info.min * np.ones((nToken, 1))
#         for j in range(0, nToken):
#             prevLike = tokenL[j, i - 1]
#             for k in range(0, nToken):
#                 tL[k] = prevLike + ac_scale * likelihood[i, j] + np.log(TM[k, j])
#                 tokenL[k, i] = log_add(tokenL[k, i], tL[k])
#     fwd = log_add(tokenL[0, len - 1], tokenL[1, len - 1])
#
#     tokenBL[0, len - 1] = np.log(0.5)
#     tokenBL[1, len - 1] = np.log(0.5)
#     for i in range(len - 2, -1, -1):
#         tL = -real_info.min * np.ones((nToken, 1))
#         for j in range(0, nToken):
#             prevLike = tokenBL[j, i + 1]
#             for k in range(0, nToken):
#                 tL[k] = prevLike + ac_scale * likelihood[i + 1, j] + np.log(TM[k, j])
#                 tokenBL[k, i] = log_add(tokenBL[k, i], tL[k])
#     bwd = log_add(tokenBL[0, 0], tokenBL[1, 0])
#
#     Q = tokenL + tokenBL - fwd
#     Q = np.exp(Q)
#     Q = Q / np.sum(Q)
#
#     return likelihood

