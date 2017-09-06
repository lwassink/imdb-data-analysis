from load_data import data
import numpy as np
import matplotlib.pyplot as plt

A = data['A']

cov = np.matmul(A.T, A)
U, s, V = np.linalg.svd(cov)
n = np.size(s)

W = np.matmul(A, U)

# cumulative variances
total = np.sum(s)
s_cum = np.zeros(n)
for i in range(n):
    s_cum[i] = sum(s[:i + 1])

# graph variances
plt.plot(s, 'ro')
plt.ylabel('Energies of Priciple Components')
plt.savefig('graphs/energy.pdf')
plt.clf()

plt.plot(s_cum, 'bo')
plt.ylabel('Cumulative Energies of Priciple Components')
plt.savefig('graphs/cumulative_engergy.pdf')
plt.clf()

# graph 2D projection
plt.plot(W[:, 0], W[:, 1], 'bo')
plt.legend('Projection onto principle axes')
plt.savefig('graphs/2d_view.pdf')
