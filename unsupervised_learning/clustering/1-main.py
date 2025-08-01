#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
kmeans = __import__('1-kmeans').kmeans

if __name__ == "__main__":
    np.random.seed(0)
    a = np.random.multivariate_normal([30, 40], [[16, 0], [0, 16]], size=50)
    b = np.random.multivariate_normal([10, 25], [[16, 0], [0, 16]], size=50)
    c = np.random.multivariate_normal([40, 20], [[16, 0], [0, 16]], size=50)
    d = np.random.multivariate_normal([60, 30], [[16, 0], [0, 16]], size=50)
    e = np.random.multivariate_normal([20, 70], [[16, 0], [0, 16]], size=50)
    X = np.concatenate((a, b, c, d, e), axis=0)
    np.random.shuffle(X)
    C, clss = kmeans(X, 5)
    print(C)
    print(clss)
    #plt.scatter(X[:, 0], X[:, 1], s=10, c=clss)
    #plt.scatter(C[:, 0], C[:, 1], s=50, marker='*', c=list(range(5)))
    #plt.show()

    # Après appel à kmeans(X, 5)
    points_1 = X[clss == 1]
    points_4 = X[clss == 4]

    print("Centroïde 1:", C[1])
    print("Centroïde 4:", C[4])

    # Distances de points dans cluster 1 au centroïde 4
    dist_to_4 = np.linalg.norm(points_1 - C[4], axis=1)
    dist_to_1 = np.linalg.norm(points_1 - C[1], axis=1)

    # Trouve points dans cluster 1 plus proches du centroïde 4 que du 1
    mask = dist_to_4 < dist_to_1
    print("Points dans cluster 1 plus proches du centroïde 4:")
    print(points_1[mask])