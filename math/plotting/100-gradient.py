#!/usr/bin/env python3
""" A function that creates a scatter plot of sampled elevations on a mountain """
import numpy as np
import matplotlib.pyplot as plt


def gradient():
    """ Plots a scatter plot of sampled elevations on a mountain """
    np.random.seed(5)

    x = np.random.randn(2000) * 10
    y = np.random.randn(2000) * 10
    z = np.random.rand(2000) + 40 - np.sqrt(np.square(x) + np.square(y))
    plt.figure(figsize=(6.4, 4.8))

    scatter_plot = plt.scatter(x, y, c=z, cmap='viridis')
    colorbar = plt.colorbar(scatter_plot)
    colorbar.set_label('elevation (m)', rotation=90)

    plt.xlabel('x coordinate (m)')
    plt.ylabel('y coordinate (m)')
    plt.title('Mountain Elevation')
    plt.show()
