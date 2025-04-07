#!/usr/bin/env python3
""" A function that plots a histogram of student scores for a project """

import numpy as np
import matplotlib.pyplot as plt


def frequency():
    """ plots a histogram of student scores for a project """
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    plt.figure(figsize=(6.4, 4.8))

    plt.hist(student_grades, range=(0, 100), bins=10, edgecolor='black')
    plt.xlim(0, 100)
    plt.ylim(0, 30)
    plt.xticks(range(0, 100, 10))
    plt.xlabel('Grades')
    plt.ylabel('Number of Students')
    plt.title('Project A')
    plt.show()
