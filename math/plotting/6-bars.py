#!/usr/bin/env python3
"""
    A function that plots a stacked bar graph representing
    the number of fruit each person possesses
"""
import numpy as np
import matplotlib.pyplot as plt


def bars():
    """ Plots a stacked bar graph """
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))
    plt.figure(figsize=(6.4, 4.8))

    people = ["Farrah", "Fred", "Felicia"]

    plt.bar(people, fruit[0], label='apples', width=0.5, color='red')
    plt.bar(people, fruit[1], bottom=fruit[0], label='bananas', width=0.5,
            color='yellow')
    plt.bar(people, fruit[2], bottom=fruit[0] + fruit[1], label='oranges',
            width=0.5, color='#ff8000')
    plt.bar(people, fruit[3], bottom=fruit[0] + fruit[1] + fruit[2],
            label='peaches', width=0.5, color='#ffe5b4')

    plt.xticks(range(len(people)), [person for person in people])
    plt.ylabel('Quantity of Fruit')
    plt.title('Number of Fruit per Person')
    plt.ylim(0, 80)
    plt.legend()
    plt.show()
