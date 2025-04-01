#!/usr/bin/env python3
""" A function that adds two arrays element-wise """


def add_arrays(arr1, arr2):
    """
        Adds two arrays elements-wise.

        Parameters:
            arr1 (list of ints or floats): The first array.
            arr2 (list of ints or floats): The second array.

        Returns:
            A new array of the sum of arr1 and arr2.
    """
    if len(arr1) != len(arr2):
        return None

    new_array = [first_list + second_list
                 for first_list, second_list
                 in zip(arr1, arr2)]

    return new_array
