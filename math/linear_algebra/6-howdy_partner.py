#!/usr/bin/env python3
""" A function that concatenates two arrays """


def cat_arrays(arr1, arr2):
    """
        concatenates two arrays.

        Parameters:
            arr1 (list of ints or floats): The first array.
            arr2 (list of ints or floats): The second array.

        Returns:
            A new list of the concatenation of arr1 and arr2.
    """
    new_list = []

    new_list = [number for number in arr1]
    new_list += [number for number in arr2]

    return new_list
