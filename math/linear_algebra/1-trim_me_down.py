#!/usr/bin/env python3
matrix = [[1, 3, 9, 4, 5, 8], [2, 4, 7, 3, 4, 0], [0, 3, 4, 6, 1, 5]]
the_middle = []
columns_3_and_4 = [element[2:4] for element in matrix]
the_middle.append(columns_3_and_4)
print("The middle columns of the matrix are: {}".format(the_middle))
