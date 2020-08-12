"""
This script takes sudokus (as numpy arrays)
and runs backtracking to solve them.

Taken and adapted from:
https://techwithtim.net/tutorials/python-programming/sudoku-solver-backtracking/
"""

def solve(bo, size=9):
    find = find_empty(bo)
    if not find:
        return True
    else:
        row, col = find

    for i in range(1, size + 1):
        # print(np.array(bo))
        if valid(bo, i, (row, col), size=size):
            bo[row][col] = i

            if solve(bo, size=size):
                return True

            bo[row][col] = 0

    return False

def valid(bo, num, pos, size=9):
    # Check row
    for i in range(len(bo[0])):
        if bo[pos[0]][i] == num and pos[1] != i:
            return False

    # Check column
    for i in range(len(bo)):
        if bo[i][pos[1]] == num and pos[0] != i:
            return False

    # Define bounds for the box
    if size == 9:
        # how many rows/columns per box
        rows_per_box = 3
        columns_per_box = 3
        # how many boxes in both directions
        column_boxes = 3
        row_boxes = 3
    elif size == 6:
        rows_per_box = 2
        columns_per_box = 3

        column_boxes = 2
        row_boxes = 3
    elif size == 4:
        rows_per_box = 2
        columns_per_box = 2

        column_boxes = 2
        row_boxes = 2

    # Check box
    box_x = pos[1] // (len(bo) // column_boxes)
    box_y = pos[0] // (len(bo) // row_boxes)

    # print(f"num: {num}, pos: {pos}, box: {box_y, box_x}")
    # print(f"box: {np.array(bo)[box_y*rows_per_box:rows_per_box*(box_y + 1),box_x * columns_per_box: columns_per_box * (box_x + 1)]}")
    for i in range(box_y*rows_per_box, rows_per_box*(box_y + 1)):
        for j in range(box_x * columns_per_box, columns_per_box * (box_x + 1)):
            # print(f"(i, j): {i, j}")
            if bo[i][j] == num and (i,j) != pos:
                return False

    return True

def find_empty(bo):
    for i in range(len(bo)):
        for j in range(len(bo[0])):
            if bo[i][j] == 0:
                return (i, j)  # row, col

    return None
