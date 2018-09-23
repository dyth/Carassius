#!/usr/bin/env python
"""
player, winning and move conditions for noughts and crosses
"""
import copy
from random import uniform, seed
seed(9001)


# game information
initialBoard = [None, None, None, None, None, None, None, None, None]
players = [1, -1]



def next_player(x):
    'return the identity of the next player'
    return players[(players.index(x) + 1) % 2]


def move(player, board, pos1, pos2):
    'board[3 * pos1 + pos2] = player'
    moved = copy.deepcopy(board)
    index = 3 * pos1 + pos2
    if (moved[index] == None):
        moved[index] = player
        return moved
    else:
        return None

    
def move_all(player, board):
    'return list of all possible next boards : a list list'
    if board == None:
        return
    moves = []
    for i in range(9):
        moved = copy.deepcopy(board)
        if (moved[i] == None):
            moved[i] = player
            moves.append(moved)
    return moves



def evaluate(board):
    'evaluate whether there are still possible moves to be played'
    if (board[0] == board[1] == board[2]) and (board[0] is not None):
        return float(board[0])
    elif (board[0] == board[3] == board[6]) and (board[0] is not None):
        return float(board[0])
    elif (board[0] == board[4] == board[8]) and (board[0] is not None):
        return float(board[0])
    elif (board[1] == board[4] == board[7]) and (board[1] is not None):
        return float(board[1])
    elif (board[3] == board[4] == board[5]) and (board[3] is not None):
        return float(board[3])
    elif (board[6] == board[7] == board[8]) and (board[6] is not None):
        return float(board[6])
    elif (board[2] == board[5] == board[8]) and (board[2] is not None):
        return float(board[2])
    elif (board[2] == board[4] == board[6]) and (board[2] is not None):
        return float(board[2])
    elif None not in board:
        return 0.0
    # if valid moves, return None
    return None


def pretty_print(board):
    printBoard = []
    for i in range(9):
        if board[i] == players[0]:
            printBoard.append("X")
        elif board[i] == players[1]:
            printBoard.append("O")
        else:
            printBoard.append(" ")
    print """
   |   |
 %s | %s | %s
___|___|___
   |   |
 %s | %s | %s
___|___|___
   |   |
 %s | %s | %s
   |   |
""" % tuple(printBoard)

    
def optimal(board):
    'optimal policy'
    if (board[0] == board[1] == board[2]) and (board[0] is not None):
        return float(board[0])
    elif (board[0] == board[3] == board[6]) and (board[0] is not None):
        return float(board[0])
    elif (board[0] == board[4] == board[8]) and (board[0] is not None):
        return float(board[0])
    elif (board[1] == board[4] == board[7]) and (board[1] is not None):
        return float(board[1])
    elif (board[3] == board[4] == board[5]) and (board[3] is not None):
        return float(board[3])
    elif (board[6] == board[7] == board[8]) and (board[6] is not None):
        return float(board[6])
    elif (board[2] == board[5] == board[8]) and (board[2] is not None):
        return float(board[2])
    elif (board[2] == board[4] == board[6]) and (board[2] is not None):
        return float(board[2])
    else:
        return 0.0


def random(board):
    'policy which selects a move at random'
    return uniform(-1, 1)
