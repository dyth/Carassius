#!/usr/bin/env python
"""
helper functions for chess
"""
from chess import *
import copy
from random import uniform, seed
seed(9001)


# game information
players = [1, -1]


def evaluate(board):
    'if checkmate, return 1 or -1'
    if board.is_checkmate():
        if board.turn:
            return 1
        else:
            return -1
    elif board.is_stalemate():
        return 0


def random(board):
    'policy which selects a move at random'
    return uniform(-1, 1)
