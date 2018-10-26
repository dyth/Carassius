#!/usr/bin/env python
"""
Datastructure for a node in a game tree. Useful in reinforcement learning
"""
from chess import *
from copy import deepcopy
from random import uniform, seed
seed(9001)


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
    'policy to select a move at random'
    return uniform(-1, 1)


class Node:

    def __init__(self, board):
        'initialise a new node with a board'
        # score is the reward given to board if this node is terminal
        self.board = deepcopy(board)#.fen()
        self.reward = evaluate(self.board)
        self.pv = None
        self.other = []
