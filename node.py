#!/usr/bin/env python
"""
Datastructure for a node in a game tree. Useful in reinforcement learning
"""
from chess import *
from copy import deepcopy
from random import uniform, seed
seed(9001)


def pretty_print(board):
    'print chessboard ascii'
    # get ascii and change numbers into spaces
    board_string = board.fen().split(' ')[0]
    for i in range(1, 9):
        board_string = board_string.replace(str(i), i*' ')
    pieces = {'R':'♜', 'N':'♞', 'B':'♝', 'Q':'♛', 'K':'♚', 'P':'♟',
              'r':'♖', 'n':'♘', 'b':'♗', 'q':'♕', 'k':'♔', 'p':'♙'}
    # print ascii board and coordinates
    print('\n')
    for i, row in enumerate(board_string.split('/')):
        print(' ', 8-i, ' '.join(pieces.get(p, p) for p in row))
    print('    a b c d e f g h \n')


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
