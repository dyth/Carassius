#!/usr/bin/env python
"""
play chess against a random agent or human opposition
"""
from engine import *
from value_network import *
from chess import *


def self_play(engines):
    'engines is a list of engines and engines[0] moves first'
    board = Board()
    index = 0
    moves = 50
    while (evaluate(board) is None) and (moves > 0):
        board = engines[index].minimax(board)
        index = int(not index)
        # print(board)
        moves -= 1
        # print(moves)
    return evaluate(board)


if __name__ == "__main__":
    e = Engine(optimal, 9, 0.7)
    u = User()
    self_play([e, u])
