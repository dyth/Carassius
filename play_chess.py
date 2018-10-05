#!/usr/bin/env python
"""
play chess against a random agent or human opposition
"""
from engine_chess import *
from value_network_chess import *
from chess import *


def self_play(engines):
    'engines is a list of engines and engines[0] moves first'
    board = Board()
    player = board.turn
    index = 0
    moves = 100
    while (evaluate(board) is None) and (moves > 0):
        board = engines[index].minimax(board, player)
        player = board.turn
        index = int(not index)
        #print(board)
        #pretty_print(board)
        moves -= 1
    return evaluate(board)


if __name__ == "__main__":
    e = Engine(optimal, 9, 0.7)
    u = User()
    self_play([e, u])
    
