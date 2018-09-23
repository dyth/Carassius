#!/usr/bin/env python
"""
play noughts and crosses against a random agent or human opposition
"""
from engine import *
from value_network import *
from noughts_crosses import *


class User:

    def query_print(self, board):
        printBoard = []
        for i in range(9):
            if board[i] == players[0]:
                printBoard.append("X")
            elif board[i] == players[1]:
                printBoard.append("O")
            else:
                printBoard.append("\033[93m" + str(i+1) + "\033[0;0m")
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

    
    def minimax(self, board, player):
        self.query_print(board)
        index = input("Type the number of the square to move ")
        board[index-1] = player
        return board
    

def self_play(engines):
    'engines is a list of engines and engines[0] moves first'
    board = initialBoard
    player = players[0]
    index = 0
    while evaluate(board) is None:
        board = engines[index].minimax(board, player)
        player = next_player(player)
        index = int(not index)
        #print(board)
        #pretty_print(board)
    return evaluate(board)


if __name__ == "__main__":
    e = Engine(optimal, 9, 0.7)
    u = User()
    self_play([e, u])
    
