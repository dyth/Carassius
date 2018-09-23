#!/usr/bin/env python
"""
Datastructure for a node in a game tree. Useful in reinforcement learning
"""
from noughts_crosses import *

class Node:

    def __init__(self, board):
        'initialise a new node with a board'
        # score is the reward given to board if this node is terminal
        self.board = board
        self.reward = evaluate(self.board)
        self.pv = None
        self.other = []
