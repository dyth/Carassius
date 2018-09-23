#!/usr/bin/env python
"""
engine utilises a function policy to choose the best move using minimax
"""
from noughts_crosses import *
from node import *


class Engine:

    def __init__(self, policy, searchDepth, discount):
        # policy : fn board -> [-1.0, 1.0]
        # searchDepth : int
        # discount : float [0.0, 1.0]
        self.policy = policy
        self.searchDepth = searchDepth
        self.discount = discount


    def create_search_tree(self, board, player):
        'create search tree from board'
        node = Node(board)
        if player == players[0]:
            self.maximise(node, self.searchDepth, True)
        else:
            self.minimise(node, self.searchDepth, True)
        return node

    
    def minimax(self, board, player):
        'find self.bestMove using minimax and principal variation'
        return self.create_search_tree(board, player).pv.board
    
    
    def maximise(self, node, depth, rootNode):
        'maximise policy score for players[0]'
        if (depth == 0) or (node.reward is not None):
            return self.policy(node.board)
        moves = move_all(players[0], node.board)
        score = -2.0
        for m in moves:
            daughter = Node(m)
            newScore = self.minimise(daughter, depth-1, False)
            if (newScore > score):
                if node.pv is not None:
                    node.other.append(node.pv)
                score = newScore
                node.pv = daughter
            else:
                node.other.append(daughter)
        return self.discount * score
    
    
    def minimise(self, node, depth, rootNode):
        'minimise policy score for players[1]'
        if (depth == 0) or (node.reward is not None):
            return self.policy(node.board)
        moves = move_all(players[1], node.board)
        score = 2.0
        for m in moves:
            daughter = Node(m)
            newScore = self.maximise(daughter, depth-1, False)
            if (newScore < score):
                if node.pv is not None:
                    node.other.append(node.pv)
                score = newScore
                node.pv = daughter
            else:
                node.other.append(daughter)
        return self.discount * score


if __name__ == "__main__":
    e = Engine(optimal, 9, 0.7)
    tree = e.create_search_tree(initialBoard, players[0])
    assert(len(tree.other) == 8)
    pretty_print(tree.pv.board)
    

