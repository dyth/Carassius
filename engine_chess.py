#!/usr/bin/env python
"""
engine utilises a function policy to choose the best move using minimax on chess
"""
from chess import *
from node_chess import *


class Engine:

    def __init__(self, policy, searchDepth, discount):
        # policy : fn board -> [-1.0, 1.0]
        # searchDepth : int
        # discount : float [0.0, 1.0]
        self.policy = policy
        self.searchDepth = searchDepth
        self.discount = discount

    
    def create_search_tree(self, board):
        'create search tree from board'
        node = Node(board)
        if board.turn:
            self.minimise(node, self.searchDepth, True)
        else:
            self.maximise(node, self.searchDepth, True)
        return node

    
    def minimax(self, board, player):
        'find self.bestMove using minimax and principal variation'
        return self.create_search_tree(board).pv.board
    
    
    def maximise(self, node, depth, rootNode):
        'maximise policy score for players[0]'
        if (depth == 0) or (node.reward is not None):
            return self.policy(node.board)
        moves = node.board.legal_moves
        score = -2.0
        for m in moves:
            node.board.push(m)
            daughter = Node(node.board)
            newScore = self.minimise(daughter, depth-1, False)
            if (newScore > score):
                if node.pv is not None:
                    node.other.append(node.pv)
                score = newScore
                node.pv = daughter
            else:
                node.other.append(daughter)
            node.board.pop()
        return self.discount * score
    
    
    def minimise(self, node, depth, rootNode):
        'minimise policy score for players[1]'
        if (depth == 0) or (node.reward is not None):
            return self.policy(node.board)
        moves = node.board.legal_moves
        score = 2.0
        for m in moves:
            node.board.push(m)
            daughter = Node(node.board)
            newScore = self.maximise(daughter, depth-1, False)
            if (newScore < score):
                if node.pv is not None:
                    node.other.append(node.pv)
                score = newScore
                node.pv = daughter
            else:
                node.other.append(daughter)
            node.board.pop()
        return self.discount * score


if __name__ == "__main__":
    e = Engine(evaluate, 9, 0.7)
    tree = e.create_search_tree(Board())
    print tree.other
