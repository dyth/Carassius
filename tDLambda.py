#!/usr/bin/env python
"""
train value_network using the TD(lambda) reinforcement algorithm
"""
from engine import *
from node import *
from value_network_large import *
from chess import *

import csv, os

games_played = 0


def create_train_sequence(engines, discount):
    'create a forest of nodes, their roots a new board position'
    global games_played
    games_played += 1
    index = 0
    moves = 0
    trace = []
    # initialise board, set of seen boards and do random move for diversity
    board = Board()
    seen_boards = set({board_to_fen(board)})
    r = Engine(random, 1, 0.7)
    board = r.minimax(board)
    # only quit if checkmate, stalemate or insufficent material for win
    while (evaluate(board) is None) and not board.is_insufficient_material():
        # get new board position, if previously seen, do random move
        node = engines[index].create_search_tree(board)
        if board_to_fen(node.pv.board) in seen_boards:
            node.pv.board = r.minimax(board)
        # add board to previously seen, update turn and move variables
        board = node.pv.board
        seen_boards.add(board_to_fen(board))
        index = int(not index)
        moves += 1
        trace.append(node)
    # append final board to trace and print final game information
    node = Node(board)
    node.reward = evaluate(board)
    trace.append(node)
    pretty_print(board)
    print(games_played, node.reward, moves)
    # if node.reward == -1:
    #     for t in trace:
    #         pretty_print(t.board)
    #         print(t.reward)
    #     print(bad)
    return trace


def TD_Lambda(engines, network, discount):
    'return sequence of boards and reward for training'
    trace = create_train_sequence(engines, discount)
    boards = [t.board for t in trace]
    reward = trace[-1].reward
    if reward is None:
        reward = network(boards[-1])
        boards = boards[:-1]
    network.temporal_difference(boards, reward, discount)


def train(engine, games):
    'train engine for self play in games'
    for _ in range(games):
        TD_Lambda([engine, engine], engine.policy, engine.discount)


if __name__ == "__main__":
    batch = 20
    learningRate = 0.01
    discount = 0.7

    directory = "tDLambda9"
    if not os.path.exists(directory):
        os.makedirs(directory)
    valueNetwork = ValueNet(learningRate, 0.7)
    e = Engine(valueNetwork, 1, discount)

    count = 0
    while True:
        torch.save(e.policy.state_dict(), f'{directory}/{count}.pt')
        train(e, batch)
        count += 1
