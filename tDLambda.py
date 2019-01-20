#!/usr/bin/env python
"""
train value_network using the TD(lambda) reinforcement algorithm
"""
from engine import *
from node import *
from value_network_large import *
from chess import *

import csv
import os


games_played = 0


def create_train_sequence(engines, discount):
    'create a forest of nodes, their roots a new board position'
    global games_played
    games_played += 1

    board = Board()
    seen_boards = set() # set of all seen boards
    seen_boards.add(board.fen().split(' ')[0])

    # to explore, do a randomly chosen first move
    r = Engine(random, 1, discount)
    board = r.minimax(board)

    trace = []
    index = 0
    moves = 0
    while (evaluate(board) is None) and not board.is_insufficient_material():
        seen_boards.add(board.fen().split(' ')[0])
        node = engines[index].create_search_tree(board)
        trace.append(node)

        if node.pv.board.fen().split(' ')[0] in seen_boards:
            node.pv.board = r.minimax(board)
        board = node.pv.board

        index = int(not index)
        moves += 1
    pretty_print(board)
    print(games_played, moves, evaluate(board))

    node = Node(board)
    node.reward = evaluate(board)
    trace.append(node)
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
    directory = "tDLambda8"
    if not os.path.exists(directory):
        os.makedirs(directory)
    valueNetwork = ValueNet(learningRate, 0.7)
    e = Engine(valueNetwork, 1, discount)

    count = 0
    while True:
        torch.save(e.policy.state_dict(), f'{directory}/{count}.pt')
        train(e, batch)
        count += 1
