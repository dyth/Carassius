#!/usr/bin/env python
"""
train value_network using the TD(lambda) reinforcement algorithm
"""
from engine import *
from node import *
# from play import *
from value_network import *
from chess import *

import matplotlib.pyplot as plt
import csv
import os


def self_play(engines):
    'engines is a list of engines and engines[0] moves first'
    board = Board()
    index = 0
    moves = 400
    while (evaluate(board) is None) and (moves > 0):
        board = engines[index].minimax(board)
        index = int(not index)
        moves -= 1
    pretty_print(board)
    print(evaluate(board), moves)
    return evaluate(board)


def create_train_sequence(engines, discount):
    'create a forest of nodes, their roots a new board position'
    board = Board()
    seen_boards = set() # set of all seen boards
    seen_boards.add(board.fen().split(' ')[0])

    # to explore, do a randomly chosen first move
    r = Engine(random, 1, discount)
    board = r.minimax(board)

    trace = []
    index = 0
    moves = 400
    while (evaluate(board) is None) and (moves > 0):
        seen_boards.add(board.fen().split(' ')[0])
        node = engines[index].create_search_tree(board)
        trace.append(node)

        if node.pv.board.fen().split(' ')[0] in seen_boards:
            node.pv.board = r.minimax(board)
        board = node.pv.board

        index = int(not index)
        moves -= 1
    pretty_print(board)
    print(evaluate(board), moves, "train\n")

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
    with open("tDLambda.csv", "w") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')

        plt.ion()
        batch = 20
        learningRate = 0.01
        discount = 0.7
        directory = "tDLambda"
        if not os.path.exists(directory):
            os.makedirs(directory)
        valueNetwork = ValueNet(learningRate, 0.7)
        e = Engine(valueNetwork, 1, discount)
        r = Engine(random, 1, discount)
        win, lose, draw = [], [], []
        testGamesNum = 10
        count = 0

        while True:
            # plot first before train
            w, l, d = 0, 0, 0
            for _ in range(testGamesNum):
                score = self_play([e, r])
                if score == 1:
                    w += 1
                elif score == -1:
                    l += 1
                else:
                    d += 1
                score = self_play([r, e])
                if score == -1:
                    w += 1
                elif score == 1:
                    l += 1
                else:
                    d += 1
            w = float(w) / (2.0 * testGamesNum)
            l = float(l) / (2.0 * testGamesNum)
            d = float(d) / (2.0 * testGamesNum)
            writer.writerow([w, l, d])
            print("Wins, Losses, Draws:", w, l, d, e.policy(Board()))
            win.append(w)
            lose.append(l)
            draw.append(d)
            x = range(0, batch*(count + 1), batch)
            plt.plot(x, win, label="P(win)")
            plt.plot(x, draw, label="P(draw)")
            plt.plot(x, lose, label="P(lose)")
            plt.legend()
            plt.title("Training vs Time")
            plt.xlabel('Self-Play Games Played')
            plt.ylabel('Probability')
            plt.pause(0.001)
            plt.clf()

            torch.save(e.policy.state_dict(), f'{directory}/{count}.pt')
            count += 1

            # train
            train(e, batch)
