#!/usr/bin/env python
"""
treestrap learning algorithm
"""
from tDLambda_chess import *
from engine_chess import *
from node_chess import *
from play_chess import *
from value_network_chess import *
from chess import *
import matplotlib.pyplot as plt
import csv


def get_trace(node, trace):
    'return list of boards and reward of principle variation'
    trace.append(node.board)
    if node.pv is not None:
        return get_trace(node.pv, trace)
    else:
        return trace, node.reward


def train_games(network, node, discount):
    'train on all possible games from node'
    boards, reward = get_trace(node, [])
    # if no reward, then need to remove final value
    if reward is None:
        reward = network(boards[-1])
        boards = boards[:-1]
    if boards != []:
        network.temporal_difference(boards, reward, discount)
    for board in node.other:
        train_games(network, board, discount)


def TreeStrap(engines, network, discount):
    'return sequence of boards and reward for training'
    board = Board()
    player = players[0]
    index = 0
    moves = 100
    while (evaluate(board) is None) and (moves > 0):
        node = engines[index].create_search_tree(board)
        train_games(network, node, discount)
        board = node.pv.board
        player = next_player(player)
        index = int(not index)
        moves -= 1


def train(engine, games):
    'train engine for self play in games'
    for _ in range(games):
        TreeStrap([engine, engine], engine.policy, engine.discount)

        
if __name__ == "__main__":
    with open("treestrap.csv", "wb") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        
        plt.ion()
        batch = 20
        learningRate = 0.01
        discount = 0.7
        directory = "treestrap"
        valueNetwork = ValueNet(learningRate, 0.7)
        e = Engine(valueNetwork, 1, discount)
        r = Engine(random, 1, discount)
        win, lose, draw = [], [], []
        testGamesNum = 10
        count = 0
        while True:
            # plot first before train
            w, l, d = 0, 0, 0
            for i in range(testGamesNum):
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
            print "Wins, Losses, Draws:", w, l, d, e.policy(Board())
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

            # train
            train(e, batch)
            if (count % 100) == 99:
                e.policy.save_weights(directory)
            count += 1

