#!/usr/bin/env python
"""
train value_network using the TD(lambda) reinforcement algorithm
"""
from engine import *
from node import *
from play import *
from value_network import *
from noughts_crosses import *
import matplotlib.pyplot as plt
import csv


def create_train_sequence(engines, discount):
    'create a forest of nodes, their roots a new board position'
    board = initialBoard
    player = players[0]

    # to explore, do a randomly chosen first move
    r = Engine(random, 1, discount)
    board = r.minimax(board, players[0])
    player = players[1]
    
    trace = []
    index = 0
    while evaluate(board) is None:
        node = engines[index].create_search_tree(board, player)
        trace.append(node)
        board = node.pv.board 
        player = next_player(player)
        index = int(not index)
    node = Node(board)
    node.reward = evaluate(board)
    trace.append(node)
    return trace


def TD_Lambda(engines, network, discount):
    'return sequence of boards and reward for training'
    trace = create_train_sequence(engines, discount)
    boards = [t.board for t in trace]
    reward = trace[-1].reward
    network.temporal_difference(boards, reward, discount)
        

def train(engine, games):
    'train engine for self play in games'
    for _ in range(games):
        TD_Lambda([engine, engine], engine.policy, engine.discount)


if __name__ == "__main__":
    with open("tDLambda.csv", "wb") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        
        plt.ion()
        batch = 20
        learningRate = 0.005
        discount = 1.0#0.7
        directory = "tDLambda"
        valueNetwork = ValueNet(learningRate, 0.7)
        e = Engine(valueNetwork, 3, discount)
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
            print "Wins, Losses, Draws:", w, l, d, e.policy(initialBoard)
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
        
        e.policy.learningRate = 0.001
        for count2 in range(1600):
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
            print "Wins, Losses, Draws:", w, l, d, e.policy(initialBoard)
            win.append(w)
            lose.append(l)
            draw.append(d)
            x = range(0, batch*(count + count2 + 2), batch)
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
            if (count2 % 100) == 99:
                e.policy.save_weights(directory)
