#!/usr/bin/env python
"""
plot training curve
"""
from engine import *
from node import *
from value_network import *
from chess import *

import matplotlib.pyplot as plt
import csv
import os
import time
import json


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
    print(evaluate(board), 400-moves)
    return evaluate(board)


def evaluate_model_performance(batch, e, r):
    'find proportion of games within batch are win, lose or draw'
    w, l, d = 0, 0, 0
    for _ in range(batch):
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
    w = float(w) / (2.0 * batch)
    l = float(l) / (2.0 * batch)
    d = float(d) / (2.0 * batch)
    return w, l, d


def add_new_values(win, lose, draw, path, name):
    'find % win, lose and draw and add to history list'
    valueNetwork = ValueNet(learningRate, 0.7)
    valueNetwork.load_state_dict(torch.load(f'{path}/{name}'))
    valueNetwork.eval()
    e = Engine(valueNetwork, 1, discount)
    w, l, d = evaluate_model_performance(batch, e, r)
    print("Wins, Losses, Draws:", w, l, d, e.policy(Board()))
    win.append(w)
    lose.append(l)
    draw.append(d)
    return win, lose, draw



path = 'tDLambda'
seen = set()
batch = 10
learningRate = 0.01
discount = 0.7
r = Engine(random, 1, discount)
win, lose, draw = [], [], []

plt.ion()
count = 0
while True:
    files = os.listdir(path)
    for name in sorted(files):
        if name not in seen:
            win, lose, draw = add_new_values(win, lose, draw, path, name)
            seen.add(name)
            count += 1
            games = range(0, batch*count, batch)

            graph_data = {
                'games': list(games),
                'win': win,
                'lose': lose,
                'draw': draw
            }
            with open(f'{path}.json', 'w') as outfile:
                json.dump(graph_data, outfile)

            plt.plot(games, win, label="P(win)")
            plt.plot(games, draw, label="P(draw)")
            plt.plot(games, lose, label="P(lose)")
            plt.legend()
            plt.title("Training vs Time")
            plt.xlabel('Self-Play Games Played')
            plt.ylabel('Probability')
            plt.pause(1)
            plt.clf()
        else:
            time.sleep(1)
