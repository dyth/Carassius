#!/usr/bin/env python
"""
plot training curve
"""
from engine import *
from node import *
from value_network_large import *
from chess import *

import matplotlib.pyplot as plt
import csv
import os
import time
import json


games_played = 0


def self_play(engines):
    'engines is a list of engines and engines[0] moves first'
    global games_played
    games_played += 1
    board = Board()
    index = 0
    moves = 400
    while (evaluate(board) is None) and (moves > 0):
        board = engines[index].minimax(board)
        index = int(not index)
        moves -= 1
    pretty_print(board)
    print(games_played, evaluate(board), 400-moves)
    return evaluate(board)


def evaluate_model_performance(batch, e, r):
    'find proportion of games within batch are win, lose or draw'
    w, l, s, d = 0, 0, 0, 0
    for _ in range(batch):
        score = self_play([e, r])
        if score == 1:
            w += 1
        elif score == -1:
            l += 1
        elif score == 0:
            s += 1
        else:
            d += 1
        score = self_play([r, e])
        if score == -1:
            w += 1
        elif score == 1:
            l += 1
        elif score == 0:
            s += 1
        else:
            d += 1
    w = float(w) / (2.0 * batch)
    l = float(l) / (2.0 * batch)
    s = float(s) / (2.0 * batch)
    d = float(d) / (2.0 * batch)
    return w, l, s, d


def add_new_values(win, lose, stalemate, draw, path, name):
    'find % win, lose and draw and add to history list'
    valueNetwork = ValueNet(learningRate, 0.7)
    valueNetwork.load_state_dict(torch.load(f'{path}/{name}'))
    valueNetwork.eval()
    e = Engine(valueNetwork, 1, discount)
    w, l, s, d = evaluate_model_performance(batch, e, r)
    print("Wins, Losses, Stalemates, Draws:", w, l, s, d, e.policy(Board()))
    win.append(w)
    lose.append(l)
    stalemate.append(s)
    draw.append(d)
    return win, lose, stalemate, draw


def sort_file_name(files):
    'sort weights by number'
    return sorted(files, key = lambda x: int(x.split('.')[0]))


path = 'tDLambda5'
seen = set()
batch = 10
learningRate = 0.01
discount = 0.7
r = Engine(random, 1, discount)

if os.path.isfile(f'{path}.json'):
    with open(f'{path}.json') as f:
        data = json.load(f)
    win, lose = data['win'], data['lose']
    stalemate, draw = data['stalemate'], data['draw']
    files = os.listdir(path)
    for name in sort_file_name(files)[:len(win)]:
        seen.add(name)
else:
    win, lose, stalemate, draw = [], [], [], []

plt.ion()
count = len(win)
while True:
    files = os.listdir(path)
    for name in sort_file_name(files):
        if name not in seen:
            win, lose, stalemate, draw = add_new_values(
                win, lose, stalemate, draw, path, name
            )
            seen.add(name)
            count += 1
            games = range(0, batch*count, batch)

            graph_data = {
                'games': list(games),
                'win': win,
                'lose': lose,
                'stalemate': stalemate,
                'draw': draw
            }
            with open(f'{path}.json', 'w') as outfile:
                json.dump(graph_data, outfile)

            plt.plot(games, win, label="P(win)")
            plt.plot(games, draw, label="P(draw)")
            plt.plot(games, stalemate, label="P(stalemate)")
            plt.plot(games, lose, label="P(lose)")
            plt.legend()
            plt.title("Training vs Time")
            plt.xlabel('Self-Play Games Played')
            plt.ylabel('Probability')
            plt.pause(0.1)
            plt.clf()
    time.sleep(0.1)
