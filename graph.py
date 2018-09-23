#!/usr/bin/env python
"""
plot graph of training
"""
import csv
import matplotlib.pyplot as plt


def read_file(name):
    'return list of win, lose draw over time'
    win, lose, draw = [], [], []
    with open(name) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            win.append(float(row[0]))
            lose.append(float(row[1]))
            draw.append(float(row[2]))
    return win, lose, draw


def make_graph(win, lose, draw, limit, batch):
    'plot graph'
    x = range(0, batch*limit, batch)
    plt.plot(x, win[:limit], label="P(win)")
    plt.plot(x, draw[:limit], label="P(draw)")
    plt.plot(x, lose[:limit], label="P(lose)")
    plt.legend()
    plt.title("Training vs Time")
    plt.xlabel('Self-Play Games Played')
    plt.ylabel('Probability')
    plt.show()


if __name__ == "__main__":
    win, lose, draw = read_file("tDLambda_9_128_32_1.csv")
    make_graph(win, lose, draw, 1000, 20)
