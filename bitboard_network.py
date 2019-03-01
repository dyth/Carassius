#!/usr/bin/env python
"""
Value Network based on Giraffe
"""
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import copy

torch.manual_seed(1729)
np.random.seed(1729)


class ValueNet(nn.Module):
    """
    Value Network Layers, Architecture and forward pass
    """
    def __init__(self, learningRate, decay):
        'initialise all the layers and activation functions needed'
        super(ValueNet, self).__init__()

        self.learningRate = learningRate
        self.weightsNum = 0
        self.decay = decay

        # three layers
        self.fc1 = nn.Linear(768, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3= nn.Linear(64, 1)

        # if cuda, use GPU
        self.gpu = torch.cuda.is_available()
        if self.gpu:
            self.cuda()
            if torch.backends.cudnn.is_available():
                torch.backends.cudnn.benchmark = True


    def board_to_bitboard_vector(self, pos, grad):
        'convert fen to a 12 * 8 * 8 = 768 bitboard'
        bitboard = np.zeros((12, 8, 8))

        # over all squares, get piece type and number
        for r in range(8):
            for f in range(8):
                square = 8*r + f
                index = pos.piece_type_at(square)

                # if piece exists in square, if white, increment plane by 6
                if index is not None:
                    piece = pos.piece_at(square).symbol()
                    offset = 0 if piece.istitle() else 6
                    bitboard[index + offset - 1, r, f] = 1.0
        vector = torch.FloatTensor(bitboard.flatten())
        if self.gpu:
            vector = vector.cuda()
        return Variable(vector, requires_grad=grad)


    def forward_pass(self, out):
        'forward pass using Variable inputLayer'
        out = self.fc1(out)
        out = F.elu(out)
        out = self.fc2(out)
        out = F.elu(out)
        out = self.fc3(out)
        out = F.tanh(out)
        return out


    def forward(self, inputLayer):
        'forward pass using Variable inputLayer'
        inputLayer = self.board_to_bitboard_vector(inputLayer, False)
        return self.forward_pass(inputLayer).data[0]


    def temporal_difference(self, boards, lastValue, discount):
        'backup values according to boards'
        traces, gradients, trace = [], [], 0.0
        # boards goes forward in time, so reverse index
        for i in range(len(boards)-1, -1, -1):
            board = self.board_to_bitboard_vector(boards[i], True)
            value = self.forward_pass(board)
            # compute eligibility trace
            difference = discount*lastValue - float(value.data)
            trace = trace*self.decay + difference # trace*discount*self.decay
            traces.append(trace)
            # zero gradients
            for p in self.parameters():
                if p.grad is not None:
                    p.grad.data.zero_()
            # compute partial differential wrt parameters
            lastValue = float(value.data)
            value.backward()
            grad = [copy.deepcopy(p.grad.data) for p in self.parameters()]
            gradients.append(grad)
            del grad, value, board
        # update the parameters of the network
        for (t, grad) in zip(traces, gradients):
            for (p, g) in zip(self.parameters(), grad):
                p.data += self.learningRate * t * g
        del traces, gradients, trace, grad, boards, lastValue
        torch.cuda.empty_cache()


if __name__ == "__main__":
    from chess import *
    v = ValueNet(0.5, 0.7)
    # print(v.board_to_feature_vector(Board(), 1))
    print(v.forward(Board()))
    v.temporal_difference([Board()], 1.0, 0.7)
    print(v.forward(Board())) # confirm that forward pass of chessboard works
    v.temporal_difference([Board()], 1.0, 0.7)
    print(v.forward(Board()))
    v.temporal_difference([Board()], 1.0, 0.7)
    print(v.forward(Board()))
