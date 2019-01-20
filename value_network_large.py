#!/usr/bin/env python
"""
Value Network based on Giraffe
"""
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torch
torch.manual_seed(1729)
np.random.seed(1729)
import copy


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
        self.fc1 = nn.Linear(384, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3= nn.Linear(64, 1)

        # if cuda, use GPU
        self.gpu = torch.cuda.is_available()
        if self.gpu:
            self.cuda()
            if torch.backends.cudnn.is_available():
                torch.backends.cudnn.benchmark = True


    def list_to_Variable(self, inputLayer, grad):
        'convert a list to Variable for use in PyTorch'
        inputLayer = [0 if vector == None else vector for vector in inputLayer]
        inputLayer = torch.FloatTensor(np.array(inputLayer))
        if self.gpu:
            inputLayer = inputLayer.cuda()
        return Variable(inputLayer, requires_grad=grad)


    def set_piece_position(self, i, vector, f, r):
        'set normalised file, rank, 8-rank for a piece'
        while vector[i+3] != 0.0:
            i += 4
            # normalise values
        vector[i] = f / 8.0
        vector[i+1] = r / 8.0
        vector[i+2] = (8-f) / 8.0
        vector[i+3] = (8-r) / 8.0
        return vector


    def board_to_feature_vector(self, board, grad):
        'full piece promotion'
        # create {piece -> vectorposition} dictionary
        # {white, black} 8*Pawn, 10*Knight, 10*Bishop, 10*Rook, 9*Queen, 1*King
        pieces = ['p', 'n', 'b', 'r', 'q', 'k', 'P', 'N', 'B', 'R', 'Q', 'K']
        count = [8, 10, 10, 10, 9, 1, 8, 10, 10, 10, 9, 1]
        countSum = [4 * sum(count[:i]) for i in range(len(count))]
        index = dict(zip(pieces, countSum))
        # create input layer for network
        vector = np.zeros(4*sum(count))#, -1.0)
        for f in range(8):
            for r in range(8):
                square = board.piece_at((8*f) + r)
                # if occupied, set places in the vector
                if square != None:
                    i = index[square.symbol()]
                    self.set_piece_position(i, vector, f, r)
        vector = torch.FloatTensor(np.array(vector))
        if self.gpu:
            vector = vector.cuda()
        return Variable(vector, requires_grad=grad)


    def forward_pass(self, out):
        'forward pass using Variable inputLayer'
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)
        #out = F.dropout(out, training=self.training)
        return F.tanh(out)


    def forward(self, inputLayer):
        'forward pass using Variable inputLayer'
        inputLayer = self.board_to_feature_vector(inputLayer, False)
        return self.forward_pass(inputLayer).data[0]


    def load_weights(self):
        'load name'
        self.load_state_dict(torch.load("weights.h5"))


    def save_weights(self, directory):
        'save the weights as the number in "directory/"'
        self.weightsNum += 1
        name = directory + "/" +  str(self.weightsNum) + ".h5"
        print(name)
        torch.save(self.state_dict(), name)


    def temporal_difference(self, boards, lastValue, discount):
        'backup values according to boards'
        traces, gradients, trace = [], [], 0.0
        # boards goes forward in time, so reverse index
        for i in range(len(boards)-1, -1, -1):
            #board = self.list_to_Variable(boards[i], True)
            board = self.board_to_feature_vector(boards[i], True)
            value = self.forward_pass(board)
            # compute eligibility trace
            difference = discount * lastValue - value.data[0]
            trace = trace * discount * self.decay + difference
            traces.append(trace)
            # zero gradients and compute partial differential wrt parameters
            for p in self.parameters():
                if p.grad is not None:
                    p.grad.data.zero_()
            lastValue = value.data[0]
            value.backward()
            grad = []
            for p in self.parameters():
                grad.append(copy.deepcopy(p.grad.data))
            gradients.append(grad)
            #print "gen", gradients[-1][0][0]
        # update the parameters of the network
        for (t, grad) in zip(traces, gradients):
            for (p, g) in zip(self.parameters(), grad):
                p.data += self.learningRate * t * g
            #print grad[0][0]


if __name__ == "__main__":
    from chess import *
    v = ValueNet(0.5, 0.7)
    print(v.board_to_feature_vector(Board(), 1))
    v.temporal_difference([Board()], 1.0, 0.7)
    print(v.forward(Board())) # confirm that forward pass of chessboard works
