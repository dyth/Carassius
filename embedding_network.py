#!/usr/bin/env python
"""
Smaller Value Network based on AlphaZero
"""
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import copy

torch.manual_seed(17)
np.random.seed(17)


class ImageBOWEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None, reduce_fn=torch.mean):
        super(ImageBOWEmbedding, self).__init__()
        self.padding_idx = padding_idx
        self.reduce_fn = reduce_fn
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)

    def forward(self, inputs):
        embeddings = self.embedding(inputs)
        embeddings = self.reduce_fn(embeddings, dim=1)
        embeddings = torch.transpose(embeddings, 1, 3)
        embeddings = torch.transpose(embeddings, 2, 3)
        return embeddings


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

        # self.embedding = nn.Embedding(13, 16, padding_idx=0)
        self.embedding = ImageBOWEmbedding(13, 4)
        self.value = nn.Sequential(
            nn.Conv2d(4, 8, 3, 1, padding=1),
            nn.GroupNorm(1, 8),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, 2),
            nn.GroupNorm(1, 16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 1, padding=1),
            nn.GroupNorm(1, 32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2),
            nn.GroupNorm(1, 64),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1, 1)
        )

        # if cuda, use GPU
        self.gpu = torch.cuda.is_available()
        if self.gpu:
            self.cuda()
            if torch.backends.cudnn.is_available():
                torch.backends.cudnn.benchmark = True


    def fen_to_mailbox(self, pos, grad):
        'convert fen to a 6 * 8 * 8 = 384 bitboard'
        bitboard = np.zeros((1, 2, 8, 8))
        # over all squares, get piece type and number
        for r in range(8):
            for f in range(8):
                square = 8*r + f
                index = pos.piece_type_at(square)
                # if piece exists in square, if white, increment else decrement
                if index is not None:
                    piece = pos.piece_at(square).symbol()
                    player = 1 if piece.istitle() else 2
                    bitboard[0, 1, r, f] = index
                    bitboard[0, 0, r, f] = player
        vector = torch.LongTensor(bitboard)
        if self.gpu:
            vector = vector.cuda()
        vector = self.embedding(vector)
        # vector = torch.transpose(vector, 1, 3)
        # vector = torch.transpose(vector, 2, 3)
        return vector


    def forward_pass(self, out):
        'forward pass using Variable inputLayer'
        # 1, 16, 8, 8
        # out = self.conv1(out)
        # out = F.relu(out)
        # out = self.conv2(out)
        # out = F.relu(out)
        # out = self.conv3(out)
        # out = F.relu(out)
        # out = self.conv4(out)
        # out = F.relu(out)
        # out = self.conv5(out)
        # out = F.relu(out)
        # out = self.conv6(out)
        # out = F.tanh(out)
        out = self.value(out)
        out = torch.tanh(out)
        return out.view(-1)


    def forward(self, inputLayer):
        'forward pass using Variable inputLayer'
        inputLayer = self.fen_to_mailbox(inputLayer, False)
        return self.forward_pass(inputLayer).data[0]


    def temporal_difference(self, boards, lastValue, discount):
        'backup values according to boards'
        traces, gradients, trace = [], [], 0.0
        # boards goes forward in time, so reverse index
        for i in range(len(boards)-1, -1, -1):
            board = self.fen_to_mailbox(boards[i], True)
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
    # v.fen_to_mailbox(Board(), 1)
    print(v.forward(Board()))
    v.temporal_difference([Board()], 1.0, 0.7)
    print(v.forward(Board())) # confirm that forward pass of chessboard works
    v.temporal_difference([Board()], 1.0, 0.7)
    print(v.forward(Board()))
    v.temporal_difference([Board()], 1.0, 0.7)
    print(v.forward(Board()))
