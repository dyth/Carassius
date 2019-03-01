#!/usr/bin/env python
"""
determine whether bitboard, piece-list coordinate or mailbox is best

bitboard (12, 8, 8): third order tensor

bitboard_vector (768): flattened third order tensor
small_bitboard_vector (384): with -1.0 values for black

piece_list (384): normalised rank, file, 8-rank and file for pieces in list
small_piece_list (192): normalised rank and file for pieces in list

mailbox (8, 8): normalised type of piece in its position in an 8 * 8 board
"""
import chess
import numpy as np


def fen_to_3D_bitboard(fen):
    'convert fen to a 12 by 8 * 8 bitboard'
    pos = chess.Board(fen)
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
    return bitboard


def fen_to_bitboard_vector(fen):
    'convert fen to a 12 * 8 * 8 = 768 bitboard'
    pos = chess.Board(fen)
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
    return bitboard.flatten()


def fen_to_small_bitboard_vector(fen):
    'convert fen to a 6 * 8 * 8 = 384 bitboard'
    pos = chess.Board(fen)
    bitboard = np.zeros((6, 8, 8))

    # over all squares, get piece type and number
    for r in range(8):
        for f in range(8):
            square = 8*r + f
            index = pos.piece_type_at(square)

            # if piece exists in square, if white, increment else decrement
            if index is not None:
                piece = pos.piece_at(square).symbol()
                player += 1.0 if piece.istitle() else -1.0
                bitboard[index-1, r, f] = player
    return bitboard.flatten()


def fen_to_piece_list(fen):
    'convert fen to 384 piece list of coordinates'
    pos = chess.Board(fen)
    pieceList = np.zeros(384)

    # {white, black} 8*Pawn, 10*Knight, 10*Bishop, 10*Rook, 9*Queen, 1*King
    pieces = ['p', 'n', 'b', 'r', 'q', 'k', 'P', 'N', 'B', 'R', 'Q', 'K']
    count = [8, 10, 10, 10, 9, 1, 8, 10, 10, 10, 9, 1]

    # create {piece -> vectorposition} dictionary
    vectorPosition = [4 * sum(count[:i]) for i in range(len(count))]
    pieceIndex = dict(zip(pieces, vectorPosition))

    # create pieceList and type
    for f in range(8):
        for r in range(8):
            square = pos.piece_at(8*f + r)

            # if piece exists in square, find first available index by adding 4
            if square is not None:
                index = pieceIndex[square.symbol()]
                while pieceList[index] != 0.0:
                    index += 4

                # set pieceList index to coordinate
                pieceList[index] = (r+1.0) / 8.0
                pieceList[index+1] = (8.0-r) / 8.0
                pieceList[index+2] = (f+1.0) / 8.0
                pieceList[index+3] = (8.0-f) / 8.0
    return pieceList


def fen_to_small_piece_list(fen):
    'convert fen to 192 piece list of coordinates'
    pos = chess.Board(fen)
    pieceList = np.zeros(192)

    # {white, black} 8*Pawn, 10*Knight, 10*Bishop, 10*Rook, 9*Queen, 1*King
    pieces = ['p', 'n', 'b', 'r', 'q', 'k', 'P', 'N', 'B', 'R', 'Q', 'K']
    count = [8, 10, 10, 10, 9, 1, 8, 10, 10, 10, 9, 1]

    # create {piece -> vectorposition} dictionary
    vectorPosition = [2 * sum(count[:i]) for i in range(len(count))]
    pieceIndex = dict(zip(pieces, vectorPosition))

    # create pieceList and type
    for f in range(8):
        for r in range(8):
            square = pos.piece_at(8*f + r)

            # if piece exists in square, find first available index by adding 4
            if square is not None:
                index = pieceIndex[square.symbol()]
                while pieceList[index] != 0.0:
                    index += 2

                # set pieceList index to coordinate
                pieceList[index] = (r+1.0) / 8.0
                pieceList[index+1] = (f+1.0) / 8.0
    return pieceList


def fen_to_mailbox(fen):
    'convert fen to 8 * 8 matrix'
    pos = chess.Board(fen)
    bitboard = np.zeros((2, 8, 8))

    # over all squares, get piece type and number
    for r in range(8):
        for f in range(8):
            square = 8*r + f
            index = pos.piece_type_at(square)

            # if piece exists in square, if white, increment plane by 6
            if index is not None:
                piece = pos.piece_at(square).symbol()
                offset = 0.0 if piece.istitle() else 6.0
                encoding = (offset + index) / 12.0
                bitboard[0, r, f] = encoding
                bitboard[1, r, f] = 1.0 - encoding
    return bitboard


def fen_to_mailbox_flat(fen):
    'convert fen to 8 * 8 matrix'
    pos = chess.Board(fen)
    bitboard = np.zeros(2 * 8 * 8)

    # over all squares, get piece type and number
    for r in range(8):
        for f in range(8):
            square = 8*r + f
            index = pos.piece_type_at(square)

            # if piece exists in square, if white, increment plane by 6
            if index is not None:
                piece = pos.piece_at(square).symbol()
                offset = 0.0 if piece.istitle() else 6.0
                encoding = (offset + index) / 12.0
                bitboard[2*square] = encoding
                bitboard[2*square+1] = 1.0 - encoding
    return bitboard


if __name__ == "__main__":
    print(fen_to_bitboard_vector(chess.Board().fen()))
    # print(fen_to_small_bitboard_vector(chess.Board().fen()))
    # print(fen_to_piece_list(chess.Board().fen()))
    # print(fen_to_small_piece_list(chess.Board().fen()))
    # print(fen_to_mailbox(chess.Board().fen()))
    # print(fen_to_mailbox_flat(chess.Board().fen()))
