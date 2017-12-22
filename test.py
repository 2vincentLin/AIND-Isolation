# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 02:54:42 2017

@author: vincent
"""

from isolation import Board
from sample_players import *
from game_agent import *
from random import randint
import numpy as np

# create an isolation board (by default 7x7)
player1 = AlphaBetaPlayer(score_fn=custom_score)
player2 = MinimaxPlayer(score_fn=improved_score)
game = Board(player1, player2)

game.apply_move([1,1])
game.get_legal_moves()
game.get_player_location(game.get_opponent(player2))

game = Board(player2, player1)

# play the remainder of the game automatically -- outcome can be "illegal
# move", "timeout", or "forfeit"
winner, history, outcome = game.play(time_limit= 150)
print("\nWinner: {}\nOutcome: {}".format(winner, outcome))
print(game.to_string())
print("Move history:\n{!s}".format(history))
print(player1== winner)





# =============================================================================
# test code
# =============================================================================

first_hand_wins= 0
second_hand_wins= 0

for i in range(10):
    player1 = AlphaBetaPlayer(score_fn=improved_score)
    player2 = AlphaBetaPlayer(score_fn=improved_score)
    game= Board(player1, player2)
    winner, history, outcome = game.play(time_limit=150)
    if player1== winner:
        first_hand_wins+= 1
    game = Board(player2, player1)
    winner, history, outcome = game.play(time_limit=150)
    if player1 == winner:
        second_hand_wins+= 1
print(first_hand_wins, second_hand_wins)




