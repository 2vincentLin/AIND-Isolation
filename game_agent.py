"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random


# =============================================================================
# 
# =============================================================================

import numpy as np

def manhattan_distance(A, B):
    return sum([abs(a-b) for a, b in zip(A, B)])

def tailing(game, player):
    '''
    If the player’s location is on one of the legal move locations of the 
    opponent,     return higher score. Because the player can keep following 
    opponent,     reduce one of the opponent’s possible moves. If the player’s 
    location is     one square away from the opponent, return lower but still 
    positive score,     because on the next move, no matter where the opponent 
    goes, the player can occupy one of the legal move locations of the opponent 
    or simply follows the opponent.
    '''
      
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    A= game.get_player_location(player)
    B= game.get_player_location(game.get_opponent(player))

    if not A:
        return 0.0

    
    distance= np.sum([abs(a-b) for a,b in zip(A,B) if abs(a-b) < 3])
    
    if distance== 3:
        return 2.0
    elif distance== 1:
        return 1.0
    return 0.0




def near_blank(game, player):
    '''
    Get blank spaces, then calculate which rows and column have the most blanks 
    squares, if the player’s location is near the row and column have the most 
    blanks squares, return higher score on the base of improved score. The 
    reason given a higher score is because around the player may contain more 
    blanks. 
    '''
    
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")
    
    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))   
    score= float(own_moves - opp_moves)
#    score= 0
    
    blanks= np.array(game.get_blank_spaces())
    A= game.get_player_location(player)
    if not A:
        return score
    for i in range(2):
        _, temp= np.unique(blanks[:,i], return_counts= True)
        if abs(A[i]-np.argmax(temp))== 1:
            score+= 1
    return score

def combines(game, player):
    '''
    combines the tailing and near_blank function
    '''
   
    
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")
    


    A= game.get_player_location(player)
    B= game.get_player_location(game.get_opponent(player))
    
    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))   
    score= float(own_moves - opp_moves)
    
    if not A:
        return score
    distance= np.sum([abs(a-b) for a,b in zip(A,B) if abs(a-b) < 3])
    blanks= np.array(game.get_blank_spaces())

    for i in range(2):
        _, temp= np.unique(blanks[:,i], return_counts= True)
        if abs(A[i]-np.argmax(temp))== 1:
            score+= 1
    if distance== 3:
        score += 2.0
    elif distance== 1:
        score += 1.0
    return score



class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # TODO: finish this function!
    

    return combines(game, player)


def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # TODO: finish this function!
    return tailing(game, player)


def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # TODO: finish this function!

    return near_blank(game, player)



class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

 

class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        legal_moves= game.get_legal_moves()
        best_move = (-1, -1)
        if not legal_moves:
            return best_move
        else:
            best_move= legal_moves[0]
        if len(game.get_blank_spaces()) == 49:
            return (random.randint(0, 6), random.randint(0, 6))
        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.

            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move
    
    def cut_off(self, game, depth):
        '''
        cut off test for minimax algorithm
        Args:
            game(Isolation.Board)
            depth(Int)
        Returns:
            True for len(legal moves)== 0 or depth== 0
            False for else
        '''
    
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        if not game.get_legal_moves() or depth == 0:
            return True
        return False     
    
    def max_value(self, game, depth):
        '''
        max_value implementation from minimax algorithm
        Args:
            game(Isolation.Board)
            depth(Int)
        Returns:
            heuristic(evaluation value) if cut_off returns True
        '''
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        if self.cut_off(game, depth):
            return self.score(game, self)
        v= float('-inf')
        for m in game.get_legal_moves():
            v= max(v, self.min_value(game.forecast_move(m), depth-1))
        return v

    def min_value(self, game, depth):
        '''
        min_value implementation from minimax algorithm
        Args:
            game(Isolation.Board)
            depth(Int)
        Returns:
            heuristic(evaluation value) if cut_off returns True        
        '''
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        if self.cut_off(game, depth):
            return self.score(game, self)
        v= float('inf')
        for m in game.get_legal_moves():
            v= min(v, self.max_value(game.forecast_move(m), depth-1))
        return v

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # TODO: finish this function!
        best_score= float('-inf')

        # in some cases, if all the moves lead to lose
        # the v will always be -inf, that means best_move will never be updated. 
        # so in this situation, I set best_move= first move
        if len(game.get_legal_moves()) != 0:
            best_move= game.get_legal_moves()[0]
        for m in game.get_legal_moves():
            v= self.min_value(game.forecast_move(m), depth-1)
            if v > best_score:
                best_score= v
                best_move= m
        return best_move
        #raise NotImplementedError


class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """
    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # TODO: finish this function!

        if not game.get_legal_moves():
            return (-1, -1)
        if len(game.get_blank_spaces()) == 49:
            # empty board, randomly choose first move
            return (random.randint(0, 6), random.randint(0, 6))
        depth= 0
        alpha= float('-inf')
        beta= float('inf')
        moves= []
        try:
            while depth <= len(game.get_blank_spaces()):
                # in end game, this loop can easily search for depth > 1000,
                # which is wasting of time since the max depth is the blank spces
                moves.append(self.alphabeta(game, depth, alpha, beta))
                depth+= 1
        except SearchTimeout:
            pass
        # if SearchTimeout, then moves[-1] will be None, in this case, return moves[-2]
        if not moves[-1]:
            return moves[-1]
        else:
            return moves[-2]

# =============================================================================
# another implementation, in some case, this implementation works better  
# =============================================================================
#        scores_and_moves= [] # [(score,(moves))]   
#        try:
#            while depth <= 49:
#                # limit the depth <= 49 so at the end game, 
#                # it won't run thousands times with the same result
#                scores_and_moves.append(self.alphabeta(game, depth, alpha, beta))
#                depth += 1
#        except SearchTimeout:
#            pass
#        #print(depth, max(scores_and_moves))
#        return max(scores_and_moves[::-1], key= lambda t: t[0])[1]  
        # backward search for highest score then return the move
 
    def cut_off(self, game, depth):
        '''
        cut off test for minimax algorithm with alpha-beta pruning
        Args:
            game(Isolation.Board)
            depth(Int)
        Returns:
            True for len(legal moves)== 0 or depth== 0
            False for else        
        '''
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        if not game.get_legal_moves() or depth == 0:
            return True
        return False
            
    def max_value(self, game, depth, alpha, beta):
        '''
        max_value implementation from minimax algorithm with alpha-beta pruning
        Args:
            game(Isolation.Board)
            depth(Int)
        Returns:
            heuristic(evaluation value) if cut_off returns True        
        '''
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        if self.cut_off(game, depth):
            return self.score(game, self)

        v= float('-inf')
        for m in game.get_legal_moves():
            v= max(v, self.min_value(game.forecast_move(m), depth-1, alpha, beta))
            if v >= beta: return v
            alpha = max(alpha, v)
        return v

    def min_value(self, game, depth, alpha, beta):
        '''
        min_value implementation from minimax algorithm with alpha-beta pruning
        Args:
            game(Isolation.Board)
            depth(Int)
        Returns:
            heuristic(evaluation value) if cut_off returns True           
        '''
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        if self.cut_off(game, depth):
            return self.score(game, self)
        v= float('inf')
        for m in game.get_legal_moves():
            v= min(v, self.max_value(game.forecast_move(m), depth-1, alpha, beta))
            if v <= alpha: return v
            beta= min(beta, v)
        return v
    
    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # TODO: finish this function!
        if self.cut_off(game, depth):
            return self.score(game, self)
        v= best_score= float('-inf')
        legal_moves= game.get_legal_moves()
        best_move= legal_moves[0]
        for m in legal_moves:
            v= self.min_value(game.forecast_move(m), depth-1, alpha, beta)
            alpha= max(alpha, v)
            if v > best_score:
                best_score= v
                best_move= m
        return best_move
        #return (best_score, best_move)
        # change to return (score, move)
        #raise NotImplementedError
