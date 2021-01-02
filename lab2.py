# MIT 6.034 Lab 2: Games
# Written by 6.034 staff

from game_api import *
from boards import *
from toytree import GAME1

INF = float('inf')

# Please see wiki lab page for full description of functions and API.

#### Part 1: Utility Functions #################################################

def is_game_over_connectfour(board):
    """Returns True if game is over, otherwise False."""
    for chain in board.get_all_chains():
        if len(chain) >= 4:
            return True

    true_count = 0
    for i in range(6):
        if board.is_column_full(i):
            true_count += 1

    return true_count == 6

def next_boards_connectfour(board):
    """Returns a list of ConnectFourBoard objects that could result from the
    next move, or an empty list if no moves can be made."""
    result = []
    if is_game_over_connectfour(board):
        return []
    for i in range(7):
        if not board.is_column_full(i):
            result.append(board.add_piece(i))
    return result

def is_connectfour(board):
    """Returns True if chain is four in a row."""
    for chain in board.get_all_chains():
        if len(chain) >= 4:
            return True
            
def endgame_score_connectfour(board, is_current_player_maximizer):
    """Given an endgame board, returns 1000 if the maximizer has won,
    -1000 if the minimizer has won, or 0 in case of a tie."""
    if not is_connectfour(board):
        return 0
    if is_current_player_maximizer:
        return -1000
    else:
        return 1000

def endgame_score_connectfour_faster(board, is_current_player_maximizer):
    """Given an endgame board, returns an endgame score with abs(score) >= 1000,
    returning larger absolute scores for winning sooner."""
    score = endgame_score_connectfour(board, is_current_player_maximizer)

    if not is_connectfour(board) or score == 0:
        return 0
    elif score < 0: 
        return -1000 - (42 - board.count_pieces(True))
    else:
        return 1000 + (42 - board.count_pieces(False))

def heuristic_connectfour(board, is_current_player_maximizer):
    """Given a non-endgame board, returns a heuristic score with
    abs(score) < 1000, where higher numbers indicate that the board is better
    for the maximizer."""  
    maximizer = heuristic_score_calculator(board, is_current_player_maximizer)
    minimizer = heuristic_score_calculator(board, not is_current_player_maximizer)
    return maximizer - minimizer

def heuristic_score_calculator(board, is_current_player_maximizer):
    score = 0
    scoring = {1: 1, 2: 10, 3: 100, 4: 500}
    for chain in board.get_all_chains(is_current_player_maximizer):
        chain_length = len(chain)
        score += scoring[chain_length]
    return score

# Now we can create AbstractGameState objects for Connect Four, using some of
# the functions you implemented above.  You can use the following examples to
# test your dfs and minimax implementations in Part 2.

# This AbstractGameState represents a new ConnectFourBoard, before the game has started:
state_starting_connectfour = AbstractGameState(snapshot = ConnectFourBoard(),
                                 is_game_over_fn = is_game_over_connectfour,
                                 generate_next_states_fn = next_boards_connectfour,
                                 endgame_score_fn = endgame_score_connectfour_faster)

# This AbstractGameState represents the ConnectFourBoard "NEARLY_OVER" from boards.py:
state_NEARLY_OVER = AbstractGameState(snapshot = NEARLY_OVER,
                                 is_game_over_fn = is_game_over_connectfour,
                                 generate_next_states_fn = next_boards_connectfour,
                                 endgame_score_fn = endgame_score_connectfour_faster)

# This AbstractGameState represents the ConnectFourBoard "BOARD_UHOH" from boards.py:
state_UHOH = AbstractGameState(snapshot = BOARD_UHOH,
                                 is_game_over_fn = is_game_over_connectfour,
                                 generate_next_states_fn = next_boards_connectfour,
                                 endgame_score_fn = endgame_score_connectfour_faster)


#### Part 2: Searching a Game Tree #############################################

# Note: Functions in Part 2 use the AbstractGameState API, not ConnectFourBoard.

def dfs_maximizing(state) :
    """Performs depth-first search to find path with highest endgame score.
    Returns a tuple containing:
     0. the best path (a list of AbstractGameState objects),
     1. the score of the leaf node (a number), and
     2. the number of static evaluations performed (a number)"""
    stack = [[state]]
    num_evals = 0
    path_score = None, None 
    
    while stack!=[]:
        path = stack.pop()
        node = path[-1]
      
        next_states = node.generate_next_states()
        
        if next_states!=[]:
            for state in next_states:
                if state not in path:
                    stack.append(path + [state])
        else:
            node_score = node.get_endgame_score(is_current_player_maximizer=True)
            num_evals += 1
            
            if path_score == (None, None) or node_score > path_score[-1]:
                    path_score = path, node_score 
    
    best_path = path_score[-2] 
    score = path_score[-1]                             
    return (best_path, score, num_evals)

# Uncomment the line below to try your dfs_maximizing on an
# AbstractGameState representing the games tree "GAME1" from toytree.py:

# pretty_print_dfs_type(dfs_maximizing(GAME1))


def minimax_endgame_search(state, maximize=True) :
    """Performs minimax search, searching all leaf nodes and statically
    evaluating all endgame scores.  Same return type as dfs_maximizing."""
    result = []
    best_path = [state]
    num_evals = 1
    if state.is_game_over():
        score = state.get_endgame_score(maximize)
        return (best_path, score, num_evals)

    moves = state.generate_next_states()
    for move in moves:
        result.append(minimax_endgame_search(move,not maximize))
    num_evals = 0 
    for i in result:
        num_evals += i[2]
    if maximize:
        result = max(result,key = lambda x: x[1])
    if not maximize:
        result = min(result,key = lambda x: x[1])
    
    score = result[1]
    best_path = [state] + result[0] 
    return (best_path, score, num_evals)


# Uncomment the line below to try your minimax_endgame_search on an
# AbstractGameState representing the ConnectFourBoard "NEARLY_OVER" from boards.py:

# pretty_print_dfs_type(minimax_endgame_search(state_NEARLY_OVER))


def minimax_search(state, heuristic_fn=always_zero, depth_limit=INF, maximize=True) :
    """Performs standard minimax search. Same return type as dfs_maximizing."""
    result = []
    best_path = [state]
    num_evals = 1
    if state.is_game_over():
        score = state.get_endgame_score(maximize)
        return (best_path, score, num_evals)

    if depth_limit == 0:
        score = heuristic_fn(state.get_snapshot(),maximize)
        return (best_path, score, num_evals)

    moves = state.generate_next_states()
    for move in moves:
        result.append(minimax_search(move, heuristic_fn, depth_limit - 1, not maximize))
    num_evals = 0 
    for i in result:
        num_evals += i[2]
    if maximize:
        result = max(result,key = lambda x: x[1])
    if not maximize:
        result = min(result,key = lambda x: x[1])
    
    score = result[1]
    best_path = [state] + result[0] 
    return (best_path, score, num_evals)


# Uncomment the line below to try minimax_search with "BOARD_UHOH" and
# depth_limit=1. Try increasing the value of depth_limit to see what happens:

# pretty_print_dfs_type(minimax_search(state_UHOH, heuristic_fn=heuristic_connectfour, depth_limit=1))


def minimax_search_alphabeta(state, alpha=-INF, beta=INF, heuristic_fn=always_zero,
                             depth_limit=INF, maximize=True) :
    """"Performs minimax with alpha-beta pruning. Same return type 
    as dfs_maximizing."""
    best_path = [state]
    num_evals = 1
    if state.is_game_over():
        score = state.get_endgame_score(maximize)
        return (best_path, score, num_evals)
        
    if depth_limit == 0:
        score = heuristic_fn(state.get_snapshot(),maximize)
        return (best_path, score, num_evals)

    if maximize:
        maxalpha = alpha
        bestresult = None
        result = []
        moves = state.generate_next_states()
        for move in moves:
            recurse =minimax_search_alphabeta(move, maxalpha, beta, heuristic_fn, depth_limit - 1, not maximize)
            result.append(recurse)
            minalpha = maxalpha
            maxalpha = max(maxalpha,recurse[1])
            if maxalpha != minalpha:
                bestresult = recurse
            if maxalpha >= beta:
                break
        
        num_evals = 0 
        for i in result:
            num_evals += i[2]
        bestresult = max(result,key = lambda x: x[1])
        
        best_path = [state] + bestresult[0] 
        score = bestresult[1]
        return (best_path, score, num_evals)

    if not maximize:
        result = []
        bestresult = None
        maxbeta = beta
        moves = state.generate_next_states()
        for move in moves:
            recurse = minimax_search_alphabeta(move, alpha, maxbeta, heuristic_fn, depth_limit - 1, not maximize)
            result.append(recurse)
            minbeta = maxbeta
            maxbeta = min(maxbeta,recurse[1])
            if maxbeta != minbeta:
                bestresult = recurse
            if maxbeta <= alpha:
                break

        num_evals = 0 
        for i in result:
            num_evals += i[2]
        bestresult = min(result,key = lambda x: x[1])
        
        best_path = [state] + bestresult[0] 
        score = bestresult[1]
        return (best_path, score, num_evals)


# Uncomment the line below to try minimax_search_alphabeta with "BOARD_UHOH" and
# depth_limit=4. Compare with the number of evaluations from minimax_search for
# different values of depth_limit.

# pretty_print_dfs_type(minimax_search_alphabeta(state_UHOH, heuristic_fn=heuristic_connectfour, depth_limit=4))


def progressive_deepening(state, heuristic_fn=always_zero, depth_limit=INF,
                          maximize=True) :
    """Runs minimax with alpha-beta pruning. At each level, updates anytime_value
    with the tuple returned from minimax_search_alphabeta. Returns anytime_value."""
    anytime_value = AnytimeValue()
    value = 0
    while value < depth_limit:
        anytime_value.set_value(minimax_search_alphabeta(state, -INF, INF, heuristic_fn,
                             value+1, maximize=True))
        value += 1
    return anytime_value


# Uncomment the line below to try progressive_deepening with "BOARD_UHOH" and
# depth_limit=4. Compare the total number of evaluations with the number of
# evaluations from minimax_search or minimax_search_alphabeta.

# progressive_deepening(state_UHOH, heuristic_fn=heuristic_connectfour, depth_limit=4).pretty_print()


# Progressive deepening is NOT optional. However, you may find that 
#  the tests for progressive deepening take a long time. If you would
#  like to temporarily bypass them, set this variable False. You will,
#  of course, need to set this back to True to pass all of the local
#  and online tests.
TEST_PROGRESSIVE_DEEPENING = True
if not TEST_PROGRESSIVE_DEEPENING:
    def not_implemented(*args): raise NotImplementedError
    progressive_deepening = not_implemented


#### Part 3: Multiple Choice ###################################################

ANSWER_1 = '4'

ANSWER_2 = '1'

ANSWER_3 = '4'

ANSWER_4 = '5'


#### SURVEY ###################################################

NAME = "Marisa Papagelis"
COLLABORATORS = "Peyton Wang"
HOW_MANY_HOURS_THIS_LAB_TOOK = 20
WHAT_I_FOUND_INTERESTING = None
WHAT_I_FOUND_BORING = None
SUGGESTIONS = None
