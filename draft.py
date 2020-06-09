import numpy as np
from random import randint

class Board():
    def __init__(self, size=(3, 3)):
        self.grid = np.zeros(size, dtype='uint8')

    def update(self, move):
        if self.grid[move[0], move[1]] == 0:
            self.grid[move[0], move[1]] = move[2]
            return True
        else:
            return False

    def simulate(self, move):
        sgrid = self.grid.copy()
        sgrid[move[0], move[1]] = move[2]
        return sgrid


def evaluate_grid(grid):
    # 1 for a good situation
    # -1 for an opponent given victory
    # 0.5 for a draw situation
    # 0 for a move with low consequences
    win_score = 15

    # win on a row
    if win_score in [grid[:, i].sum() if 0 not in grid[:, i] else 0 for i in range(0, grid.shape[1])]:
        return 1

        # win on a col
    if win_score in [grid[i, :].sum() if 0 not in grid[i, :] else 0 for i in range(0, grid.shape[0])]:
        return 1

    # win on the diag
    if win_score == np.diag(grid).sum() and not 0 in np.diag(grid):
        return 1

    # win on the inverse diag
    if win_score == np.diag(np.fliplr(grid)).sum() and not 0 in np.diag(np.fliplr(grid)):
        return 1

        # draw if all is filled and nobody has win
    if len(np.where(grid == 0)[0]) == 0:
        return 0.5

    return 0


def get_possible_actions(grid, pawns):
    result = []
    availables_moves = np.array(np.where(grid == 0)).T

    for i, j in availables_moves:
        for k in pawns:
            result.append(tuple((i, j, k, 0, "0")))

    return result


def recursive_exploration(grid, pfirst_pawns, psecond_pawns, action=(0, 0, 0, 0, "0"), factor=-1, actions=[], limit=3):
    value = evaluate_grid(grid)

    # not ended game
    if value == 0 and len(action[4].split(".")) < limit:
        actions.append(tuple(action))
        branch = action[4]

        k = 0
        # evaluate all possible actions
        for action in get_possible_actions(grid, pfirst_pawns):
            # increment the branch
            k += 1
            # remove the pawn used in the action
            first_pawns = [pawn for pawn in pfirst_pawns if pawn != action[2]]
            # update the grid
            new_grid = grid.copy()
            new_grid[action[0], action[1]] = action[2]
            # continue exploration
            recursive_exploration(new_grid, psecond_pawns, first_pawns,
                                  (action[0], action[1], action[2], value * factor, branch + "." + str(k)),
                                  factor * -1, actions, limit)
    # no more places to play
    else:
        # actions.append(tuple(action))
        actions.append(tuple((action[0], action[1], action[2], value * factor, action[4])))

    return actions


def get_best_move(possibilities):
    # get (path, score, level, row, col, pawn) from possibilities
    values = [(node[4],
               round(float(node[3] / len(node[4].split("."))), 2),
               len(node[4].split("."))-1,
               node[0], node[1], node[2])
              for node in possibilities]
    dtype = [('path', 'S10'), ('score', float), ('level', int),
             ('row', int), ('col', int), ('pawn', int)]
    values = np.array(values, dtype=dtype)
    values = np.sort(values, order=['score'])

    # if the worst move could lead to a bad situation
    if values[0]['score'] < 0 and values[0]['level'] == 2:
        # Search for all the moves which can avoid that situation
        preselected_nodes = [value for value in values if value[3] == values[0][3]
                             and value[4] == values[0][4] and value[2] == 1]
        print("preselected_nodes: ", preselected_nodes)
    else:
        max_value = values[-1][1]
        current_value = max_value
        preselected_nodes = []
        k=0
        for k in range(0, 10000):
            current_value = values[-1-k][1]
            if current_value == max_value:
                preselected_nodes.append(values[-1-k])
                k += 1
            else:
                break

    node = preselected_nodes[randint(0, len(preselected_nodes))]

    parents_paths = node[0].decode("utf-8").split(".")
    selected_path = parents_paths[0] + "." + parents_paths[1]

    for possibility in possibilities:
        if possibility[4] == selected_path:
            return possibility

    return None


def game():
    board = Board()
    quit = False
    turn = 0
    pawns = [[2, 4, 6, 8], [1, 3, 5, 7, 9]]
    players = [{'name': 'Player 2', 'id': 2, 'pawns': [2, 4, 6, 8], 'type': 'HUM'},
               {'name': 'Player 1', 'id': 1, 'pawns': [1, 3, 5, 7, 9], 'type': 'CPU'}]

    while evaluate_grid(board.grid) == 0:
        turn += 1
        print("turn {}".format(turn))
        print("{} ({}): ".format(players[1]['name'], players[1]['type']), players[1]['pawns'],
              "\t{} ({}): ".format(players[0]['name'], players[0]['type']), players[0]['pawns'])
        print("player {} turn.".format(players[turn % 2]['name']))
        print(board.grid)

        move_validate = False
        while not move_validate:
            if players[turn % 2]['type'] == "HUM":
                option = input("[Q] Quit [row,col,pawn] To play\n")

                quit = option == "Q" or option == "q"

                if quit:
                    print("leaving")
                    break;

                move = [int(value) for value in option.split(",")]

                if move[2] in players[turn % 2]['pawns']:
                    move_validate = board.update(move)

                    if not move_validate:
                        print("Invalid move")
                    else:
                        players[turn % 2]['pawns'] = [pawn for pawn in players[turn % 2]['pawns'] if pawn != move[2]]
                else:
                    print("Invalid pawn for {}. Valid are: ".format(players[turn % 2]['name']),
                          players[turn % 2]['pawns'])
            else:
                # get possibles actions
                possibilities = recursive_exploration(board.grid, players[turn % 2]['pawns'],
                                                      players[(turn + 1) % 2]['pawns'], limit=4, actions=[])

                # get the best move
                action = get_best_move(possibilities)
                move = [action[0], action[1], action[2]]

                if move[2] in players[turn % 2]['pawns']:
                    move_validate = board.update(move)

                    if not move_validate:
                        print("Invalid move")
                    else:
                        players[turn % 2]['pawns'] = [pawn for pawn in players[turn % 2]['pawns'] if pawn != move[2]]
                else:
                    print("Invalid pawn for {}. Valid are: ".format(players[turn % 2]['name']),
                          players[turn % 2]['pawns'])
        if quit:
            break

    print(board.grid)

    if evaluate_grid(board.grid) == 1:
        print("{} WINS !".format(players[turn % 2]['name']))

    print("Thanks for playing")


game()
