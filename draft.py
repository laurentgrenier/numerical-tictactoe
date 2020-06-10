import numpy as np
from random import randint
import tqdm as tq

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


class AI():
    def __init__(self, level=1):
        self.level = 1

    @staticmethod
    def evaluate(grid):
        # 1 for a good situation
        # -1 for an opponent given victory
        # 0.5 for a draw situation
        # 0 for a move with low consequences
        win_score = 15
        result = 0

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

        return result

    def get_actions(self, grid, pawns):
        result = []
        availables_moves = np.array(np.where(grid == 0)).T

        for i, j in availables_moves:
            for k in pawns:
                result.append(tuple((i, j, k, 0, "0")))

        return result

    def get_actions_tree(self, grid, pfirst_pawns, psecond_pawns, action=(0, 0, 0, 0, "0"), factor=-1, actions=[], limit=3):
        value = self.evaluate(grid)

        # not ended game
        if value == 0 and len(action[4].split(".")) < limit:
            actions.append(tuple(action))
            branch = action[4]

            k = 0
            # evaluate all possible actions
            for action in self.get_actions(grid, pfirst_pawns):
                # increment the branch
                k += 1
                # remove the pawn used in the action
                first_pawns = [pawn for pawn in pfirst_pawns if pawn != action[2]]
                # update the grid
                new_grid = grid.copy()
                new_grid[action[0], action[1]] = action[2]
                # continue exploration
                self.get_actions_tree(new_grid, psecond_pawns, first_pawns,
                                      (action[0], action[1], action[2], value * factor, branch + "." + str(k)),
                                      factor * -1, actions, limit)
        # no more places to play
        else:
            # actions.append(tuple(action))
            actions.append(tuple((action[0], action[1], action[2], value * factor, action[4])))

        return actions

    def get_best_move(self, actions=None):
        # get (path, score, level, row, col, pawn) from possibilities
        values = [(node[4],
                   round(float(node[3] / len(node[4].split("."))), 2),
                   len(node[4].split("."))-1,
                   node[0], node[1], node[2])
                  for node in actions]
        dtype = [('path', 'S10'), ('score', float), ('level', int),
                 ('row', int), ('col', int), ('pawn', int)]
        values = np.array(values, dtype=dtype)
        values = np.sort(values, order=['score'])

        # if the worst move could lead to a bad situation
        if values[0]['score'] < 0 and values[0]['level'] == 1:
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

        node = preselected_nodes[randint(0, len(preselected_nodes)-1)]

        parents_paths = node[0].decode("utf-8").split(".")
        selected_path = parents_paths[0] + "." + parents_paths[1]

        for action in actions:
            if action[4] == selected_path:
                return action

        return None

class Game():

    def __init__(self, mode=1, player_one="player 1", player_two="player 2"):
        # mode 1: human vs human
        # mode 2: human vs computer
        # mode 3: computer vs computer
        self.mode = mode
        self.__set_players(player_one, player_two)
        self.board = Board()
        self.quit = False
        self.turn = 0

    def __set_players(self, player_one_name, player_two_name):
        self.ai_arbiter = AI()
        if self.mode == 1:
            self.players = [{'name': player_two_name, 'id': 2, 'pawns': [2, 4, 6, 8], 'type': 'HUM'},
                            {'name': player_one_name, 'id': 1, 'pawns': [1, 3, 5, 7, 9], 'type': 'HUM'}]
        elif self.mode == 2:
            self.ai_one = AI()
            self.players = [{'name': player_two_name, 'id': 2, 'pawns': [2, 4, 6, 8], 'type': 'HUM'},
                            {'name': player_one_name, 'id': 1, 'pawns': [1, 3, 5, 7, 9], 'type': 'CPU'}]
        elif self.mode == 3:
            self.ai_one = AI()
            self.ai_two = AI()
            self.players = [{'name': player_two_name, 'id': 2, 'pawns': [2, 4, 6, 8], 'type': 'CPU'},
                            {'name': player_one_name, 'id': 1, 'pawns': [1, 3, 5, 7, 9], 'type': 'CPU'}]

    def launch(self, verbose=True):
        quit = False
        while self.ai_arbiter.evaluate(self.board.grid) == 0:
            self.turn += 1
            if verbose:
                print("turn {}".format(self.turn))
                print("{} ({}): ".format(self.players[1]['name'], self.players[1]['type']), self.players[1]['pawns'],
                      "\t{} ({}): ".format(self.players[0]['name'], self.players[0]['type']), self.players[0]['pawns'])
                print("player {} turn.".format(self.players[self.turn % 2]['name']))
                print(self.board.grid)

            move_validate = False
            while not move_validate:
                if self.players[self.turn % 2]['type'] == "HUM":
                    option = input("[Q] Quit [row,col,pawn] To play\n")

                    quit = option == "Q" or option == "q"

                    if quit:
                        print("leaving")
                        break;

                    move = [int(value) for value in option.split(",")]

                    if move[2] in self.players[self.turn % 2]['pawns']:
                        move_validate = self.board.update(move)

                        if not move_validate:
                            if verbose:
                                print("Invalid move")
                        else:
                            self.players[self.turn % 2]['pawns'] = [pawn for pawn in self.players[self.turn % 2]['pawns'] if
                                                          pawn != move[2]]
                    else:
                        if verbose:
                            print("Invalid pawn for {}. Valid are: ".format(self.players[self.turn % 2]['name']),
                                  self.players[self.turn % 2]['pawns'])
                else:
                    if self.turn % 2 == 1:
                        ai = self.ai_one
                    elif self.turn % 2 == 0:
                        ai = self.ai_two

                    # get possibles actions
                    actions_tree = ai.get_actions_tree(self.board.grid, self.players[self.turn % 2]['pawns'],
                                                       self.players[(self.turn + 1) % 2]['pawns'], limit=4, actions=[])

                    # get the best move
                    action = ai.get_best_move(actions_tree)
                    move = [action[0], action[1], action[2]]

                    if move[2] in self.players[self.turn % 2]['pawns']:
                        move_validate = self.board.update(move)

                        if not move_validate:
                            if verbose:
                                print("Invalid move")
                        else:
                            if verbose:
                                self.players[self.turn % 2]['pawns'] = [pawn for pawn in self.players[self.turn % 2]['pawns'] if
                                                          pawn != move[2]]
                    else:
                        if verbose:
                            print("Invalid pawn for {}. Valid are: ".format(self.players[self.turn % 2]['name']),
                                  self.players[self.turn % 2]['pawns'])
            if quit:
                break

        if verbose:
            print("FINAL STATE:\n", self.board.grid)

        if self.ai_arbiter.evaluate(self.board.grid) == 1:
            if verbose:
                print("{} WINS !".format(self.players[self.turn % 2]['name']))
            return self.players[self.turn % 2]['id']
        elif self.ai_arbiter.evaluate(self.board.grid) == 0:
            print("DRAW GAME !")

        if verbose:
            print("Thanks for playing")
        return 0




# draws, player one wins, player two wins
results = np.array([0,0,0])
NUMBER_OF_GAMES = 1
for i in tq.tqdm(range(0, NUMBER_OF_GAMES), total=NUMBER_OF_GAMES):
    results[Game(3).launch(verbose=True)] += 1

print(results)

