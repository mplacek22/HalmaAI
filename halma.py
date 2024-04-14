# Player 1 : top-left corner
import random

p1_camp = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4),
           (1, 0), (1, 1), (1, 2), (1, 3), (1, 4),
           (2, 0), (2, 1), (2, 2), (2, 3),
           (3, 0), (3, 1), (3, 2),
           (4, 0), (4, 1)]

# Player 2 : bottom-right corner
p2_camp = [(15, 15), (15, 14), (15, 13), (15, 12), (15, 11),
           (14, 15), (14, 14), (14, 13), (14, 12), (14, 11),
           (13, 15), (13, 14), (13, 13), (13, 12),
           (12, 15), (12, 14), (12, 13),
           (11, 15), (11, 14)]


def get_opposing_camp(player):
    if player == 1:
        return p2_camp
    else:
        return p1_camp


def is_in_opposing_camp(position, player):
    x, y = position
    camp_squares = get_opposing_camp(player)
    return (x, y) in camp_squares


class Halma:
    def __init__(self, num_players=2):
        self.size = 16
        self.board = [[0] * self.size for _ in range(self.size)]
        self.players = num_players
        self.current_player = 1
        self.setup_players()
        self.winner = 0
        self.round = 0

    def setup_players(self):
        if self.players != 2:
            raise ValueError("Only 2 players are supported")

        for x, y in p1_camp:
            self.board[x][y] = 1

        for x, y in p2_camp:
            self.board[x][y] = 2

    def print_board(self):
        for row in self.board:
            print(' '.join(str(x).rjust(2) for x in row))
        print(f"Current player: {self.current_player}")
        print(f"Round: {self.round}")
        print()

    def is_valid_move(self, start, end):
        x1, y1 = start
        x2, y2 = end
        dx, dy = x2 - x1, y2 - y1

        # move on board
        if not (0 <= x2 < self.size and 0 <= y2 < self.size):
            return False

        # end field is empty
        if self.board[x2][y2] != 0:
            return False

        # Once in the opposing camp, cannot leave
        if is_in_opposing_camp(start, self.current_player) and not is_in_opposing_camp(end, self.current_player):
            return False

        # adjacent move
        if abs(dx) <= 1 and abs(dy) <= 1:
            return True

        # jump
        if abs(dx) > 1 or abs(dy) > 1:
            return self.can_jump(start, end)

    def can_jump(self, start, end):
        x1, y1 = start
        x2, y2 = end
        path = self.find_jump_path(x1, y1, x2, y2)
        return path is not None

    def find_jump_path(self, x1, y1, x2, y2, visited=None):
        if visited is None:
            visited = set()

        visited.add((x1, y1))

        if x1 == x2 and y1 == y2:
            return []

        for dx, dy in [(-2, -2), (-2, 0), (-2, 2), (0, -2), (0, 2), (2, -2), (2, 0), (2, 2)]:
            midx, midy = x1 + dx // 2, y1 + dy // 2
            newx, newy = x1 + dx, y1 + dy
            if 0 <= newx < self.size and 0 <= newy < self.size and self.board[midx][midy] != 0 and (newx, newy) not in visited:
                subpath = self.find_jump_path(newx, newy, x2, y2, visited)
                if subpath is not None:
                    return [(newx, newy)] + subpath

        return None

    def move(self, start, end):
        if not self.is_valid_move(start, end):
            return False
        x1, y1 = start
        x2, y2 = end
        self.board[x2][y2] = self.board[x1][y1]
        self.board[x1][y1] = 0
        self.current_player = 3 - self.current_player
        self.winner = self.check_win()
        self.round += 1
        if self.winner != 0:
            print(f"Gracz {self.winner} wygrał!")
        return True

    def check_win(self):
        if all(self.board[i][j] == 1 for i, j in p2_camp):
            return 1

        if all(self.board[i][j] == 2 for i, j in p1_camp):
            return 2
        return 0

    def generate_possible_moves(self):
        moves = []
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        player = self.current_player

        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] == player:
                    # Simple moves
                    for dx, dy in directions:
                        ni, nj = i + dx, j + dy
                        if 0 <= ni < self.size and 0 <= nj < self.size and self.board[ni][nj] == 0:
                            moves.append(((i, j), (ni, nj)))

                    # Recursive function to handle multiple jumps
                    visited = set()
                    self.find_jumps(i, j, i, j, visited, [(i, j)], moves)

        return moves

    def find_jumps(self, si, sj, i, j, visited, path, moves):
        directions = [(-2, -2), (-2, 0), (-2, 2), (0, -2), (0, 2), (2, -2), (2, 0), (2, 2)]
        has_jumped = False

        for dx, dy in directions:
            ni, nj = i + dx, j + dy
            mi, mj = i + dx // 2, j + dy // 2  # Middle piece to jump over
            if 0 <= ni < self.size and 0 <= nj < self.size and self.board[ni][nj] == 0 and self.board[mi][mj] != 0:
                if (ni, nj) not in visited:  # Avoid cycles in jumping
                    visited.add((ni, nj))
                    path.append((ni, nj))
                    moves.append((path[0], (ni, nj)))
                    self.find_jumps(si, sj, ni, nj, visited, path, moves)
                    path.pop()
                    visited.remove((ni, nj))
                    has_jumped = True

        if not has_jumped and len(path) > 1:
            moves.append((path[0], path[-1]))

    @staticmethod
    def distance_heuristic(board, player):
        size = len(board)
        target_x, target_y = (0, 0) if player == 2 else (size - 1, size - 1)
        total_distance = 0
        count = 0
        for i in range(size):
            for j in range(size):
                if board[i][j] == player:
                    total_distance += abs(target_x - i) + abs(target_y - j)
                    count += 1

        return -total_distance / count if count > 0 else float('inf')

    @staticmethod
    def clustering_heuristic(board, player):
        target_x, _ = (0, 0) if player == 2 else (board.size - 1, board.size - 1)
        centroid_x, centroid_y = 0, 0
        count = 0

        for i in range(board.size):
            for j in range(board.size):
                if board[i][j] == player:
                    centroid_x += i
                    centroid_y += j
                    count += 1

        if count == 0:
            return float('-inf')

        centroid_x /= count
        centroid_y /= count
        clustering_score = 0

        for i in range(board.size):
            for j in range(board.size):
                if board[i][j] == player:
                    clustering_score += abs(centroid_x - i) + abs(centroid_y - j)

        return -clustering_score

    @staticmethod
    def progress_heuristic(board, player):
        mid_line = board.size // 2
        progress_score = 0

        for i in range(board.size):
            for j in range(board.size):
                if board[i][j] == player:
                    if player == 1 and i >= mid_line:
                        progress_score += 1
                    elif player == 2 and i <= mid_line:
                        progress_score += 1

        return progress_score

    def evaluate(self):
        if self.winner == 1:
            return float('inf')  # player 1 wins
        elif self.winner == 2:
            return float('-inf')  # player 2 wins
        else:
            return Halma.distance_heuristic(self.board, self.current_player)  # game not over

    def minimax(self, depth, alpha, beta, maximizing_player):
        if depth == 0 or self.winner != 0:
            return self.evaluate(), None  # Return the evaluation and no move since it's a terminal node

        best_move = None

        if maximizing_player:
            max_eval = float('-inf')
            moves = self.generate_possible_moves()
            for move in moves:
                self.move(move[0], move[1])
                evaluation, _ = self.minimax(depth - 1, alpha, beta, False)
                self.undo_move(move[0], move[1])
                if evaluation > max_eval:
                    max_eval, best_move = evaluation, move
                alpha = max(alpha, evaluation)
                if beta <= alpha:
                    break
            return max_eval, best_move
        else:
            min_eval = float('inf')
            moves = self.generate_possible_moves()
            for move in moves:
                self.move(move[0], move[1])
                evaluation, _ = self.minimax(depth - 1, alpha, beta, True)
                self.undo_move(move[0], move[1])
                if evaluation < min_eval:
                    min_eval, best_move = evaluation, move
                beta = min(beta, evaluation)
                if beta <= alpha:
                    break
            return min_eval, best_move

    def undo_move(self, start, end):
        self.board[start[0]][start[1]] = self.board[end[0]][end[1]]
        self.board[end[0]][end[1]] = 0
        self.current_player = 3 - self.current_player
        self.round -= 1

    def play_turn(self):
        # If the game is already won, no more moves are needed
        if self.winner != 0:
            print(f"Game over! Player {self.winner} has won.")
            return

        if self.current_player == 1:
            move_score, best_move = self.minimax(3, float('-inf'), float('inf'), True)
            print(f"Best move: {best_move} with evaluation: {move_score}")
            self.move(best_move[0], best_move[1])
        else:
            moves = self.generate_possible_moves()
            move = random.choice(moves)
            self.move(move[0], move[1])
        self.print_board()


    def run(self):
        while self.winner == 0:
            self.play_turn()
