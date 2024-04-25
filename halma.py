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


def is_valid_move(board, start, end, player):
    x1, y1 = start
    x2, y2 = end
    dx, dy = x2 - x1, y2 - y1
    size = len(board)

    # Ensure move is on the board
    if not (0 <= x2 < size and 0 <= y2 < size):
        return False

    # Destination must be empty
    if board[x2][y2] != 0:
        return False

    # Cannot leave the opposing camp once entered
    if is_in_opposing_camp(start, player) and not is_in_opposing_camp(end, player):
        return False

    # Adjacent move
    if abs(dx) <= 1 and abs(dy) <= 1:
        return True

    # Jump
    if abs(dx) > 1 or abs(dy) > 1:
        return can_jump(board, start, end, size)

    return False


def can_jump(board, start, end, size):
    path = find_jump_path(board, start[0], start[1], end[0], end[1], size)
    return path is not None


def find_jump_path(board, x1, y1, x2, y2, size, visited=None):
    if visited is None:
        visited = set()
    visited.add((x1, y1))

    if x1 == x2 and y1 == y2:
        return []

    for dx, dy in [(-2, -2), (-2, 0), (-2, 2), (0, -2), (0, 2), (2, -2), (2, 0), (2, 2)]:
        midx, midy = x1 + dx // 2, y1 + dy // 2
        newx, newy = x1 + dx, y1 + dy
        if 0 <= newx < size and 0 <= newy < size and board[midx][midy] != 0 and (newx, newy) not in visited:
            subpath = find_jump_path(board, newx, newy, x2, y2, size, visited)
            if subpath is not None:
                return [(newx, newy)] + subpath

    return None


def check_win(board):
    if all(board[i][j] == 1 for i, j in p2_camp):
        return 1

    if all(board[i][j] == 2 for i, j in p1_camp):
        return 2
    return 0


def find_jumps(si, sj, i, j, visited, path, moves, current_player, size, board):
    directions = [(-2, -2), (-2, 0), (-2, 2), (0, -2), (0, 2), (2, -2), (2, 0), (2, 2)]
    opposing_camp = get_opposing_camp(current_player)
    in_opposing_camp = (si, sj) in opposing_camp  # Check if starting position is in the opposing camp

    for dx, dy in directions:
        ni, nj = i + dx, j + dy
        mi, mj = i + dx // 2, j + dy // 2  # Middle piece to jump over
        if 0 <= ni < size and 0 <= nj < size and board[ni][nj] == 0 and board[mi][mj] != 0:
            if (ni, nj) not in visited:  # Avoid cycles in jumping
                if not in_opposing_camp or (ni, nj) in opposing_camp:  # Check camp condition
                    visited.add((ni, nj))
                    path.append((ni, nj))
                    moves.append((path[0], (ni, nj)))
                    find_jumps(si, sj, ni, nj, visited, path, moves, current_player, size, board)  # Continue jumping
                    path.pop()
                    visited.remove((ni, nj))


def apply_move(board, start, end):
    x1, y1 = start
    x2, y2 = end
    board[x2][y2] = board[x1][y1]
    board[x1][y1] = 0


def generate_possible_moves(current_player, size, board):
    moves = []
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    player = current_player
    opposing_camp = get_opposing_camp(player)

    for i in range(size):
        for j in range(size):
            if board[i][j] == player:
                # Check if the piece is in the opposing camp
                in_opposing_camp = (i, j) in opposing_camp

                # Simple moves
                for dx, dy in directions:
                    ni, nj = i + dx, j + dy
                    if 0 <= ni < size and 0 <= nj < size and board[ni][nj] == 0:
                        # If in opposing camp, check the new position is also in the camp
                        if not in_opposing_camp or (ni, nj) in opposing_camp:
                            moves.append(((i, j), (ni, nj)))

                # Recursive function to handle multiple jumps
                visited = set()
                find_jumps(i, j, i, j, visited, [(i, j)], moves, current_player, size, board)

    return moves


def distance_heuristic(board, player):
    size = len(board)
    target_camp = get_opposing_camp(player)
    total_distance = 0
    count = 0
    reward_for_being_in_target = -100  # Define a reward for each piece in the target camp

    for i in range(size):
        for j in range(size):
            if board[i][j] == player:
                if (i, j) in target_camp:
                    # Apply the reward for being in the target position
                    total_distance += reward_for_being_in_target
                else:
                    # Calculate and accumulate distances for pieces not yet in the target camp
                    distances_to_camp = [abs(x - i) + abs(y - j) for x, y in target_camp]
                    min_distance = min(distances_to_camp)
                    total_distance += min_distance
                count += 1

    return -total_distance / count if count > 0 else float('inf')


def group_movement_reward(board, player):
    size = len(board)
    grouped_score = 0
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Adjacent positions: up, down, left, right

    for i in range(size):
        for j in range(size):
            if board[i][j] == player:
                # Check for adjacent allied pieces to increase group strength score
                for di, dj in directions:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < size and 0 <= nj < size and board[ni][nj] == player:
                        grouped_score += 1  # Increment for each adjacent ally

    return grouped_score


def border_safety_index(board, player):
    size = len(board)
    border_risk_penalty = 0

    # Check only the outermost rows and columns for player pieces
    for i in range(size):
        for j in [0, size - 1]:  # Check first and last column
            if board[i][j] == player:
                border_risk_penalty += 1
        for j in range(1, size - 1):
            for i in [0, size - 1]:  # Check first and last row
                if board[i][j] == player:
                    border_risk_penalty += 1

    return -border_risk_penalty  # Negative because it's a penalty


def cluster_heuristic(board, player):
    group_reward = group_movement_reward(board, player)
    border_penalty = border_safety_index(board, player)
    return group_reward + border_penalty


def path_clearance_heuristic(board, player):
    size = len(board)
    path_clearance_score = 0
    target_camp = get_opposing_camp(player)
    for i in range(size):
        for j in range(size):
            if board[i][j] == player:
                distances = [abs(x - i) + abs(y - j) for x, y in target_camp]
                min_distance = min(distances)
                obstacles = count_obstacles(board, i, j, target_camp, min_distance)
                # Subtract the number of obstacles from the path clearance score
                path_clearance_score -= obstacles

    return path_clearance_score


def count_obstacles(board, start_i, start_j, target_camp, min_distance):
    obstacles = 0
    size = len(board)
    direction_i = 1 if target_camp[0][0] > start_i else -1
    direction_j = 1 if target_camp[0][1] > start_j else -1

    for step in range(1, min_distance + 1):
        checking_i = start_i + step * direction_i
        checking_j = start_j + step * direction_j
        if 0 <= checking_i < size and 0 <= checking_j < size:
            if board[checking_i][checking_j] != 0:
                obstacles += 1
        else:
            break

    return obstacles


def evaluate(board, current_player, recent_moves):
    move_penalty = sum(1 for move in recent_moves if move in recent_moves) * 100  # Penalize repeated moves
    # Combine heuristics with the penalty
    heuristic_value = distance_heuristic(board, current_player) - move_penalty
    winner = check_win(board)
    if winner == 1:
        return float('inf')  # player 1 wins
    elif winner == 2:
        return float('-inf')  # player 2 wins
    else:
        return heuristic_value


def minimax(board, depth, alpha, beta, player, size, recent_moves):
    winner = check_win(board)
    if depth == 0 or winner != 0:
        return evaluate(board, player, recent_moves), []

    if player == 1:
        max_eval = float('-inf')
        best_moves = []
        moves = generate_possible_moves(player, size, board)
        random.shuffle(moves)
        for start, end in moves:
            temp_board = [row[:] for row in board]
            apply_move(temp_board, start, end)
            evaluation, _ = minimax(temp_board, depth - 1, alpha, beta, 2, size, recent_moves)
            if evaluation > max_eval:
                max_eval = evaluation
                best_moves = [(start, end)]
            elif evaluation == max_eval:
                best_moves.append((start, end))
            alpha = max(alpha, evaluation)
            if beta <= alpha:
                break
        return max_eval, best_moves
    else:
        min_eval = float('inf')
        best_moves = []
        moves = generate_possible_moves(player, size, board)
        for start, end in moves:
            temp_board = [row[:] for row in board]
            apply_move(temp_board, start, end)
            evaluation, _ = minimax(temp_board, depth - 1, alpha, beta, 1, size, recent_moves)
            if evaluation < min_eval:
                min_eval = evaluation
                best_moves = [(start, end)]
            elif evaluation == min_eval:
                best_moves.append((start, end))
            beta = min(beta, evaluation)
            if alpha >= beta:
                break
        return min_eval, best_moves


class Halma:
    def __init__(self, num_players=2):
        self.size = 16
        self.board = [[0] * self.size for _ in range(self.size)]
        self.players = num_players
        self.current_player = 1
        self.setup_players()
        self.winner = 0
        self.round = 0
        self.recent_moves = []
        self.max_moves_repetition = 4

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

    def move(self, start, end):
        valid = is_valid_move(self.board, start, end, self.current_player)
        if not valid:
            raise ValueError(f"Invalid move {start} -> {end}")
        if len(self.recent_moves) >= 4:
            self.recent_moves.pop(0)  # Remove the oldest move
        self.recent_moves.append((start, end))
        x1, y1 = start
        x2, y2 = end
        self.board[x2][y2] = self.board[x1][y1]
        self.board[x1][y1] = 0
        self.current_player = 3 - self.current_player
        self.winner = check_win(self.board)
        self.round += 1
        return True

    def play_turn(self):
        if self.winner != 0:
            print(f"Game over! Player {self.winner} has won.")
            return

        if self.current_player == 1:
            move_score, best_moves = minimax(self.board, 4, float('-inf'), float('inf'), self.current_player, self.size,
                                             self.recent_moves)
            best_move = random.choice(best_moves)
            self.move(best_move[0], best_move[1])
            print(f"Best move: {best_move} with evaluation: {move_score}")
        else:
            moves = generate_possible_moves(self.current_player, self.size, self.board)
            if not moves:
                print(f"Player {self.current_player} has no moves left.")
                return
            move = random.choice(moves)
            self.move(move[0], move[1])
        self.print_board()

    def run(self):
        while self.winner == 0:
            self.play_turn()
        print(f"Game over! Player {self.winner} has won.")

    def setup_almost_winning(self):
        self.board = [[0] * self.size for _ in range(self.size)]

        # Set almost all of Player 1's pieces in Player 2's camp
        p1_almost_winning = [(15, 15), (15, 14), (15, 13), (15, 12), (14, 15),
                             (14, 14), (14, 13), (14, 12), (13, 15), (13, 14),
                             (13, 13), (12, 15), (12, 14), (11, 15)]

        # Place remaining Player 1's pieces near Player 2's camp
        p1_remaining = [(15, 11), (15, 10), (15, 9), (14, 11), (13, 12)]

        # Set almost all of Player 2's pieces in Player 1's camp
        p2_almost_winning = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0),
                             (1, 1), (1, 2), (1, 3), (2, 0), (2, 1),
                             (2, 2), (3, 0), (3, 1), (4, 0)]

        # Place remaining Player 2's pieces near Player 1's camp
        p2_remaining = [(0, 4), (0, 5), (0, 6), (1, 4), (2, 3)]

        # Place almost winning positions on the board for Player 1
        for x, y in p1_almost_winning + p1_remaining:
            self.board[x][y] = 1

        # Place almost winning positions on the board for Player 2
        for x, y in p2_almost_winning + p2_remaining:
            self.board[x][y] = 2


if __name__ == '__main__':
    game = Halma()
    game.setup_almost_winning()
    game.print_board()
    game.run()
