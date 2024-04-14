class Halma:
    def __init__(self, num_players=2):
        self.size = 16
        self.board = [[0] * self.size for _ in range(self.size)]
        self.setup_players(num_players)
        self.current_player = 1
        self.players = num_players
        self.winner = 0

    def setup_players(self, num_players):
        if num_players == 2:
            for i in range(5):
                for j in range(5 - i):
                    self.board[i][j] = 1
                    self.board[self.size - 1 - i][self.size - 1 - j] = 2

    def print_board(self):
        for row in self.board:
            print(' '.join(str(x).rjust(2) for x in row))
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

        # no jump
        if abs(dx) <= 1 and abs(dy) <= 1:
            return True

        # jump
        if abs(dx) == 2 and abs(dy) in [0, 2] or abs(dy) == 2 and abs(dx) in [0, 2]:
            midx, midy = (x1 + x2) // 2, (y1 + y2) // 2
            # mid field is not empty
            if self.board[midx][midy] != 0:
                return True

        return False

    def move(self, start, end):
        if not self.is_valid_move(start, end):
            return False
        x1, y1 = start
        x2, y2 = end
        self.board[x2][y2] = self.board[x1][y1]
        self.board[x1][y1] = 0
        self.current_player = 3 - self.current_player
        self.winner = self.check_win()
        if self.winner != 0:
            print(f"Gracz {self.winner} wygrał!")
        return True

    def check_win(self):
        if all(self.board[i][j] == 1 for i in range(11, 16) for j in range(11, 16)):
            return 1

        if all(self.board[i][j] == 2 for i in range(5) for j in range(5)):
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
                        if self.size > ni >= 0 == self.board[ni][nj] and 0 <= nj < self.size:
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
            if self.size > ni >= 0 == self.board[ni][nj] and self.size > nj >= 0 != self.board[mi][mj]:
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
