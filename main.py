from halma import Halma

if __name__ == '__main__':
    game = Halma()
    game.print_board()
    game.move((0, 1), (2, 3))
    game.print_board()
