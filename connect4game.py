

class FullColumnException(Exception):
    """Raised when one attempts place a counter in an already a full column."""
    pass


class Board:
    """Stores a 6*7 2D array of integers representing the board,
    with 1 being a red counter, -1 being yellow and 0 for no counter."""
    def __init__(self):
        """initialises the board in a blank state by default"""
        self.array = [[0] * 7 for i in range(6)]

    def print(self):
        """Prints a representation of the board into the console."""
        for i in range(7):
            print(i + 1, end=" ")
        print()
        for row in self.array:
            for cell in row:
                if cell == 0:
                    character = "#"
                elif cell == 1:
                    character = "R"
                else:
                    character = "Y"
                print(character, end=" ")
            print()

    def column_is_full(self, column):
        """Determines whether a column is full."""
        if self.array[0][column] == 0:
            return False
        else:
            return True

    def place_counter(self, reds_move, column):
        """Places a in a given column,
         returns a tuple representing where it was placed."""
        # raises FullColumnException if the column is full
        if self.column_is_full(column):
            raise FullColumnException
        # otherwise, checks up from the bottom for a free cell
        for i in range(5, -1, -1):
            if self.array[i][column] == 0:
                new_counter_position = i, column
                if reds_move:
                    self.array[i][column] = 1
                else:
                    self.array[i][column] = -1
                return new_counter_position

    def win_check(self, reds_move, position):
        """Returns True if a player won with their last move."""
        if reds_move:
            counter_code = 1
        else:
            counter_code = -1
        # checks the row of the position for line of length for
        if self.check_full_line(counter_code, position, -1, 0):
            return True
        # checks the column of the position, which is only
        # necessary when the 'y' value is greater than or equal to 4
        if position[0] < 3:
            if self.check_line(counter_code, position, 1, 0, 1) == 4:
                return True
        # checks the diagonals, in some cases where the counter is close
        # a corner of the board, checking one of the diagonals is unnecessary
        position_sum = position[0] + position[1]
        position_difference = position[0] - position[1]
        if 2 < position_sum < 9:
            if self.check_full_line(counter_code, position, -1, 1):
                return True
        if -4 < position_difference < 3:
            if self.check_full_line(counter_code, position, 1, 1):
                return True

    def check_full_line(self, counter_code, position,
                        column_index_iterator, row_index_iterator):
        """Used by the win_check method, returns True if there is a line of at least
         four like coloured counters either way in the specified direction."""
        full_line_length_counter = 1
        full_line_length_counter = self.check_line(
            counter_code, position, full_line_length_counter,
            column_index_iterator, row_index_iterator, )
        if full_line_length_counter == 4:
            return True
        # counts the length of the line in the opposite direction
        full_line_length_counter = self.check_line(
            counter_code, position, full_line_length_counter,
            -column_index_iterator, -row_index_iterator)
        if full_line_length_counter == 4:
            return True
        return False

    def check_line(self, counter_code, position, line_length_counter,
                   column_index_iterator, row_index_iterator):
        """Used by check full line method, returns the length of a line of
         like coloured counters from a point in one specified direction"""
        for i in range(1, 4):
            column_index = position[1] + i * column_index_iterator
            row_index = position[0] + i * row_index_iterator
            if -1 < column_index < 7:
                if -1 < row_index < 6:
                    if self.array[row_index][column_index] == counter_code:
                        line_length_counter += 1
                        if line_length_counter == 4:
                            return 4
                    else:
                        break
                else:
                    break
            else:
                break
        return line_length_counter


class GameState:
    """An abstract class which manages games in play."""
    def __init__(self, red, yellow, reds_move, board, starting_move_list):
        self.red = red
        self.yellow = yellow
        self.current_board = board
        self.move_list = starting_move_list
        self.reds_move = reds_move
        self.is_drawn = False

    def start_game_procedure(self):
        raise NotImplementedError("Please Implement this method")

    def won_game_procedure(self):
        raise NotImplementedError("Please Implement this method")

    def next_move_procedure(self):
        raise NotImplementedError("Please Implement this method")

    def change_clock_and_win_check(self):
        raise NotImplementedError("Please Implement this method")

    def make_move(self, move):
        """Places a counter in the given column on the board."""
        if self.place_counter_and_win_check(move):
            self.won_game_procedure()
        elif self.change_clock_and_win_check():
            self.won_game_procedure()
        else:
            self.next_move_procedure()
            current_player = self.get_current_player()
            current_player.play_chosen_move(self)

    def place_counter_and_win_check(self, move):
        self.move_list.append(move)
        try:
            new_move = self.current_board.place_counter(self.reds_move, move)
        except FullColumnException:
            self.red_winner = not self.reds_move
            self.is_drawn = False
            self.timeout = False
            return True
        else:
            if self.current_board.win_check(self.reds_move, new_move):
                self.red_winner = self.reds_move
                self.is_drawn = False
                self.timeout = False
                return True
            if len(self.move_list) >= 42:
                self.is_drawn = True
                self.timeout = False
                return True
        self.reds_move = not self.reds_move
        return False

    def play(self):
        self.start_game_procedure()
        if self.reds_move:
            self.red.play_chosen_move(self)
        else:
            self.yellow.play_chosen_move(self)

    def get_current_player(self):
        if self.reds_move:
            return self.red
        else:
            return self.yellow

    def get_enemy_player(self):
        if self.reds_move:
            return self.yellow
        else:
            return self.red

    def get_possible_moves(self):
        """Returns a list of integers representing
         which columns can be played in."""
        possible_moves = []
        for column in range(7):
            if not self.current_board.column_is_full(column):
                possible_moves.append(column)
        return possible_moves

    def get_normalised_clocks(self):
        """Abstract method used by the TimedUCT AI to return tuple containing
        how much time each player has left on a normalised scale."""
        raise NotImplementedError("Please Implement this method")


