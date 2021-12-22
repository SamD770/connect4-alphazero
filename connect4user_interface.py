import time
import os
import math
import tkinter as tk
from copy import deepcopy

import connect4AI as AI
import connect4game as game

DEFAULT_PLAYER_SECONDS = 300
HUMAN_PLAYER_STRING = "human player"
RESNET_STRING = "flat residual network"
MINIMAX_STRING = "MiniMax"
AZ_CLONE_STRING = "Alpha Zero clone"
TIMED_UCT_STRING = "Timed UCT"
AI_TYPE_STRINGS = (HUMAN_PLAYER_STRING, MINIMAX_STRING, RESNET_STRING,
                   AZ_CLONE_STRING, TIMED_UCT_STRING)


def AI_type_dict(string):
    return{
        HUMAN_PLAYER_STRING: PlayingGUI.GUIHumanPlayer,
        MINIMAX_STRING: AI.MiniMax,
        RESNET_STRING: AI.ResidualNN,
        AZ_CLONE_STRING: AI.AZclone,
        TIMED_UCT_STRING: AI.TimedUCT
    }[string]


def AI_default_time_dict(AI_type):
    return{
        PlayingGUI.GUIHumanPlayer: 300,
        AI.MiniMax: 20,
        AI.ResidualNN: 10,
        AI.AZclone: 20,
        AI.TimedUCT: 20
    }[AI_type]


class PlayingClock:
    """Stores how much time each player has left."""
    def __init__(self, player_seconds):
        """ Player seconds is a tuple saying how much time each player has."""
        self.current_player_seconds = player_seconds[0]
        self.enemy_player_seconds = player_seconds[1]
        # the start method needs to be applied before the clock can be used
        self.start_time = None

    def start(self):
        """Sets the clock's start time to the current time"""
        self.start_time = time.time()

    def change_player(self):
        """Takes away the correct amount of time from the current player
        and switches the current and other player."""
        new_start_time = time.time()
        self.current_player_seconds = self.current_player_seconds\
                                      + self.start_time - new_start_time
        self.start_time = new_start_time
        self.enemy_player_seconds, self.current_player_seconds = \
            self.current_player_seconds, self.enemy_player_seconds


class PlayingGameState(game.GameState):
    """Game state used when playing games with a user interface."""
    def __init__(self, red, yellow, parent_UI, reds_move=True, board=None,
                 starting_move_list=None, player_seconds=None):
        # the default values for board and starting move list are for a new game
        if board is None:
            board = game.Board()
        if starting_move_list is None:
            starting_move_list = []
        super().__init__(red, yellow, reds_move, board, starting_move_list)
        self.parent_UI = parent_UI
        # if no value is given for player seconds, the game is not timed
        if player_seconds is None:
            self.is_timed = False
        else:
            self.is_timed = True
            self.clock = PlayingClock(player_seconds)

    def start_game_procedure(self):
        """Overridden from game module."""
        if self.is_timed:
            self.clock.start()
        self.parent_UI.display_state(self)

    def won_game_procedure(self):
        """Overridden from game module."""
        self.parent_UI.display_winner(self)

    def next_move_procedure(self):
        """Overridden from game module."""
        self.parent_UI.display_state(self)


    def change_clock_and_win_check(self):
        """Returns True if the current player runs out of time."""
        if self.is_timed:
            self.clock.change_player()
            if self.clock.enemy_player_seconds < 0:
                self.red_winner = self.reds_move
                self.is_drawn = False
                self.timeout = True
                return True
            else:
                return False
        else:
            return False

    def get_normalised_clocks(self):
        """Returns a tuple containing how much time each player has divided by
        their default amount."""
        current_player_type = type(self.get_current_player())
        default_current_player_seconds = AI_default_time_dict(current_player_type)
        enemy_player_type = type(self.get_enemy_player())
        default_enemy_player_seconds = AI_default_time_dict(enemy_player_type)
        normalised_current_player_clock = \
            self.clock.current_player_seconds / default_current_player_seconds
        normalised_enemy_player_clock = \
            self.clock.enemy_player_seconds / default_enemy_player_seconds
        print("normalised clocks: ", normalised_current_player_clock, normalised_enemy_player_clock)
        return normalised_current_player_clock, normalised_enemy_player_clock


class ConnectFourUserInterface:
    """abstract interface for different types of UI"""
    def display_state(self, game_state):
        """Abstract method that is called after a move is made."""
        raise NotImplementedError("Please Implement this method")

    def display_winner(self, game_state):
        raise NotImplementedError("Please Implement this method")


class CommandLineUI(ConnectFourUserInterface):
    """abstract interface for different types of UI"""
    class CommandLineHumanPlayer(AI.ConnectFourEntity):
        """For a human player to play."""
        def choose_move(self, game_state):
            """prompts the player to enter an integer between 1 and 7"""
            invalid_input = True
            while invalid_input:
                player_input = input("Select a column")
                if player_input.isdigit():
                    player_input = int(player_input)
                    if player_input in range(1, 8):
                        if game_state.current_board.column_is_full(player_input - 1):
                            print("that column is full")
                        else:
                            invalid_input = False
                    else:
                        print("please enter a number between 1 and 7")
                else:
                    print("the imput need to be an integer")
            return player_input-1

    def display_state(self, game_state):
        """Abstract method that is called after a move is made."""
        current_player_name = game_state.get_current_player().name
        enemy_player_name = game_state.get_enemy_player().name
        #at this time, the current and enemy players have been switched on the clock
        print("{0}'s move, \n"
              "{0} has {1} seconds left and {2} has {3} seconds left"
              .format(current_player_name, game_state.clock.current_player_seconds,
                      enemy_player_name, game_state.clock.enemy_player_seconds))
        game_state.current_board.print()

    def display_winner(self, game_state):
        if game_state.is_drawn:
            print("draw!")
        else:
            if game_state.timeout:
                print("timeout!")
            if game_state.red_winner:
                print(game_state.red.name, " wins!")
            else:
                print(game_state.yellow.name, " wins!")
        game_state.current_board.print()


class GUIwidget:
    """Interface used so all the widgets can be updated in a loop."""
    def display_state(self, game_state):
        """Abstract method which displays information about the current game."""
        raise NotImplementedError("Please Implement this method")

    def transition_to_playing(self):
        """Abstract method which changes the widget to the 'playing' state."""
        raise NotImplementedError("Please Implement this method")

    def transition_to_paused(self):
        """Abstract method which changes the widget to the 'paused' state."""
        raise NotImplementedError("Please Implement this method")

    def display_winner(self, game_state):
        """Abstract method which changes the widget to the 'won game' state."""
        raise NotImplementedError("Please Implement this method")


class NonResizableButton(tk.Frame):
    """Used by MoveButton and PausePlayReset button to create a button which
    is not automatically re-sized by tkinter."""
    def __init__(self, master, width, height, command, text=""):
        tk.Frame.__init__(self, master, width=width, height=height, bg="white")
        self.grid_propagate(False)
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        self.button = tk.Button(self, command=command, text=text)
        self.show()

    def show(self):
        self.button.grid(column=0, row=0, sticky="nesw")

    def hide(self):
        self.button.grid_remove()


class TransitioningWidget(GUIwidget, tk.Frame):
    """Abstract class for widgets which become labels in the 'playing' state"""
    def __init__(self, master, *options, **kwargs):
        """Options is a tuple of strings and kwargs is a dictionary of tkinter
        settings for a given widget."""
        tk.Frame.__init__(self, master)
        self.child_text = tk.StringVar(value=options[0])
        self.child_widget = self.get_child_widget(*options, **kwargs)
        # use the default font unless it is passed as a keyword argument
        self.child_label = tk.Label(self, text=options[0],
                                    font=kwargs.get("font", None))

    def display_winner(self, game_state):
        self.transition_to_playing()

    def transition_to_playing(self):
        # Replaces the widget with the label and updates the label text.
        self.child_widget.grid_remove()
        self.child_label.configure(text=self.child_text.get())
        self.child_label.grid()

    def transition_to_paused(self):
        # Replaces the label with the widget.
        self.child_label.grid_remove()
        self.child_widget.grid()

    def get_child_widget(self, options, command, font):
        """Abstract method which gets the widget which is being used."""
        raise NotImplementedError

    def display_state(self, game_state):
        pass


class TransitioningDropdown(TransitioningWidget):
    def get_child_widget(self, *options, **kwargs):
        return tk.OptionMenu(self, self.child_text, *options, **kwargs)


class TransitioningEntry(TransitioningWidget):
    def get_child_widget(self, *options, **kwargs):
        return tk.Entry(self, textvariable=self.child_text, **kwargs)


class TransitioningPlayerTimer(TransitioningEntry):
    """Displays how much thinking time one player has left."""
    def __init__(self, master, is_reds_clock, *options):
        # validation is used to ensure the entry contents are numeric
        validate_command = (master.register(self.validate_user_entry), "%S")
        super().__init__(master, *options, font=("Times", 35, "bold"),
                         validate="key", validatecommand=validate_command, width=5)
        self.is_reds_clock = is_reds_clock

    def display_state(self, game_state):
        # if in the 'playing' state, the player's time left updated
        if game_state.is_timed:
            if game_state.reds_move == self.is_reds_clock:
                player_seconds = game_state.clock.current_player_seconds
                background_colour = "green"
            else:
                player_seconds = game_state.clock.enemy_player_seconds
                background_colour = "grey"
            self.child_label.configure(
                text=str(round(player_seconds, 1)), bg=background_colour)

    def validate_user_entry(self, new_character):
        """Returns true if the new contents could be converted to a float."""
        try:
            float(self.child_text.get() + new_character)
        except ValueError:
            return False
        else:
            return True

    def transition_to_paused(self):
        # changes the default text in the entry to how much time is left.
        super().transition_to_paused()
        self.child_text.set(self.child_label.cget('text'))

    def display_winner(self, game_state):
        # changes te background to red if this timer ran out.
        if game_state.timeout:
            if game_state.red_winner != self.is_reds_clock:
                self.child_label.configure(text="0.0", bg="red")


class PlayingGUI(ConnectFourUserInterface):
    """Uses the tkinter library to create a GUI for playing connect4."""
    def __init__(self, master):
        master.title("connect4")
        # widgets is an array of TransitioningWidgets to be looped over
        self.widgets = []
        self.setup_base_frames(master)
        self.board_canvas = PlayingGUI.BoardCanvas(master)
        self.widgets.append(self.board_canvas)
        self.move_buttons = []
        self.setup_move_button_array()
        self.pause_play_reset_button = PlayingGUI.PausePlayResetButton(self)
        self.widgets.append(self.pause_play_reset_button)
        self.game_info_tab = PlayingGUI.GameInfoTab(self)
        self.widgets.append(self.game_info_tab)
        self.timer = PlayingGUI.GUItimer(self)
        self.widgets.append(self.timer)
        self.player_info_tab = PlayingGUI.PlayerInfoTab(self)
        self.widgets.append(self.player_info_tab)
        # sets up the board with a new game and in the paused state
        self.reset()

    def copy_game(self, red, yellow, timed_game):
        """Used by pause and play to create a copy of current_game"""
        board_copy = deepcopy(self.current_game.current_board)
        move_list_copy = deepcopy(self.current_game.move_list)
        if timed_game:
            self.current_game = PlayingGameState(
                red, yellow, self, self.current_game.reds_move,
                board_copy, move_list_copy,
                self.timer.get_player_seconds(self.current_game.reds_move))
        else:
            self.current_game = PlayingGameState(
                red, yellow, self, self.current_game.reds_move,
                board_copy, move_list_copy)

    def display_state(self, game_state):
        for widget in self.widgets:
            widget.display_state(game_state)

    def display_winner(self, game_state):
        for widget in self.widgets:
            widget.display_winner(game_state)

    def pause(self):
        """Stops the timer and allows for the editing of the current game."""
        # copies the current_game,
        # but makes both players human so that the board can be edited
        red = PlayingGUI.GUIHumanPlayer("red", self)
        yellow = PlayingGUI.GUIHumanPlayer("yellow", self)
        self.copy_game(red, yellow, False)
        for widget in self.widgets:
            widget.transition_to_paused()
        self.current_game.play()

    def play(self):
        """Plays the game from the current game displayed on the GUI."""
        red = self.player_info_tab.get_AI(True)
        yellow = self.player_info_tab.get_AI(False)
        self.copy_game(red, yellow, True)
        for widget in self.widgets:
            widget.transition_to_playing()
        self.current_game.play()

    def reset(self):
        """Clears the board and transitions the GUI to a 'paused' state."""
        dummy = AI.Dummy()
        self.current_game = PlayingGameState(dummy, dummy, self)
        self.pause()
        self.timer.reset_timer()
        self.display_state(self.current_game)

    def show_move_buttons(self, game_state):
        for move in game_state.get_possible_moves():
            self.move_buttons[move].show()

    def hide_move_buttons(self):
        for button in self.move_buttons:
            button.hide()

    def create_move_button_command(self, move):
        """Hides the move buttons and makes the chosen move on the board"""
        self.hide_move_buttons()
        self.current_game.make_move(move)

    def setup_move_button_array(self):
        """Creates an array of buttons for human players to make moves with."""
        for i in range(7):
            self.move_buttons.append(PlayingGUI.MoveButton(self, i))
            self.move_buttons[i].grid(column=i, row=0, padx=10, pady=10)
        self.hide_move_buttons()

    def setup_base_frames(self, master):
        """Places three frames on the root frame to fill them with widgets."""
        master.resizable(False, False)
        self.master_frame = tk.Frame(master)
        self.master_frame.grid()
        self.above_board_frame = tk.Frame(
            master, width=700, height=100, bg="white")
        self.right_hand_frame = tk.Frame(
            master, width=250, height=700, bg="white")
        self.above_board_frame.grid_propagate(False)
        self.above_board_frame.grid(column=0, row=0)
        self.right_hand_frame.grid_propagate(False)
        self.right_hand_frame.grid(column=1, row=0, rowspan=2)

    class GUIHumanPlayer(AI.ConnectFourEntity):
        """For a human player to play using the move buttons."""

        def __init__(self, name, parent_GUI):
            super().__init__(name)
            self.parent_GUI = parent_GUI

        def play_chosen_move(self, game_state):
            self.parent_GUI.show_move_buttons(game_state)

    class BoardCanvas(tk.Canvas, GUIwidget):
        """Displays the board in it's current state"""

        def __init__(self, root):
            super().__init__(root, width=700, height=600, bg="dark blue")
            # makes a 2D array of ovals of width 80 and a spacing of 20.
            self.oval_array = [[self.create_oval((10 + 100 * i, 10 + 100 * j,
                                                  100 * i + 90, 100 * j + 90),
                                                 fill="white")
                                for i in range(7)] for j in range(6)]
            self.grid(column=0, row=1)

        def display_state(self, game_state):
            for i in range(6):
                for j in range(7):
                    self.update_canvas_location(
                        (i, j), game_state.current_board.array[i][j])

        def update_canvas_location(self, position, counter_code):
            """Updates a specific position on the canvas with a counter code."""
            if counter_code == 0:
                self.itemconfig(self.oval_array[position[0]][position[1]],
                                fill="white")
            elif counter_code == 1:
                self.itemconfig(self.oval_array[position[0]][position[1]],
                                fill="red")
            else:
                self.itemconfig(self.oval_array[position[0]][position[1]],
                                fill="yellow")

            def transition_to_playing(self):
                pass

            def transition_to_paused(self):
                pass

            def display_winner(self, game_state):
                if not game_state.timeout:
                    self.display_state(game_state)

        class MoveButton(NonResizableButton):
            """Enables a human player to place counters on the board"""

            def __init__(self, parent_GUI, i):
                command = lambda move=i: parent_GUI.create_move_button_command(move)
                super().__init__(parent_GUI.above_board_frame, 80, 80, command, "")

        class PausePlayResetButton(NonResizableButton, GUIwidget):
            """Enables the player to transition the GUI state."""

            def __init__(self, parent_GUI):
                self.play_command = lambda: parent_GUI.play()
                self.pause_command = lambda: parent_GUI.pause()
                self.reset_command = lambda: parent_GUI.reset()
                super().__init__(parent_GUI.right_hand_frame, 230, 80,
                                 self.play_command, text="play from here")
                self.grid(row=0, column=0, padx=10, pady=10)

            def display_state(self, game_state):
                pass

            def transition_to_playing(self):
                self.button.configure(text="pause", command=self.pause_command)

            def transition_to_paused(self):
                self.button.configure(text="play from here", command=self.play_command)

            def display_winner(self, game_state):
                self.button.configure(text="reset", command=self.reset_command)

        class GUItimer(GUIwidget):
            """Displays or enables editing of how much time each player has left."""

            def __init__(self, parent_GUI):
                # creates a frame to place the timers in
                self.timer_frame = tk.Frame(parent_GUI.right_hand_frame,
                                            width=230, height=80, bg="white")
                self.timer_frame.grid(row=1, column=0, padx=10, pady=10)
                self.timer_frame.grid_propagate(False)
                self.red_title_label = tk.Label(self.timer_frame,
                                                text="red time:")
                self.yellow_title_label = tk.Label(self.timer_frame,
                                                   text="yellow time:")
                # Initialises TransitioningPlayerTimer objects for each player
                self.red_timer = TransitioningPlayerTimer(
                    self.timer_frame, True, str(DEFAULT_PLAYER_SECONDS))
                self.yellow_timer = TransitioningPlayerTimer(
                    self.timer_frame, False, str(DEFAULT_PLAYER_SECONDS))
                self.red_title_label.grid(row=0, column=0)
                self.yellow_title_label.grid(row=0, column=1)
                self.red_timer.grid(row=1, column=0, sticky="e")
                self.yellow_timer.grid(row=1, column=1, sticky="w")

            def display_state(self, game_state):
                self.red_timer.display_state(game_state)
                self.yellow_timer.display_state(game_state)

            def transition_to_playing(self):
                self.red_timer.transition_to_playing()
                self.yellow_timer.transition_to_playing()

            def transition_to_paused(self):
                self.red_timer.transition_to_paused()
                self.yellow_timer.transition_to_paused()

            def display_winner(self, game_state):
                self.red_timer.display_winner(game_state)
                self.yellow_timer.display_winner(game_state)

            def get_player_seconds(self, reds_move):
                """Returns a tuple containing how much time each player has."""
                if reds_move:
                    current_player_timer = self.red_timer
                    enemy_player_timer = self.yellow_timer
                else:
                    enemy_player_timer = self.red_timer
                    current_player_timer = self.yellow_timer
                current_player_seconds = float(current_player_timer.child_text.get())
                enemy_player_seconds = float(enemy_player_timer.child_text.get())
                return current_player_seconds, enemy_player_seconds

            def reset_timer(self):
                self.red_timer.child_text.set(str(DEFAULT_PLAYER_SECONDS))
                self.yellow_timer.child_text.set(str(DEFAULT_PLAYER_SECONDS))

    class GameInfoTab(GUIwidget):
        """Displays whose move it is and the list of previous moves"""
        def __init__(self, parent_GUI):
            self.game_info_frame = tk.Frame(parent_GUI.right_hand_frame,
                                            width=230, height=230)
            self.game_info_frame.grid_configure()
            self.game_info_frame.grid(row=2, column=0, padx=10, pady=10)
            self.game_info_frame.grid_propagate(False)
            self.player_move_label = tk.Label(self.game_info_frame, text="")
            # creates arrays of labels to show the list of previous moves
            self.red_move_labels = [
                tk.Label(self.game_info_frame, text="  ") for _ in range(21)]
            self.yellow_move_labels = [
                tk.Label(self.game_info_frame, text="  ") for _ in range(21)]
            self.player_move_label.grid(row=0, columnspan=6)
            # places the move labels in a grid
            for i in range(7):
                for j in range(3):
                    self.red_move_labels[i+j*7].grid(row=i+1, column=j,
                                                     padx=13, pady=4)
                    self.yellow_move_labels[i+j*7].grid(row=i+1, column=j+3,
                                                        padx=13, pady=4)

        def display_state(self, game_state):
            # updates whose move it is and the
            if game_state.reds_move:
                current_player_name = game_state.red.name
                text_colour = "red"
            else:
                current_player_name = game_state.yellow.name
                text_colour = "yellow"
            current_player_name += "'s move"
            self.player_move_label.configure(text=current_player_name,
                                             fg=text_colour)
            self.update_move_labels(game_state.move_list)

        def transition_to_playing(self):
            pass

        def transition_to_paused(self):
            pass

        def display_winner(self, game_state):
            if game_state.timeout:
                text_colour = "black"
                message = "timeout!"
            elif game_state.is_drawn:
                text_colour = "black"
                message = "draw!"
            elif game_state.red_winner:
                text_colour = "red"
                message = "red wins!"
            else:
                text_colour = "yellow"
                message = "yellow wins!"
            self.player_move_label.configure(text=message, fg=text_colour)
            if not game_state.timeout:
                self.update_move_labels(game_state.move_list)

        def update_move_labels(self, move_list):
            """Prints the move list into the grid of labels"""

            def get_move_label(counter):
                if counter % 2 == 0:
                    return self.red_move_labels[int(counter / 2)]
                else:
                    return self.yellow_move_labels[int((counter - 1) / 2)]

            for i, move in enumerate(move_list):
                get_move_label(i).configure(text=str(move + 1))
            for i in range(len(move_list), 42):
                get_move_label(i).configure(text="  ")

        class PlayerInfoTab(GUIwidget):
            """Lets user change player types when the game in a 'paused' state."""

            def __init__(self, parent_GUI):
                self.transitioning_widgets = []
                self.parent_GUI = parent_GUI
                self.player_info_frame = tk.Frame(parent_GUI.right_hand_frame,
                                                  width=230, height=220)
                self.player_info_frame.grid(row=3, column=0, padx=10, pady=10)
                self.player_info_frame.grid_propagate(False)

                # creates dropdown menus which show a load parameter entry box if
                # the AI selected is trainable
                def red_type_dropdown_command(value):
                    self.show_if_trainable_AI(True, value)

                self.red_type_dropdown = TransitioningDropdown(
                    self.player_info_frame, *AI_TYPE_STRINGS,
                    command=red_type_dropdown_command, )
                self.transitioning_widgets.append(self.red_type_dropdown)

                def yellow_type_dropdown_command(value):
                    self.show_if_trainable_AI(False, value)

                self.yellow_type_dropdown = TransitioningDropdown(
                    self.player_info_frame, *AI_TYPE_STRINGS,
                    command=yellow_type_dropdown_command, )
                # Creates widgets for a filepath to a model's weights be entered.
                self.transitioning_widgets.append(self.yellow_type_dropdown)
                self.red_weights_entry = TransitioningEntry(
                    self.player_info_frame, "")
                self.transitioning_widgets.append(self.red_weights_entry)
                self.yellow_weights_entry = TransitioningEntry(
                    self.player_info_frame, "")
                # Creates widgets for players' names to be entered.
                self.transitioning_widgets.append(self.yellow_weights_entry)
                self.red_name_entry = TransitioningEntry(
                    self.player_info_frame, "red")
                self.transitioning_widgets.append(self.red_name_entry)
                self.yellow_name_entry = TransitioningEntry(
                    self.player_info_frame, "yellow")
                self.transitioning_widgets.append(self.yellow_name_entry)
                self.setup_labels()
                self.place_widgets()

            def show_if_trainable_AI(self, red_AI, string):
                """Makes the weights entry widget show if the AI is trainable."""
                if red_AI:
                    parameter_filepath_label = self.red_weights_filepath_label
                    parameter_entry = self.red_weights_entry
                    row = 4
                else:
                    parameter_filepath_label = self.yellow_weights_filepath_label
                    parameter_entry = self.yellow_weights_entry
                    row = 9
                if AI_type_dict(string).is_trainable:
                    parameter_filepath_label.configure(text="weights filepath:")
                    parameter_entry.grid(row=row, column=1, sticky="w")
                else:
                    parameter_filepath_label.configure(text="                      ")
                    parameter_entry.grid_remove()

            def setup_labels(self):
                self.title_label = tk.Label(self.player_info_frame,
                                            text="player information")
                self.red_label = tk.Label(self.player_info_frame, text="red")
                self.yellow_label = tk.Label(self.player_info_frame, text="yellow")
                self.name_label_1 = tk.Label(self.player_info_frame, text="name:")
                self.name_label_2 = tk.Label(self.player_info_frame, text="name:")
                self.type_label_1 = tk.Label(self.player_info_frame, text="type:")
                self.type_label_2 = tk.Label(self.player_info_frame, text="type:")
                self.red_weights_filepath_label = tk.Label(
                    self.player_info_frame, text="                      ")
                self.yellow_weights_filepath_label = tk.Label(
                    self.player_info_frame, text="                      ")

            def place_widgets(self):
                self.title_label.grid(row=0, columnspan=2)
                self.red_label.grid(row=1, column=0, sticky="w")
                self.name_label_1.grid(row=2, column=0, sticky="e")
                self.red_name_entry.grid(row=2, column=1, sticky="w")
                self.type_label_1.grid(row=3, column=0, sticky="e")
                self.red_type_dropdown.grid(row=3, column=1, sticky="w")
                self.red_weights_filepath_label.grid(row=4, column=0, sticky="e")
                self.player_info_frame.grid_rowconfigure(5, weight=10)
                self.yellow_label.grid(row=6, column=0, sticky="w")
                self.name_label_2.grid(row=7, column=0, sticky="e")
                self.yellow_name_entry.grid(row=7, column=1, sticky="w")
                self.type_label_2.grid(row=8, column=0, sticky="e")
                self.yellow_type_dropdown.grid(row=8, column=1, sticky="w")
                self.yellow_weights_filepath_label.grid(row=9, column=0, sticky="e")

            def get_AI(self, red_AI):
                """Returns an AI generated form the user's entry in the tab"""
                # Gets the correct widgets and variables for if the requested
                # AI is the red player or the yellow player.
                if red_AI:
                    AI_type_string = self.red_type_dropdown.child_text.get()
                    name = self.red_name_entry.child_text.get()
                    weights_entry = self.red_weights_entry
                else:
                    AI_type_string = self.yellow_type_dropdown.child_text.get()
                    name = self.yellow_name_entry.child_text.get()
                    weights_entry = self.yellow_weights_entry
                # Gets the AI type from the string retrieved from the dropdown menu.
                AI_type = AI_type_dict(AI_type_string)
                # If the AI type is trainable, an attempt is made to load the model
                # in from the filepath entered, otherwise random weights are used.
                if AI_type.is_trainable:
                    AI = AI_type(name, False)
                    weights_filepath = weights_entry.child_text.get()
                    if os.path.isfile(weights_filepath):
                        AI.load_parameters(weights_filepath)
                    else:
                        weights_entry.child_text.set("no weights found.")
                    return AI
                elif AI_type == PlayingGUI.GUIHumanPlayer:
                    return AI_type(name, self.parent_GUI)
                else:
                    return AI_type(name)

            def display_state(self, game_state):
                pass

            def transition_to_playing(self):
                for widget in self.transitioning_widgets:
                    widget.transition_to_playing()

            def transition_to_paused(self):
                for widget in self.transitioning_widgets:
                    widget.transition_to_paused()

            def display_winner(self, game_state):
                self.transition_to_playing()


if __name__ == "__main__":
    # If the user interface module is explicitly run, the GUI is opened.
    root = tk.Tk()
    PlayingGUI(root)
    root.mainloop()

