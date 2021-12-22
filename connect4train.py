import random

import numpy as np
from matplotlib import pyplot as plt

import connect4game as game
import connect4user_interface

INITIAL_COMPUTATION_TOKENS = 2000
EXAMPLE_CONTAINER_SIZE = 10000
BATCH_SIZE = 1024


class FullContainerException(Exception):
    """Raised when an attempt is made to place an example into the container
    when it is already full."""
    pass


class TrainingExampleContainer:
    def __init__(self, container_size=EXAMPLE_CONTAINER_SIZE):
        self.container_size = container_size
        self.empty_index_stack = [i for i in range(self.container_size)]
        self.empty_index_set = set(self.empty_index_stack)
        self.training_examples = [0 for _ in range(self.container_size)]

    def place(self, training_example):
        if self.is_full():
            raise FullContainerException
        else:
            empty_index = self.empty_index_stack.pop()
            self.empty_index_set.remove(empty_index)
            self.training_examples[empty_index] = training_example

    def retrieve(self):
        """Returns a randomly selected training example."""
        while True:
            random_index = random.randint(0, self.container_size-1)
            if random_index not in self.empty_index_set:
                self.empty_index_stack.append(random_index)
                self.empty_index_set.add(random_index)
                return self.training_examples[random_index]

    def retrieve_batch(self, batch_size=BATCH_SIZE):
        """Returns a randomly selected batch of training examples."""
        batch = []
        for _ in range(batch_size):
            batch.append(self.retrieve())
        return batch

    def is_full(self):
        if self.empty_index_stack:
            return False
        else:
            return True

    def get_percent_full(self):
        proportion = (EXAMPLE_CONTAINER_SIZE -
                      len(self.empty_index_stack))/EXAMPLE_CONTAINER_SIZE
        return proportion*100

class TrainingClock:
    """Keeps track of how many 'computation tokens' each player has remaining
    To ensure training games are fair."""
    def __init__(self):
        self.current_player_tokens = INITIAL_COMPUTATION_TOKENS
        self.enemy_tokens = INITIAL_COMPUTATION_TOKENS

    def change_player(self, current_player_used_tokens):
        """Returns true if the current player runs out of tokens."""
        self.enemy_tokens = self.current_player_tokens
        self.current_player_tokens = \
            INITIAL_COMPUTATION_TOKENS - current_player_used_tokens
        if self.current_player_tokens < 0:
            return True
        else:
            return False


class TrainingGameState(game.GameState):
    """Type of GameState that does not have a User Interface for training."""
    def __init__(self, red, yellow, reds_move=True, board=None,
                 starting_move_list=None):
        if board is None:
            board = game.Board()
        if starting_move_list is None:
            starting_move_list = []
        super().__init__(red, yellow, reds_move, board, starting_move_list)
        self.clock = TrainingClock()

    def start_game_procedure(self):
        self.red.used_computation_tokens = 0
        self.yellow.used_computation_tokens = 0
        self.clock = TrainingClock()

    def won_game_procedure(self):
        pass

    def next_move_procedure(self):
        pass

    def change_clock_and_win_check(self):
        """Returns true if a player runs out of computation tokens."""
        if self.clock.change_player(
                self.get_current_player().used_computation_tokens):
            self.red_winner = self.reds_move
            self.is_drawn = False
            self.timeout = True
            return True
        else:
            return False

    def get_normalised_clocks(self):
        """Used by the TimedUCT AI to return tuple containing
        how much time each player has left on a normalised scale."""
        normalised_current_player_tokens = \
            self.clock.current_player_tokens/INITIAL_COMPUTATION_TOKENS
        normalised_enemy_player_tokens = \
            self.clock.current_player_tokens/INITIAL_COMPUTATION_TOKENS
        return normalised_current_player_tokens, normalised_enemy_player_tokens


class TrainingEnvironment:
    """Used to train AI."""
    def __init__(self, AI, epoch_count=0):
        self.AI = AI
        self.epoch_count = epoch_count
        self.training_example_container = TrainingExampleContainer()
        # loss_history stores average loss from each batch of training.
        self.loss_history = []

    def execute_episode(self, red, yellow):
        """Executes a game of self play and places examples in the container."""
        new_game = TrainingGameState(red, yellow)
        new_game.play()
        if new_game.timeout:
            self.timeout_counter += 1
        if new_game.is_drawn:
            red_reward = 0
        elif new_game.red_winner:
            red_reward = 1
        else:
            red_reward = -1
        self.assign_rewards_and_place(red.get_training_examples(), red_reward)
        self.assign_rewards_and_place(yellow.get_training_examples(), -red_reward)

    def assign_rewards_and_place(self, training_examples, reward):
        # If the container is already full, no more examples are placed into it.
        for training_example in training_examples:
            training_example.set_reward(reward)
            try:
                self.training_example_container.place(training_example)
            except FullContainerException:
                break

    def execute_epoch(self):
        """Saves the current AI, executes episodes until the container is full,
        and then trains the AI using a batch retrieved from the container."""
        self.timeout_counter = 0
        self.episode_counter = 0
        print("epoch: ", self.epoch_count)
        filepath = self.AI.name+"_"+str(self.epoch_count)+"_epochs"
        self.epoch_count += 1
        # saves the current AI
        self.AI.save_parameters(filepath)
        self.fill_container(filepath)
        self.AI.punish_for_timeouts(self.timeout_counter, self.episode_counter)
        metric_history = self.AI.train_on_batch(
            self.training_example_container.retrieve_batch())
        self.loss_history.append(metric_history.history["loss"])

    def fill_container(self, filepath):
        """Plays games until the container is full."""
        red, yellow = self.create_copies(filepath)
        i = 0
        print("filling container")
        while not self.training_example_container.is_full():
            # Prints how full the container is every fifth game.
            if i % 5 == 0:
                print(str(round(
                    self.training_example_container.get_percent_full(), 1)),
                    "percent full")
            i += 1
            self.episode_counter += 1
            self.execute_episode(red, yellow)
        print("games played: ", i)

    def create_copies(self, filepath):
        """Returns a tuple of copies of the current AI being trained."""
        red = type(self.AI)(self.AI.name, True)
        yellow = type(self.AI)(self.AI.name, True)
        red.load_parameters(filepath)
        yellow.load_parameters(filepath)
        return red, yellow

    def evaluate(red, yellow, num_games_per_colour=50):
        """Returns the percentage of non-drawn
         games won by the AI that is currently red"""
        draw_count = 0
        player_one_win_count = 0
        timeout_count = 0
        # Plays 50 games and logs the results.
        for _ in range(num_games_per_colour):
            new_game = TrainingGameState(red, yellow)
            new_game.play()
            if new_game.is_drawn:
                draw_count += 1
            elif new_game.red_winner:
                player_one_win_count += 1
            if new_game.timeout:
                timeout_count += 1
        # Switches the sides and then logs the results again.
        red, yellow = yellow, red
        for _ in range(num_games_per_colour):
            new_game = TrainingGameState(red, yellow)
            new_game.play()
            if new_game.is_drawn:
                draw_count += 1
            elif not new_game.red_winner:
                player_one_win_count += 1
            if new_game.timeout:
                timeout_count += 1
        print("red win count: ", player_one_win_count)
        print("draw count: ", draw_count)
        print("timeout count: ", timeout_count)
        try:
            red_win_percentage = \
                100 * player_one_win_count / (2 * num_games_per_colour - draw_count)
        except ZeroDivisionError:
            red_win_percentage = 50
        return red_win_percentage
