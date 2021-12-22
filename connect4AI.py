import math
import random
import pickle
from copy import deepcopy

import numpy as np
import tensorflow as tf
from tensorflow import keras

import connect4game as game


class ConnectFourEntity:
    """Interface for different types of AI to play connect 4."""
    is_trainable = False

    def __init__(self, name):
        self.name = name

    def play_chosen_move(self, game_state):
        chosen_move = self.choose_move(game_state)
        game_state.make_move(chosen_move)

    def choose_move(self, game_state):
        """Abstract method that returns an integer between 0 and 6
         for the column that the AI wants to play in"""
        raise NotImplementedError("Please Implement this method")


class Dummy(ConnectFourEntity):
    def __init__(self):
        pass

    def choose_move(self, game_state):
        pass


class TrainableAI(ConnectFourEntity):
    """An abstract class that can interact with the training module as well
     as the game module."""
    is_trainable = True

    def __init__(self, name, is_training):
        self.name = name
        # Training examples is an array that stores data from the current game.
        self.training_examples = []
        self.root_node_initialised = False
        self.model = self.build_and_compile_model()
        self.is_training = is_training
        # Used computation tokes counts how many times the model has been used
        # to generate a prediction in the current game.
        self.used_computation_tokens = 0

    def get_training_examples(self):
        """Returns and clears the list of currently stored training examples."""
        training_examples = self.training_examples
        self.training_examples = []
        return training_examples


    def train_on_batch(self, examples):
        """Abstract method that takes an array of training examples
         and trains the AI's model on them."""
        raise NotImplementedError("Please Implement this method")

    def load_parameters(self, filepath):
        self.model.load_weights(filepath+".h5")

    def save_parameters(self, filepath):
        self.model.save_weights(filepath+".h5")

    @staticmethod
    def build_and_compile_model():
        """Abstract method which returns an instance of this AI's model
        architecture with randomly initialised weights."""
        raise NotImplementedError("Please Implement this method")

    def prepare_input_tensor(self, game_state):
        """Abstract method that Returns a numpy array representing the
        game_state that can be inputted into the model."""
        raise NotImplementedError("Please Implement this method")

    def punish_for_timeouts(self, timeout_count, game_count):
        pass


class FlatNeuralNetwork(TrainableAI):
    """An abstract class for an AI that simply looks one move deep and uses
     a neural network to analyse the resulting positions."""
    class TrainingExample:
        """Class which stores examples from games to train the AI's model on."""
        def __init__(self, game_state, parent_AI):
            self.state_tensor = parent_AI.prepare_input_tensor(game_state)
            self.reward_tensor = np.array([0])

        def set_reward(self, reward):
            self.reward_tensor = np.array([reward])

    def choose_move(self, game_state):
        if self.is_training:
            # Places the current game_state into the training_examples array.
            self.training_examples.append(FlatNeuralNetwork.TrainingExample(
                game_state, self))
        # A move is counted of having a rating in the range  [1 ,-1] depending
        # on how good it is for the current player (1 being a win).
        # Predictions stores an array of these ratings.
        predictions = []
        best_move_rating = -1
        best_move = 0
        possible_moves = game_state.get_possible_moves()
        for move in possible_moves:
            game_state_copy = make_copy(game_state)
            # If making the move on the board results in a finished game,
            # use the default move ratings
            if game_state_copy.place_counter_and_win_check(move):
                if game_state_copy.is_drawn:
                    move_rating = 0
                else:
                    move_rating = 1
            # Otherwise, the move rating is the output of the neural network.
            else:
                input_tensor = self.prepare_input_tensor(game_state_copy)
                input_tensor = np.expand_dims(input_tensor, 0)
                # The neural network predicts how favourable the game state is
                # for the current player, so this rating must be flipped.
                move_rating = np.squeeze(-self.model.predict(input_tensor))
                self.used_computation_tokens += 1
            # If the game is longer than 6 moves, the AI plays what it thinks is
            # the best move, otherwise it will sample from the prediction array.
            if len(game_state.move_list) > 6:
                if move_rating > best_move_rating:
                    best_move = move
                    best_move_rating = move_rating
            else:
                predictions.append(move_rating)
        if len(game_state.move_list) > 6:
            chosen_move = best_move
        else:
            probability_distribution = softmax(predictions)
            chosen_move = possible_moves[sample(probability_distribution)]
        return chosen_move

    def train_on_batch(self, examples):
        state_tensors = []
        reward_tensors = []
        for example in examples:
            state_tensors.append(example.state_tensor)
            reward_tensors.append(example.reward_tensor)
        return self.model.fit(np.array(state_tensors), np.array(reward_tensors))


class ResidualNN(FlatNeuralNetwork):
    """Uses a residual neural network architecture to predict how favourable
    a given game state is for the current player."""
    @staticmethod
    def build_and_compile_model(residual_blocks=5, filter_size=3, filter_no=75):
        # Uses the keras functional api to build a residual model.
        input_layer = keras.Input(shape=(6, 7, 1))
        resnet = add_resnet(input_layer, residual_blocks, filter_size, filter_no)
        resnet = add_dense_head(resnet)
        output_layer = keras.layers.Dense(1)(resnet)
        model = keras.Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer=tf.train.AdamOptimizer(),
                      loss=keras.losses.mean_squared_error,
                      metrics=['accuracy'])
        return model

    def prepare_input_tensor(self, game_state):
        board_array = deepcopy(game_state.current_board.array)
        # the board has to be inverted  if it is yellow's move to make it from
        # the perspective of the current player.
        if not game_state.reds_move:
            invert_board(board_array)
        input_tensor = np.array(board_array)
        # Keras' convolutional layer expects an array of values per cell.
        input_tensor = np.expand_dims(input_tensor, 3)
        return input_tensor


class MiniMax(ConnectFourEntity):
    """Uses the MiniMax algorithm to choose moves."""
    MOVE_PRIORITIES = (3, 4, 2, 5, 1, 6, 0)
    DEFAULT_SEARCH_DEPTH = 6

    def __init__(self, name, search_depth):
        self.name = name
        self.search_depth = MiniMax.DEFAULT_SEARCH_DEPTH
        # transposition_table is a dictionary indexed by boards
        # that is used to prevent repeated analysis of the same board.
        self.transposition_table = {}

    def choose_move(self, game_state):
        # The transposition_table has to be cleared for every new move.
        self.transposition_table = {}
        winning_moves_list = []
        drawing_moves_list = []
        losing_moves_list = []
        winning_score = MiniMax.get_winning_score(game_state)
        self.get_move_evaluations(
            winning_moves_list, drawing_moves_list, losing_moves_list,
            winning_score, game_state, self.search_depth)
        # If there are some winning moves, randomly choose from them. Otherwise,
        # choose the drawing move that is highest in the move priorities
        if winning_moves_list:
            chosen_move = random.choice(winning_moves_list)
        elif drawing_moves_list:
            for move in MiniMax.MOVE_PRIORITIES:
                if move in drawing_moves_list:
                    chosen_move = move
                    break
        else:
            chosen_move = random.choice(losing_moves_list)
        return chosen_move

    def analyse(self, game_state, depth):
        """Returns either 1, -1 or 0  to represent a win, loss or draw for the
        current player as the evaluation of this game state."""
        if depth == 0:
            return 0
        else:
            winning_moves_list = []
            drawing_moves_list = []
            losing_moves_list = []
            winning_score = MiniMax.get_winning_score(game_state)
            # Recursively calls get_move_evaluations.
            self.get_move_evaluations(
                winning_moves_list, drawing_moves_list, losing_moves_list,
                winning_score, game_state, depth)
            if winning_moves_list:
                return winning_score
            if drawing_moves_list:
                return 0
            elif losing_moves_list:
                return -winning_score
            else:
                # In the case where the board is full, returns zero for a draw.
                return 0

    def get_move_evaluations(self, winning_moves_list, drawing_moves_list,
                             losing_moves_list, winning_score, game_state, depth):
        """Fills the move lists with the moves of the given evaluation by
         recursively calling analyse."""
        possible_moves = game_state.get_possible_moves()
        for move in possible_moves:
            game_state_copy = make_copy(game_state)
            if game_state_copy.place_counter_and_win_check(move):
                if game_state_copy.is_drawn:
                    drawing_moves_list.append(move)
                else:
                    winning_moves_list.append(move)
            else:
                # If making the move does not result in a terminal game state,
                # first check if this position has aready been analysed.
                hashable_board = get_board_tuple(game_state_copy)
                if hashable_board in self.transposition_table:
                    move_analysis = self.transposition_table[hashable_board]
                else:
                    # If the position is not in the transposition table,
                    # recursively call analyse.
                    move_analysis = self.analyse(game_state_copy, depth - 1)
                    self.transposition_table[hashable_board] = move_analysis
                if move_analysis == winning_score:
                    winning_moves_list.append(move)
                elif move_analysis == 0:
                    drawing_moves_list.append(move)
                else:
                    losing_moves_list.append(move)

    @staticmethod
    def get_winning_score(game_state):
        if game_state.reds_move:
            return 1
        else:
            return -1


class AZtypeAI(TrainableAI):
    """An abstract class for AI which use types of the AlphaZero algorithm."""
    class TrainingExample(FlatNeuralNetwork.TrainingExample):
        """Used to store data used to train AlphaZero's neural network."""
        def __init__(self, game_state, improved_policy, parent_AI):
            super().__init__(game_state, parent_AI)
            self.policy_tensor = np.array(improved_policy)

    def __init__(self, name, is_training):
        super().__init__(name, is_training)
        self.root_node_initialised = False

    def choose_move(self, game_state):
        self.setup_for_simulations(game_state)
        self.set_root_node(game_state)
        # Plays simulated games until it is decided enough have been played.
        while self.more_simulations_needed():
            self.root_node.expand(self)
            self.used_computation_tokens += 1
        # Sets the improved policy to be the normalised counts
        # of the possible moves and 0 otherwise
        improved_policy = [0 for _ in range(7)]
        for child_node in self.root_node.child_nodes:
            improved_policy[child_node.preceding_action] = \
                child_node.visit_count/(self.root_node.visit_count-1)
        if self.is_training:
            self.training_examples.append(AZtypeAI.TrainingExample(
                make_copy(game_state), improved_policy, self))
        # Chooses the child node either deterministically (if the game is longer
        # than 4 moves) or probabilistically using the improved policy.
        chosen_child_node = self.get_chosen_child_node(
            len(game_state.move_list), improved_policy)
        self.root_node = chosen_child_node
        return chosen_child_node.preceding_action

    def more_simulations_needed(self):
        """Abstract method returns False when computation should be cut off."""
        raise NotImplementedError("Please Implement this method")

    def setup_for_simulations(self, game_state):
        """Abstract method called at the start of the choose move method."""
        raise NotImplementedError("Please Implement this method")

    def get_chosen_child_node(self, move_list_length, improved_policy):
        """Chooses the child node either deterministically (if the game is longer
        than 4 moves) or probabilistically using the improved policy."""
        if move_list_length < 5:
            chosen_action = sample(improved_policy)
            for child_node in self.root_node.child_nodes:
                if child_node.preceding_action == chosen_action:
                    return child_node
        else:
            chosen_child_node_list = []
            max_visit_count = 0
            # Keeps a list if there are multiple child nodes with the same
            # highest visit count and chooses randomly in the event of a tie.
            for child_node in self.root_node.child_nodes:
                if child_node.visit_count > max_visit_count:
                    max_visit_count = child_node.visit_count
                    chosen_child_node_list = [child_node]
                elif child_node.visit_count == max_visit_count:
                    chosen_child_node_list.append(child_node)
            return random.choice(chosen_child_node_list)

    def set_root_node(self, game_state):
        # If the root node is initialised, checks all the child nodes to see
        # if the game state they represent is the same as the passed game state.
        if self.root_node_initialised:
            for child_node in self.root_node.child_nodes:
                if child_node.game_state.current_board.array ==\
                        game_state.current_board.array:
                    self.root_node = child_node
                    break
            else:
                # If no such child node is found, a new root node is made.
                self.root_node_initialised = False
        # Makes a new root node using a copy of the game state.
        if not self.root_node_initialised:
            game_copy = make_copy(game_state)
            self.root_node = UCTsearchNode(None, game_copy, None)
            self.root_node.expand(self)
            self.root_node_initialised = True

    def train_on_batch(self, examples):
        """Trains the neural network on a list of training examples."""
        state_tensors = []
        reward_tensors = []
        policy_tensors = []
        for example in examples:
            state_tensors.append(example.state_tensor)
            reward_tensors.append(example.reward_tensor)
            policy_tensors.append(example.policy_tensor)
        state_tensors = np.array(state_tensors)
        reward_tensors = np.array(reward_tensors)
        policy_tensors = np.array(policy_tensors)
        return self.model.fit(state_tensors, {"policy_head": policy_tensors,
                                              "value_head": reward_tensors})

    @staticmethod
    def build_AZ_network(input_shape, residual_blocks, filter_size, filter_no):
        """Uses the Keras functional api to make an alphazero-type
        residual neural network model with a specified input shape."""
        input_layer = keras.Input(shape=input_shape)
        resnet = add_resnet(input_layer, residual_blocks, filter_size, filter_no)
        policy_head = add_dense_head(resnet)
        policy_head = keras.layers.Dense(7, name="policy_head",
                                         activation="sigmoid")(policy_head)
        value_head = add_dense_head(resnet)
        value_head = keras.layers.Dense(1, name="value_head",
                                        activation="tanh")(value_head)
        model = keras.Model(inputs=input_layer,
                            outputs=[policy_head, value_head])
        model.compile(optimizer=tf.train.AdamOptimizer(),
                      loss={"policy_head": keras.losses.binary_crossentropy,
                            "value_head": keras.losses.mean_squared_error},
                      metrics=['accuracy'])
        return model


class AZclone(AZtypeAI):
    """ An implementation of the AlphaZero algorithm for connect4."""
    def more_simulations_needed(self, simulations_per_move=100):
        # Returns false when 100 simulations have been used for this move.
        if self.simulation_count == simulations_per_move:
            return False
        else:
            self.simulation_count += 1
            return True

    def setup_for_simulations(self, game_state):
        self.simulation_count = 0

    def __init__(self, name, is_training):
        super().__init__(name, is_training)
        self.root_node_initialised = False

    @staticmethod
    def build_and_compile_model(residual_blocks=5, filter_size=3, filter_no=75):
        return AZtypeAI.build_AZ_network((6, 7, 1),
                                         residual_blocks, filter_size, filter_no)

    def prepare_input_tensor(self, game_state):
        return ResidualNN.prepare_input_tensor(self, game_state)


class TimedUCT(AZtypeAI):
    DEFAULT_TIMER_CONSTANT = 3

    def __init__(self, name, is_training):
        super().__init__(name, is_training)
        self.timer_constant = TimedUCT.DEFAULT_TIMER_CONSTANT

    def setup_for_simulations(self, game_state):
        # collects data used when preparing the input tensor and
        # deciding whether more simulations are needed.
        self.simulation_count = 0
        self.initially_reds_move = game_state.reds_move
        self.remaining_move_count = 42 - len(game_state.move_list)
        self.normalised_clocks = game_state.get_normalised_clocks()
        self.current_player_clock_tensor = \
            np.array([[[self.normalised_clocks[0]]
                       for _ in range(7)] for _ in range(6)])
        self.enemy_player_clock_tensor = \
            np.array([[[self.normalised_clocks[1]]
                       for _ in range(7)] for _ in range(6)])

    def more_simulations_needed(self):
        # Atleast 5 simulations are done to prevent anomolies.
        self.simulation_count += 1
        if self.simulation_count < 5:
            return True
        # calculates the lower confidence bound of the "best" node
        most_negative_UCT_node = self.root_node.get_most_negative_UCT_node()
        lower_confidence_bound = \
            most_negative_UCT_node.average_value() \
            + most_negative_UCT_node.confidence_interval(
                self.root_node.visit_count)
        # Calculates how much this bound overlaps with the UCT values
        # of the other child nodes.
        total_confidence_overlap = 0
        for child_node in self.root_node.child_nodes:
            if child_node is not most_negative_UCT_node:
                total_confidence_overlap += self.confidence_overlap(
                    child_node, lower_confidence_bound)
        # calculates timer_value, a score of how much
        # could be gained from doing more simulations
        timer_value = (total_confidence_overlap * self.normalised_clocks[0]*42)\
                      / (self.remaining_move_count)
        # compares this value with a learned parameter
        # to decide whether to cut of calculation
        # The simulation count is limited to mitigate against anomolies.
        if self.simulation_count > 500:
            return False
        elif timer_value < self.timer_constant:
            return False
        else:
            return True

    def confidence_overlap(self, child_node, lower_confidence_bound):
        """Returns the difference between this node's UCT value
         and the lower confidence bound."""
        child_UCT = child_node.average_value() - child_node.confidence_interval(
                self.root_node.visit_count)
        confidence_overlap = lower_confidence_bound - child_UCT
        if confidence_overlap > 0:
            return confidence_overlap
        else:
            return 0

    def punish_for_timeouts(self, timeout_count, game_count):
        # If the AI timed out in over 5 percent of the games, the timer
        # constant is increased to make it use less time per move.
        if timeout_count / game_count > 0.05:
            # We multiply rather than add to the timer constant so that
            # it will converge on an appropriate value quicker.
            self.timer_constant = self.timer_constant*1.01
        else:
            self.timer_constant = self.timer_constant*0.99
        print("timer_constant: ", self.timer_constant)

    @staticmethod
    def build_and_compile_model(residual_blocks=2, filter_size=3, filter_no=75):
        return AZtypeAI.build_AZ_network((6, 7, 3),
                                         residual_blocks, filter_size, filter_no)

    def prepare_input_tensor(self, game_state):
        board_tensor = ResidualNN.prepare_input_tensor(self, game_state)
        if game_state.reds_move == self.initially_reds_move:
            return np.concatenate((board_tensor,
                             self.current_player_clock_tensor,
                             self.enemy_player_clock_tensor), 2)
        else:
            return np.concatenate((board_tensor,
                             self.enemy_player_clock_tensor,
                             self.current_player_clock_tensor), 2)

    def load_parameters(self, filepath):
        """Overrides TrainableAI's load_parameters method to
        load the timer constant as well"""
        self.model.load_weights(filepath + ".h5")
        timer_constant_file = open(filepath + ".p", "rb")
        self.timer_constant = pickle.load(timer_constant_file)

    def save_parameters(self, filepath):
        """Overrides TrainableAI's save_parameters method to
         save the timer constant as well"""
        self.model.save_weights(filepath + ".h5")
        timer_constant_file = open(filepath + ".p", "wb")
        pickle.dump(self.timer_constant, timer_constant_file)


class UCTsearchNode:
    """Used by the AZclone and TimedUCT classes, each node represents a game
    state with child nodes that are the result of different moves from this state."""
    C_PUCT = 4

    def __init__(self, preceding_action, game_state, node_policy):
        """Initialises this node as a leaf node"""
        self.preceding_action = preceding_action
        self.game_state = game_state
        self.is_leaf_node = True
        self.visit_count = 0
        self.total_value = 0
        self.node_policy = node_policy
        self.child_nodes = []

    def expand(self, AI):
        """Returns the change in this node's value resulting from the simulation."""
        if self.is_leaf_node:
            # get prediction from neural network
            input_tensor = AI.prepare_input_tensor(self.game_state)
            input_tensor = np.expand_dims(input_tensor, 0)
            model_output = AI.model.predict(input_tensor)
            policy_head = np.squeeze(model_output[0])
            value_head = np.squeeze(model_output[1])
            # use prediction to estimate own value and likely moves
            self.initialise_child_nodes(policy_head)
            value_change = value_head
            self.is_leaf_node = False
        else:
            # recursively expand the child node with the most negative
            # upper confidence value, which corresponds to the
            # game state that is worst for the other player
            most_negative_UCT_node = self.get_most_negative_UCT_node()
            value_change = most_negative_UCT_node.expand(AI)
        self.total_value += value_change
        self.visit_count += 1
        # the parent node is from the perspective of the other player,
        # hence the value change returned is negative
        return -value_change

    def get_most_negative_UCT_node(self):
        """Returns the child node with the highest upper confidence (UCT) value"""
        most_negative_UCT = 1
        most_negative_UCT_node = None
        for child_node in self.child_nodes:
            child_UCT = child_node.average_value() \
                        - child_node.confidence_interval(self.visit_count)
            if child_UCT < most_negative_UCT:
                most_negative_UCT_node = child_node
                most_negative_UCT = child_UCT
        return most_negative_UCT_node

    def initialise_child_nodes(self, policy_head):
        # get possible moves and normalise the policy head
        # to only take into account allowed moves
        policy_sum = 0
        possible_moves = []
        for i, policy in enumerate(policy_head):
            if not self.game_state.current_board.column_is_full(i):
                policy_sum += policy
                possible_moves.append(i)
        # initialise child nodes corresponding to game states
        # resulting from each possible move
        for move in possible_moves:
            game_state_copy = make_copy(self.game_state)
            adjusted_policy = policy_head[move] / policy_sum
            if game_state_copy.place_counter_and_win_check(move):
                if game_state_copy.is_drawn:
                    node_value = 0
                else:
                    node_value = -1
                self.child_nodes.append(TerminalNode(
                    node_value, move, game_state_copy, adjusted_policy))
            else:
                self.child_nodes.append(UCTsearchNode(
                    move, game_state_copy, adjusted_policy))

    def confidence_interval(self, parent_n):
        return (UCTsearchNode.C_PUCT * self.node_policy *
                math.sqrt(parent_n)) / (self.visit_count + 1)

    def average_value(self):
        if self.is_leaf_node:
            return 0
        else:
            return self.total_value / self.visit_count


class TerminalNode(UCTsearchNode):
    """Class for nodes that represent a won or drawn position, a terminal node's
    value is -1 for a loss for the current player and 0 for a draw."""
    def __init__(self, value, preceding_action, game_state, node_policy):
        super().__init__(preceding_action, game_state, node_policy)
        self.value = value

    def average_value(self):
        return self.value

    def confidence_interval(self, parent_n):
        """We can be one hundred percent sure of this node's value;
         therefore the confidence interval on it's value is zero."""
        return 0

    def expand(self, AI):
        self.visit_count += 1
        return -self.value


def add_dense_head(y):
    """Adds a set of dense layers using the Keras functional api."""
    y = keras.layers.Flatten()(y)
    y = keras.layers.Dense(20)(y)
    y = keras.layers.LeakyReLU()(y)
    y = keras.layers.Dense(20)(y)
    y = keras.layers.LeakyReLU()(y)
    return y


def add_resnet(y, residual_blocks, filter_size, filter_no):
    """Uses the Keras functional api to add a convolutional residual model."""
    y = add_convolutional_layer(y, filter_size, filter_no)
    y = add_convolutional_layer(y, filter_size, filter_no)
    for i in range(residual_blocks):
        y = add_residual_block(y, filter_size, filter_no)
    return y


def add_residual_block(block_input, filter_size, filter_no):
    """Uses the Keras functional api to add a convolutional residual block."""
    y = add_convolutional_layer(block_input, filter_size, filter_no)
    y = add_convolutional_layer(y, filter_size, filter_no)
    return keras.layers.add([block_input, y])


def add_convolutional_layer(y, filter_size, filter_no):
    # the padding is "same" to keep the input and output a constant shape
    y = keras.layers.Conv2D(filter_no, filter_size, padding="same")(y)
    y = keras.layers.BatchNormalization()(y)
    return keras.layers.LeakyReLU()(y)


def invert_board(board):
    """Returns a list with the counter codes switched."""
    for i in range(6):
        for j in range(7):
            board[i][j] = -board[i][j]
    return board

def softmax(prediction):
    """Turns an array of expected rewards into a probability distribution."""
    prediction = np.exp(prediction)
    probability_sum = sum(prediction)
    for i, probability in enumerate(prediction):
        prediction[i] = probability / probability_sum
    return prediction

def sample(probability_distribution):
    """Returns the index of the randomly selected probability."""
    random_probability = random.random()
    for i, probability in enumerate(probability_distribution):
        random_probability -= probability
        if random_probability < 0:
            return i
    return len(probability_distribution) - 1


def make_copy(game_state):
    dummy = Dummy()
    board_copy = deepcopy(game_state.current_board)
    move_list_copy = deepcopy(game_state.move_list)
    return game.GameState(dummy, dummy,
                          game_state.reds_move, board_copy, move_list_copy)


def get_board_tuple(game_state):
    return tuple(tuple(row) for row in game_state.current_board.array)
