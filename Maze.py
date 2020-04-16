import random
import numpy as np
from MazeGenerator import MazeGenerator, bcolors


class Maze:
    ACTION_UP = 0
    ACTION_LEFT = 1
    ACTION_DOWN = 2
    ACTION_RIGHT = 3

    ACTIONS = [ACTION_UP, ACTION_LEFT, ACTION_DOWN, ACTION_RIGHT]
    ACTIONS_NAMES = ['UP','LEFT','DOWN','RIGHT']

    MOVEMENTS = {
        ACTION_UP: (1, 0),
        ACTION_RIGHT: (0, 1),
        ACTION_LEFT: (0, -1),
        ACTION_DOWN: (-1, 0)
    }

    num_actions = len(ACTIONS)

    def __init__(self, n, m, wrong_action_p=0.1, alea=False):
        self.n = n
        self.m = m
        self.wrong_action_p = wrong_action_p
        self.alea = alea
        self.path = []
        self.generate_maze()

    def _position_to_id(self, x, y):
        return x + y * self.n

    def _id_to_position(self, id):
        return (id % self.n, id // self.n)

    def generate_maze(self):
        cases = [(x, y) for x in range(self.n) for y in range(self.m)]
        #build walls
        mazeGenerator = MazeGenerator(self.n, self.m)
        maze = mazeGenerator.generate_maze()
        walls=[]
        for i in range(self.m):
            for j in range(self.n):
                if (maze[i][j] == 0):
                    walls.append((i, j))
                    cases.remove((i, j))

        #choose start
        start = random.choice(cases)
        cases.remove(start)

        #choose end
        end = random.choice(cases)
        cases.remove(end)

        self.position = start
        self.end = end
        self.walls = walls
        self.counter = 0
        if not self.alea:
            self.start = start

        return self._get_state()

    def reset(self):
        if not self.alea:
            self.position = self.start
            self.counter = 0
            return self._get_state()
        else:
            return self.generate_maze()

    def _get_grille(self, x, y):
        grille = [
            [0] * self.n for i in range(self.m)
        ]
        grille[x][y] = 1
        return grille

    def _get_state(self):
        if self.alea:
            return [self._get_grille(x, y) for (x, y) in
                    [self.position, self.end, self.walls]]
        return self._position_to_id(*self.position)

    def move(self, action):
        """
        input: id of the action to play
        output: ((state_id, end, walls), reward, is_final, actions)
        """
        self.counter += 1
        if action not in self.ACTIONS:
            raise Exception('Invalid action')

        choice = random.random()
        if choice < self.wrong_action_p :
            action = (action + 1) % 4
        elif choice < 2 * self.wrong_action_p:
            action = (action - 1) % 4

        d_x, d_y = self.MOVEMENTS[action]
        x, y = self.position
        new_x, new_y = x + d_x, y + d_y

        if self.end == (new_x, new_y):
            self.position = new_x, new_y
            return self._get_state(), 10, True, self.ACTIONS
        elif (new_x, new_y) in self.walls:
            return self._get_state(), -1, False, self.ACTIONS
        elif new_x >= self.n or new_y >= self.m or new_x < 0 or new_y < 0:
            return self._get_state(), -1, False, self.ACTIONS
        elif self.counter > 190:
            self.position = new_x, new_y
            return self._get_state(), -10, True, self.ACTIONS
        else:
            self.position = new_x, new_y
            return self._get_state(), -1, False, self.ACTIONS

    def solve(self, lr, y, num_episodes):
        """ lr: learning_reate
            y: Gamma
            num_episodes: number of total episodes """

        num_states = self.n * self.m
        num_actions = 4
        Q = np.zeros((num_states, num_actions))
        cumul_reward_list = []
        actions_list = []
        states_list = []
        print("Initial State of the Maze")
        self.print()

        for i in range(num_episodes):
            actions = []
            s = self.reset()
            states = []
            cumul_reward = 0
            e = False

            while not e:
                Q2 = Q[s, :]
                a = np.argmax(Q2)
                s1, reward, e, _ = self.move(a)
                Q[s, a] = Q[s, a] + lr * (reward + y * np.max(Q[s1, :]) - Q[s, a])
                cumul_reward += reward
                s = s1
                actions.append(a)
                states.append(s)
            states_list.append(states)
            actions_list.append(actions)
            cumul_reward_list.append(cumul_reward)

        print("Score over time: " + str(sum(cumul_reward_list[-100:])/100))
        #print actions
        actions_name = []
        for i in range(len(actions)):
            actions_name.append(self.ACTIONS_NAMES[actions[i]])
        print("Moves: " + str(actions_name))
        #get path
        for i in range(len(states)):
            states[i] = self._id_to_position(states[i])

        self.print(states)


    def print(self, states=None):
        if (states == None):
            str = ""
            for i in range(self.n - 1, -1, -1):
                for j in range(self.m):
                    if (i, j) == self.position:
                        str += bcolors.OKBLUE + "X " + bcolors.ENDC
                    elif (i, j) in self.walls:
                        str += bcolors.FAIL + "¤ " + bcolors.ENDC
                    elif (i, j) == self.end:
                        str += bcolors.OKGREEN + "@ " + bcolors.ENDC
                    else:
                        str += ". "
                str += "\n"
        else:
            str = ""
            for i in range(self.n - 1, -1, -1):
                for j in range(self.m):
                    if (i, j) == self.start:
                        str += bcolors.OKBLUE + "X " + bcolors.ENDC
                    elif (i, j) == self.end:
                        str += bcolors.OKGREEN + "@ " + bcolors.ENDC
                    elif (i, j) in states:
                        str += bcolors.WARNING + "# " + bcolors.ENDC
                    elif (i, j) in self.walls:
                        str += bcolors.FAIL + "¤ " + bcolors.ENDC

                    else:
                        str += ". "
                str += "\n"
        print(str)




if __name__=="__main__":

    lr = 0.85 # Alpha / learning rate
    y = 0.99 # Gamma
    num_episodes = 1000
    print(bcolors.CYAN + '''
    __  ___                    _____ ____  __
   /  |/  /___ _____  ___     / ___// __ \/ /   _____  _____
  / /|_/ / __ `/_  / / _ \    \__ \/ / / / / | / / _ \/ ___/
 / /  / / /_/ / / /_/  __/   ___/ / /_/ / /| |/ /  __/ /
/_/  /_/\__,_/ /___/\___/   /____/\____/_/ |___/\___/_/
''' + bcolors.ENDC)
    game = Maze(10,10,0)
    game.solve(lr, y, num_episodes)
