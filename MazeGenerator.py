import math
import random

# Random self.maze Generator using Depth-first Search (Recursive Version)
# http://en.wikipedia.org/wiki/self.maze_generation_algorithm
# http://code.activestate.com/recipes/578356-random-maze-generator/
# FB - 20121205

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    CYAN = '\u001b[36m'

class MazeGenerator:
    def __init__(self, n, m):
        self.n = n
        self.m = m
        self.maze = [[0 for x in range(self.n)] for y in range(self.m)]

        # directions to move in the self.maze
        self.dx = [0, 1, 0, -1]
        self.dy = [-1, 0, 1, 0]

    def generate_maze(self, cx=0, cy=0):
        self.maze[cy][cx] = 1
        while True:
            # find a new cell to add
            nlst = [] # list of available neighbors
            for i in range(4):
                nx = cx + self.dx[i]; ny = cy + self.dy[i]
                if nx >= 0 and nx < self.n and ny >= 0 and ny < self.m:
                    if self.maze[ny][nx] == 0:
                        # of occupied neighbors of the candidate cell must be 1
                        ctr = 0
                        for j in range(4):
                            ex = nx + self.dx[j]; ey = ny + self.dy[j]
                            if ex >= 0 and ex < self.n and ey >= 0 and ey < self.m:
                                if self.maze[ey][ex] == 1: ctr += 1
                        if ctr == 1: nlst.append(i)
            # if 1 or more available neighbors then randomly select one and add
            if len(nlst) > 0:
                ir = nlst[random.randint(0, len(nlst) - 1)]
                cx += self.dx[ir]; cy += self.dy[ir]
                self.generate_maze(cx, cy)
            else:
                break
        return self.maze
