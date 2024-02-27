import numpy as np
import time
import sys

if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk

UNIT = 40
MAZE_H = 8
MAZE_W = 8


class Hell:
    def __init__(self, canvas, center, size=15, fill_color='black'):
        self.canvas = canvas
        self.center = center
        self.size = size
        self.fill_color = fill_color
        self._create_hell()

    def _create_hell(self):
        x0, y0 = self.center[0] - self.size, self.center[1] - self.size
        x1, y1 = self.center[0] + self.size, self.center[1] + self.size
        self.rect = self.canvas.create_rectangle(x0, y0, x1, y1, fill=self.fill_color)

    def get_coords(self):
        return self.canvas.coords(self.rect)


class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.n_features = 2
        self.title('maze')
        self.geometry('{0}x{1}'.format(MAZE_H * UNIT, MAZE_H * UNIT))
        self._build_maze()

    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg='white', height=MAZE_H * UNIT, width=MAZE_W * UNIT)

        for c in range(0, MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_W * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        origin = np.array([20, 20])

        hell1_center = origin + np.array([UNIT, UNIT * 2])
        self.hell1 = Hell(self.canvas, center=hell1_center).get_coords()

        hell2_center = origin + np.array([UNIT * 2, UNIT])
        self.hell2 = Hell(self.canvas, center=hell2_center).get_coords()

        hell3_center = origin + np.array([UNIT * 2, UNIT * 6])
        self.hell3 = Hell(self.canvas, center=hell3_center).get_coords()

        hell4_center = origin + np.array([UNIT * 3, UNIT * 7])
        self.hell4 = Hell(self.canvas, center=hell4_center).get_coords()

        hell5_center = origin + np.array([UNIT * 4, UNIT * 4])
        self.hell5 = Hell(self.canvas, center=hell5_center).get_coords()

        hell6_center = origin + np.array([UNIT * 4, UNIT * 1])
        self.hell6 = Hell(self.canvas, center=hell6_center).get_coords()

        hell7_center = origin + np.array([UNIT * 3, UNIT * 2])
        self.hell7 = Hell(self.canvas, center=hell7_center).get_coords()

        hell8_center = origin + np.array([UNIT * 1, UNIT * 3])
        self.hell8 = Hell(self.canvas, center=hell8_center).get_coords()

        hell9_center = origin + np.array([UNIT * 6, UNIT * 4])
        self.hell9 = Hell(self.canvas, center=hell9_center).get_coords()

        self.hells = [self.hell1, self.hell2, self.hell3, self.hell4, self.hell5, self.hell6, self.hell7, self.hell8,
                      self.hell9]
        #print([hell for hell in self.hells])

        oval_center = origin + np.array([UNIT * 3, UNIT * 3])
        self.oval = self.canvas.create_oval(
            oval_center[0] - 15, oval_center[1] - 15,
            oval_center[0] + 15, oval_center[1] + 15,
            fill='yellow')

        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red')

        self.canvas.pack()

    def reset(self):
        self.update()  # 刷新页面的布局（与外部的update函数无关）
        time.sleep(0.5)
        self.canvas.delete(self.rect)
        origin = np.array([20, 20])
        self.rect = self.canvas.create_rectangle(origin[0] - 15, origin[1] - 15, origin[0] + 15, origin[1] + 15,
                                                 fill='red')

        return (np.array(self.canvas.coords(self.rect)[:2])-np.array(self.canvas.coords(self.oval)[:2]))/(MAZE_H*UNIT)

    def step(self, action):
        s = self.canvas.coords(self.rect)
        base_action = np.array([0, 0])
        if action == 0:  # up
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:  # down
            if s[1] < UNIT * (MAZE_H - 1):
                base_action[1] += UNIT
        elif action == 2:  # left
            if s[0] > UNIT:
                base_action[0] -= UNIT
        elif action == 3:  # right
            if s[0] < UNIT * (MAZE_W - 1):
                base_action[0] += UNIT

        self.canvas.move(self.rect, base_action[0], base_action[1])  # move红色正方形

        next_coords = self.canvas.coords(self.rect)  # 确定玩家（红色矩形）新状态

        # print([self.canvas.coords(hell.get_coords()) for hell in self.hells])

        # 奖励机制
        if next_coords == self.canvas.coords(self.oval):
            reward = 1
            done = True
            s_ = 'Terminal'
        elif any(next_coords == hell for hell in self.hells):
            reward = -1
            done = True
            s_ = 'Terminal'
            print("ooh!!")
        else:
            reward = 0
            done = False
        s_=(np.array(next_coords[:2])-np.array(self.canvas.coords(self.oval)[:2]))/(MAZE_H*UNIT)
        return s_, reward, done

    def render(self):
        time.sleep(0.1)
        self.update()


def update():  #
    for t in range(10):
        s = env.reset()
        a = 1
        while True:
            env.render()
            a = (a + 1) % 4
            s, r, done = env.step(a)
            if done: break


if __name__ == '__main__':
    env = Maze()
    env.after(100, update)
    env.mainloop()
