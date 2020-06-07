# -*- coding: utf-8 -*-
import imageio
import numpy as np
import pandas as pd
from random import randint
from numpy.random import random
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec


class SimulationParameters:
    def __init__(self):
        pass

    def parameterize_brownian_motion(self, remove_rate, infect_rate, motion_speed, space_size):
        self.remove_rate = remove_rate
        self.infect_rate = infect_rate
        self.motion_speed = motion_speed
        self.space_size = space_size

    def parameterize_sir(self, S, I, R, days):
        self.S = S
        self.I = I
        self.R = R
        self.days = days


class BrownianMotionBasedSimulation:

    def __init__(self, parameters, plot):

        self.N = parameters.days
        self.S = parameters.S
        self.I = parameters.I
        self.R = parameters.R
        self.v = parameters.motion_speed
        self.D = parameters.space_size
        self.remove_rate = parameters.remove_rate
        self.infect_rate = parameters.infect_rate
        self.plot = plot
        self.start_people()
        self.start_history()
        self.cmap = list('grk')

    def start_history(self):
        """ Initializes the historical data of an infected """

        self.history = pd.DataFrame(
            {'S': [self.S-self.I], 'I': [self.I], 'R': [self.R]})

    def generate_random_location(self):
        """ Generate a random location """

        return random(self.N)*self.D

    def start_people(self):
        """ Starts a population """

        self.people = pd.DataFrame({
            'x': self.generate_random_location(),
            'y': self.generate_random_location(),
            'status': np.zeros(self.N, dtype=int)
        })
        # init the first infector
        self.people.loc[randint(0, self.N-1), 'status'] = 1

    def display(self):

        def colors():
            def color_map(code): return self.cmap[int(code)]
            return list(map(color_map, self.people['status']))

        # 2 rows 3 cols
        grid = self.plot.GridSpec(2, 3)

        # top left
        self.plot.subplot(grid[0, :1])
        self.plot.scatter(self.people['x'], self.people['y'], c=colors())
        self.plot.title('Velocidade = {}, Densidade = {}'.format(
            self.v, self.D), fontsize=16)

        days = range(len(self.history))

        # top center
        self.plot.subplot(grid[0, 1])
        sir = []
        for i, label in enumerate(list('SIR')):
            value = self.history.iloc[len(self.history)-1][label]
            self.plot.bar(i, value)
            sir.append(value)

        self.plot.title(
            f'S: {sir[0]} - I: {sir[1]} - R: {sir[2]}', fontsize=16)
        self.plot.legend(list('SIR'))
        self.plot.ylim(0, self.N)

        # top right
        self.plot.subplot(grid[0, 2])

        for i, label in enumerate(list('SIR')):
            self.plot.bar(days, self.history[label])
        self.plot.legend(list('SIR'))

        # bottom
        self.plot.subplot(grid[1, :3])
        for i, label in enumerate(list('SIR')):
            self.plot.plot(
                days, self.history[label], self.cmap[i], label=label, linewidth=3, linestyle='--')

        self.plot.title(f'SIR - day: {len(self.history)}', fontsize=16)
        self.plot.legend(loc="upper right")
        self.plot.xlim(0, self.N)
        self.plot.xlabel('day')
        self.plot.ylabel('count')

    def daily(self):
        self.move()
        self.infect()
        self.remove()
        self.summerize()

    def move(self):
        def random_sign(i): return np.sign(random(i) - .5)
        xs = random(self.N) * random_sign(self.N)
        ys = (1 - xs ** 2) ** .5 * random_sign(self.N)
        dps = pd.DataFrame({'x': xs*self.v, 'y': ys*self.v})
        self.people.loc[:, ['x', 'y']] += dps

    def infect(self):
        infectors = self.people.query('status == 1')
        # verifiy the infectors len
        infected = len(infectors)
        for i in range(infected):
            ordinarys = self.people.query('status == 0')
            xs, ys = ordinarys['x'], ordinarys['y']
            x, y = infectors.iloc[i, 0], infectors.iloc[i, 1]
            dxs, dys = xs-x, ys-y
            ds = (dxs**2+dys**2)
            ps = np.exp(-self.infect_rate*ds)
            Ss = self.people['status'] == 0
            infectment = [1 if v < min(
                p, .8) and p > 0.1 else 0 for v, p in zip(random(len(Ss)), ps)]
            self.people.loc[Ss, 'status'] += infectment

    def remove(self):
        infected = len(self.people.query('status == 1'))
        removements = [
            1 if i < self.remove_rate else 0 for i in random(infected)]
        self.people.loc[self.people['status'] == 1, 'status'] += removements

    def summerize(self):
        dic = {}
        for i, key in enumerate(list('SIR')):
            dic[key] = [len(self.people.query('status == {}'.format(i)))]
        summary = pd.DataFrame(dic)
        self.history = self.history.append(summary)


class Visualization:

    def __init__(self, days, plot):
        self.days = days
        self.plot = plot

    def save(self, day, is_last=False):
        if self.days == (day+1):
            self.plot.savefig(r'output/day{}.png'.format(day+1))

        self.plot.savefig(r'plots/day{}.png'.format(day+1))

    def generate_gif(self):
        frames = []
        for i in range(self.days):
            frames.append(imageio.imread(r'plots/day{}.png'.format(i+1)))
        imageio.mimsave(r'output/{}days.gif'.format(self.days),
                        frames, 'GIF', duration=0.1)


if __name__ == '__main__':

    parameters = SimulationParameters()
    parameters.parameterize_brownian_motion(
        remove_rate=.01,
        infect_rate=.5,
        motion_speed=2,
        space_size=10
    )
    parameters.parameterize_sir(
        S=100,
        I=1,
        R=0,
        days=30
    )

    plt.style.use('ggplot')
    fig = plt.figure(figsize=(16, 12))

    visualization = Visualization(days=parameters.days, plot=plt)

    motion_simulation = BrownianMotionBasedSimulation(parameters, plot=plt)

    plt.ion()
    for i in range(parameters.days):
        plt.cla()
        # displays the current plot
        motion_simulation.display()
        # generates a new simulation for the updated period
        motion_simulation.daily()
        # automatically adjust subplot parameters
        fig.tight_layout()
        plt.pause(.05)
        # save an result
        visualization.save(i)

    # generate gif and save the output
    visualization.generate_gif()
    plt.ioff()
    plt.show()
