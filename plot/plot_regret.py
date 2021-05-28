import os
import pickle

import numpy as np
import matplotlib as mpl
import pandas as pd

# mpl.use('pdf')
import matplotlib.pyplot as plt

from experiments.experiments import evaluate
from agent.switching_agents import FixedSwitchingHuman


def latexify(fig_width, fig_height, font_size=7, legend_size=5.6):
    """Set up matplotlib's RC params for LaTeX plotting."""
    params = {
        'backend': 'ps',
        'text.latex.preamble': ['\\usepackage{amsmath,amsfonts,amssymb,bbm,amsthm, mathtools,times}'],
        'axes.labelsize': font_size,
        'axes.titlesize': font_size,
        'font.size': font_size,
        'legend.fontsize': legend_size,
        'xtick.labelsize': font_size,
        'ytick.labelsize': font_size,
        'text.usetex': True,
        'figure.figsize': [fig_width, fig_height],
        'font.family': 'serif',
        'xtick.minor.size': 0.5,
        'xtick.major.pad': 1.5,
        'xtick.major.size': 1,
        'ytick.minor.size': 0.5,
        'ytick.major.pad': 1.5,
        'ytick.major.size': 1
    }

    mpl.rcParams.update(params)
    plt.rcParams.update(params)


SPINE_COLOR = 'grey'


def format_axes(ax):
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color(SPINE_COLOR)
        ax.spines[spine].set_linewidth(0.5)

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_tick_params(direction='out', color=SPINE_COLOR)

    return ax


COLORS = ['#3E9651', '#CC2529', '#396AB1', '#535154']
golden_ratio = (np.sqrt(5) - 1.0) / 2


def plot_regret(alg2_regret, greedy_regret, file_name):
    width = 1.3
    height = width * golden_ratio

    latexify(fig_height=height, fig_width=width, font_size=5, legend_size=4)

    fig, ax = plt.subplots()
    ax = format_axes(ax)

    ax.plot(np.cumsum(alg2_regret), COLORS[1], label=r'Algorithm 2', linestyle='solid',
            linewidth=1)
    ax.plot(np.cumsum(greedy_regret), COLORS[0], label=r'Greedy Baseline', linestyle='solid',
            linewidth=1)

    ax.set_ylabel(r'Regret, $R(T)$')
    ax.set_xlabel(r'Episode, $k$')
    ax.legend(frameon=False)
    fig.savefig(file_name, bbox_inches='tight')


def plot_performance(agents, optimal_c, human, eval_set, root_dir):
    costs_dict = {}
    n_episodes = 0
    for agent in agents:
        config = agent.split('_')
        on_off = config[3]
        with open(f'{root_dir}/{agent}/costs_{on_off}', 'rb') as file :
            costs = pickle.load(file)
            costs_dict[agent] = costs
        if len(config) > 6:
            on_off = config[5]
            with open(f'{root_dir}/{agent}/costs_{on_off}', 'rb') as file :
                costs = pickle.load(file)
                costs_dict[agent].extend(costs)
        n_episodes = max(len(costs_dict[agent]), n_episodes)
        
    costs_dict['optimal'] = [optimal_c]* n_episodes
    costs_dict['human'] = [evaluate(FixedSwitchingHuman(), [human],eval_set, n_try=1) for _ in range(n_episodes)]

    df = pd.DataFrame({k:pd.Series(v) for k,v in costs_dict.items()} )
    df.plot(style='.-')
