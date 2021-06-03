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


def plot_performance(root_dir, eval_set, agents, optimal_c=None, human_cost_flat=True, human=None):
    costs_dict = {}
    ratios_dict = {}
    n_episodes = 0
    for agent in agents:
        config = agent.split('_')
        on_off = config[3]
        with open(f'{root_dir}/{agent}/costs_{on_off}', 'rb') as file :
            costs = pickle.load(file)
            costs_dict[agent] = costs
        try:
           with open(f'{root_dir}/{agent}/ratios_{on_off}', 'rb') as file :
            ratios = pickle.load(file)
            ratios_dict[agent] = ratios
        except:
            pass 
        if 'on' in config and 'off' in config:
            with open(f'{root_dir}/{agent}/costs_on', 'rb') as file :
                costs = pickle.load(file)
                costs_dict[agent].extend(costs)
            try:
                with open(f'{root_dir}/{agent}/ratios_on', 'rb') as file :
                    ratios = pickle.load(file)
                    ratios_dict[agent].extend(ratios)
            except:
                pass
        n_episodes = max(len(costs_dict[agent]), n_episodes)

    if optimal_c:    
        costs_dict['optimal'] = [optimal_c]* n_episodes
    if human:
        human_only = FixedSwitchingHuman()
        if human_cost_flat:
            human_c = evaluate(human_only, [human],eval_set, n_try=3)[0]
            costs_dict['human'] = [human_c]*n_episodes 
        else:
            costs_dict['human'] = [evaluate(human_only, [human],eval_set, n_try=1)[0] for _ in range(n_episodes)]
    df = pd.DataFrame({k:pd.Series(v) for k,v in costs_dict.items()} )
    if ratios_dict:
        df_ratios = pd.DataFrame({k:pd.Series(v) for k,v in ratios_dict.items()} )
        df_ratios.plot(style='.-')


    df.plot(style='.-')
    return df, df_ratios
