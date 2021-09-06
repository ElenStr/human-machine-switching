import pickle
import numpy as np
import matplotlib as mpl
import pandas as pd
import matplotlib.pyplot as plt

from experiments.experiments import evaluate
from agent.switching_agents import FixedSwitchingHuman

def plot_performance(root_dir, eval_set, agents, optimal_c=None, human_cost_flat=True, human=None):
    costs_dict = {r'$\textsc{Human}$': [],r'$\textsc{Machine}$':[], r'$\textsc{FixSwitch}$':[], r'$\textsc{Triage}$':[],r'$\textsc{Opt}$':[]}
    ratios_dict = {}
    mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath,amsfonts}'
    mpl.rcParams['axes.formatter.use_mathtext'] = True
    plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 28,
    "figure.figsize":(9,7)
})
    agent_name = {'switch': r'$\textsc{Triage}$', 'fxd':r'$\textsc{FixSwitch}$', 'auto':r'$\textsc{Machine}$'}
    n_episodes = 0
    for agent in agents:
        config = agent.split('_')
        on_off = config[np.argwhere(np.array(config)=='We' ).flatten()[0] + 1].strip('T')
        name = agent_name[agent.split('V')[0]]
        with open(f'{root_dir}/{agent}/costs_{on_off}', 'rb') as file :
            costs = pickle.load(file)
            costs_dict[name] = costs
        try:
           with open(f'{root_dir}/{agent}/ratios_{on_off}', 'rb') as file :
            ratios = pickle.load(file)
            ratios_dict[name] = ratios
        except:
            pass 
        if 'on' in config and ('off' in config or 'offT' in config):
            try:
                with open(f'{root_dir}/{agent}/costs_on', 'rb') as file :
                    costs = pickle.load(file)
                    costs_dict[name].extend(costs)
            except:
                pass
            try:
                with open(f'{root_dir}/{agent}/ratios_on', 'rb') as file :
                    ratios = pickle.load(file)
                    ratios_dict[name].extend(ratios)
            except:
                pass
        n_episodes = max(len(costs_dict[name]), n_episodes)

    if optimal_c:    
        costs_dict[r'$\textsc{Opt}$'] = [optimal_c]* n_episodes
    if human:
        human_only = FixedSwitchingHuman()
        if human_cost_flat:
            human_c = evaluate(human_only, [human],eval_set, n_try=3)[0]
            costs_dict[r'$\textsc{Human}$'] = [human_c]*n_episodes 
        else:
            costs_dict[r'$\textsc{Human}$'] = [evaluate(human_only, [human],eval_set, n_try=1)[0] for _ in range(n_episodes)]
    df = pd.DataFrame({k:pd.Series(v) for k,v in costs_dict.items()} )
    df_ratios = pd.DataFrame({k:pd.Series(v) for k,v in ratios_dict.items()} )
    if ratios_dict:
        ax1 = df_ratios.plot(style=['--','-.' ], lw=4)#.legend(loc='lower left', bbox_to_anchor=(1.0, 0.5))
        ax1.legend(fontsize=11)
        ax1.set_xlabel(r'Number of episodes ($\times$ 1000)')


    ax = df.plot(style=['-.','--',':','-', '-|'], markevery=20,lw=4,ms=10,color=[ r'#9467bd',r'#2ca02c',r'#ff7f0e',r'#1f77b4', r'#e377c2'])#.legend(loc=', bbox_to_anchor=(1.0, 0.5))
    ax.legend(fontsize=24, loc='upper right')
  
    ax.set_xlabel(r'Number of episodes ($\times$ $1000$)')
    ax.set_ylabel(r'Average trajectory cost')
    if '2' in config[0] :
        plt.yticks(list(plt.yticks()[0]) + [50])    

    if '7' in config[0]:
        plt.xticks(list(range(0,260, 50)))
    scen = {'2':1, '3':2,'7':3}
    plt.savefig(f'C:/Users/user/Desktop/final_plots/synthetic{scen[config[0][-1]]}_costs_new.pdf')
    return df, df_ratios
