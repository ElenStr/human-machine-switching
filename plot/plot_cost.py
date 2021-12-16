from collections import defaultdict
import pickle
import numpy as np
import matplotlib as mpl
from numpy.core.fromnumeric import size
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from experiments.experiments import evaluate
from agent.switching_agents import FixedSwitchingHuman

def plot_performance(root_dir, eval_set, agents, optimal_c=None, human_cost_flat=True, human=None):
    # costs_dict = {r'$\textsc{Human}$': {i:[] for i in range(10) },r'$\textsc{Machine}$':{i:[] for i in range(10)}, r'$\textsc{FixSwitch}$':{i:[] for i in range(10)}, r'$\textsc{Triage}$':{i:[] for i in range(10)},r'$\textsc{Opt}$':{i: [] for i in range(10)}}
    costs_list = []
    ratios_dict = defaultdict(lambda:[])
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
    n_episodes =0
    on_size = 0
    off_size = 0
    for agent in agents:
        config = agent.split('_')
        on_off = config[np.argwhere(np.array(config)=='We' ).flatten()[0] + 1].strip('T')
        name = agent_name[agent.split('V')[0]]
        for run in range(10):
            try:
                with open(f'{root_dir}/{agent}_run{run}/costs_{on_off}', 'rb') as file :
                    costs = pickle.load(file)
                    off_size = len(costs)
                    for i, cost in enumerate(costs):
                        costs_list.append((name,run, i ,cost))

                        

                # try:
                #     with open(f'{root_dir}/{agent}_run{run}/ratios_{on_off}', 'rb') as file :
                #         ratios = pickle.load(file)
                #         ratios_dict[name][run] = ratios
                # except:
                #     pass 
                if 'on' in config and ('off' in config or 'offT' in config):
                    try:
                        with open(f'{root_dir}/{agent}_run{run}/costs_on', 'rb') as file :
                            costs = pickle.load(file)
                            on_size = len(costs)
                            for i, cost in enumerate(costs):
                                costs_list.append((name,run, off_size+i ,cost))


                            # [run].extend(costs)
                    except:
                        pass
                    # try:
                    #     with open(f'{root_dir}/{agent}_run{run}/ratios_on', 'rb') as file :
                    #         ratios = pickle.load(file)
                    #         ratios_dict[name][run].extend(ratios)
                    # except:
                    #     pass
            except:
                pass
        n_episodes = max(on_size+off_size, n_episodes)

    if optimal_c: 
        for i in range(n_episodes):
            costs_list.append((r'$\textsc{Opt}$',0, i ,optimal_c))
        # costs_dict[r'$\textsc{Opt}$'] = [[optimal_c]* n_episodes for _ in range(10)]
    if human:
        human_only = FixedSwitchingHuman()
        if human_cost_flat:
            human_c = evaluate(human_only, [human],eval_set, n_try=3)[0]
            for i in range(n_episodes):
                costs_list.append((r'$\textsc{Human}$',0, i ,human_c))
            # costs_dict[r'$\textsc{Human}$'] = [[human_c]*n_episodes for _ in range(10)]
        # else:
            # costs_dict[r'$\textsc{Human}$'] = [evaluate(human_only, [human],eval_set, n_try=1)[0] for _ in range(n_episodes)]
    costs_df = pd.DataFrame(costs_list,columns=['method', 'run','step','cost'])

    
    # df_ratios = pd.DataFrame({k:pd.Series(v) for k,v in ratios_dict.items()} )
    if ratios_dict:
        ax1 = df_ratios.plot(style=['--','-.' ], lw=4)#.legend(loc='lower left', bbox_to_anchor=(1.0, 0.5))
        ax1.legend(fontsize=11)
        ax1.set_xlabel(r'Number of episodes ($\times$ 1000)')


    # ax = df.plot(style=['-.','--',':','-', '-|'], markevery=20,lw=4,ms=10,color=[ r'#9467bd',r'#2ca02c',r'#ff7f0e',r'#1f77b4', r'#e377c2'])#.legend(loc=', bbox_to_anchor=(1.0, 0.5))
    palette = {r'$\textsc{Triage}$': r'#1f77b4', r'$\textsc{FixSwitch}$':r'#ff7f0e', r'$\textsc{Machine}$':r'#2ca02c',r'$\textsc{Human}$':r'#9467bd', r'$\textsc{Opt}$': r'#e377c2'}

    ax = sns.lineplot(data=costs_df,x="step", y="cost", hue="method",style="method", 
                        markevery=20,lw=4,ms=10, palette=palette, 
                        hue_order=[r'$\textsc{Human}$',r'$\textsc{Machine}$', r'$\textsc{FixSwitch}$', r'$\textsc{Triage}$',r'$\textsc{Opt}$'],
                        style_order=[r'$\textsc{Triage}$', r'$\textsc{Machine}$',r'$\textsc{FixSwitch}$', r'$\textsc{Opt}$',r'$\textsc{Human}$'])
    ax.legend(fontsize=24, loc='upper right')
  
    ax.set_xlabel(r'Number of episodes ($\times$ $1000$)')
    ax.set_ylabel(r'Average trajectory cost')
    if '2' in config[0] :
        plt.yticks(list(plt.yticks()[0][:-1]))    

    if '7' in config[0]:
        plt.xticks(list(range(0,260, 50)))
    scen = {'2':1, '3':2,'7':3}
    # plt.savefig(f'C:/Users/user/Desktop/final_plots/synthetic{scen[config[0][-1]]}_costs_new.pdf')
    return costs_df #, df_ratios
