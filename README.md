# Reinforcement Learning Under Algorithmic Triage

## Requirements
 To install requirements:

 ```setup
 pip install -r requirements.txt
 ```
## Run experiments
To run experiments:
1. Specify the configuration in `config.py`. **Note:** *Scenario I* in paper corresponds to `setting = 2` in code, *Scenario II* in paper corresponds to `setting = 
3` in code,*Scenario III* in paper corresponds to `setting = 7` in code. To choose method  MACHINE , FIxSWITCH , TRIAGE in paper, set `agent` to {`auto`, `fxd`, `switch`} respectively.
2. Run: ``` python \run_config.py```
 

## Code structure of key components

 - `agents/`
     - `agent/agents.py` contains various actor policies (human, machine, random) and an agent that acts optimally given a specific episode horizon.

     - `agent/switching_agents.py` contains implementation of various switching agents (human_only, machine_only, switching (triage)).

 - `environments/` contains the code to produce lane driving environment used in the paper as well as other synthetic lane driving environment types.

 - `experiments/`
     - `experiments/experiments.py` contains the main 2-stage training loop with evaluation and saving of intermediate results as specified in `config.py`
     - `experiments/utils.py` contains the implementation of the offline and the online algorithms per episode.



## Plots

See the notebook `plot.ipynb` to produce quantitative and qualitative plots.

```notebook
jupyter notebook plot.ipynb
```
