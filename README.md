# Reinforcement Learning Under Algorithmic Triage

## Requirements
 To install requirements:

 ```setup
 pip install -r requirements.txt
 ```
## Run experiments
To run experiments:
1. Specify the configuration in `config.py`. **Note:** *Scenario I* in paper corresponds to `setting = 2` in code, *Scenario II* in paper corresponds to `setting = 
3` in code,*Scenario III* in paper corresponds to `setting = 7` in code. To choose method  MACHINE , FIXSWITCH , TRIAGE in paper, set `agent` to `auto`, `fxd`, `switch` respectively.
2. Run: ``` python \run_config.py```

## Outputs and results

Running the above will produce the below:
1. `outputs/` directory, which will contain the online training grids, the test grids and the recorded human trajectories and agent (the last two under `outputs/trajectories/`).
2. `results/` directory, which will contain the experiment results directory with the saved agents and metrics.
3. experiment output and error log files <experiment_name>.out and <experiment_name>\_err.out.

The experiment name's stucture is:

<*method*>V<*scenario*><*env*>\_b<*batch_size*>\_<*entropy*>e

\_off<*true_human*>\_D<*off_grids*>K<*n_try*>R_on\_D<*episodes*>K\_h<*human_noise*>\_run<*#run*>

With the following interpretation:
* <*method*> is one of `auto`, `fxd`, `switch`
* V is a separator to facilitate ploting
* <*scenario*> is the value of `setting` in `config.py` (2 --> *Scenario I*, 3 --> *Scenario II*, 7 --> *Scenario III*)
* <*env*> is the name of the environment type as defined in `scen_postfix` in `config.py` ('_scGen')
* <*batch_size*> value of `batch_size` in `config.py` (always 1)
* <*entropy*> is 'W' if entropy regularization is used in online 'N' otherwise ('W')
* <*true_human*> is 'T' if there is access to true human policy or 'F' otherwise ('T')
* <*off_grids*> is the number of grids in which we recorded the human trajectories (in thousands) (60.0)
* <*n_try*> is the number of recorded human rollouts (human trajectories) per grid (1)
* <*episodes*> is the number of online training episodes (100.0 for *Scenario I* and *Scenario II* and 200.0 for *Scenario III*)
* <*human_noise*> is the probability of the human acting at random (always 0.0)
* <*#run*> is the number of run (0-9 with the given configuration, one run per seed)
 

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
