# Learning to Switch Between Machines and Humans

## Requirements
 To install requirements:

 ```setup
 pip install -r requirements.txt
 ```

## Code structure (Training and Evaluation)

 - `agents/`
     - `agent/agents.py` contains human/machine action policies, including UCRL2 algorithm.

     - `agent/switching_agents.py` contains implementation of UCRL2-MC, and UCRL2 for switching agents.

 - `environments/` contains the code to produce the episodic MDP for the lane driving environment used in the paper.

 - `experiments/` contains the single team and multiple teams experiment.

## Results

See the notebook `plots.ipynb` to reproduce the results in the paper:

```notebook
jupyter notebook plots.ipynb
```
