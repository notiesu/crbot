path to battle logic: py-clash-bot/pyclashbot/bot/fight.py

ISSUE:
GAME REPLAYS TERMINATING BEFORE SCHEDULED STEPS. THIS COULD INDICATE A MISMATCH IN THE GAME ENVIRONMENTS AND INFERENCE LOOPS.

They play the latest policy against itself for 80% of games, and play against older
policies for 20% of games (for details of opponent sampling, see Appendix N). The rollout machines
run the game engine but not the policy; they communicate with a separate pool of GPU machines
which run forward passes in larger batches of approximately 60. These machines frequently poll the
controller to gather the newest parameters.

These experiments on the early part of training indicate that high quality data matters even
more than compute consumed; small degradations in data quality have severe effects on learning.
Full details of the experiment setup can be found in Appendix M.


TODO:
CHANGE BATCH SIZE TO N_ENVS * AVG_NUM_TIMESTEPS

Notes:

Some RL Learnings:

Gymnasium environment - standard interface for RL environments in python. If someone wanted to train an RL bot on some game, you create it as a gymnasium environment to standardize the behavior/pipeline for training models.

Vectorized Env - parallel copies of the RL environment for batching observations and faster traaining.