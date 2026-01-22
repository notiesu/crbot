path to battle logic: py-clash-bot/pyclashbot/bot/fight.py

ISSUE:
GAME REPLAYS TERMINATING BEFORE SCHEDULED STEPS. THIS COULD INDICATE A MISMATCH IN THE GAME ENVIRONMENTS AND INFERENCE LOOPS.

TODO:
EXPOSE RecurrentPPO.logger for logging training stats
Multi-agent inference
    - Create abstraction for opponent policy input
    - Tranpose observation space as player 0 instead of player 1
    - Inverse transpose to play move as player 1
Simple training test - if we fix the opponent policy with a series of game actions, how fast do we converge on 100% win rate?
    - Log statistics - this could be good for determing how much compute to use
Check for existence of basic smart parameters:
    - Opponent cycle
    - Opponent elixir


Notes:
card_detection.py
It looks like card detection is handled by using color schemes, card classifications
It detects coordinates for each card - this can be passed on to battle logic to decide plays


Some RL Learnings:

Gymnasium environment - standard interface for RL environments in python. If someone wanted to train an RL bot on some game, you create it as a gymnasium environment to standardize the behavior/pipeline for training models.

Vectorized Env - parallel copies of the RL environment for batching observations and faster traaining.
