path to battle logic: py-clash-bot/pyclashbot/bot/fight.py

TODO:
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
