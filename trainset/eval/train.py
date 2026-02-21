
from wrappers.recurrentppo import RecurrentPPOInferenceModel
from wrappers.randompolicy import RandomPolicyInferenceModel
from wrappers.rppo_onnx import RecurrentPPOONNXInferenceModel

from eval_env import EvalVectorEnv
from src.clasher.gym_env import ClashRoyaleGymEnv
from src.clasher.vec_model import VecInferenceModel
from src.clasher.model_state import State, ONNXRPPOState
import numpy as np
"""
This script evaluates the trained model for n steps

"""



if __name__ == "__main__":

    DECK = ["Cannon", "Fireball", "HogRider", "IceGolemite", "IceSpirits", "Musketeer", "Skeletons", "Log"]
    NUM_ENVS = 10
    NUM_EPISODES = 100
    ONNX_MODEL_DIR = "train/recurrentppo.onnx"
    MODEL_DIR = "train/recurrentppo_1lr_checkpoint_7.zip"

    RANDOM_MODEL = RandomPolicyInferenceModel(no_op_pct=50/51) #plays a move about every 300 steps
    # RPPO_ONNX_MODEL = VecInferenceModel(RecurrentPPOONNXInferenceModel(ONNX_MODEL_DIR, deterministic=True))
    RPPO_MODEL = VecInferenceModel(RecurrentPPOInferenceModel(MODEL_DIR, deterministic=True))
    OPPONENT_POLICIES = [RANDOM_MODEL]
    OPPONENT_STATES = [None] * len(OPPONENT_POLICIES)
    
    #RPPO State
    # onnx_state_template = ONNXRPPOState()
    rppo_state_template = None
    eval_env = EvalVectorEnv(num_envs=NUM_ENVS, opponent_policies=OPPONENT_POLICIES, opponent_states=OPPONENT_STATES, initial_state=rppo_state_template, deck0 = DECK, deck1 = DECK)
    eval_env.evaluate(model=RPPO_MODEL, num_episodes=NUM_EPISODES)






