from utils import *
from solver import *

def partA():
    for path_prefix in [
                "envs/known_envs/doorkey-5x5-normal",
                "envs/known_envs/doorkey-6x6-direct",
                "envs/known_envs/doorkey-6x6-normal",
                "envs/known_envs/doorkey-6x6-shortcut",
                "envs/known_envs/doorkey-8x8-direct",
                "envs/known_envs/doorkey-8x8-normal",
                "envs/known_envs/doorkey-8x8-shortcut",
                "envs/known_envs/example-8x8"]:
        solver = DoorKeySolver(path_prefix+".env")
        solver.solve()
        _, seq = solver.result()
        draw_gif_from_seq(seq, solver.env, path_prefix+".gif")

def partB():
    env_folder = "envs/random_envs"
    env, info, env_path = load_all_random_env(env_folder)

if __name__ == "__main__":
    # example_use_of_gym_env()
    # partA()
    partB()

