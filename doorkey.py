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
        env, info = load_env(path_prefix+".env")
        solver = DoorKeySolver_1(env, info) # had to pass in env for wall check
        solver.solve()
        _, seq = solver.query(solver.gen_init_state())
        draw_gif_from_seq(seq, solver.env, path_prefix+".gif")

def partB():
    # offline solve policy
    solver = DoorKeySolver_2() 
    solver.solve() 
    # online lookup policy
    print("obtained policy, writing out")
    env_folder = "envs/random_envs"
    for env, info, path in load_all_random_env(env_folder):
        _, seq = solver.query(solver.gen_init_state(info))
        draw_gif_from_seq(seq, env, path[:-4]+".gif")

if __name__ == "__main__":
    partA()
    partB()
