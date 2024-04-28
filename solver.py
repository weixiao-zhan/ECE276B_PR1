from utils import *
from minigrid.core.world_object import Wall
from tqdm import tqdm
from typing import Tuple

class XY:
    def __init__(self, x = 0, y = 0):
        self.x = x
        self.y = y
    
    def init(self, x, y):
        self.x = x
        self.y = y
        return self
    
    def init_from(self, other):
        self.x = other.x
        self.y = other.y
        return self
    
    def __add__(self, other):
        if not isinstance(other, XY):
            return NotImplemented
        return XY(self.x+other.x, self.y+other.y)
    
    def __mul__(self, dir):
        if dir == TL:
            return XY(self.y, - self.x)
        elif dir == TR:
            return XY(- self.y, self.x)
        else:
            raise ValueError() 
        
    def __eq__(self, other: "XY"):
        return self.x == other.x and self.y == other.y

    def __repr__(self):
        return (f"({self.x}, {self.y})")
    
    def __hash__(self):
        return hash((self.x, self.y))

class State:
    def __init__(self, door_num): 
        """
        n: num of doors
        """
        self.agent_pos = XY()
        self.agent_dir = XY()
        self.goal_pos  = XY()
        
        self.key_pos   = XY()
        self.key_carrying = False

        self.door_num = door_num
        self.door_pos  = tuple([XY() for _ in range(door_num)])
        self.door_is_open = tuple([False for _ in range(door_num)])

    def init(self, agent_pos: XY, agent_dir: XY, goal_pos: XY, 
             key_pos: XY, key_carrying: bool, 
             door_pos: Tuple[XY, ...], door_is_open: Tuple[bool, ...]):
        self.agent_pos.init_from(agent_pos)
        self.agent_dir.init_from(agent_dir)
        self.goal_pos.init_from(goal_pos)

        self.key_pos.init_from(key_pos)
        self.key_carrying = key_carrying
        
        for i in range(self.door_num):
            self.door_pos[i].init_from(door_pos[i])
        self.door_is_open = door_is_open
        return self

    def init_from_state(self, other: "State"):
        self.agent_pos.init_from(other.agent_pos)
        self.agent_dir.init_from(other.agent_dir)
        self.goal_pos.init_from(other.goal_pos)
        self.key_pos.init_from(other.key_pos)
        self.key_carrying = other.key_carrying
        for i in range(self.door_num):
            self.door_pos[i].init_from(other.door_pos[i])
        self.door_is_open = other.door_is_open
        return self

    def __hash__(self):
            return hash((self.agent_pos, self.agent_dir, 
                         self.goal_pos, 
                         self.key_pos, self.key_carrying, 
                         self.door_pos, self.door_is_open))

    def __eq__(self, other):
        if not isinstance(other, State):
            return NotImplemented
        return self.__hash__() == other.__hash__()
    
    def __repr__(self) -> str:
        return f"State:{self.__hash__()}\n\tAgent:{self.agent_pos}, {self.agent_dir}\n\tGoal:{self.goal_pos}\n\tKey:{self.key_pos}, {self.key_carrying}\n\tDoor:{self.door_pos},{self.door_is_open}\n"

class DoorKeySolver:
    def __init__(self, env_path) -> None:
        self.env, self.info = load_env(env_path)
    
    def is_wall(self, pos):
        return isinstance(self.env.get_wrapper_attr('grid').get(pos.x, pos.y), Wall)

    def motion_model(self, state: State, control):
        new_state = State(1).init_from_state(state)
        
        if control == ST: # stay
            if state.agent_pos == state.goal_pos:
                return new_state, 0
            else:
                return None, float("inf")
        
        if control == TL: # Turn Left
            new_state.agent_dir = state.agent_dir * TL
            return new_state, 1
        
        if control == TR: # Turn Right
            new_state.agent_dir = state.agent_dir * TR
            return new_state, 1

        front_pos = state.agent_pos + state.agent_dir

        if control == MF: # Move Forward
            if self.is_wall(front_pos) or \
                any(front_pos == state.door_pos[i] and not state.door_is_open[i] 
                    for i in range(state.door_num)):
                return None, float("inf")
            new_state.agent_pos = front_pos
            return new_state, 2

        if control == PK: # Pickup Key
            if (front_pos != state.key_pos):
                return None, float("inf")
            new_state.key_carrying = True
            return new_state, 1

        if control == UD: # Unlock Door
            for i in range(state.door_num):
                if front_pos == state.door_pos[i] and not state.door_is_open[i] and state.key_carrying:
                    tmp_list = list(state.door_is_open)
                    tmp_list[i] = True
                    new_state.door_is_open = tuple(tmp_list)
                    new_state.key_carrying = False
                    return new_state, 1
        return None, float("inf")
    
    def terminate_cost(self, state:State):
        if state.agent_pos == state.goal_pos:
            return 0
        return float("inf")
    
    def iter_state_space(self):
        key_pos = XY(*self.info["key_pos"])
        door_pos = XY(*self.info["door_pos"])
        goal_pos = XY(*self.info["goal_pos"])
        for x in range(self.info["height"]):
            for y in range(self.info["width"]):
                agent_pos = XY(x, y)
                if self.is_wall(agent_pos):
                    continue
                for agent_dir in [XY(0,1), XY(1,0),XY(0,-1),XY(-1,0)]:
                    yield State(1).init(agent_pos, agent_dir, goal_pos, key_pos, False, (door_pos,), (False,))
                    yield State(1).init(agent_pos, agent_dir, goal_pos, key_pos, True, (door_pos,), (False,))
                    yield State(1).init(agent_pos, agent_dir, goal_pos, key_pos, False, (door_pos,), (True,))
                    yield State(1).init(agent_pos, agent_dir, goal_pos, key_pos, True, (door_pos,), (True,)) # not possible

    def iter_control_space(self):
        for u in [ST,MF,TL,TR,PK,UD]:
            yield u

    def solve(self, time_horizon = 0):
        if time_horizon > 0:
            self.time_horizon = time_horizon
        else:
            self.time_horizon = (self.info["height"]-2) * (self.info["width"]-2) * 4 # ? is this enough?
        print("solve: time_horizon",self.time_horizon)
        self.V_t = dict()
        self.Pi_t = dict()

        V = dict()
        for state in self.iter_state_space():
            V[state] = self.terminate_cost(state)
        self.V_t[self.time_horizon] = V

        for t in tqdm(range(self.time_horizon-1,-1,-1)):
            V_next = self.V_t[t + 1]  # Get the value function of the next time step
            V = dict()
            Pi = dict()
            for state in self.iter_state_space():
                # print(state.agent_pos, state.agent_dir)
                best_cost = float('inf')
                best_control = None
                for control in self.iter_control_space():
                    new_state, stage_cost = self.motion_model(state, control)
                    if new_state is not None:
                        # print(f"* --{control_str[control]}({stage_cost})--> {new_state.agent_pos}, {new_state.agent_dir}")
                        cost = stage_cost + V_next.get(new_state, float("inf"))
                    else:
                        cost = float("inf")
                    if cost < best_cost:
                        best_cost = cost
                        best_control = control
                if best_control is not None:
                    # print(f"** best: {control_str[best_control]}({best_cost})")
                    V[state] = best_cost
                    Pi[state] = best_control
            
            self.V_t[t] = V
            self.Pi_t[t] = Pi

    def result(self):

        state = State(1).init(
            XY(
                self.info["init_agent_pos"][0],
                self.info["init_agent_pos"][1]),
            XY(
                self.info["init_agent_dir"][0],
                self.info["init_agent_dir"][1]),
            XY(
                self.info["goal_pos"][0],
                self.info["goal_pos"][1]),
            XY(
                self.info["key_pos"][0],
                self.info["key_pos"][1]),
            False,
            (
                XY(
                    self.info["door_pos"][0],
                    self.info["door_pos"][1]
                ),
            ),
            (
                self.env.get_wrapper_attr('grid').get(self.info["door_pos"][0], self.info["door_pos"][1]).is_open,
            )
        )
        
        total_cost = self.V_t[0][state]
        controls = []
        for t in range(0, self.time_horizon):
            best_control = self.Pi_t[t][state]
            if best_control == ST:
                break
            if state is None:
                raise TypeError("should never happen")
            controls.append(best_control)
            state, _ = self.motion_model(state, best_control)
        print(f"result: optimal cost: {total_cost}\noptimal control policy: {[control_str[control] for control in controls]}")
        return total_cost, controls