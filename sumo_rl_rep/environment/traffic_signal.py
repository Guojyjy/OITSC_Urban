import os
import sys
from typing import Callable, List, Union

import sumolib

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import traci
import numpy as np
from gymnasium import spaces

from sumo_rl_rep.util.data_processing import Dict


class TrafficSignal:
    """
    This class represents a Traffic Signal of an intersection
    It is responsible for retrieving information and changing the traffic phase using Traci API

    IMPORTANT: It assumes that the traffic phases defined in the .net file are of the form:
        [green_phase, yellow_phase, green_phase, yellow_phase, ...]
    Currently it is not supporting all-red phases (but should be easy to implement it).

    Default observation space is a vector R^(#greenPhases + 2 * #lanes)
    s = [current phase one-hot encoded, density for each lane, queue for each lane]
    You can change this by modifing self.observation_space and the method _compute_observations()

    Action space is which green phase is going to be open for the next delta_time seconds
    """
    # Default min gap of SUMO (see https://sumo.dlr.de/docs/Simulation/Safety.html). Should this be parameterized? 
    MIN_GAP = 2.5

    def __init__(self,
                 env,
                 ts_id: List[str],
                 delta_time: int,
                 yellow_time: int,
                 min_green: int,
                 max_green: int,
                 begin_time: int,
                 reward_fn: Union[str, Callable],
                 sumo):
        self.id = ts_id
        self.env = env
        self.delta_time = delta_time
        self.yellow_time = yellow_time
        self.min_green = min_green
        self.max_green = max_green
        self.green_phase = 0
        self.is_yellow = False
        self.time_since_last_phase_change = 0
        self.next_action_time = begin_time
        self.last_measure = 0.0
        self.last_reward = None
        self.reward_fn = reward_fn
        self.sumo = sumo

        if type(self.reward_fn) is str:
            if self.reward_fn in TrafficSignal.reward_fns.keys():
                self.reward_fn = TrafficSignal.reward_fns[self.reward_fn]
            else:
                raise NotImplementedError(f'Reward function {self.reward_fn} not implemented')

        if isinstance(self.env.observation_fn, Callable):
            self.observation_fn = self.env.observation_fn
        else:
            if self.env.observation_fn in TrafficSignal.observation_fns.keys():
                self.observation_fn = TrafficSignal.observation_fns[self.env.observation_fn]
            else:
                raise NotImplementedError(f'Observation function {self.env.observation_fn} not implemented')

        self.build_phases()

        self.lanes = list(
            dict.fromkeys(self.sumo.trafficlight.getControlledLanes(self.id)))  # Remove duplicates and keep order
        self.out_lanes = [link[0][1] for link in self.sumo.trafficlight.getControlledLinks(self.id) if link]
        self.out_lanes = list(set(self.out_lanes))
        self.lanes_lenght = {lane: self.sumo.lane.getLength(lane) for lane in self.lanes + self.out_lanes}

        self.observation_space = spaces.Box(
            low=np.zeros(self.num_green_phases + 1 + 2 * len(self.lanes), dtype=np.float32),
            high=np.ones(self.num_green_phases + 1 + 2 * len(self.lanes), dtype=np.float32))
        self.discrete_observation_space = spaces.Tuple((
            spaces.Discrete(self.num_green_phases),  # Green Phase
            spaces.Discrete(2),  # Binary variable active if min_green seconds already elapsed
            *(spaces.Discrete(10) for _ in range(2 * len(self.lanes)))  # Density and stopped-density for each lane
        ))
        self.action_space = spaces.Discrete(self.num_green_phases)

    def build_phases(self):
        phases = self.sumo.trafficlight.getAllProgramLogics(self.id)[0].phases
        if self.env.fixed_ts:
            self.num_green_phases = len(
                phases) // 2  # Number of green phases == number of phases (green+yellow) divided by 2
            return

        self.green_phases = []
        self.yellow_dict = {}
        for phase in phases:
            state = phase.state
            if 'y' not in state and (state.count('r') + state.count('s') != len(state)):
                self.green_phases.append(self.sumo.trafficlight.Phase(60, state))
        self.num_green_phases = len(self.green_phases)
        self.all_phases = self.green_phases.copy()

        for i, p1 in enumerate(self.green_phases):
            for j, p2 in enumerate(self.green_phases):
                if i == j: continue
                yellow_state = ''
                for s in range(len(p1.state)):
                    if (p1.state[s] == 'G' or p1.state[s] == 'g') and (p2.state[s] == 'r' or p2.state[s] == 's'):
                        yellow_state += 'y'
                    else:
                        yellow_state += p1.state[s]
                self.yellow_dict[(i, j)] = len(self.all_phases)
                self.all_phases.append(self.sumo.trafficlight.Phase(self.yellow_time, yellow_state))

        programs = self.sumo.trafficlight.getAllProgramLogics(self.id)
        logic = programs[0]
        logic.type = 0
        logic.phases = self.all_phases
        self.sumo.trafficlight.setProgramLogic(self.id, logic)
        self.sumo.trafficlight.setRedYellowGreenState(self.id, self.all_phases[0].state)

    @property
    def time_to_act(self):
        return self.next_action_time == self.env.sim_step

    def update(self):
        self.time_since_last_phase_change += 1
        if self.is_yellow and self.time_since_last_phase_change == self.yellow_time:
            # self.sumo.trafficlight.setPhase(self.id, self.green_phase)
            self.sumo.trafficlight.setRedYellowGreenState(self.id, self.all_phases[self.green_phase].state)
            self.is_yellow = False

    def set_next_phase(self, new_phase):
        """
        Sets what will be the next green phase and sets yellow phase if the next phase is different than the current

        :param new_phase: (int) Number between [0..num_green_phases] 
        """
        new_phase = int(new_phase)
        if self.green_phase == new_phase or self.time_since_last_phase_change < self.yellow_time + self.min_green:
            # self.sumo.trafficlight.setPhase(self.id, self.green_phase)
            self.sumo.trafficlight.setRedYellowGreenState(self.id, self.all_phases[self.green_phase].state)
            self.next_action_time = self.env.sim_step + self.delta_time
        else:
            # self.sumo.trafficlight.setPhase(self.id, self.yellow_dict[(self.green_phase, new_phase)])  # turns yellow
            self.sumo.trafficlight.setRedYellowGreenState(self.id, self.all_phases[
                self.yellow_dict[(self.green_phase, new_phase)]].state)
            self.green_phase = new_phase
            self.next_action_time = self.env.sim_step + self.delta_time
            self.is_yellow = True
            self.time_since_last_phase_change = 0

    def compute_observation(self):
        return self.observation_fn(self)

    def compute_reward(self):
        self.last_reward = self.reward_fn(self)
        return self.last_reward

    def _pressure_reward(self):
        return -self.get_pressure()

    def _average_speed_reward(self):
        return self.get_average_speed()

    def _queue_reward(self):
        return -self.get_total_queued()

    def _diff_waiting_time_reward(self):
        ts_wait = sum(self.get_accumulated_waiting_time_per_lane()) / 100.0
        reward = self.last_measure - ts_wait
        self.last_measure = ts_wait
        return reward

    def _observation_fn_default(self):
        phase_id = [1 if self.green_phase == i else 0 for i in range(self.num_green_phases)]  # one-hot encoding
        min_green = [0 if self.time_since_last_phase_change < self.min_green + self.yellow_time else 1]
        density = self.get_lanes_density()
        queue = self.get_lanes_queue()
        observation = np.array(phase_id + min_green + density + queue, dtype=np.float32)
        return observation

    def get_accumulated_waiting_time_per_lane(self):
        wait_time_per_lane = []
        for lane in self.lanes:
            veh_list = self.sumo.lane.getLastStepVehicleIDs(lane)
            wait_time = 0.0
            for veh in veh_list:
                veh_lane = self.sumo.vehicle.getLaneID(veh)
                acc = self.sumo.vehicle.getAccumulatedWaitingTime(veh)
                if veh not in self.env.vehicles:
                    self.env.vehicles[veh] = {veh_lane: acc}
                else:
                    self.env.vehicles[veh][veh_lane] = acc - sum(
                        [self.env.vehicles[veh][lane] for lane in self.env.vehicles[veh].keys() if lane != veh_lane])
                wait_time += self.env.vehicles[veh][veh_lane]
            wait_time_per_lane.append(wait_time)
        return wait_time_per_lane

    def get_average_speed(self):
        avg_speed = 0.0
        vehs = self._get_veh_list()
        if len(vehs) == 0:
            return 1.0
        for v in vehs:
            avg_speed += self.sumo.vehicle.getSpeed(v) / self.sumo.vehicle.getAllowedSpeed(v)
        return avg_speed / len(vehs)

    def get_pressure(self):
        return sum(self.sumo.lane.getLastStepVehicleNumber(lane) for lane in self.out_lanes) - sum(
            self.sumo.lane.getLastStepVehicleNumber(lane) for lane in self.lanes)
        # TODO: pressure
        #  return abs(sum(traci.lane.getLastStepVehicleNumber(lane) for lane in self.lanes) -
        #             sum(traci.lane.getLastStepVehicleNumber(lane) for lane in self.out_lanes))

    def get_out_lanes_density(self):
        lanes_density = [self.sumo.lane.getLastStepVehicleNumber(lane) / (
                self.lanes_lenght[lane] / (self.MIN_GAP + self.sumo.lane.getLastStepLength(lane))) for lane in
                         self.out_lanes]
        return [min(1, density) for density in lanes_density]

    def get_lanes_density(self):
        lanes_density = [self.sumo.lane.getLastStepVehicleNumber(lane) / (
                self.lanes_lenght[lane] / (self.MIN_GAP + self.sumo.lane.getLastStepLength(lane))) for lane in
                         self.lanes]
        return [min(1, density) for density in lanes_density]

    def get_lanes_queue(self):
        lanes_queue = [self.sumo.lane.getLastStepHaltingNumber(lane) / (
                self.lanes_lenght[lane] / (self.MIN_GAP + self.sumo.lane.getLastStepLength(lane))) for lane in
                       self.lanes]
        return [min(1, queue) for queue in lanes_queue]

    def get_total_queued(self):
        return sum(self.sumo.lane.getLastStepHaltingNumber(lane) for lane in self.lanes)

    def _get_veh_list(self):
        veh_list = []
        for lane in self.lanes:
            veh_list += self.sumo.lane.getLastStepVehicleIDs(lane)
        return veh_list

    @classmethod
    def register_reward_fn(cls, fn):
        if fn.__name__ in cls.reward_fns.keys():
            raise KeyError(f'Reward function {fn.__name__} already exists')

        cls.reward_fns[fn.__name__] = fn

    @classmethod
    def register_observation_fn(cls, fn):
        if fn.__name__ in cls.observation_fns.keys():
            raise KeyError(f'Observation function {fn.__name__} already exists')

        cls.observation_fns[fn.__name__] = fn

    reward_fns = {
        'diff-waiting-time': _diff_waiting_time_reward,
        'average-speed': _average_speed_reward,
        'queue': _queue_reward,
        'pressure': _pressure_reward
    }

    observation_fns = {
        'default': _observation_fn_default
    }


#      ------- OITSC_Urban -------

class TrafficSignalUrban:
    """
            This class represents a Traffic Signal of an intersection in urban scenarios
            It is responsible for retrieving information and changing the traffic phase using Traci API
    """

    # Default min gap of SUMO (see https://sumo.dlr.de/docs/Simulation/Safety.html). Should this be parameterized?
    MIN_GAP = 2.5

    def __init__(self,
                 env,
                 ts_id: str,
                 phases: list,
                 delta_time: int,
                 yellow_time: int,
                 min_green: int,
                 max_green: int,
                 sumo):
        self.env = env
        self.id = ts_id
        self.phases = phases
        self.delta_time = delta_time
        self.yellow_time = yellow_time
        self.min_green = min_green
        self.max_green = max_green
        self.sumo = sumo

        self.lanes = list(
            dict.fromkeys(self.sumo.trafficlight.getControlledLanes(self.id)))  # Remove duplicates and keep order
        self.out_lanes = [link[0][1] for link in self.sumo.trafficlight.getControlledLinks(self.id) if link]
        self.out_lanes = list(set(self.out_lanes))
        self.lanes_length = {lane: self.sumo.lane.getLength(lane) for lane in self.lanes + self.out_lanes}

        self.is_yellow = False
        self.time_since_last_phase_change = 0
        self.num_green_phases = self._count_green_phases(self.phases)  # TODO: needed?

    def update(self):
        # start after warmup
        # TODO: not accurate in time_since_last_phase_change, cuz not starting from simulation beginning
        self.time_since_last_phase_change += 1
        if self.sumo.simulation.getTime() < 110:
            now_state = self.sumo.trafficlight.getRedYellowGreenState(self.id)
            if 'y' in now_state:
                self.is_yellow = True
        else:
            # check the yellow light switching automatically
            if self.is_yellow and self.time_since_last_phase_change == self.yellow_time:
                self.is_yellow = False
                self.time_since_last_phase_change = 0

    def _count_green_phases(self, phases):
        if len(phases) % 3 == 0:  # green -> yellow -> red
            return len(phases) / 3
        if len(phases) % 2 == 0:  # green + yellow
            return len(phases) / 2

    def phase_switching(self, action):
        """
        only consider the phase switching for green light
        """
        switch = action > 0
        now_state = self.sumo.trafficlight.getRedYellowGreenState(self.id)
        state_index = self.phases.index(now_state)
        if switch and 'G' in now_state:  # proposed to switch to yellow light
            if self.time_since_last_phase_change > self.min_green:
                self.sumo.trafficlight.setPhase(self.id, state_index + 1)
                self.time_since_last_phase_change = 0
                self.is_yellow = True
        if switch and 'G' not in now_state:  # current: yellow or red
            self.is_yellow = False
        if not switch and 'G' in now_state:  # extend the current green light
            if self.time_since_last_phase_change < self.max_green - self.delta_time:
                self.sumo.trafficlight.setPhase(self.id, state_index)
                self.is_yellow = False

    def phase_switching_important_observation(self, action, proposed_phase):
        """
        only consider the phase switching for green light
        """
        switch = action > 0
        now_state = self.sumo.trafficlight.getRedYellowGreenState(self.id)
        state_index = self.phases.index(now_state)
        if switch and 'G' in now_state:  # proposed to switch to yellow light
            if proposed_phase == now_state:  # oitsc: should extend
                switch = False
            else:
                if self.time_since_last_phase_change > self.min_green:
                    self.sumo.trafficlight.setPhase(self.id, state_index + 1)
                    self.time_since_last_phase_change = 0
                    self.is_yellow = True
        if switch and 'G' not in now_state:  # current: yellow or red
            self.is_yellow = False
            next_phase_index = 0 if state_index+1 >= len(self.phases) else state_index+1
            if proposed_phase != self.phases[next_phase_index]:  # oitsc: switch to green
                self.sumo.trafficlight.setPhase(self.id, self.phases.index(proposed_phase))
                self.time_since_last_phase_change = 0
        if not switch and 'G' in now_state:  # extend the current green light
            if now_state != proposed_phase:  # oitsc: switch to yellow before the proposed one
                proposed_index = self.phases.index(proposed_phase)
                if proposed_index == 0:
                    before_index = len(self.phases)-1
                else:
                    before_index = proposed_index - 1
                if self.time_since_last_phase_change > self.min_green:
                    self.sumo.trafficlight.setPhase(self.id, before_index)
                    self.time_since_last_phase_change = 0
                    self.is_yellow = True
            else:
                if self.time_since_last_phase_change < self.max_green - self.delta_time:
                    self.sumo.trafficlight.setPhase(self.id, state_index)
                    self.is_yellow = False

    def get_phase_index(self):
        return self.sumo.trafficlight.getPhase(self.id)

    def get_lanes_density(self):
        lanes_density = [self.sumo.lane.getLastStepVehicleNumber(lane) / (
                self.lanes_length[lane] / (self.MIN_GAP + self.sumo.lane.getLastStepLength(lane))) for lane in
                         self.lanes]
        return [min(1, density) for density in lanes_density]

    def get_lanes_queue(self):
        lanes_queue = [self.sumo.lane.getLastStepHaltingNumber(lane) / (
                self.lanes_length[lane] / (self.MIN_GAP + self.sumo.lane.getLastStepLength(lane))) for lane in
                       self.lanes]
        return [min(1, queue) for queue in lanes_queue]

    def green_lanes_per_phase(self, net_file):
        green_index_per_phase = {}
        green_lanes_per_phase = {}
        for phase in self.phases:
            phase = str(phase)
            if ('G' or 'g') in phase:
                green_lanes_per_phase.update({phase: []})
                green_index_per_phase.update({phase: []})
                for i in range(len(list(phase))):
                    if list(phase)[i] == 'G' or list(phase)[i] == 'g':
                        green_index_per_phase[phase].append(i)
        for conn in sumolib.output.parse(net_file, ['connection']):
            if conn.tl:
                if conn.tl == self.id:
                    for phase, green_index in green_index_per_phase.items():
                        if int(conn.linkIndex) in green_index:  # have duplicate lanes
                            green_lanes_per_phase[phase].append(conn.attr_from+'_'+conn.fromLane)
        return green_lanes_per_phase


class TrafficSignalImplicitObservation:
    """
        This class represents a Traffic Signal of an intersection
        It is responsible for retrieving information and changing the traffic phase using Traci API
    """

    def __init__(self, env, ts_id, delta_time, yellow_time, min_green, max_green, phases):
        self.id = ts_id  # TODO: for each traffic light
        self.env = env
        self.time_on_phase = 0.0
        self.delta_time = delta_time
        self.yellow_time = yellow_time
        self.min_green = min_green
        self.max_green = max_green
        self.green_phase = 0
        self.num_green_phases = len(phases) // 2
        self.lanes = list(
            dict.fromkeys(traci.trafficlight.getControlledLanes(self.id)))  # remove duplicates and keep order
        self.out_lanes = [link[0][1] for link in traci.trafficlight.getControlledLinks(self.id)]
        self.out_lanes = list(set(self.out_lanes))

        logic = traci.trafficlight.Logic("new-program", 0, 0, phases=phases)
        traci.trafficlight.setCompleteRedYellowGreenDefinition(self.id, logic)  # TODO: why need set new progrem

    @property
    def phase(self):
        return traci.trafficlight.getPhase(self.id)

    def set_next_phase(self, new_phase):
        """
            Sets what will be the next green phase and sets yellow phase if the next phase is different from the current

            :param new_phase: (int) Number between [0..num_green_phases]
            """

        keep, give_up = self.test_keep()  # TODO: ambulance
        new_phase *= 2  # 0: NS_green; 2: WE_green
        if self.phase == new_phase or self.time_on_phase < self.min_green:  # or keep == 1) and give_up != 1:
            self.time_on_phase += self.delta_time
            self.green_phase = self.phase
        else:
            self.time_on_phase = self.delta_time - self.yellow_time
            self.green_phase = new_phase
            traci.trafficlight.setPhase(self.id, self.phase + 1)  # turns yellow

    def update_phase(self):
        """
            Change the next green_phase after it is set by set_next_phase method
            """
        traci.trafficlight.setPhase(self.id, self.green_phase)

    def get_waiting_time_per_lane(self):
        wait_time_per_lane = []
        for lane in self.lanes:
            veh_list = traci.lane.getLastStepVehicleIDs(lane)
            wait_time = 0.0
            for veh in veh_list:
                veh_lane = traci.vehicle.getLaneID(veh)
                acc = traci.vehicle.getAccumulatedWaitingTime(veh)
                if veh not in self.env.vehicles:
                    self.env.vehicles[veh] = {veh_lane: acc}
                else:
                    self.env.vehicles[veh][veh_lane] = acc - sum(
                        [self.env.vehicles[veh][lane] for lane in self.env.vehicles[veh].keys() if
                         lane != veh_lane])
                wait_time += self.env.vehicles[veh][veh_lane]
            wait_time_per_lane.append(wait_time)
        return wait_time_per_lane

    def get_pressure(self):
        return abs(sum(traci.lane.getLastStepVehicleNumber(lane) for lane in self.lanes) - sum(
            traci.lane.getLastStepVehicleNumber(lane) for lane in self.out_lanes))

    def get_out_lanes_density(self):
        vehicle_size_min_gap = 7.5  # 5(vehSize) + 2.5(minGap)
        return [
            min(1, traci.lane.getLastStepVehicleNumber(lane) / (traci.lane.getLength(lane) / vehicle_size_min_gap))
            for lane in self.out_lanes]

    def get_lanes_density(self):
        vehicle_size_min_gap = 7.5  # 5(vehSize) + 2.5(minGap)
        return [
            min(1, traci.lane.getLastStepVehicleNumber(lane) / (traci.lane.getLength(lane) / vehicle_size_min_gap))
            for lane in self.lanes]

    def get_lanes_queue(self):
        vehicle_size_min_gap = 7.5  # 5(vehSize) + 2.5(minGap)
        return [
            min(1, traci.lane.getLastStepHaltingNumber(lane) / (traci.lane.getLength(lane) / vehicle_size_min_gap))
            for lane in self.lanes]

    def get_total_queued(self):
        return sum([traci.lane.getLastStepHaltingNumber(lane) for lane in self.lanes])

    def _get_veh_list(self, p):
        veh_list = []
        for lane in self.lanes:
            veh_list += traci.lane.getLastStepVehicleIDs(lane)
        return veh_list

    # ------- Implicit Observation -------

    def get_complete_data(self):
        veh_complete_data = Dict()
        total_veh_list = []
        p_list = []
        for p in range(self.num_green_phases):
            if p not in p_list:
                p_list.append(p)
            veh_list = self._get_veh_list(p)
            total_veh_list += veh_list
            for veh in veh_list:
                veh_lane = self.get_edge_id(traci.vehicle.getLaneID(veh))
                veh_position = traci.vehicle.getLanePosition(veh)
                veh_type = traci.vehicle.getTypeID(veh)
                acc = traci.vehicle.getAccumulatedWaitingTime(veh)
                if veh not in self.env.vehicles:
                    self.env.vehicles_details[veh][veh_type][veh_position] = {veh_lane: acc}
                    # TODO: why need veh_position for this lane
                else:
                    self.env.vehicles_details[veh][veh_type][veh_position][veh_lane] = acc - sum(
                        [self.env.vehicles_details[veh][veh_type][veh_position][lane] for lane in
                         self.env.vehicles_details[veh][veh_type][veh_position].keys() if lane != veh_lane])
                    # TODO: reduce the waiting time on other lanes
                veh_complete_data[veh][veh_type][veh_position][veh_lane] = \
                    self.env.vehicles_details[veh][veh_type][veh_position][veh_lane]
                # TODO: data structure
        return veh_complete_data

    def get_position_data(self):
        veh_position_data = {}
        for p in range(self.num_green_phases):
            veh_list = self._get_veh_list(p)
            for veh in veh_list:
                veh_position = traci.vehicle.getLanePosition(veh)
                veh_position_data[veh] = veh_position
        return veh_position_data

    def get_waiting_time(self):
        veh_waiting_time = Dict()
        for p in range(self.num_green_phases):
            veh_list = self._get_veh_list(p)
            for veh in veh_list:
                veh_lane = self.get_edge_id(traci.vehicle.getLaneID(veh))
                veh_position = traci.vehicle.getLanePosition(veh)
                veh_type = traci.vehicle.getTypeID(veh)
                acc = traci.vehicle.getAccumulatedWaitingTime(veh)
                if veh not in self.env.vehicles:
                    self.env.vehicles_details[veh][veh_type][veh_position] = {veh_lane: acc}
                else:
                    self.env.vehicles_details[veh][veh_type][veh_position][veh_lane] = acc - sum(
                        [self.env.vehicles_details[veh][veh_type][veh_position][lane] for lane in
                         self.env.vehicles_details[veh][veh_type][veh_position].keys() if lane != veh_lane])
                veh_waiting_time[veh] = self.env.vehicles_details[veh][veh_type][veh_position][veh_lane]
        # TODO: same as veh_complete_data, but only need a pair {veh: waiting time}
        return veh_waiting_time

    def get_edge_id(self, lane):
        """ Get edge Id from lane Id
            :param lane: id of the lane
            :return: the edge id of the lane
            """
        return lane[:-2]

    def test_keep(self):
        # 3 cases of keep+give up: 1+0, 0+1, 0+0
        ambulance_lane_list, trailer_lane_list, fueltruck_lane_list = self._get_important_objects_lane_list()
        keep = 0
        give_up = 0
        for lane in self.lanes:
            k = lane[:-2]  # TODO: edge name
            if k in self.env.last_distance_reward:
                if self.env.last_distance_reward[k] > 0:
                    keep = 1
                if self.env.last_distance_reward[k] == 0 and k in ambulance_lane_list:
                    give_up = 1
        return keep, give_up

    def get_distance_val(self):
        veh_list_part1, veh_list_part2 = self._get_veh_list_per_edge()
        # key is the name of the road, same in veh_list_part1, veh_list_part2
        for key in veh_list_part1:
            self.env.distance_val_per_road[key] = {}
            for veh in veh_list_part1[key]:
                veh_type = traci.vehicle.getTypeID(veh)
                if veh_type == "Ambulance" or veh_type == "fueltruck" or veh_type == "trailer":
                    self.env.distance_val_per_road[key][veh] = traci.vehicle.getLanePosition(veh)
        for key in veh_list_part2:
            for veh in veh_list_part2[key]:
                veh_type = traci.vehicle.getTypeID(veh)
                if veh_type == "Ambulance" or veh_type == "fueltruck" or veh_type == "trailer":
                    self.env.distance_val_per_road[key][veh] = traci.vehicle.getLanePosition(veh)
        return self.env.distance_val_per_road

    def get_default_vehicles_distance_val(self):
        veh_list_part1, veh_list_part2 = self._get_veh_list_per_edge()
        for key in veh_list_part1:
            self.env.default_vehicles_distance_val_per_road[key] = {}
            for veh in veh_list_part1[key]:
                veh_type = traci.vehicle.getTypeID(veh)
                if veh_type == "DEFAULT_VEHTYPE":
                    self.env.default_vehicles_distance_val_per_road[key][veh] = traci.vehicle.getLanePosition(veh)
        for key in veh_list_part2:
            for veh in veh_list_part2[key]:
                veh_type = traci.vehicle.getTypeID(veh)
                if veh_type == "DEFAULT_VEHTYPE":
                    self.env.default_vehicles_distance_val_per_road[key][veh] = traci.vehicle.getLanePosition(veh)
        return self.env.default_vehicles_distance_val_per_road

    def _get_veh_list_per_edge(self):
        # TODO: why part1 and part2 for lanes to summary all edges of this intersection
        veh_list_part1 = {}
        veh_list_part2 = {}
        veh_list = {}
        cnt_part = 0
        for p in self.lanes:
            pl = p[:-2]
            if cnt_part % 2 == 0:
                veh_list_part1[pl] = traci.lane.getLastStepVehicleIDs(p)
            elif cnt_part % 2 != 0:
                veh_list_part2[pl] = traci.lane.getLastStepVehicleIDs(p)
            veh_list[pl] = traci.lane.getLastStepVehicleIDs(p)  # TODO: not used
            cnt_part += 1
        return veh_list_part1, veh_list_part2

    def get_importance_weight_val(self):
        """
        importance_weight, dict: the count of important vehicle for each road
        """
        veh_list_part1, veh_list_part2 = self._get_veh_list_per_edge()
        importance_weight_val_per_road = {}
        cnt = {}
        for key in veh_list_part1:
            cnt[key] = 0
            if cnt[key] == 0:
                importance_weight_val_per_road[key] = 0
            for veh in veh_list_part1[key]:
                veh_type = traci.vehicle.getTypeID(veh)
                if veh_type == "Ambulance" or veh_type == "fueltruck" or veh_type == "trailer":
                    importance_weight_val_per_road[key] = importance_weight_val_per_road[key] + 1
                    cnt[key] = 1
        for key in veh_list_part2:
            if cnt[key] == 0:
                importance_weight_val_per_road[key] = 0
            for veh in veh_list_part2[key]:
                veh_type = traci.vehicle.getTypeID(veh)
                if veh_type == "Ambulance" or veh_type == "fueltruck" or veh_type == "trailer":
                    importance_weight_val_per_road[key] = importance_weight_val_per_road[key] + 1
                    cnt[key] = 1
        return importance_weight_val_per_road

    def get_importance_weight_val_per_ts(self):
        """
                importance_weight, int: the count of important vehicle for each tl
        """
        veh_list_part1, veh_list_part2 = self._get_veh_list_per_edge()
        importance_weight_val_per_ts = 0
        for key in veh_list_part1:
            for veh in veh_list_part1[key]:
                veh_type = traci.vehicle.getTypeID(veh)
                if veh_type == "Ambulance" or veh_type == "fueltruck" or veh_type == "trailer":
                    importance_weight_val_per_ts = importance_weight_val_per_ts + 1
        for key in veh_list_part2:
            for veh in veh_list_part2[key]:
                veh_type = traci.vehicle.getTypeID(veh)
                if veh_type == "Ambulance" or veh_type == "fueltruck" or veh_type == "trailer":
                    importance_weight_val_per_ts = importance_weight_val_per_ts + 1
        return importance_weight_val_per_ts

    def _get_important_objects_lane_list(self):
        ambulance_lane_list = []  # [edge_name_for_ambulance], len = # veh_ambulance
        trailer_lane_list = []
        fueltruck_lane_list = []
        for p in self.lanes:
            veh_list = self._get_veh_list(p)
            for veh in veh_list:
                veh_lane = self.get_edge_id(traci.vehicle.getLaneID(veh))  # edge!
                veh_type = traci.vehicle.getTypeID(veh)
                if veh_type == "Ambulance": ambulance_lane_list.append(veh_lane)
                if veh_type == "trailer": trailer_lane_list.append(veh_lane)
                if veh_type == "fueltruck": fueltruck_lane_list.append(veh_lane)
        return ambulance_lane_list, trailer_lane_list, fueltruck_lane_list
