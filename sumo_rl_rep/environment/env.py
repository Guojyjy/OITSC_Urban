import os
import sys
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import gymnasium as gym
import numpy as np
import pandas as pd
import sumolib
import traci

from .traffic_signal import TrafficSignal, TrafficSignalUrban, TrafficSignalImplicitObservation
from gym import spaces

LIBSUMO = 'LIBSUMO_AS_TRACI' in os.environ  # TODO: how to use LIBSUMO


class SumoEnvironment(gym.Env):
    """
    SUMO Environment for Traffic Signal Control

    :param net_file: (str) SUMO .net.xml file
    :param route_file: (str) SUMO .rou.xml file
    :param out_csv_name: (Optional[str]) name of the .csv output with simulation results. If None no output is generated
    :param use_gui: (bool) Wheter to run SUMO simulation with GUI visualisation
    :param begin_time: (int) The time step (in seconds) the simulation starts -- removed?
    :param num_seconds: (int) Number of simulated seconds on SUMO. The time in seconds the simulation must end.
    :param max_depart_delay: (int) Vehicles are discarded if they could not be inserted after max_depart_delay seconds
    :param waiting_time_memory: (int) Number of seconds to remember the waiting time of a vehicle
    (see https://sumo.dlr.de/pydoc/traci._vehicle.html#VehicleDomain-getAccumulatedWaitingTime)  -- removed?
    :param delta_time: (int) Simulation seconds between actions
    :param min_green: (int) Minimum green time in a phase
    :param max_green: (int) Max green time in a phase
    :single_agent: (bool) If true, it behaves like a regular gym.Env. Else, it behaves like a MultiagentEnv
    (https://github.com/ray-project/ray/blob/master/python/ray/rllib/env/multi_agent_env.py)

    :reward_fn: (str/function/dict) String with the name of the reward function used by the agents, a reward function,
     or dictionary with reward functions assigned to individual traffic lights by their keys
    :observation_fn: (str/function) String with the name of the observation function or a callable observation function itself
    :add_system_info: (bool) If true, it computes system metrics (total queue, total waiting time, average speed) in the info dictionary
    :add_per_agent_info: (bool) If true, it computes per-agent (per-traffic signal) metrics (average accumulated waiting time, average queue) in the info dictionary
    :sumo_seed: (int/string) Random seed for sumo. If 'random' it uses a randomly chosen seed.
    :fixed_ts: (bool) If true, it will follow the phase configuration in the route_file and ignore the actions.
    :sumo_warnings: (bool) If False, remove SUMO warnings in the terminal
    :additional_sumo_cmd: (list) Additional command line arguments for SUMO

    newly added
    :param phases: (traci.trafficlight.Phase list) Traffic Signal phases definition
    :param time_to_load_vehicles: (int) Number of simulation seconds ran before learning begins
    """

    CONNECTION_LABEL = 0  # For traci multi-client support

    def __init__(
            self,
            net_file: str,
            route_file: str,
            out_csv_name: Optional[str] = None,
            use_gui: bool = False,
            begin_time: int = 0,  # TODO: begin_time needed?
            num_seconds: int = 20000,
            max_depart_delay: int = 100000,  # TODO: max_depart_delay needed?
            waiting_time_memory: int = 1000,
            time_to_teleport: int = -1,  # TODO: time_to_teleport, non-positive values disable teleporting
            delta_time: int = 5,
            yellow_time: int = 2,
            min_green: int = 5,
            max_green: int = 50,
            single_agent: bool = False,
            reward_fn: Union[str, Callable, dict] = 'diff-waiting-time',  # TODO: reward_fn skipped
            observation_fn: Union[str, Callable] = 'default',  # TODO: observation_fn skipped
            add_system_info: bool = True,  # TODO: add_system_info skipped
            add_per_agent_info: bool = True,  # TODO: add_per_agent_info skipped
            sumo_seed: Union[str, int] = 'random',
            fixed_ts: bool = False,  # TODO: need fixed_ts?
            sumo_warnings: bool = True,
            additional_sumo_cmd: Optional[str] = None,
    ) -> None:

        self._net = net_file
        self._route = route_file
        self.use_gui = use_gui
        if self.use_gui:
            self._sumo_binary = sumolib.checkBinary('sumo-gui')
            # img = self.sumo.gui.screenshot(traci.gui.DEFAULT_VIEW,
            #                          f"temp/img{self.sim_step}.jpg",
            #                          width=self.virtual_display[0],
            #                          height=self.virtual_display[1])
        else:
            self._sumo_binary = sumolib.checkBinary('sumo')

        assert delta_time > yellow_time, "Time between actions must be at least greater than yellow time."

        self.begin_time = begin_time
        self.sim_max_time = num_seconds
        self.delta_time = delta_time  # seconds on sumo at each step
        self.max_depart_delay = max_depart_delay  # Max wait time to insert a vehicle
        self.waiting_time_memory = waiting_time_memory
        # Number of seconds to remember the waiting time of a vehicle
        # (see https://sumo.dlr.de/pydoc/traci._vehicle.html#VehicleDomain-getAccumulatedWaitingTime)
        self.time_to_teleport = time_to_teleport
        self.min_green = min_green
        self.max_green = max_green
        self.yellow_time = yellow_time
        self.single_agent = single_agent
        self.reward_fn = reward_fn
        self.sumo_seed = sumo_seed
        self.fixed_ts = fixed_ts
        self.sumo_warnings = sumo_warnings
        self.additional_sumo_cmd = additional_sumo_cmd
        self.add_system_info = add_system_info
        self.add_per_agent_info = add_per_agent_info
        self.label = str(SumoEnvironment.CONNECTION_LABEL)
        SumoEnvironment.CONNECTION_LABEL += 1
        self.sumo = None

        if LIBSUMO:
            traci.start(
                [sumolib.checkBinary('sumo'), '-n', self._net])  # Start only to retrieve traffic light information
            conn = traci
        else:
            traci.start([sumolib.checkBinary('sumo'), '-n', self._net], label='init_connection' + self.label)
            conn = traci.getConnection('init_connection' + self.label)
        self.ts_ids = list(conn.trafficlight.getIDList())
        self.observation_fn = observation_fn

        if isinstance(self.reward_fn, dict):
            self.traffic_signals = {ts: TrafficSignal(self,
                                                      ts,
                                                      self.delta_time,
                                                      self.yellow_time,
                                                      self.min_green,
                                                      self.max_green,
                                                      self.begin_time,
                                                      self.reward_fn[ts],
                                                      conn) for ts in self.reward_fn.keys()}
        else:
            self.traffic_signals = {ts: TrafficSignal(self,
                                                      ts,
                                                      self.delta_time,
                                                      self.yellow_time,
                                                      self.min_green,
                                                      self.max_green,
                                                      self.begin_time,
                                                      self.reward_fn,
                                                      conn) for ts in self.ts_ids}

        conn.close()

        self.vehicles = dict()
        self.reward_range = (-float('inf'), float('inf'))
        self.run = 0
        self.metrics = []
        self.out_csv_name = out_csv_name
        self.observations = {ts: None for ts in self.ts_ids}
        self.rewards = {ts: None for ts in self.ts_ids}

    def _start_simulation(self):
        sumo_cmd = [self._sumo_binary,
                    '-n', self._net,
                    '-r', self._route,
                    '--max-depart-delay', str(self.max_depart_delay),
                    '--waiting-time-memory', str(self.waiting_time_memory),
                    '--time-to-teleport', str(self.time_to_teleport)]
        if self.begin_time > 0:
            sumo_cmd.append('-b {}'.format(self.begin_time))
        if self.sumo_seed == 'random':
            sumo_cmd.append('--random')
        else:
            sumo_cmd.extend(['--seed', str(self.sumo_seed)])
        if not self.sumo_warnings:
            sumo_cmd.append('--no-warnings')
        if self.additional_sumo_cmd is not None:
            sumo_cmd.extend(self.additional_sumo_cmd.split())
        if self.use_gui:
            sumo_cmd.extend(['--start', '--quit-on-end'])

        if LIBSUMO:
            traci.start(sumo_cmd)
            self.sumo = traci
        else:
            traci.start(sumo_cmd, label=self.label)
            self.sumo = traci.getConnection(self.label)

        if self.use_gui:
            self.sumo.gui.setSchema(traci.gui.DEFAULT_VIEW, "real world")

    def reset(self, seed: Optional[int] = None, **kwargs):
        super().reset(seed=seed, **kwargs)

        if self.run != 0:
            self.close()
            self.save_csv(self.out_csv_name, self.run)
        self.run += 1
        self.metrics = []

        if seed is not None:
            self.sumo_seed = seed
        self._start_simulation()

        if isinstance(self.reward_fn, dict):
            self.traffic_signals = {ts: TrafficSignal(self,
                                                      ts,
                                                      self.delta_time,
                                                      self.yellow_time,
                                                      self.min_green,
                                                      self.max_green,
                                                      self.begin_time,
                                                      self.reward_fn[ts],
                                                      self.sumo) for ts in self.reward_fn.keys()}
        else:
            self.traffic_signals = {ts: TrafficSignal(self,
                                                      ts,
                                                      self.delta_time,
                                                      self.yellow_time,
                                                      self.min_green,
                                                      self.max_green,
                                                      self.begin_time,
                                                      self.reward_fn,
                                                      self.sumo) for ts in self.ts_ids}

        self.vehicles = dict()

        if self.single_agent:
            return self._compute_observations()[self.ts_ids[0]], self._compute_info()
        else:
            return self._compute_observations()

    @property
    def sim_step(self):
        """
        Return current simulation second on SUMO
        """
        return self.sumo.simulation.getTime()

    def step(self, action):
        # No action, follow fixed TL defined in self.phases
        if action is None or action == {}:
            for _ in range(self.delta_time):
                self._sumo_step()
        else:
            self._apply_actions(action)
            self._run_steps()  # To skip all tls in yellow

        observations = self._compute_observations()
        rewards = self._compute_rewards()
        dones = self._compute_dones()
        info = self._compute_info()

        terminated = False  # there are no 'terminal' states in this environment # TODO: remove?
        truncated = dones['__all__']  # episode ends when sim_step >= max_steps # TODO: remove?

        if self.single_agent:
            return observations[self.ts_ids[0]], rewards[self.ts_ids[0]], terminated, truncated, info
        else:
            return observations, rewards, dones, info

    def _run_steps(self):
        """
        To skip all tls in yellow
        Only when any time_to_act = True in the network, to compute the corresponding observation and reward,
        _apply_actions
        """
        time_to_act = False
        while not time_to_act:
            self._sumo_step()
            for ts in self.ts_ids:
                self.traffic_signals[ts].update()
                if self.traffic_signals[ts].time_to_act:
                    time_to_act = True

    def _apply_actions(self, actions):
        """
        Set the next green phase for the traffic signals
        :param actions: If single-agent, actions is an int between 0 and self.num_green_phases (next green phase)
                        If multiagent, actions is a dict {ts_id : greenPhase}
        """
        if self.single_agent:
            if self.traffic_signals[self.ts_ids[0]].time_to_act:
                self.traffic_signals[self.ts_ids[0]].set_next_phase(actions)
        else:
            for ts, action in actions.items():
                if self.traffic_signals[ts].time_to_act:
                    self.traffic_signals[ts].set_next_phase(action)

    def _compute_dones(self):
        dones = {ts_id: False for ts_id in self.ts_ids}
        dones['__all__'] = self.sim_step > self.sim_max_time
        return dones

    def _compute_info(self):
        info = {'step': self.sim_step}
        if self.add_system_info:
            info.update(self._get_system_info())
        if self.add_per_agent_info:
            info.update(self._get_per_agent_info())
        self.metrics.append(info)
        return info

    def _compute_observations(self):
        self.observations.update({ts: self.traffic_signals[ts].compute_observation() for ts in self.ts_ids if
                                  self.traffic_signals[ts].time_to_act})
        return {ts: self.observations[ts].copy() for ts in self.observations.keys() if
                self.traffic_signals[ts].time_to_act}

    def _compute_rewards(self):
        self.rewards.update({ts: self.traffic_signals[ts].compute_reward() for ts in self.ts_ids if
                             self.traffic_signals[ts].time_to_act})
        return {ts: self.rewards[ts] for ts in self.rewards.keys() if self.traffic_signals[ts].time_to_act}

    @property
    def observation_space(self):
        # TODO: observation space defined in traffic_signal
        return self.traffic_signals[self.ts_ids[0]].observation_space

    @property
    def action_space(self):
        # TODO: action space defined in traffic_signal
        return self.traffic_signals[self.ts_ids[0]].action_space

    def _sumo_step(self):
        self.sumo.simulationStep()

    def _get_system_info(self):
        vehicles = self.sumo.vehicle.getIDList()
        speeds = [self.sumo.vehicle.getSpeed(vehicle) for vehicle in vehicles]
        waiting_times = [self.sumo.vehicle.getWaitingTime(vehicle) for vehicle in vehicles]
        return {
            # In SUMO, a vehicle is considered halting if its speed is below 0.1 m/s
            'system_total_stopped': sum(int(speed < 0.1) for speed in speeds),
            'system_total_waiting_time': sum(waiting_times),
            'system_mean_waiting_time': np.mean(waiting_times),
            'system_mean_speed': 0.0 if len(vehicles) == 0 else np.mean(speeds)
        }

    def _get_per_agent_info(self):
        stopped = [self.traffic_signals[ts].get_total_queued() for ts in self.ts_ids]
        accumulated_waiting_time = [sum(self.traffic_signals[ts].get_accumulated_waiting_time_per_lane()) for ts in
                                    self.ts_ids]
        average_speed = [self.traffic_signals[ts].get_average_speed() for ts in self.ts_ids]
        info = {}
        for i, ts in enumerate(self.ts_ids):
            info[f'{ts}_stopped'] = stopped[i]
            info[f'{ts}_accumulated_waiting_time'] = accumulated_waiting_time[i]
            info[f'{ts}_average_speed'] = average_speed[i]
        info['agents_total_stopped'] = sum(stopped)
        info['agents_total_accumulated_waiting_time'] = sum(accumulated_waiting_time)
        return info

    def close(self):
        if self.sumo is None:
            return
        if not LIBSUMO:
            traci.switch(self.label)
        traci.close()
        self.sumo = None

    def __del__(self):
        self.close()

    def save_csv(self, out_csv_name, run):
        if out_csv_name is not None:
            df = pd.DataFrame(self.metrics)
            Path(Path(out_csv_name).parent).mkdir(parents=True, exist_ok=True)
            df.to_csv(out_csv_name + '_conn{}_run{}'.format(self.label, run) + '.csv', index=False)

    # Below functions are for discrete state space

    def encode(self, state, ts_id):
        phase = int(np.where(state[:self.traffic_signals[ts_id].num_green_phases] == 1)[0])
        min_green = state[self.traffic_signals[ts_id].num_green_phases]
        density_queue = [self._discretize_density(d) for d in state[self.traffic_signals[ts_id].num_green_phases + 1:]]
        # tuples are hashable and can be used as key in python dictionary
        return tuple([phase, min_green] + density_queue)

    def _discretize_density(self, density):
        return min(int(density * 10), 9)


#      ------- OITSC_Urban -------
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.typing import MultiAgentDict


class UrbanEnv(MultiAgentEnv):
    """
    SUMO Environment for Traffic Signal Control

    :param net_file: (str) SUMO .net.xml file
    :param route_file: (str) SUMO .rou.xml file
    :param horizon: (int) Number of simulated seconds on SUMO
    :param warmup: (int) Number of simulation seconds ran before learning begins
    :param delta_time: (int) Simulation seconds between actions
    :parma yellow_time: (int) Fixed yellow time
    :param min_green: (int) Minimum green time in a phase
    :param max_green: (int) Max green time in a phase
    :param single_agent: (bool) If true, it behaves like a regular gym.Env. Else, it behaves like a MultiagentEnv
    (https://github.com/ray-project/ray/blob/master/python/ray/rllib/env/multi_agent_env.py)
    :param use_gui: (bool) Whether to run SUMO simulation with GUI visualisation
    :param sumo_seed: sumo_seed: (int/string) Random seed for sumo. If 'random' it uses a randomly chosen seed
    :param waiting_time_memory: (int) Number of seconds to remember the waiting time of a vehicle
    (see https://sumo.dlr.de/pydoc/traci._vehicle.html#VehicleDomain-getAccumulatedWaitingTime)
    :param sumo_warnings: (bool) If False, remove SUMO warnings in the terminal
    :param additional_sumo_cmd: (list) Additional command line arguments for SUMO
    :param out_csv_name: (str) name of the .csv output with simulation results. If None no output is generated
    """
    CONNECTION_LABEL = 0  # For traci multi-client support

    def __init__(
            self,
            net_file: str,
            route_file: str,
            horizon: int = 1000,
            warmup: int = 0,
            delta_time: int = 5,
            yellow_time: int = 2,
            min_green: int = 5,
            max_green: int = 50,
            single_agent: bool = False,
            use_gui: bool = False,
            sumo_seed: Union[str, int] = 'random',
            waiting_time_memory: int = 10000,
            time_to_teleport: int = -1,
            sumo_warnings: bool = True,
            additional_sumo_cmd: Optional[list] = None,
            out_csv_name: Optional[str] = None,
    ) -> None:
        self._net = net_file
        self._route = route_file
        self.use_gui = use_gui
        if self.use_gui:
            self._sumo_binary = sumolib.checkBinary('sumo-gui')
        else:
            self._sumo_binary = sumolib.checkBinary('sumo')

        assert delta_time > yellow_time, "Time between actions must be at least greater than yellow time."
        self.horizon = horizon
        self.warmup = warmup
        self.delta_time = delta_time
        self.yellow_time = yellow_time
        self.min_green = min_green
        self.max_green = max_green

        self.single_agent = single_agent

        self.sumo_seed = sumo_seed
        self.waiting_time_memory = waiting_time_memory
        self.time_to_teleport = time_to_teleport
        self.sumo_warnings = sumo_warnings
        self.additional_sumo_cmd = additional_sumo_cmd

        self.run = 0
        self.sumo = None

        # Start a sumo connection, only to retrieve traffic light information
        self.label = str(UrbanEnv.CONNECTION_LABEL)
        UrbanEnv.CONNECTION_LABEL += 1
        traci.start([sumolib.checkBinary('sumo'), '-n', self._net], numRetries=100,
                    label='init_connection' + self.label)
        info_sumo = traci.getConnection('init_connection' + self.label)
        self.ts_ids = list(info_sumo.trafficlight.getIDList())
        self.agents = self.ts_ids  # for PettingZooEnv
        self.max_num_lanes = self._get_max_num_lanes(self.ts_ids, info_sumo)
        info_sumo.close()

        # process scenario file to retrieve scenario info
        self.ts_phases = self._get_phases_all_ts(self._net)
        self.routes = self._get_routes_choice()

        self.observations = {ts: None for ts in self.ts_ids}
        self.rewards = {ts: 0 for ts in self.ts_ids}
        self.traffic_signals = dict()
        self.veh_type = dict()
        self.veh_waiting_per_lane = dict()
        self.veh_waiting_all = dict()
        self.veh_waiting_diff = dict()

        # unexpected events
        self.ambulance_count = 0
        self.fueltruck_count = 0
        self.trailer_count = 0

        # output
        self.metrics = []
        self.out_csv_name = out_csv_name

    def _start_simulation(self):
        sumo_cmd = [self._sumo_binary,
                    '-n', self._net,
                    '-r', self._route,
                    '--waiting-time-memory', str(self.waiting_time_memory),
                    '--time-to-teleport', str(self.time_to_teleport)]
        if self.sumo_seed == 'random':
            sumo_cmd.append('--random')
        else:
            sumo_cmd.extend(['--seed', str(self.sumo_seed)])
        if not self.sumo_warnings:
            sumo_cmd.append('--no-warnings')
        if self.additional_sumo_cmd is not None:
            sumo_cmd.append('--tripinfo-output')
            sumo_cmd.extend([self.additional_sumo_cmd[1] + '{}.xml'.format(self.run)])
            # sumo_cmd.extend(self.additional_sumo_cmd)
        if self.use_gui:
            sumo_cmd.extend(['--start', '--quit-on-end'])

        traci.start(sumo_cmd, numRetries=100, label=str(self.run))
        self.run += 1
        self.sumo = traci

        if self.use_gui:
            self.sumo.gui.setSchema(traci.gui.DEFAULT_VIEW, "real world")

    def reset(self,
              *,
              seed: Optional[int] = None,
              options: Optional[dict] = None) -> MultiAgentDict:

        if self.run != 0:
            self.close()
            if self.out_csv_name:
                self.save_csv(self.out_csv_name, self.run)
        self.metrics = []
        self.veh_type = dict()
        self.veh_waiting_per_lane = dict()
        self.veh_waiting_all = dict()
        self.veh_waiting_diff = dict()
        # unexpected events
        self.ambulance_count = 0
        self.fueltruck_count = 0
        self.trailer_count = 0

        self._start_simulation()
        self.traffic_signals = {ts: TrafficSignalUrban(self, ts, self.ts_phases[ts], self.delta_time,
                                                       self.yellow_time, self.min_green, self.max_green,
                                                       self.sumo) for ts in self.ts_ids}

        # warmup before learning
        for _ in range(self.warmup):
            self._sumo_step()

        if self.single_agent:
            return self._compute_observations()[self.ts_ids[0]]
        else:
            return self._compute_observations(), self._compute_info_rllib()  # TODO: rllib

    @property
    def sim_step(self):
        """
        Return current simulation second on SUMO
        """
        return self.sumo.simulation.getTime()

    def step(
            self, action_dict: MultiAgentDict
    ) -> Tuple[MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict]:

        for _ in range(self.delta_time):
            self._sumo_step()
            for ts in self.ts_ids:
                self.traffic_signals[ts].update()
        if action_dict:
            self._apply_actions(action_dict)

        # print("step_num:", self.sim_step)
        if self.sim_step % 500 == 0:  # unexpected events occur per 5 minutes
            self._insert_important_vehicles()

        observations = self._compute_observations()
        self.get_waiting_vehs()
        rewards = self._compute_rewards()
        dones = self._compute_dones()
        truncateds = self._compute_truncateds()
        info = self._compute_info_rllib()

        if self.single_agent:
            return observations[self.ts_ids[0]], rewards[self.ts_ids[0]], dones['__all__'], {}
        else:
            return observations, rewards, dones, truncateds, info  # TODO: changed cuz rllib
            # self.veh_waiting_all, self.veh_type, for save evaluation data
            # obs, rewards, terminateds, truncateds, infos = env.step(agent_dict)
            # Truncated values for each ready agent.

    def _apply_actions(self, actions):
        if self.single_agent:
            self.traffic_signals[self.ts_ids[0]].phase_switching(int(actions))
        else:
            for ts, action in actions.items():
                self.traffic_signals[ts].phase_switching(action)

    def _compute_dones(self):
        dones = {ts_id: False for ts_id in self.ts_ids}
        dones['__all__'] = self.sim_step > self.horizon
        return dones

    def _compute_truncateds(self):
        truncateds = {ts_id: False for ts_id in self.ts_ids}
        truncateds['__all__'] = self.sim_step > self.horizon
        return truncateds

    def _compute_info(self):
        info = {'step': self.sim_step}
        info.update({'avg_reward': np.mean(list(self.rewards.values()))})
        info.update(
            {'avg_waiting_time': np.sum(list(self.veh_waiting_all.values())) / len(self.veh_waiting_all.keys())})
        self.metrics.append(info)
        return info

    def _compute_info_rllib(self):
        return {}

    def _compute_observations(self):
        observations = {}
        for ts in self.ts_ids:
            phase_index = self.traffic_signals[ts].get_phase_index() / len(self.ts_phases[ts])
            density = self.traffic_signals[ts].get_lanes_density()
            queue = self.traffic_signals[ts].get_lanes_queue()

            # padding for various number of lanes
            if len(self.traffic_signals[ts].lanes) < self.max_num_lanes:
                diff = self.max_num_lanes - len(self.traffic_signals[ts].lanes)
                density.extend([0] * diff)
                queue.extend([0] * diff)

            observations.update({ts: [phase_index] + density + queue})
        return observations

    def _compute_rewards(self):
        rewards = {}
        for ts in self.ts_ids:
            sum_waiting = 0
            for veh in self.veh_waiting_diff.keys():
                lane = list(self.veh_waiting_diff[veh].keys())[0]
                if lane in self.traffic_signals[ts].lanes:
                    sum_waiting += self.veh_waiting_diff[veh][lane]
            rewards.update({ts: sum_waiting})
        self.rewards = rewards
        return rewards

    def get_waiting_vehs(self):
        """
        self.veh_type = {veh_id: veh_type}
        self.veh_waiting_per_lane = {veh_id: {lane_name: waiting time}}
        self.veh_waiting_diff = {veh_id: {lane_name: waiting time}}
        self.veh_waiting_all = {veh_id: waiting time}
        """
        self.veh_waiting_diff = {}
        for veh in self.sumo.vehicle.getIDList():
            lane_now = self.sumo.vehicle.getLaneID(veh)
            waiting_all = self.sumo.vehicle.getAccumulatedWaitingTime(veh)

            if veh not in self.veh_waiting_all.keys():  # new vehicle
                self.veh_waiting_per_lane.update({veh: {lane_now: waiting_all}})
                self.veh_waiting_diff.update({veh: {lane_now: -waiting_all}})
                self.veh_type.update({veh: self.sumo.vehicle.getTypeID(veh)})
            else:
                waiting_new = waiting_all - self.veh_waiting_all[veh]
                self.veh_waiting_diff.update({veh: {lane_now: -waiting_new}})
                if lane_now not in self.veh_waiting_per_lane[veh].keys():  # new road
                    self.veh_waiting_per_lane[veh].update({lane_now: waiting_new})
                else:
                    pre_waiting = self.veh_waiting_per_lane[veh][lane_now]
                    self.veh_waiting_per_lane[veh].update({lane_now: pre_waiting + waiting_new})

            self.veh_waiting_all.update({veh: waiting_all})

    @property
    def observation_space(self):
        # keep consistent for all ts in the network
        # self.discrete_observation_space = spaces.Tuple((
        #     spaces.Discrete(self._get_max_num_phases()),
        #     *(spaces.Discrete(10) for _ in range(2 * self.max_num_lanes))
        # ))
        return spaces.Box(low=np.zeros(1 + 2 * self.max_num_lanes, dtype=np.float32),
                          high=np.ones(1 + 2 * self.max_num_lanes, dtype=np.float32))

    @property
    def action_space(self):
        # keep consistent for all ts in the network
        return spaces.Discrete(2)  # switch or not

    def _sumo_step(self):
        self.sumo.simulationStep()

    def close(self):
        if self.sumo is None:
            return
        traci.close()
        self.sumo = None

    def save_csv(self, out_csv_name, run):
        if out_csv_name is not None:
            df = pd.DataFrame(self.metrics)
            Path(Path(out_csv_name).parent).mkdir(parents=True, exist_ok=True)
            df.to_csv(out_csv_name + '_iter{}'.format(run) + '.csv', index=False)

    # Below functions are for discrete state space

    def encode(self, state):
        phase = state[0]
        density_queue = [self._discretize_density(d) for d in state[1:]]
        # tuples are hashable and can be used as key in python dictionary
        return tuple([phase] + density_queue)

    def _discretize_density(self, density):
        # density and queue length
        return min(int(density * 10), 9)

    # Discrete

    # Collect network info from scenario file
    def _get_phases_all_ts(self, net_file):
        ts_phases = {}
        for tlLogic in sumolib.output.parse(net_file, ['tlLogic']):
            phases = []
            for each in tlLogic['phase']:
                phases.append(each.state)
            ts_phases.update({tlLogic.id: phases})
        return ts_phases

    # Collect info from sumo info connection
    def _get_max_num_lanes(self, ts_ids, info_sumo):
        max_num_lanes = 0
        for tl in ts_ids:
            max_num_lanes = max(len(info_sumo.trafficlight.getControlledLanes(tl)), max_num_lanes)
        return max_num_lanes

    def _get_max_num_phases(self):
        max_num_phases = 0
        for tl in self.traffic_signals:
            max_num_phases = max(len(self.traffic_signals[tl].phases), max_num_phases)
        return max_num_phases

    # insert important vehicles, e.g., ambulances, fuel trucks, and trailer trucks
    def _insert_important_vehicles(self):
        now_ambulance_count = 0
        now_fueltruck_count = 0
        now_trailer_count = 0
        for veh in self.sumo.vehicle.getIDList():
            if self.sumo.vehicle.getTypeID(veh) == "Ambulance":
                now_ambulance_count += 1
            elif self.sumo.vehicle.getTypeID(veh) == "fueltruck":
                now_fueltruck_count += 1
            elif self.sumo.vehicle.getTypeID(veh) == "trailer":
                now_trailer_count += 1

        if now_ambulance_count < 1:
            random_route_id = random.randint(0, len(self.routes) - 1)
            traci.vehicle.add("Ambulance_" + str(self.ambulance_count), str(self.routes[random_route_id]),
                              "Ambulance", None, "random", "base", "0", "current", "max", "current", "", "", "", 0,
                              0)
            self.ambulance_count += 1
        if now_fueltruck_count < 1:
            random_route_id = random.randint(0, len(self.routes) - 1)
            traci.vehicle.add("FuelTruck_" + str(self.fueltruck_count), str(self.routes[random_route_id]),
                              "fueltruck", None, "random", "base", "0", "current", "max", "current", "", "", "", 0,
                              0)
            self.fueltruck_count += 1
        if now_trailer_count < 1:
            random_route_id = random.randint(0, len(self.routes) - 1)
            traci.vehicle.add("Trailer_" + str(self.trailer_count), str(self.routes[random_route_id]), "trailer",
                              None, "random", "base", "0", "current", "max", "current", "", "", "", 0, 0)
            self.trailer_count += 1

    def _get_routes_choice(self):
        routes = []
        for route in sumolib.output.parse(self._route, ['route']):
            routes.append(route.id)
        return routes


class UrbanEnvImplicitObservation(UrbanEnv):

    def __init__(
            self,
            net_file: str,
            route_file: str,
            horizon: int = 1000,
            warmup: int = 0,
            delta_time: int = 5,
            yellow_time: int = 2,
            min_green: int = 5,
            max_green: int = 50,
            single_agent: bool = False,
            use_gui: bool = False,
            sumo_seed: Union[str, int] = 'random',
            waiting_time_memory: int = 10000,
            time_to_teleport: int = -1,
            sumo_warnings: bool = True,
            additional_sumo_cmd: Optional[list] = None,
            out_csv_name: Optional[str] = None,
            oitsc: bool = False
    ) -> None:
        super().__init__(net_file, route_file, horizon, warmup, delta_time, yellow_time, min_green, max_green,
                         single_agent, use_gui, sumo_seed, waiting_time_memory, time_to_teleport, sumo_warnings,
                         additional_sumo_cmd, out_csv_name)
        self.oitsc = oitsc
        self.ts_phase = {}
        self.ts_phase_dura = {}

    def _compute_rewards(self):
        """
        cannot observe the waiting time of 20% of vehicles for computing rewards
        oitsc: replaces the waiting time of stationary vehicles
               with the current traffic signal phase elapsed time
        """
        rewards = {}
        pre_ts_phase = self.ts_phase
        # count = 0
        all_veh_list = self.sumo.vehicle.getIDList()
        for ts in self.ts_ids:
            self.ts_phase.update({ts: self.sumo.trafficlight.getRedYellowGreenState(ts)})

            sum_waiting = 0
            for veh in self.veh_waiting_diff.keys():
                if veh in all_veh_list:
                    lane = list(self.veh_waiting_diff[veh].keys())[0]
                    if lane in self.traffic_signals[ts].lanes:
                        if random.randint(0, 4) == 2:  # cannot get waiting time
                            if self.oitsc and self.sumo.vehicle.isStopped(veh):
                                # count += 1
                                if pre_ts_phase[ts] == self.ts_phase[ts]:  # same phase
                                    sum_waiting += -self.delta_time
                                else:
                                    sum_waiting += -self.traffic_signals[ts].time_since_last_phase_change
                        else:
                            sum_waiting += self.veh_waiting_diff[veh][lane]
                            # count += 1

            rewards.update({ts: sum_waiting})
        # print("{} vehs counted, total: {}".format(count, len(all_veh_list)))
        self.rewards = rewards
        return rewards


class UrbanEnvImportantObservation(UrbanEnv):

    def __init__(
            self,
            net_file: str,
            route_file: str,
            horizon: int = 1000,
            warmup: int = 0,
            delta_time: int = 5,
            yellow_time: int = 2,
            min_green: int = 5,
            max_green: int = 50,
            single_agent: bool = False,
            use_gui: bool = False,
            sumo_seed: Union[str, int] = 'random',
            waiting_time_memory: int = 10000,
            time_to_teleport: int = -1,
            sumo_warnings: bool = True,
            additional_sumo_cmd: Optional[list] = None,
            out_csv_name: Optional[str] = None,
            oitsc: bool = False
    ) -> None:
        super().__init__(net_file, route_file, horizon, warmup, delta_time, yellow_time, min_green, max_green,
                         single_agent, use_gui, sumo_seed, waiting_time_memory, time_to_teleport, sumo_warnings,
                         additional_sumo_cmd, out_csv_name)
        self.oitsc = oitsc
        self.ts_phase_green_lanes = {}

    def reset(self) -> MultiAgentDict:
        if self.run != 0:
            self.close()
            if self.out_csv_name:
                self.save_csv(self.out_csv_name, self.run)
        self.metrics = []
        self.veh_type = dict()
        self.veh_waiting_per_lane = dict()
        self.veh_waiting_all = dict()
        self.veh_waiting_diff = dict()
        # unexpected events
        self.ambulance_count = 0
        self.fueltruck_count = 0
        self.trailer_count = 0

        self._start_simulation()
        self.traffic_signals = {ts: TrafficSignalUrban(self, ts, self.ts_phases[ts], self.delta_time,
                                                       self.yellow_time, self.min_green, self.max_green,
                                                       self.sumo) for ts in self.ts_ids}

        self.ts_phase_green_lanes = self._get_green_lanes_per_phase_all_ts()

        # warmup before learning
        for _ in range(self.warmup):
            self._sumo_step()

        if self.single_agent:
            return self._compute_observations()[self.ts_ids[0]]
        else:
            return self._compute_observations()

    def _apply_actions(self, actions):
        if self.single_agent:
            self.traffic_signals[self.ts_ids[0]].phase_switching(int(actions))
        else:
            if self.sim_step % 100 == 0 and self.oitsc:
                print(self.sim_step)
                ts_phase_count_important_vehicles = self.count_important_vehicles()
            for ts, action in actions.items():
                if self.sim_step % 100 == 0 and self.oitsc:
                    print(self.sim_step)
                    phase_count_important_vehicles = ts_phase_count_important_vehicles[ts]
                    proposed_phase = max(phase_count_important_vehicles, key=phase_count_important_vehicles.get)
                    self.traffic_signals[ts].phase_switching_important_observation(action, proposed_phase)
                else:
                    self.traffic_signals[ts].phase_switching(action)

    def _get_green_lanes_per_phase_all_ts(self):
        ts_phase_green_lanes = {}  # {tl: {green_phase: [green_lanes]}}
        for ts in self.ts_ids:
            ts_phase_green_lanes.update({ts: self.traffic_signals[ts].green_lanes_per_phase(self._net)})
        return ts_phase_green_lanes

    def count_important_vehicles(self):
        # TODO: add weight for different important vehicles
        ts_phase_count_important_vehicles = {}  # {tl: {green_phase: #important_vehicles}}
        for ts in self.ts_ids:
            ts_phase_count_important_vehicles.update({ts: {}})
            green_phases = list(self.ts_phase_green_lanes[ts].keys())
            for green_phase in green_phases:
                ts_phase_count_important_vehicles[ts].update({green_phase: 0})
        for veh in self.sumo.vehicle.getIDList():
            veh_type = self.sumo.vehicle.getTypeID(veh)
            if veh_type == "Ambulance" or veh_type == "fueltruck" or veh_type == "trailer":
                lane_now = self.sumo.vehicle.getLaneID(veh)
                for ts, green_phase_lanes in self.ts_phase_green_lanes.items():
                    for phase, lanes in green_phase_lanes.items():
                        if lane_now in lanes:
                            ts_phase_count_important_vehicles[ts][phase] += 1
        return ts_phase_count_important_vehicles


class UrbanEnvSampleObservation(UrbanEnv):

    def __init__(
            self,
            net_file: str,
            route_file: str,
            horizon: int = 1000,
            warmup: int = 0,
            delta_time: int = 5,
            yellow_time: int = 2,
            min_green: int = 5,
            max_green: int = 50,
            single_agent: bool = False,
            use_gui: bool = False,
            sumo_seed: Union[str, int] = 'random',
            waiting_time_memory: int = 10000,
            time_to_teleport: int = -1,
            sumo_warnings: bool = True,
            additional_sumo_cmd: Optional[list] = None,
            out_csv_name: Optional[str] = None,
            oitsc: bool = False
    ) -> None:
        super().__init__(net_file, route_file, horizon, warmup, delta_time, yellow_time, min_green, max_green,
                         single_agent, use_gui, sumo_seed, waiting_time_memory, time_to_teleport, sumo_warnings,
                         additional_sumo_cmd, out_csv_name)
        self.oitsc = oitsc
        self.pre_veh_waiting_per_lane = {}

    def reset(self) -> MultiAgentDict:
        if self.run != 0:
            self.close()
            if self.out_csv_name:
                self.save_csv(self.out_csv_name, self.run)
        self.metrics = []
        self.veh_type = dict()
        self.veh_waiting_per_lane = dict()
        self.pre_veh_waiting_per_lane = dict()
        self.veh_waiting_all = dict()
        self.veh_waiting_diff = dict()
        # unexpected events
        self.ambulance_count = 0
        self.fueltruck_count = 0
        self.trailer_count = 0

        self._start_simulation()
        self.traffic_signals = {ts: TrafficSignalUrban(self, ts, self.ts_phases[ts], self.delta_time,
                                                       self.yellow_time, self.min_green, self.max_green,
                                                       self.sumo) for ts in self.ts_ids}

        # warmup before learning
        for _ in range(self.warmup):
            self._sumo_step()

        if self.single_agent:
            return self._compute_observations()[self.ts_ids[0]]
        else:
            return self._compute_observations()

    def get_waiting_vehs(self):
        """
        self.veh_type = {veh_id: veh_type}
        self.veh_waiting_per_lane = {veh_id: {lane_name: waiting time}}
        self.pre_veh_waiting_per_lane
        self.veh_waiting_diff = {veh_id: {lane_name: waiting time}}
        self.veh_waiting_all = {veh_id: waiting time}
        """
        self.pre_veh_waiting_per_lane = self.veh_waiting_per_lane
        super().get_waiting_vehs()

    def _compute_rewards(self):
        """
        The waiting time of 50% of vehicles will be corrupted.
        oitsc: sample the waiting time for undetected vehicles
        """
        rewards = {}
        for ts in self.ts_ids:
            sum_waiting = 0
            sample = {}  # {lane_id: sample_veh_waiting}
            for veh in self.veh_waiting_diff.keys():
                lane = list(self.veh_waiting_diff[veh].keys())[0]
                if lane in self.traffic_signals[ts].lanes:
                    if random.randint(0, 1) == 1:
                        if self.oitsc and lane in sample.keys():
                            sum_waiting += self.pre_veh_waiting_per_lane[veh][lane] - sample[lane]
                    else:
                        sum_waiting += self.veh_waiting_diff[veh][lane]
                        if lane not in sample.keys():
                            sample.update({lane: self.veh_waiting_per_lane[veh][lane]})

            rewards.update({ts: sum_waiting})
        self.rewards = rewards
        return rewards


import random
from datetime import datetime
from collections import defaultdict
from sumo_rl_rep.util.data_processing import Dict, random_sample


class SumoEnvironmentImplicitObservation(MultiAgentEnv):
    """
    SUMO Environment for Traffic Signal Control

    :param net_file: (str) SUMO .net.xml file
    :param route_file: (str) SUMO .rou.xml file
    :param phases: (traci.trafficlight.Phase list) Traffic Signal phases definition
    :param out_csv_name: (str) name of the .csv output with simulation results. If None no output is generated
    :param use_gui: (bool) Whether to run SUMO simulation with GUI visualisation
    :param num_seconds: (int) Number of simulated seconds on SUMO
    :param max_depart_delay: (int) Vehicles are discarded if they could not be inserted after max_depart_delay seconds
    :param time_to_load_vehicles: (int) Number of simulation seconds ran before learning begins
    :param delta_time: (int) Simulation seconds between actions
    :param min_green: (int) Minimum green time in a phase
    :param max_green: (int) Max green time in a phase
    :single_agent: (bool) If true, it behaves like a regular gym.Env. Else, it behaves like a MultiagentEnv (https://github.com/ray-project/ray/blob/master/python/ray/rllib/env/multi_agent_env.py)
    """

    # TODO: max_depart_delay is not necessary
    def __init__(self, net_file, route_file, tripinfo_output, phases, out_csv_name=None, use_gui=False,
                 num_seconds=20000, max_depart_delay=100000,
                 time_to_teleport=-1, time_to_load_vehicles=0, delta_time=5, yellow_time=2, min_green=5, max_green=100,
                 single_agent=False):

        self._net = net_file
        self._route = route_file
        self._tripinfo = tripinfo_output
        self.use_gui = use_gui
        if self.use_gui:
            self._sumo_binary = sumolib.checkBinary('sumo-gui')
        else:
            self._sumo_binary = sumolib.checkBinary('sumo')

        traci.start([sumolib.checkBinary('sumo'), '-n', self._net])  # start only to retrieve information

        self.single_agent = single_agent
        # TODO: diff on single-agent and multi-agent
        self.ts_ids = traci.trafficlight.getIDList()
        self.lanes_per_ts = len(set(traci.trafficlight.getControlledLanes(self.ts_ids[0])))
        self.traffic_signals = dict()
        self.phases = phases
        self.num_green_phases = len(phases) // 2
        # Number of green phases == number of phases (green+yellow) divided by 2
        # TODO: traci to get # green phases, dict of all tls
        self.vehicles = dict()
        self.last_measure = dict()  # used to reward function remember last measure
        self.last_reward = {i: 0 for i in self.ts_ids}
        self.sim_max_time = num_seconds
        self.time_to_load_vehicles = time_to_load_vehicles
        self.delta_time = delta_time
        self.max_depart_delay = max_depart_delay
        self.time_to_teleport = time_to_teleport
        self.min_green = min_green  # TODO: fixed for all tls?
        self.max_green = max_green
        self.yellow_time = yellow_time

        # ------- Implicit Observation -------
        self.vehicles_details = Dict()  # TODO: not found the usage in this env

        self.ambulance_count = 0
        self.fueltruck_count = 0
        self.trailer_count = 0

        self.ambulance_count_0to1 = 0
        self.ambulance_count_4to5 = 0
        self.ambulance_count_9to10 = 0
        self.ambulance_count_14to15 = 0
        self.ambulance_count_21to4 = 0
        self.ambulance_count_0to4 = 0
        self.ambulance_count_8to9 = 0
        self.ambulance_count_5to9 = 0
        self.ambulance_count_13to14 = 0
        self.ambulance_count_10to14 = 0

        self.distance_val_per_road = {}  # unexpected: {road_name: {veh: getLanePos}}
        self.default_vehicles_distance_val_per_road = {}
        self.last_distance = defaultdict(dict)  # TODO: used to distance function remember last distance
        self.last_distance_reward = dict()
        self.neighbors_list = dict()
        self.controlledlanes_list = dict()
        self.importance_weight_list = dict()
        self.last_default_vehicles_distance_reward = dict()
        self.last_default_vehicles_distance = defaultdict(dict)

        self.counter = 0
        # ------- Implicit Observation -------

        """
        Default observation space is a vector R^(#greenPhases + 2 * #lanes)
        s = [current phase one-hot encoded, density for each lane, queue for each lane]
        You can change this by modifying self.observation_space and the method _compute_observations()

        Action space is which green phase is going to be open for the next delta_time seconds
        """
        # observation space for each tl
        self.observation_space = spaces.Box(low=np.zeros(self.num_green_phases + 2 * self.lanes_per_ts),
                                            high=np.ones(self.num_green_phases + 2 * self.lanes_per_ts))

        self.discrete_observation_space = spaces.Tuple((
            spaces.Discrete(self.num_green_phases),  # Green Phase
            # spaces.Discrete(self.max_green//self.delta_time),               # Elapsed time of phase
            *(spaces.Discrete(10) for _ in range(2 * self.lanes_per_ts))  # Density and stopped-density for each lane
        ))
        #
        self.action_space = spaces.Discrete(self.num_green_phases)

        self.reward_range = (-float('inf'), float('inf'))  # TODO: need reward_range?
        self.spec = ''  # TODO: what is the meaning of spec

        self.radix_factors = [s.n for s in self.discrete_observation_space.spaces]

        self.run = 0
        self.metrics = []
        self.out_csv_name = out_csv_name

        traci.close()

    def reset(self):
        if self.run != 0:
            traci.close()
            self.save_csv(self.out_csv_name, self.run)
        self.run += 1
        self.metrics = []

        sumo_cmd = [self._sumo_binary,
                    '-n', self._net,
                    '-r', self._route,
                    '--tripinfo-output', self._tripinfo,
                    '--max-depart-delay', str(self.max_depart_delay),
                    '--waiting-time-memory', '10000',
                    '--time-to-teleport', str(self.time_to_teleport),
                    '--random']
        if self.use_gui:
            sumo_cmd.append('--start')

        traci.start(sumo_cmd)

        for ts in self.ts_ids:
            self.traffic_signals[ts] = TrafficSignalImplicitObservation(self, ts, self.delta_time, self.yellow_time,
                                                                        self.min_green,
                                                                        self.max_green, self.phases)
            self.last_measure[ts] = 0.0

            # ------- Implicit Observation -------
            lanes = self.traffic_signals[ts].getControlledLanes()
            self.last_distance[ts] = {}
            self.last_default_vehicles_distance[ts] = {}
            for lane in lanes:
                key = lane[:-2]  # edge name
                self.distance_val_per_road[key] = {}
                self.default_vehicles_distance_val_per_road = {}
            # ------- Implicit Observation -------

        self.vehicles = dict()

        # Load vehicles
        for _ in range(self.time_to_load_vehicles):
            self._sumo_step()

        if self.single_agent:
            return self._compute_observations()[self.ts_ids[0]]
        else:
            return self._compute_observations()

    @property
    def sim_step(self):
        """
        Return current simulation second on SUMO
        """
        return traci.simulation.getTime()

    def step(self, action):
        # ------- Implicit Observation -------
        random_id_list = random_sample(120, 2, 1e+30)  # count = #veh: ambulance, fueltruck, trailer
        self.counter = self.counter + 1

        now = datetime.now()
        experiment_time = now.minute

        # '''Intuitive part-generated randomly-scenario 5-high frequency
        if experiment_time % 2 == 0:
            while self.ambulance_count < 5:
                traci.vehicle.add(str(self.counter + random_id_list[0 + self.ambulance_count]), "routedist1",
                                  "Ambulance", None, "random", "base", "0", "current", "max", "current", "", "", "", 0,
                                  0)
                self.ambulance_count = self.ambulance_count + 1

            while self.fueltruck_count < 5:
                traci.vehicle.add(str(self.counter + random_id_list[5 + self.fueltruck_count]), "routedist1",
                                  "fueltruck", None, "random", "base", "0", "current", "max", "current", "", "", "", 0,
                                  0)
                self.fueltruck_count = self.fueltruck_count + 1

            while self.trailer_count < 5:
                traci.vehicle.add(str(self.counter + random_id_list[10 + self.trailer_count]), "routedist1", "trailer",
                                  None, "random", "base", "0", "current", "max", "current", "", "", "", 0, 0)
                self.trailer_count = self.trailer_count + 1

        # TODO: arrived within two minutes?
        if experiment_time % 2 != 0:
            self.ambulance_count = 0
            self.fueltruck_count = 0
            self.trailer_count = 0
        # '''

        '''generated-scenario 1-high frequency
        if experiment_time % 2 == 0 and self.ambulance_count_0to1 < 5:
            traci.vehicle.add(str(self.counter + random_id_list[0+self.ambulance_count_0to1]), "route0to1", "Ambulance", None, "random", "base", "0", "current", "max","current","","","",0, 0)
            self.ambulance_count_0to1 = self.ambulance_count_0to1 + 1
        if experiment_time % 2 != 0: self.ambulance_count_0to1 = 0
        '''
        '''generated-scenario 2-high frequency
        if experiment_time % 2 == 0 and self.ambulance_count_0to1 < 5:
            traci.vehicle.add(str(self.counter + random_id_list[0+self.ambulance_count_0to1]), "route0to1", "Ambulance", None, "random", "base", "0", "current", "max","current","","","",0, 0)
            self.ambulance_count_0to1 = self.ambulance_count_0to1 + 1
        if experiment_time % 2 == 0 and self.ambulance_count_4to5 < 5:
            traci.vehicle.add(str(self.counter + random_id_list[0+self.ambulance_count_4to5]), "route4to5", "Ambulance", None, "first", "base", "0", "current", "max","current","","","",0, 0)
            self.ambulance_count_4to5 = self.ambulance_count_4to5 + 1
        if experiment_time % 2 != 0: self.ambulance_count_0to1 = 0
        if experiment_time % 2 != 0: self.ambulance_count_4to5 = 0
        '''
        '''generated-scenario 3-high frequency
        if experiment_time % 2 == 0 and self.ambulance_count_0to1 < 5:
            traci.vehicle.add(str(self.counter + random_id_list[0+self.ambulance_count_0to1]), "route0to1", "Ambulance", None, "random", "base", "0", "current", "max","current","","","",0, 0)
            self.ambulance_count_0to1 = self.ambulance_count_0to1 + 1

        if experiment_time % 2 == 0 and self.ambulance_count_4to5 < 5:
            traci.vehicle.add(str(self.counter + random_id_list[0+self.ambulance_count_4to5]), "route4to5", "Ambulance", None, "first", "base", "0", "current", "max","current","","","",0, 0)
            self.ambulance_count_4to5 = self.ambulance_count_4to5 + 1

        if experiment_time % 2 == 0 and self.ambulance_count_9to10 < 5:
            traci.vehicle.add(str(self.counter + random_id_list[0+self.ambulance_count_9to10]), "route9to10", "Ambulance", None, "random", "base", "0", "current", "max","current","","","",0, 0)
            self.ambulance_count_9to10 = self.ambulance_count_9to10 + 1

        if experiment_time % 2 == 0 and self.ambulance_count_14to15 < 5:
            traci.vehicle.add(str(self.counter + random_id_list[0+self.ambulance_count_14to15]), "route14to15", "Ambulance", None, "first", "base", "0", "current", "max","current","","","",0, 0)
            self.ambulance_count_14to15 = self.ambulance_count_14to15 + 1

        if experiment_time % 2 != 0: 
            self.ambulance_count_0to1 = 0
            self.ambulance_count_4to5 = 0
            self.ambulance_count_9to10 = 0
            self.ambulance_count_14to15 = 0
        '''
        '''generated-scenario 4-high frequency
        if experiment_time % 2 == 0 and self.ambulance_count_21to4 < 5:
            traci.vehicle.add(str(self.counter + random_id_list[0+self.ambulance_count_21to4]), "route21to4", "Ambulance", None, "random", "base", "0", "current", "max","current","","","",0, 0)
            self.ambulance_count_21to4 = self.ambulance_count_21to4 + 1

        if experiment_time % 2 == 0 and self.ambulance_count_0to4 < 5:
            traci.vehicle.add(str(self.counter + random_id_list[5+self.ambulance_count_0to4]), "route0to4", "Ambulance", None, "first", "base", "0", "current", "max","current","","","",0, 0)
            self.ambulance_count_0to4 = self.ambulance_count_0to4 + 1

        if experiment_time % 2 == 0 and self.ambulance_count_8to9 < 5:
            traci.vehicle.add(str(self.counter + random_id_list[10+self.ambulance_count_8to9]), "route8to9", "Ambulance", None, "random", "base", "0", "current", "max","current","","","",0, 0)
            self.ambulance_count_8to9 = self.ambulance_count_8to9 + 1

        if experiment_time % 2 == 0 and self.ambulance_count_5to9 < 5:
            traci.vehicle.add(str(self.counter + random_id_list[15+self.ambulance_count_5to9]), "route5to9", "Ambulance", None, "first", "base", "0", "current", "max","current","","","",0, 0)
            self.ambulance_count_5to9 = self.ambulance_count_5to9 + 1

        if experiment_time % 2 == 0 and self.ambulance_count_13to14 < 5:
            traci.vehicle.add(str(self.counter + random_id_list[20+self.ambulance_count_13to14]), "route13to14", "Ambulance", None, "first", "base", "0", "current", "max","current","","","",0, 0)
            self.ambulance_count_13to14 = self.ambulance_count_13to14 + 1

        if experiment_time % 2 == 0 and self.ambulance_count_10to14 < 5:
            traci.vehicle.add(str(self.counter + random_id_list[25+self.ambulance_count_10to14]), "route10to14", "Ambulance", None, "first", "base", "0", "current", "max","current","","","",0, 0)
            self.ambulance_count_10to14 = self.ambulance_count_10to14 + 1


        if experiment_time % 2 != 0: 
            self.ambulance_count_21to4 = 0
            self.ambulance_count_0to4 = 0
            self.ambulance_count_8to9 = 0
            self.ambulance_count_5to9 = 0
            self.ambulance_count_13to14 = 0
            self.ambulance_count_10to14 = 0
        '''
        # ------- Implicit Observation -------

        # No action, follow fixed TL defined in self.phases
        if action is None or action == {}:
            for _ in range(self.delta_time):
                self._sumo_step()
        else:
            self._apply_actions(action)

            for _ in range(self.yellow_time):
                self._sumo_step()
            for ts in self.ts_ids:
                self.traffic_signals[ts].update_phase()
            for _ in range(self.delta_time - self.yellow_time):
                self._sumo_step()

        # observe new state and reward
        observation = self._compute_observations()
        reward = self._compute_rewards()
        done = {'__all__': self.sim_step > self.sim_max_time}
        info = self._compute_step_info()
        self.metrics.append(info)
        self.last_reward = reward

        # ------- Implicit Observation -------
        veh_complete_data = self._get_complete_data()
        distance_reward, distance_val_per_road = self._compute_distance()
        default_vehicles_distance_reward, default_vehicles_distance_val_per_road = self._compute_default_vehicles_distance()
        self.last_distance_reward = distance_reward
        self.last_default_vehicles_distance_reward = default_vehicles_distance_reward
        phase_id_list = self._compute_phase_id_list()
        importance_weight_list = self._compute_importance_weight_list()
        test_keep_list, test_give_up_list = self._compute_test_keep_list()
        time_on_phase_list = self._compute_time_on_phase_list()
        self.neighbors_list = self._get_neighbors_list_importance()
        self.controlledlanes_list = self._get_controlledlanes_list()
        self.importance_weight_list = self._get_importance_weight_list()
        veh_position_data = self._get_position_data()
        veh_waiting_time = self._get_waiting_time()
        # ------- Implicit Observation -------

        if self.single_agent:
            return observation[self.ts_ids[0]], reward[self.ts_ids[0]], done['__all__'], {}
        else:
            # TODO: why need these information
            return self.neighbors_list, time_on_phase_list, phase_id_list, self.counter, veh_position_data, veh_waiting_time, veh_complete_data, observation, reward, \
                   done['__all__'], {}

    def _apply_actions(self, actions):
        """
        Set the next green phase for the traffic signals
        :param actions: If single-agent, actions is an int between 0 and self.num_green_phases (next green phase)
                        If multiagent, actions is a dict {ts_id : greenPhase}
        """
        if self.single_agent:
            self.traffic_signals[self.ts_ids[0]].set_next_phase(actions)
        else:
            for ts, action in actions.items():
                self.traffic_signals[ts].set_next_phase(action)

    def _compute_observations(self):
        """
        Return the current observation for each traffic signal
        """
        observations = {}
        phase_id_list = {}
        for ts in self.ts_ids:
            phase_id = [1 if self.traffic_signals[ts].phase // 2 == i else 0 for i in
                        range(self.num_green_phases)]  # one-hot encoding
            # TODO: phase = range(4), num_green_phases = 2, meaning?
            # phase = 0: phase_id = [1,0]
            # phase = 1: phase_id = [1,0]
            # phase = 2: phase_id = [0,1]
            # phase = 3: phase_id = [0,1]
            # elapsed = self.traffic_signals[ts].time_on_phase / self.max_green
            density = self.traffic_signals[ts].get_lanes_density()
            queue = self.traffic_signals[ts].get_lanes_queue()
            observations[ts] = phase_id + density + queue
            phase_id_list[ts] = phase_id
        return observations

    def _compute_phase_id_list(self):
        """
        Return the current phase indicator for each traffic signal
        """
        phase_id_list = {}
        for ts in self.ts_ids:
            phase_id = self.traffic_signals[ts].phase
            phase_id_list[ts] = phase_id
        return phase_id_list

    def _compute_rewards(self):
        return self._waiting_time_reward()
        # return self._pressure_reward()
        # return self._queue_reward()
        # return self._waiting_time_reward2()
        # return self._queue_average_reward()

    def _pressure_reward(self):
        rewards = {}
        for ts in self.ts_ids:
            rewards[ts] = -self.traffic_signals[ts].get_pressure()
        return rewards

    def _queue_average_reward(self):
        rewards = {}
        for ts in self.ts_ids:
            new_average = np.mean(self.traffic_signals[ts].get_stopped_vehicles_num())
            rewards[ts] = self.last_measure[ts] - new_average
            self.last_measure[ts] = new_average
        return rewards

    def _queue_reward(self):
        rewards = {}
        for ts in self.ts_ids:
            rewards[ts] = - (sum(self.traffic_signals[ts].get_stopped_vehicles_num())) ** 2
        return rewards

    def _waiting_time_reward(self):
        rewards = {}
        for ts in self.ts_ids:
            ts_wait = sum(self.traffic_signals[ts].get_waiting_time_per_lane())
            rewards[ts] = self.last_measure[ts] - ts_wait
            self.last_measure[ts] = ts_wait
        return rewards

    def _waiting_time_reward2(self):
        rewards = {}
        for ts in self.ts_ids:
            ts_wait = sum(self.traffic_signals[ts].get_waiting_time())
            self.last_measure[ts] = ts_wait
            if ts_wait == 0:
                rewards[ts] = 1.0
            else:
                rewards[ts] = 1.0 / ts_wait
        return rewards

    def _waiting_time_reward3(self):
        rewards = {}
        for ts in self.ts_ids:
            ts_wait = sum(self.traffic_signals[ts].get_waiting_time())
            rewards[ts] = -ts_wait
            self.last_measure[ts] = ts_wait
        return rewards

    def _sumo_step(self):
        traci.simulationStep()

    def _compute_step_info(self):
        return {
            'step_time': self.sim_step,
            'reward': self.last_reward[self.ts_ids[0]],
            'total_stopped': sum(self.traffic_signals[ts].get_total_queued() for ts in self.ts_ids),
            'total_wait_time': sum(self.last_measure[ts] for ts in self.ts_ids)
            # 'total_wait_time': sum([sum(self.traffic_signals[ts].get_waiting_time()) for ts in self.ts_ids])
        }

    def close(self):
        traci.close()

    def save_csv(self, out_csv_name, run):
        if out_csv_name is not None:
            df = pd.DataFrame(self.metrics)
            df.to_csv(out_csv_name + '_run{}'.format(run) + '.csv', index=False)

    # Below functions are for discrete state space

    def encode(self, state):
        phase = state[:self.num_green_phases].index(1)
        # elapsed = self._discretize_elapsed_time(state[self.num_green_phases])
        density_queue = [self._discretize_density(d) for d in state[self.num_green_phases:]]
        return self.radix_encode([phase] + density_queue)

    def _discretize_density(self, density):
        return min(int(density * 10), 9)

    def _discretize_elapsed_time(self, elapsed):
        elapsed *= self.max_green
        for i in range(self.max_green // self.delta_time):
            if elapsed <= self.delta_time + i * self.delta_time:
                return i
        return self.max_green // self.delta_time - 1

    def radix_encode(self, values):
        res = 0
        for i in range(len(self.radix_factors)):
            res = res * self.radix_factors[i] + values[i]
        return int(res)

    def radix_decode(self, value):
        res = [0 for _ in range(len(self.radix_factors))]
        for i in reversed(range(len(self.radix_factors))):
            res[i] = value % self.radix_factors[i]
            value = value // self.radix_factors[i]
        return res

    # ------- Implicit Observation -------

    def _get_complete_data(self):
        """
        Return the current observation for each traffic signal
        """
        veh_complete_data = Dict()
        for ts in self.ts_ids:  # ts means all traffic signals in the network
            veh_complete_data[ts] = self.traffic_signals[ts].get_complete_data()
        return veh_complete_data

    def _get_position_data(self):
        """
        Return the current observation for each traffic signal
        """
        veh_position_data = Dict()  # {tl: {veh_id: getLanePosition}}
        for ts in self.ts_ids:  # ts means all traffic signals in the network
            veh_position_data[ts] = self.traffic_signals[ts].get_position_data()
        return veh_position_data

    def _get_waiting_time(self):
        """
        Return the current observation for each traffic signal
        """
        veh_waiting_time = Dict()
        for ts in self.ts_ids:  # ts means all traffic signals in the network
            veh_waiting_time[ts] = self.traffic_signals[ts].get_waiting_time()
        return veh_waiting_time

    def _compute_distance(self):
        rewards = {}
        for ts in self.ts_ids:
            distance_val_per_road = self.traffic_signals[ts].get_distance_val()  # {road: {veh: getLanePos}}
            for key in distance_val_per_road:  # key = road name
                distance_val_per_vehicle = distance_val_per_road[key]
                rewards[key] = 0
                set_reward_zero = 0
                for veh in distance_val_per_vehicle:
                    ts_distance = distance_val_per_vehicle[veh]
                    if veh in self.last_distance[ts]:
                        reward = ts_distance - self.last_distance[ts][veh]  # positive if approaching the intersection
                    else:
                        reward = ts_distance
                    self.last_distance[ts][veh] = ts_distance
                    if reward == 0:
                        set_reward_zero = 1
                    rewards[key] += reward  # accumulative
                if set_reward_zero == 1:
                    rewards[key] = 0  # but when any vehicle is standstill
        return rewards, distance_val_per_road

    def _compute_default_vehicles_distance(self):
        rewards = {}
        for ts in self.ts_ids:
            default_vehicles_distance_val_per_road = self.traffic_signals[ts].get_default_vehicles_distance_val()
            for key in default_vehicles_distance_val_per_road:
                default_vehicles_distance_val_per_vehicle = default_vehicles_distance_val_per_road[key]
                rewards[key] = 0
                set_reward_zero = 0
                for veh in default_vehicles_distance_val_per_vehicle:
                    ts_distance = default_vehicles_distance_val_per_vehicle[veh]
                    if veh in self.last_default_vehicles_distance[ts]:
                        reward = ts_distance - self.last_default_vehicles_distance[ts][veh]
                    else:
                        reward = ts_distance
                    self.last_default_vehicles_distance[ts][veh] = ts_distance
                    if reward == 0:
                        set_reward_zero = 1
                    rewards[key] += reward
                if set_reward_zero == 1:
                    rewards[key] = 0
        return rewards, default_vehicles_distance_val_per_road

    def _compute_importance_weight_list(self):
        """
        Return the current phase indicator for each traffic signal
        """
        importance_weight_list = {}
        for ts in self.ts_ids:
            importance_weight_list[ts] = self.traffic_signals[ts].get_importance_weight_val()
        return importance_weight_list

    def _compute_test_keep_list(self):
        """
        for active phase switching on considering important vehicles
        """
        test_keep_list = {}
        test_give_up_list = {}
        for ts in self.ts_ids:
            keep, give_up = self.traffic_signals[ts].test_keep()
            test_keep_list[ts] = keep
            test_give_up_list[ts] = give_up
        return test_keep_list, test_give_up_list

    def _compute_time_on_phase_list(self):
        """
        Return the current phase indicator for each traffic signal
        """
        time_on_phase_list = {}
        for ts in self.ts_ids:
            time_on_phase = self.traffic_signals[ts].time_on_phase
            time_on_phase_list[ts] = time_on_phase
        return time_on_phase_list

    def _get_neighbors_list_random(self):
        """
        randomly select some intersection in the network as neighbors
        set by "number_of_neighbors = 1"
        """
        rnd = str(random.randint(0, 15))
        number_of_neighbors = 1
        neighbors_list = {}
        for ts in self.ts_ids:
            neighbors_list_temp = []
            cnt = 0
            while cnt < number_of_neighbors:
                while rnd == ts or rnd in neighbors_list_temp:
                    rnd = str(random.randint(0, 15))
                neighbors_list_temp.append(rnd)
                cnt = cnt + 1
            neighbors_list[ts] = neighbors_list_temp
        return neighbors_list

    def _get_neighbors_list_closeness(self):
        number_of_neighbors = 3  # TODO: why 3 not 4?
        neighbors_list = {}

        # defining list include position of each traffic signal
        point = {}
        point['0'] = np.array((1, 4))
        point['1'] = np.array((2, 4))
        point['2'] = np.array((3, 4))
        point['3'] = np.array((4, 4))
        point['4'] = np.array((1, 3))
        point['5'] = np.array((2, 3))
        point['6'] = np.array((3, 3))
        point['7'] = np.array((4, 3))
        point['8'] = np.array((1, 2))
        point['9'] = np.array((2, 2))
        point['10'] = np.array((3, 2))
        point['11'] = np.array((4, 2))
        point['12'] = np.array((1, 1))
        point['13'] = np.array((2, 1))
        point['14'] = np.array((3, 1))
        point['15'] = np.array((4, 1))

        for ts in self.ts_ids:
            distance = {}
            neighbors_list_temp = []
            for neighbor_id in self.ts_ids:
                distance[neighbor_id] = np.linalg.norm(point[ts] - point[neighbor_id])
            sorted_closeness_list = sorted(distance.items(), key=lambda x: x[1], reverse=False)
            cnt = 0
            for key in sorted_closeness_list:
                if cnt < number_of_neighbors and key[0] != ts:
                    neighbors_list_temp.append(key[0])
                    cnt = cnt + 1
            neighbors_list[ts] = neighbors_list_temp
        return neighbors_list

    def _get_neighbors_list_importance(self):
        # TODO: what is the three stage? why the neighbor selected depends on the previous stage?
        number_of_neighbors_selected_in_first_stage = 1
        number_of_neighbors_selected_in_second_stage = 1
        number_of_neighbors_selected_in_third_stage = 1
        neighbors_list = {}
        importance_weight_list_per_ts = self._get_importance_weight_list_per_ts()

        # defining list include position of each traffic signal
        point = {}
        point['0'] = np.array((1, 4))
        point['1'] = np.array((2, 4))
        point['2'] = np.array((3, 4))
        point['3'] = np.array((4, 4))
        point['4'] = np.array((1, 3))
        point['5'] = np.array((2, 3))
        point['6'] = np.array((3, 3))
        point['7'] = np.array((4, 3))
        point['8'] = np.array((1, 2))
        point['9'] = np.array((2, 2))
        point['10'] = np.array((3, 2))
        point['11'] = np.array((4, 2))
        point['12'] = np.array((1, 1))
        point['13'] = np.array((2, 1))
        point['14'] = np.array((3, 1))
        point['15'] = np.array((4, 1))

        for ts in self.ts_ids:
            importance_weight_list_with_distance = {}
            neighbors_list_temp = []
            for neighbor_id in self.ts_ids:
                distance = np.linalg.norm(point[ts] - point[neighbor_id])
                if distance != 0:
                    reverse_distance = 1 / distance
                else:
                    reverse_distance = 0
                importance_weight_list_with_distance[neighbor_id] = [reverse_distance,
                                                                     importance_weight_list_per_ts[neighbor_id]]

            sorted_importance_weight_list_with_distance = sorted(importance_weight_list_with_distance.items(),
                                                                 key=lambda x: x[1], reverse=True)

            cnt = 0
            for key in sorted_importance_weight_list_with_distance:
                if cnt < number_of_neighbors_selected_in_first_stage and key[0] != ts:
                    neighbors_list_temp.append(key[0])
                    first_selected_neighbor = key[0]
                    cnt = cnt + 1

            importance_weight_list_with_distance = {}
            for neighbor_id in self.ts_ids:
                distance = np.linalg.norm(point[first_selected_neighbor] - point[neighbor_id])
                if distance != 0:
                    reverse_distance = 1 / distance
                else:
                    reverse_distance = 0
                importance_weight_list_with_distance[neighbor_id] = [reverse_distance,
                                                                     importance_weight_list_per_ts[neighbor_id]]

            sorted_importance_weight_list_with_distance = sorted(importance_weight_list_with_distance.items(),
                                                                 key=lambda x: x[1], reverse=True)

            cnt = 0
            for key in sorted_importance_weight_list_with_distance:
                if cnt < number_of_neighbors_selected_in_second_stage and key[0] != ts:
                    neighbors_list_temp.append(key[0])
                    second_selected_neighbor = key[0]
                    cnt = cnt + 1

            importance_weight_list_with_distance = {}
            for neighbor_id in self.ts_ids:
                distance = np.linalg.norm(point[second_selected_neighbor] - point[neighbor_id])
                if distance != 0:
                    reverse_distance = 1 / distance
                else:
                    reverse_distance = 0
                importance_weight_list_with_distance[neighbor_id] = [reverse_distance,
                                                                     importance_weight_list_per_ts[neighbor_id]]

            sorted_importance_weight_list_with_distance = sorted(importance_weight_list_with_distance.items(),
                                                                 key=lambda x: x[1], reverse=True)

            cnt = 0
            for key in sorted_importance_weight_list_with_distance:
                if cnt < number_of_neighbors_selected_in_third_stage and key[0] != ts:
                    neighbors_list_temp.append(key[0])
                    cnt = cnt + 1
            neighbors_list[ts] = neighbors_list_temp
        return neighbors_list

    def _get_controlledlanes_list(self):
        # TODO: same as self.traffic_signals[ts].lanes?
        controlledlanes_list = {}
        for ts in self.ts_ids:
            controlledlanes_list[ts] = self.traffic_signals[ts].getControlledLanes()
        return controlledlanes_list

    def _get_importance_weight_list(self):
        importance_weight_list = {}
        for ts in self.ts_ids:
            importance_weight = self.traffic_signals[ts].get_importance_weight_val()
            importance_weight_list[ts] = importance_weight
        return importance_weight_list

    def _get_importance_weight_list_per_ts(self):
        importance_weight_list_per_ts = {}
        for ts in self.ts_ids:
            importance_weight = self.traffic_signals[ts].get_importance_weight_val_per_ts()
            importance_weight_list_per_ts[ts] = importance_weight
        return importance_weight_list_per_ts
