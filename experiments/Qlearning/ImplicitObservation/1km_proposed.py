import csv
import os
import sys

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

ABS_DIR = os.path.abspath(os.path.dirname(__file__)).split('experiments')[0]  # absolute path for the project
sys.path.append(ABS_DIR)
from sumo_rl_rep.environment.env import UrbanEnvImplicitObservation
from sumo_rl_rep.agents.ql_agent import QLAgent
from sumo_rl_rep.exploration.epsilon_greedy import EpsilonGreedy

if __name__ == '__main__':

    # training parameters
    alpha = 0.1
    gamma = 0.99
    decay = 1
    runs = 2

    env = UrbanEnvImplicitObservation(net_file=ABS_DIR + 'nets/1km/selected1km.net.xml',
                                      route_file=ABS_DIR + 'nets/1km/selected1km_1kSec.rou.xml',
                                      horizon=1800,
                                      warmup=100,
                                      delta_time=5,
                                      yellow_time=2,
                                      min_green=5,
                                      max_green=100,
                                      single_agent=False,
                                      use_gui=False,
                                      additional_sumo_cmd=['--tripinfo-output',
                                                           'ImplicitObservation-QL-tripinfo-proposed-run{}-iter'.format(
                                                               runs)],
                                      out_csv_name='ImplicitObservation-QL-output-proposed-run{}'.format(runs),
                                      oitsc=True
                                      )

    for run in range(1, runs + 9):
        initial_states = env.reset()  # return obs

        ql_agents = {ts: QLAgent(starting_state=env.encode(initial_states[ts]),
                                 state_space=env.observation_space,
                                 action_space=env.action_space,
                                 alpha=alpha,
                                 gamma=gamma,
                                 exploration_strategy=EpsilonGreedy(initial_epsilon=0.05, min_epsilon=0.005,
                                                                    decay=decay)) for ts in env.ts_ids}

        done = {'__all__': False}

        while not done['__all__']:
            actions = {ts: ql_agents[ts].act() for ts in ql_agents.keys()}

            veh_waiting_all, veh_type, observations, rewards, done, info = env.step(action_dict=actions)
            output_data = [["veh_id", "veh_type", "waiting_time_total"]]
            for veh in veh_waiting_all.keys():
                each_veh_info = [veh, veh_type[veh], veh_waiting_all[veh]]
                output_data.append(each_veh_info)

            for agent_id in ql_agents.keys():
                reward = rewards[agent_id]
                next_state = str(env.encode(observations[agent_id]))
                ql_agents[agent_id].learn(next_state, reward)

        outfile = open('ImplicitObservation-QL-evaluation-proposed_run{}_iter{}.csv'.format(runs, run), 'w')
        writer = csv.writer(outfile)
        writer.writerows(output_data)
        outfile.close()

        env.close()
