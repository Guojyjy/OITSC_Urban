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
from sumo_rl_rep.agents.sarsa_lambda import TrueOnlineSarsaLambda

if __name__ == '__main__':

    # training parameters
    alpha = 0.000000001
    gamma = 0.95
    epsilon = 0.05
    lamb = 0.1
    fourier_order = 7
    runs = 1

    env = UrbanEnvImplicitObservation(net_file=ABS_DIR + 'nets/1km/selected1km.net.xml',
                                      route_file=ABS_DIR + 'nets/1km/selected1km_1kSec.rou.xml',
                                      horizon=1800,
                                      warmup=100,
                                      delta_time=5,
                                      yellow_time=2,  # TODO: yellow setting:4?
                                      min_green=5,
                                      max_green=100,
                                      single_agent=False,
                                      use_gui=False,
                                      additional_sumo_cmd=['--tripinfo-output',
                                                           'ImplicitObservation-SARSA-tripinfo-proposed-run{}-iter'.format(
                                                               runs)],
                                      out_csv_name='ImplicitObservation-SARSA-output-proposed-run{}'.format(runs),
                                      oitsc=True
                                      )

    for run in range(1, runs + 2):
        initial_states = env.reset()  # return obs
        last_obs = {}

        sarsa_agents = {ts: TrueOnlineSarsaLambda(state_space=env.observation_space,
                                                  action_space=env.action_space,
                                                  alpha=alpha,
                                                  gamma=gamma,
                                                  epsilon=epsilon,
                                                  lamb=lamb,
                                                  fourier_order=fourier_order) for ts in env.ts_ids}

        done = {'__all__': False}

        while not done['__all__']:

            if last_obs == {}:
                last_obs = initial_states

            actions = {ts: sarsa_agents[ts].act(sarsa_agents[ts].get_features(last_obs[ts]))
                       for ts in sarsa_agents.keys()}

            veh_waiting_all, veh_type, observations, rewards, done, info = env.step(action_dict=actions)
            output_data = [["veh_id", "veh_type", "waiting_time_total"]]
            for veh in veh_waiting_all.keys():
                each_veh_info = [veh, veh_type[veh], veh_waiting_all[veh]]
                output_data.append(each_veh_info)

            for agent_id in sarsa_agents.keys():
                sarsa_agents[agent_id].learn(state=last_obs[agent_id],
                                             action=actions[agent_id],
                                             reward=rewards[agent_id],
                                             next_state=observations[agent_id],
                                             done=done['__all__'])

            last_obs = observations

        outfile = open('ImplicitObservation-SARSA-evaluation-proposed_run{}_iter{}.csv'.format(runs, run), 'w')
        writer = csv.writer(outfile)
        writer.writerows(output_data)
        outfile.close()

        env.close()
