import os
import sys

import ray
from ray.tune.registry import register_env
from ray.rllib.algorithms import dqn
from ray.rllib.policy.policy import PolicySpec

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

ABS_DIR = os.path.abspath(os.path.dirname(__file__)).split('experiments')[0]  # absolute path for the project
sys.path.append(ABS_DIR)
from sumo_rl_rep.environment.env import UrbanEnvImplicitObservation


if __name__ == '__main__':
    runs = 2
    ray.init()
    env = UrbanEnvImplicitObservation(net_file=ABS_DIR + 'nets/1km/selected1km.net.xml',
                                      route_file=ABS_DIR + 'nets/1km/selected1km_1kSec.rou.xml',
                                      horizon=1800,
                                      warmup=100,
                                      delta_time=5,
                                      yellow_time=2,  # TODO: yellow light = 4sec?
                                      min_green=5,
                                      max_green=100,
                                      single_agent=False,
                                      use_gui=False,
                                      additional_sumo_cmd=['--tripinfo-output',
                                                           'ImplicitObservation-DQN-tripinfo-proposed-run{}-iter'.format
                                                           (runs)],
                                      out_csv_name='ImplicitObservation-DQN-output-proposed-run{}'.format(runs),
                                      oitsc=True
                                      )

    register_env(name="1km", env_creator=(lambda env_config=dict(): env))

    trainer = dqn.DQN(
        env="1km_proposed_ImplicitObservation",
        config=(
            dqn.DQNConfig()
            .multi_agent(
                policies={
                    "0": PolicySpec(dqn.DQNTFPolicy, env.observation_space, env.action_space, {})
                },
                policy_mapping_fn=(lambda agent_id, episode, worker, **kwargs: "0"),
            )
            .training(
                lr=1e-3,
            )
        )
    )

    while True:
        print(trainer.train())
