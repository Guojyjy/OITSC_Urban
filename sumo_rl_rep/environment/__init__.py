from gymnasium.envs.registration import register

register(
    id='sumo-rl-v0',
    entry_point='sumo_rl_rep.environment.env:SumoEnvironment',
    kwargs={'single_agent': True},
)
