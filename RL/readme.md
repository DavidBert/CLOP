To run an experiment on procgen:
`
python train.py --exp_name [XP_NAME] --env_name [GAME]
`
Or with more parameters:
`
python train.py --exp_name [XP_NAME] --env_name [GAME] --param_name [PARAM] --num_levels [NUM_LEVEL] --distribution_mode [MODE] --num_timesteps [TIMESTEPS] --start_level [START]
`
*  XP_NAME: experiment name for logs
*  GAME: environment name 
*  PARAM: one of the configurations available in `hyperparams/procgen/config.yml`
* NUM_LEVEL: number of unique training levels
* START: starting training level
* MODE: `easy` or `hard`
* TIMESTEPS  umber of training steps

Exemple:
`
python train.py --exp_name xp_bigfish --env_name bigfish --param_name easy-200 --num_levels 200 --distribution_mode easy --num_timesteps 25000000 --start_level 0 --clop 0.6
`

To train a clop agent add `--clop [alpha]` with $0 \leq \alpha \leq 1$.  
To train an agent with dropout add `--dropout [alpha]` with $0 \leq \alpha \leq 1$.