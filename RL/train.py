from common.env.procgen_wrappers import *
from common.logger import Logger
from common.storage import Storage
from common.model import Policy, ImpalaModel
from common import set_global_seeds, set_global_log_levels
from agents.ppo import PPO
import os, time, yaml, argparse
from procgen import ProcgenEnv
import random
import torch

import torch.backends.cudnn as cudnn

cudnn.benchmark = True


def boolean_string(s):
    if s not in {"False", "True"}:
        raise ValueError("Not a valid boolean string")
    return s == "True"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_name", type=str, default="experiment", help="experiment name"
    )
    parser.add_argument(
        "--env_name", type=str, default="coinrun", help="environment ID"
    )
    parser.add_argument(
        "--start_level", type=int, default=int(0), help="start-level for environment"
    )
    parser.add_argument(
        "--num_levels",
        type=int,
        default=int(200),
        help="number of training levels for environment",
    )
    parser.add_argument(
        "--distribution_mode",
        type=str,
        default="easy",
        help="distribution mode for environment",
    )
    parser.add_argument(
        "--param_name", type=str, default="easy-200", help="hyper-parameter ID"
    )
    parser.add_argument(
        "--device", type=str, default="gpu", required=False, help="whether to use gpu"
    )
    # parser.add_argument('--gpu_device',       type=int, default = int(0), required = False, help = 'visible device in CUDA')
    parser.add_argument(
        "--num_timesteps",
        type=int,
        default=int(25000000),
        help="number of training timesteps",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=random.randint(0, 9999),
        help="Random generator seed",
    )
    parser.add_argument("--log_level", type=int, default=int(40), help="[10,20,30,40]")
    parser.add_argument(
        "--num_checkpoints",
        type=int,
        default=int(1),
        help="number of checkpoints to store",
    )

    parser.add_argument(
        "--restrict_themes",
        type=boolean_string,
        default="False",
        help="games will only use a single theme",
    )
    parser.add_argument(
        "--use_backgrounds",
        type=boolean_string,
        default="True",
        help="distribution mode for environment",
    )
    parser.add_argument(
        "--use_monochrome_assets",
        type=boolean_string,
        default="False",
        help="games will use monochromatic rectangles instead of human designed assets",
    )
    parser.add_argument("--clop", type=float, default=float(0), help="clop layer")
    parser.add_argument("--dropout", type=float, default=float(0), help="dropout")

    args = parser.parse_args()
    exp_name = args.exp_name
    env_name = args.env_name
    start_level = args.start_level
    num_levels = args.num_levels
    distribution_mode = args.distribution_mode
    param_name = args.param_name
    device = args.device
    # gpu_device = args.gpu_device
    num_timesteps = args.num_timesteps
    seed = args.seed
    log_level = args.log_level
    num_checkpoints = args.num_checkpoints

    restrict_themes = args.restrict_themes
    use_backgrounds = args.use_backgrounds
    use_monochrome_assets = args.use_monochrome_assets
    dropout = args.dropout
    clop = args.clop

    set_global_seeds(seed)
    set_global_log_levels(log_level)

    ####################
    ## HYPERPARAMETERS #
    ####################
    print("[LOADING HYPERPARAMETERS...]")
    with open("hyperparams/procgen/config.yml", "r") as f:
        hyperparameters = yaml.safe_load(f)[param_name]
    for key, value in hyperparameters.items():
        print(key, ":", value)

    print(f"Use backround:{use_backgrounds}")
    print(f"Restrict_themes:{restrict_themes}")
    print(f"Monochrome Assets:{use_monochrome_assets}")

    ############
    ## DEVICE ##
    ############
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_device)
    device = torch.device("cuda")

    #################
    ## ENVIRONMENT ##
    #################
    print("INITIALIZAING ENVIRONMENTS...")
    n_steps = hyperparameters.get("n_steps", 256)
    n_envs = hyperparameters.get("n_envs", 64)
    # By default, pytorch utilizes multi-threaded cpu
    # Procgen is able to handle thousand of steps on a single core
    torch.set_num_threads(1)
    env = ProcgenEnv(
        num_envs=n_envs,
        env_name=env_name,
        start_level=start_level,
        num_levels=num_levels,
        distribution_mode=distribution_mode,
        restrict_themes=restrict_themes,
        use_backgrounds=use_backgrounds,
        use_monochrome_assets=use_monochrome_assets,
    )
    normalize_rew = hyperparameters.get("normalize_rew", True)
    env = VecExtractDictObs(env, "rgb")
    if normalize_rew:
        env = VecNormalize(
            env, ob=False
        )  # normalizing returns, but not the img frames.
    env = TransposeFrame(env)
    env = ScaledFloatFrame(env)

    # test env
    if distribution_mode == "hard":
        test_start_level = start_level + num_levels
        test_num_level = 1000
    else:
        test_start_level = 0
        test_num_level = 0

    test_env = ProcgenEnv(
        num_envs=256,
        env_name=env_name,
        start_level=test_start_level,
        num_levels=test_num_level,
        distribution_mode=distribution_mode,
        use_backgrounds=True,
    )
    test_env = VecExtractDictObs(test_env, "rgb")
    if normalize_rew:
        test_env = VecNormalize(test_env, ob=False)
    test_env = TransposeFrame(test_env)
    test_env = ScaledFloatFrame(test_env)

    ############
    ## LOGGER ##
    ############
    print("INITIALIZAING LOGGER...")
    logdir = (
        "procgen/"
        + env_name
        + "/"
        + exp_name
        + "/"
        + "seed"
        + "_"
        + str(seed)
        + "_"
        + time.strftime("%d-%m-%Y_%H-%M-%S")
    )
    logdir = os.path.join("logs", logdir)
    if not (os.path.exists(logdir)):
        os.makedirs(logdir)
    logger = Logger(n_envs, logdir, {**hyperparameters, **vars(args)})
    
    ###########

    ###########
    print("INTIALIZING MODEL...")
    observation_space = env.observation_space
    observation_shape = observation_space.shape
    architecture = hyperparameters.get("architecture", "impala")
    in_channels = observation_shape[0]
    action_space = env.action_space

    model = ImpalaModel(in_channels=in_channels)

    action_size = action_space.n
    policy = Policy(model, action_size, dropout, clop)

    if torch.cuda.device_count() > 1:
        print(f"RUNNING ON {torch.cuda.device_count()} GPUs")
        policy = torch.nn.DataParallel(policy)
    policy.to(device)

    #############
    ## STORAGE ##
    #############
    print("INITIALIZAING STORAGE...")
    storage = Storage(observation_shape, n_steps, n_envs, device)

    ###########
    ## AGENT ##
    ###########
    print("INTIALIZING AGENT...")
    agent = PPO(
        env,
        test_env,
        policy,
        logger,
        storage,
        device,
        num_checkpoints,
        **hyperparameters,
    )

    ##############
    ## TRAINING ##
    ##############
    print("START TRAINING...")
    agent.train(num_timesteps)
