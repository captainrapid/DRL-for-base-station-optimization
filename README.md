ProgramIntroduction

  Based on the reinforcement learning framework, this project designs and compares a variety of classic and improved algorithms (DQN, DDPG, TD3, MADDPG, Greedy, Conservative) for the resource allocation problem of heterogeneous cellular networks powered by renewable energy.
  The performance of the algorithms in terms of throughput, cost, energy efficiency, QoS and other indicators is evaluated through simulation experiments.




Files

  train_n_evaluate_2.py   # Main script for training, evaluation and visualization
  environment_2.py        # Gym Environment Customization: BaseStationEnv
  agents_2.py             # Implementation for different algorithms: DQN、DDPG、TD3、MADDPG、Greedy、Conservative
  safety_2.py             # Safety Projection: Guarantee to meet the minimum requirements of QoS
  common_2.py             # Common contents for different algorithms:Actor,Critic,ReplayMemory,SoftUpdate,etc.




EnvironmentDependency
  
  Python 3.8+
  numpy
  torch
  gym
  matplotlib
  scipy




Parameters

  train_n_evaluate_2.py
  
    SEED = 5                     # Random Seed
    N = 2                        # Base Station Amount
    TOTAL_STEPS = 5000           # Total Training Steps
    EVAL_INTERVAL = 100          # Steps Between Each Two Evaluations
    EVAL_EPISODES = 100          # Episodes in Every Evaluation
    MAX_STEPS = 50               # Steps in Every Episode
    batch_size = 128             # Agent Batch Size
    tau = 0.005                  # Soft Update Factor（TD3）
    policy_noise = 0.2           # TD3 Policy Noise
    noise_clip = 0.5             # TD3 Noise Clip
    policy_delay = 2             # TD3 Policy Update Delay
  
  
  environment_2.py
  
    max_energy = 40.0            # Maximum Transit Power for a Base Station
    max_bandwidth = 20e6         # Base Station Max Bandwidth
    max_renew = 20.0             # Max Renewable Energy
    grid_price = [0.3,0.8,1.2]   # Grid Electricity Price
    obs_noise_sigma = 0.02       # Standard Deviation for Observation
    reward_noise_sigma = 0.05    # Standard Deviation for Rewards
    users_per_bs random [2,6]    # User Amount for each Base Station


