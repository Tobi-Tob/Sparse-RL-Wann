from collections import namedtuple
import numpy as np

Game = namedtuple('Game', ['env_name', 'time_factor', 'actionSelect',
                           'input_size', 'output_size', 'layers', 'i_act', 'h_act',
                           'o_act', 'weightCap', 'noise_bias', 'output_noise', 'max_episode_length', 'in_out_labels'])

games = {}

# -- Car Racing  --------------------------------------------------------- -- #

# > 32 latent vectors (includes past frames)
vae_racing_stack = Game(env_name='VAERacingStack-v0',
                        actionSelect='all',  # all, soft, hard
                        input_size=32,
                        output_size=3,
                        time_factor=0,
                        layers=[10, 0],
                        i_act=np.full(32, 1),
                        h_act=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                        o_act=np.full(3, 1),
                        weightCap=2.0,
                        noise_bias=0.0,
                        max_episode_length=1000,
                        output_noise=[False, False, False],
                        in_out_labels=['latent01', 'latent02', 'latent03', 'latent04', 'latent05',
                                       'latent06', 'latent07', 'latent08', 'latent09', 'latent10',
                                       'latent11', 'latent12', 'latent13', 'latent14', 'latent15',
                                       'latent16', 'latent17', 'latent18', 'latent19', 'latent20',
                                       'latent21', 'latent22', 'latent23', 'latent24', 'latent25',
                                       'latent26', 'latent27', 'latent28', 'latent29', 'latent30',
                                       'latent31', 'latent32', 'steer', 'gas', 'brakes']
                        )
games['vae_racing_stack'] = vae_racing_stack

# > 16 latent vectors (current frame only)
vae_racing = vae_racing_stack._replace(
    env_name='VAERacing-v0', input_size=16, i_act=np.full(16, 1),
    in_out_labels=['latent01', 'latent02', 'latent03', 'latent04', 'latent05',
                   'latent06', 'latent07', 'latent08', 'latent09', 'latent10',
                   'latent11', 'latent12', 'latent13', 'latent14', 'latent15',
                   'latent16', 'steer', 'gas', 'brakes'])
games['vae_racing'] = vae_racing

# -- Digit Classification ------------------------------------------------ -- #

# > Scikit learn digits data set
classify = Game(env_name='Classify_digits',
                actionSelect='softmax',  # all, soft, hard
                input_size=64,
                output_size=10,
                time_factor=0,
                layers=[128, 9],
                i_act=np.full(64, 1),
                h_act=[1, 3, 4, 5, 6, 7, 8, 9, 10],  # No step function
                o_act=np.full(10, 1),
                weightCap=2.0,
                noise_bias=0.0,
                output_noise=[False, False, False],
                max_episode_length=0,
                in_out_labels=[]
                )
L = [list(range(1, classify.input_size)),
     list(range(0, classify.output_size))]
label = [item for sublist in L for item in sublist]
classify = classify._replace(in_out_labels=label)
games['digits'] = classify

# > MNIST [16x16] data set
mnist256 = classify._replace(
    env_name='Classify_mnist256', input_size=256, i_act=np.full(256, 1))
L = [list(range(1, mnist256.input_size)),
     list(range(0, mnist256.output_size))]
label = [item for sublist in L for item in sublist]
mnist256 = mnist256._replace(in_out_labels=label)
games['mnist256'] = mnist256

# -- Cart-pole Swingup --------------------------------------------------- -- #

# > Slower reaction speed
cartpole_swingup = Game(env_name='CartPoleSwingUp_Hard',
                        actionSelect='all',  # all, soft, hard
                        input_size=5,
                        output_size=1,
                        time_factor=0,
                        layers=[5, 5],
                        i_act=np.full(5, 1),
                        h_act=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                        o_act=np.full(1, 5),
                        weightCap=2.0,
                        noise_bias=0.0,
                        output_noise=[False, False, False],
                        max_episode_length=200,
                        in_out_labels=['x', 'x_dot', 'cos(theta)', 'sin(theta)', 'theta_dot',
                                       'force']
                        )
games['swingup_hard'] = cartpole_swingup

# > Normal reaction speed
cartpole_swingup = cartpole_swingup._replace(
    env_name='CartPoleSwingUp', max_episode_length=1000)
games['swingup'] = cartpole_swingup

# -- Bipedal Walker ------------------------------------------------------ -- #

# > Flat terrain
biped = Game(env_name='BipedalWalker-v2',
             actionSelect='all',  # all, soft, hard
             input_size=24,
             output_size=4,
             time_factor=0,
             layers=[40, 40],
             i_act=np.full(24, 1),
             h_act=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
             o_act=np.full(4, 1),
             weightCap=2.0,
             noise_bias=0.0,
             output_noise=[False, False, False],
             max_episode_length=1600,
             in_out_labels=[
                 'hull_angle', 'hull_vel_angle', 'vel_x', 'vel_y',
                 'hip1_angle', 'hip1_speed', 'knee1_angle', 'knee1_speed', 'leg1_contact',
                 'hip2_angle', 'hip2_speed', 'knee2_angle', 'knee2_speed', 'leg2_contact',
                 'lidar_0', 'lidar_1', 'lidar_2', 'lidar_3', 'lidar_4',
                 'lidar_5', 'lidar_6', 'lidar_7', 'lidar_8', 'lidar_9',
                 'hip_1', 'knee_1', 'hip_2', 'knee_2']
             )
games['biped'] = biped

# -- Sparse Mountain Car ------------------------------------------------- -- #

sparse_mountain_car = Game(env_name='SparseMountainCar',
                           actionSelect='all',  # all, soft, hard
                           input_size=2,  # position, velocity
                           output_size=3,  # accelerate left, no acceleration, accelerate right (discrete values)
                           time_factor=0,
                           layers=[5, 5],
                           i_act=np.full(2, 1),
                           h_act=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                           o_act=np.full(3, 1),
                           weightCap=1.0,  # 2.0
                           noise_bias=0.0,
                           output_noise=[False, False, False],
                           max_episode_length=200,  # must be the same in sparse_mountain_car.py for stop return value
                           in_out_labels=['position', 'velocity',
                                          'accelerate\nleft', 'no\nacceleration', 'accelerate\nright']
                           )
games['sparse_mountain_car'] = sparse_mountain_car

sparse_mountain_car_conti = Game(env_name='SparseMountainCarConti',
                                 actionSelect='all',  # all, soft, hard
                                 input_size=2,  # position, velocity
                                 output_size=1,  # acceleration (continuous value)
                                 time_factor=0,
                                 layers=[5, 5],
                                 i_act=np.full(2, 1),
                                 h_act=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                                 o_act=np.full(1, 1),
                                 weightCap=1.0,  # 2.0
                                 noise_bias=0.0,
                                 output_noise=[False, False, False],
                                 max_episode_length=200,
                                 # must be the same in sparse_mountain_car_conti.py for stop return value
                                 in_out_labels=['position', 'velocity',
                                                'acceleration']
                                 )
games['sparse_mountain_car_conti'] = sparse_mountain_car_conti

# -- Lunar Lander ------------------------------------------------------------- -- #
lunar_lander = Game(env_name='LunarLander',
                    actionSelect='all',  # all, soft, hard
                    input_size=8,
                    # x, y coords, velocities x', y', angle, angular velocity, two booleans that represent whether each leg is in contact with the ground or not
                    output_size=4,  # do nothing, fire left engine, fire main engine, fire right engine
                    time_factor=0,
                    layers=[5, 5],  # not used...
                    i_act=np.full(8, 1),
                    h_act=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    o_act=np.full(4, 1),
                    weightCap=2.0,
                    noise_bias=0.0,
                    output_noise=[False, False, False],
                    max_episode_length=1000,  # must be the same in lunar_lander.py for stop return value
                    in_out_labels=['x', 'y', 'dx', 'dy', 'angle', 'angular\nvel', 'leg1', 'leg2',
                                   'do\nnothing', 'fire\nleft', 'fire\nmain', 'fire\nright']
                    )
games['lunar_lander'] = lunar_lander

lunar_lander_conti = Game(env_name='LunarLanderConti',
                          actionSelect='all',  # all, soft, hard
                          input_size=8,
                          # x, y coords, velocities x', y', angle, angular velocity, two booleans that represent whether each leg is in contact with the ground or not
                          output_size=2,  # main engine, lateral engine
                          time_factor=0,
                          layers=[5, 5],
                          i_act=np.full(8, 1),
                          h_act=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                          o_act=np.full(2, 1),
                          weightCap=2.0,
                          noise_bias=0.0,
                          output_noise=[False, False, False],
                          max_episode_length=1000,  # must be the same in lunar_lander.py for stop return value
                          in_out_labels=['x', 'y', 'dx', 'dy', 'angle', 'angular\nvel', 'leg1', 'leg2',
                                         'main', 'lateral']
                          )
games['lunar_lander_conti'] = lunar_lander_conti
