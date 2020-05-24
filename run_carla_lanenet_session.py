
# Code is a modified version of https://github.com/carla-simulator/data-collector collect.py
# Code Modified by: Hamptonjc


# Imports
from __future__ import print_function

from data_collector import collect

import os
import sys
import argparse
import logging
import random
import time

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

from data_collector.carla.client import make_carla_client

from data_collector.carla.tcp import TCPConnectionError
from data_collector.carla_game.carla_game import CarlaGame
from data_collector.carla.planner import Planner
from data_collector.carla.agent import HumanAgent, ForwardAgent, CommandFollower, LaneFollower
from data_collector.carla import image_converter

from data_collector.modules import data_writer as writer
from data_collector.modules.noiser import Noiser
from data_collector.modules.collision_checker import CollisionChecker

from lanenet_lane_detection.lanenet_model import lanenet_postprocess_carla
postprocessor = lanenet_postprocess_carla.LaneNetPostProcessor()


import tensorflow as tf
import cv2

WINDOW_WIDTH = 512
WINDOW_HEIGHT = 256
MINI_WINDOW_WIDTH = 320
MINI_WINDOW_HEIGHT = 180
NUMBER_OF_FRAMES_CAR_FLIES = 25




def run_sim(client, args):
    """
    The main loop for running lanenet with CARLA.
    Args:
        client: carla client object
        args: arguments passed on the data collection main.

    Returns:
        None

    """
    settings_module = __import__('data_collector.dataset_configurations.' + (args.data_configuration_name),
                                 fromlist=['dataset_configurations'])
    if not args.verbose:
        collect.suppress_logs(args.episode_number)
    carla_game = CarlaGame(False, args.debug, WINDOW_WIDTH, WINDOW_HEIGHT, MINI_WINDOW_WIDTH,
                           MINI_WINDOW_HEIGHT)
    collision_checker = CollisionChecker()

    ##### Start the episode #####
    # ! This returns all the aspects from the episodes.
    episode_aspects = collect.reset_episode(client, carla_game,
                                    settings_module, args.debug)
    planner = Planner(episode_aspects["town_name"])
    # We instantiate the agent, depending on the parameter
    controlling_agent = collect.make_controlling_agent(args, episode_aspects["town_name"])

    # The noise object to add noise to some episodes is instanced
    longitudinal_noiser = Noiser('Throttle', frequency=15, intensity=10, min_noise_time_amount=2.0)
    lateral_noiser = Noiser('Spike', frequency=25, intensity=4, min_noise_time_amount=0.5)

    episode_lateral_noise, episode_longitudinal_noise = collect.check_episode_has_noise(
        settings_module.lat_noise_percent,
        settings_module.long_noise_percent)

    # We start the episode number with the one set as parameter
    episode_number = args.episode_number
    try:
        image_count = 0

        ######################### LANENET SETUP #############################

        FROZEN_MODEL_PATH = './lanenet_lane_detection/frozen_lanenet/lanenet.pb'

        with tf.compat.v1.Session() as sess:
            with tf.io.gfile.GFile(FROZEN_MODEL_PATH, 'rb') as f:
                graph_def = tf.compat.v1.GraphDef()
                graph_def.ParseFromString(f.read())
                sess.graph.as_default()
                g_in = tf.import_graph_def(graph_def)
            input_tensor = sess.graph.get_tensor_by_name('import/lanenet/input_tensor:0')
            bianary_output = sess.graph.get_tensor_by_name('import/lanenet/final_binary_output:0')
            pixel_embedding_output = sess.graph.get_tensor_by_name('import/lanenet/final_pixel_embedding_output:0')

        ######################### RUN CARLA ################################
            maximun_episode = int(args.number_of_episodes) + int(args.episode_number)
            while carla_game.is_running() and episode_number < maximun_episode:

                # we add the vehicle and the connection outside of the game.
                measurements, sensor_data = client.read_data()
                input_image = image_converter.to_rgb_array(sensor_data['CameraRGB'])
                image = input_image
                input_image_processed = np.array(input_image)
                image = cv2.resize(np.array(image), (512, 256), interpolation=cv2.INTER_LINEAR)
                image = image / 127.5 - 1.0            

            ######################### LANENET PREDICT #########################
                binary_seg_image, instance_seg_image = sess.run(
                    [bianary_output, pixel_embedding_output],
                    feed_dict={input_tensor: [image]})


            ####################### LANENET POSTPROCESSING ####################
                input_with_mask = input_image[:, :, (2, 1, 0)]
                try:
                    postprocess_result = postprocessor.postprocess(
                        binary_seg_result=binary_seg_image,
                        instance_seg_result=instance_seg_image,
                        source_image=input_image_processed)
                    mask_image = postprocess_result['mask_image']
                    input_image_processed = postprocess_result['source_image']
                    input_with_mask = cv2.add(input_image[:, :, (2, 1, 0)], # <-change to input_image_processed for points (bug)
                        cv2.resize(mask_image[:, :, (2, 1, 0)],(800, 600), interpolation=cv2.INTER_LINEAR))
                except:
                    pass

            ######################### DISPLAY ################################

                cv2.imshow('img', input_with_mask)
                #cv2.imshow('lanenet_img', lanenet_img) #cv2.cvtColor(lanenet_img, cv2.COLOR_BGR2RGB))
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break


                # run a step for the agent. regardless of the type
                control, controller_state = controlling_agent.run_step(measurements,
                                                           sensor_data,
                                                           [],
                                                           episode_aspects['player_target_transform'])
                # Get the directions, also important to save those for future training

                directions = collect.get_directions(measurements,
                                            episode_aspects['player_target_transform'], planner)

                controller_state.update({'directions': directions})

                # if this is a noisy episode, add noise to the controls
                #TODO add a function here.
                if episode_longitudinal_noise:
                    control_noise, _, _ = longitudinal_noiser.compute_noise(control,
                                                measurements.player_measurements.forward_speed * 3.6)
                else:
                    control_noise = control

                if episode_lateral_noise:
                    control_noise_f, _, _ = lateral_noiser.compute_noise(control_noise,
                                                measurements.player_measurements.forward_speed * 3.6)
                else:
                    control_noise_f = control_noise



                # Set the player position
                # if you want to debug also render everything
                if args.debug:
                    objects_to_render = controller_state.copy()
                    objects_to_render['player_transform'] = measurements.player_measurements.transform
                    objects_to_render['agents'] = measurements.non_player_agents
                    objects_to_render["draw_pedestrians"] = args.draw_pedestrians
                    objects_to_render["draw_vehicles"] = args.draw_vehicles
                    objects_to_render["draw_traffic_lights"] = args.draw_traffic_lights
                    # Comment the following two lines to see the waypoints and routes.
                    objects_to_render['waypoints'] = None
                    objects_to_render['route'] = None

                    # Render with the provided map
                    carla_game.render(sensor_data['CameraRGB'], objects_to_render)

                # Check two important conditions for the episode, if it has ended
                # and if the episode was a success
                episode_ended = collision_checker.test_collision(measurements.player_measurements) or \
                                collect.reach_timeout(measurements.game_timestamp / 1000.0,
                                              episode_aspects["timeout"]) or \
                                carla_game.is_reset(measurements.player_measurements.transform.location)
                episode_success = not (collision_checker.test_collision(
                                       measurements.player_measurements) or
                                       collect.reach_timeout(measurements.game_timestamp / 1000.0,
                                                     episode_aspects["timeout"]))

                # Check if there is collision
                # Start a new episode if there is a collision but repeat the same by not incrementing
                # episode number.

                if episode_ended:
                    if episode_success:
                        episode_number += 1
                    else:
                        # If the episode did go well and we were recording, delete this episode
                        if not args.not_record:
                            writer.delete_episode(args.data_path, str(episode_number-1).zfill(5))

                    episode_lateral_noise, episode_longitudinal_noise = collect.check_episode_has_noise(
                        settings_module.lat_noise_percent,
                        settings_module.long_noise_percent)

                    # We reset the episode and receive all the characteristics of this episode.
                    episode_aspects = collect.reset_episode(client, carla_game,
                                                    settings_module, args.debug)

                    writer.add_episode_metadata(args.data_path, str(episode_number).zfill(5),
                                                episode_aspects)

                    # Reset the image count
                    image_count = 0

                # We do this to avoid the frames that the car is coming from the sky.
                if image_count >= NUMBER_OF_FRAMES_CAR_FLIES and not args.not_record:
                    writer.add_data_point(measurements, control, control_noise_f, sensor_data,
                                          controller_state,
                                          args.data_path, str(episode_number).zfill(5),
                                          str(image_count - NUMBER_OF_FRAMES_CAR_FLIES),
                                          settings_module.sensors_frequency)
                # End the loop by sending control
                client.send_control(control_noise_f)
                # Add one more image to the counting
                image_count += 1
        sess.close()


    except TCPConnectionError as error:
        """
        If there is any connection error we delete the current episode, 
        This avoid incomplete episodes
        """
        import traceback
        traceback.print_exc()
        if not args.not_record:
            writer.delete_episode(args.data_path, str(episode_number).zfill(5))

        raise error

    except KeyboardInterrupt:
        import traceback
        traceback.print_exc()
        if not args.not_record:
            writer.delete_episode(args.data_path, str(episode_number).zfill(5))




def main():
    """
    The main function of the data collection process

    """
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='verbose',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='localhost',
        help='IP of the host server (default: localhost)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-pt','--data-path',
        metavar='H',
        default='.',
        dest='data_path',
        help=' Where the recorded data will be placed')
    argparser.add_argument(
        '--data-configuration-name',
        metavar='H',
        default='coil_training_dataset_singlecamera',
        dest='data_configuration_name',
        help=' Name of the data configuration file that should be place on .dataset_configurations')
    argparser.add_argument(
        '-c', '--controlling_agent',
        default='CommandFollower',
        help='the controller that is going to be used by the main vehicle.'
             ' Options: '
             ' HumanAgent - Control your agent with a keyboard.'
             ' ForwardAgent - A trivial agent that goes forward'
             ' LaneFollower - An agent that follow lanes and stop obstacles'
             ' CommandFollower - A lane follower agent that follow commands from the planner')
    argparser.add_argument(
        '-db', '--debug',
        action='store_true',
        help='enable the debug screen mode, on this mode a rendering screen will show'
             'information about the agent')
    argparser.add_argument(
        '-dp', '--draw-pedestrians',
        dest='draw_pedestrians',
        action='store_true',
        help='add pedestrians to the debug screen')
    argparser.add_argument(
        '-dv', '--draw-vehicles',
        dest='draw_vehicles',
        action='store_true',
        help='add vehicles dots to the debug screen')
    argparser.add_argument(
        '-dt', '--draw-traffic-lights',
        dest='draw_traffic_lights',
        action='store_true',
        help='add traffic lights dots to the debug screen')
    argparser.add_argument(
        '-nr', '--not-record',
        action='store_true',
        default=False,
        help='flag for not recording the data ( Testing purposes)')
    argparser.add_argument(
        '-e', '--episode-number',
        metavar='E',
        dest='episode_number',
        default=0,
        type=int,
        help='The episode number that it will start to record.')
    argparser.add_argument(
        '-n', '--number-episodes',
        metavar='N',
        dest='number_of_episodes',
        default=999999999,
        help='The number of episodes to run, default infinite.')

    args = argparser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    while True:
        try:

            with make_carla_client(args.host, args.port) as client:
                run_sim(client, args)
                break

        except TCPConnectionError as error:
            logging.error(error)
            time.sleep(1)



if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')