# ************** STUDENTS EDIT THIS FILE **************

from SteeringBehaviors import Wander
import SimulationEnvironment as sim

import numpy as np

def collect_training_data(total_actions):
    #set-up environment
    sim_env = sim.SimulationEnvironment()

    #robot control
    action_repeat = 100
    steering_behavior = Wander(action_repeat)

    num_params = 7
    #STUDENTS: network_params will be used to store your training data
    # a single sample will be comprised of: sensor_readings, action, collision
    network_params = [[],[],[]]


    for action_i in range(total_actions):
        progress = 100*float(action_i)/total_actions
        print(f'Collecting Training Data {progress}%   ', end="\r", flush=True)

        #steering_force is used for robot control only
        action, steering_force = steering_behavior.get_action(action_i, sim_env.robot.body.angle)

        for action_timestep in range(action_repeat):
            if action_timestep == 0:
                _, collision, sensor_readings = sim_env.step(steering_force)
            else:
                _, collision, _ = sim_env.step(steering_force)

            if collision:
                steering_behavior.reset_action()
                #STUDENTS NOTE: this statement only EDITS collision of PREVIOUS action
                #if current action is very new.
                if action_timestep < action_repeat * .3: #in case prior action caused collision
                    network_params[-1][-1] = collision #share collision result with prior action
                break

        #STUDENTS: Update network_params.
        # print("Sensor readings:")
        # print(sensor_readings)
        # print("Action:")
        # print(action)
        # print("Collision:")
        # print(collision)
        network_params[0].append(sensor_readings)
        network_params[1].append(action)
        network_params[2].append(collision)


    #STUDENTS: Save .csv here. Remember rows are individual samples, the first 5
    #columns are sensor values, the 6th is the action, and the 7th is collision.
    #Do not title the columns. Your .csv should look like the provided sample.
    
    # import csv
    # f = open('./saved/training_data.csv', 'a')
    # writer = csv.writer(f)
    # for i in range(0,len(network_params[0])):
    #     writer.writerow([network_params[0][i][0], network_params[0][i][1], network_params[0][i][2], network_params[0][i][3], network_params[0][i][4], network_params[1][i], network_params[2][i]])
    # f.close()


if __name__ == '__main__':
    total_actions = 100
    collect_training_data(total_actions)
