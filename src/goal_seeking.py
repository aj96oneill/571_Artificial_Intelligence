from SteeringBehaviors import Wander, Seek
import SimulationEnvironment as sim
from Networks import Action_Conditioned_FF

import pickle
import numpy as np
import torch
import numpy.linalg as la


def get_network_param(sim_env, action, scaler):
    sensor_readings = sim_env.raycasting()
    network_param = np.append(sensor_readings, [action, 0]) #unutilized 0 added to match shape of scaler
    network_param = scaler.transform(network_param.reshape(1,-1))
    network_param = network_param.flatten()[:-1]
    network_param = torch.as_tensor(network_param, dtype=torch.float32)
    return network_param

def goal_seeking(goals_to_reach):
    sim_env = sim.SimulationEnvironment()
    action_repeat = 100
    # steering_behavior = Wander(action_repeat)
    steering_behavior = Seek(sim_env.goal_body.position)

    #load model
    model = Action_Conditioned_FF()
    model.load_state_dict(torch.load('saved/saved_model.pkl'))
    model.eval()

    #load normalization parameters
    scaler = pickle.load( open("saved/scaler.pkl", "rb"))

    accurate_predictions, false_positives, missed_collisions = 0, 0, 0
    robot_turned_around = False
    actions_checked = []
    goals_reached = 0
    while goals_reached < goals_to_reach:

        seek_vector = sim_env.goal_body.position - sim_env.robot.body.position
        if la.norm(seek_vector) < 50:
            sim_env.move_goal()
            steering_behavior.update_goal(sim_env.goal_body.position)
            goals_reached += 1
            continue

        action_space = np.arange(-5,6)
        actions_available = []
        for action in action_space:
            network_param = get_network_param(sim_env, action, scaler)
            prediction = model(network_param)
            if prediction.item() < .25:
                actions_available.append(action)

        if len(actions_available) == 0:
            sim_env.turn_robot_around()
            continue

        action, _ = steering_behavior.get_action(sim_env.robot.body.position, sim_env.robot.body.angle)
        min, closest_action = 9999, 9999
        for a in actions_available:
            diff = abs(action - a)
            if diff < min:
                min = diff
                closest_action = a

        steering_force = steering_behavior.get_steering_force(closest_action, sim_env.robot.body.angle)
        for action_timestep in range(action_repeat):
            _, collision, _ = sim_env.step(steering_force)
            if collision:
                steering_behavior.reset_action()
                break


if __name__ == '__main__':
    goals_to_reach = 10
    goal_seeking(goals_to_reach)
