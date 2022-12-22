from Helper import vector, PI

import noise
import random
import numpy as np
import numpy.linalg as la


class Wander():
    def __init__(self, action_repeat):
        #action parameters
        self.action_repeat = action_repeat
        self.wander_range = .1 * PI
        self.max_scaler = 5
        #perlin noise parameters
        self.offset0, self.scale0 = random.randint(0,1000000), 250
        self.offset1, self.scale1 = random.randint(0,1000000), 2000


    def get_action(self, timestep_i, current_orientation, actions_checked=[]):

        perlin_noise = noise.pnoise1( (float(timestep_i*self.action_repeat)+self.offset0) / self.scale0 )
        perlin_noise += noise.pnoise1( (float(timestep_i*self.action_repeat)+self.offset1) / self.scale1 )

        action = int(perlin_noise * self.max_scaler)
        if action > self.max_scaler:
            action = self.max_scaler
        elif action < -self.max_scaler:
            action = -self.max_scaler

        action_samples = 0
        while action in actions_checked and action_samples < 50:
            action_samples += 1
            self.reset_action()
            perlin_noise = noise.pnoise1( (float(timestep_i*self.action_repeat)+self.offset0) / self.scale0 )
            perlin_noise += noise.pnoise1( (float(timestep_i*self.action_repeat)+self.offset1) / self.scale1 )

            action = int(perlin_noise * self.max_scaler)
            if action > self.max_scaler:
                action = self.max_scaler
            elif action < -self.max_scaler:
                action = -self.max_scaler

        steering_force = vector(action * self.wander_range + current_orientation)
        return action, steering_force


    def reset_action(self):
        self.offset0, self.offset1 = random.randint(0,1000000), random.randint(0,1000000)

    def get_steering_force(self, action, current_orientation):
        steering_force = vector(action * self.wander_range + current_orientation)
        return steering_force




class Seek():
    def __init__(self, target_position):
        self.target_position = target_position
        self.wander_range = .1 * PI
        self.max_scaler = 5

    def update_goal(self, new_goal_pos):
        self.target_position = new_goal_pos

    def get_action(self, current_position, current_orientation):
        seek_vector = self.target_position - current_position
        # if la.norm(seek_vector) < 50:
        #     print('GOAL')
        #     pdb.set_trace()
        # print(la.norm(seek_vector))

        steering_vector = seek_vector - vector(current_orientation)

        action_space = np.arange(-5,6)
        min_diff = 9999999
        min_a = 0
        for a in action_space:
            steering_force = vector(a * self.wander_range + current_orientation)
            diff = la.norm(steering_force - steering_vector)
            if diff <= min_diff:
                min_a = a
                min_diff = diff


        steering_force = vector(min_a * self.wander_range + current_orientation)

        return min_a, steering_force

    def reset_action(self):
        pass

    def get_steering_force(self, action, current_orientation):
        steering_force = vector(action * self.wander_range + current_orientation)
        return steering_force
