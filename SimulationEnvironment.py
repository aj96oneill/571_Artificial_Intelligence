from Helper import radians, degrees, angle, vector, PI, PIx2, GRAVITY

import random
import math
import noise
import time

import numpy as np
import numpy.linalg as la

import matplotlib.pyplot as plt




HEADLESS = False
if HEADLESS:
    import os
    os.environ['SDL_VIDEODRIVER'] = 'dummy'

#https://stackoverflow.com/questions/51464455/why-when-import-pygame-it-prints-the-version-and-welcome-message-how-delete-it
import os, sys
with open(os.devnull, 'w') as f:
    # disable stdout
    oldstdout = sys.stdout
    sys.stdout = f

    import pymunkoptions
    pymunkoptions.options["debug"] = False
    import pymunk
    import pymunk as pm
    from pymunk.pygame_util import DrawOptions
    from pymunk.vec2d import Vec2d
    import pygame as pg

    # enable stdout
    sys.stdout = oldstdout




BLACK = (0,0,0,1)
WHITE = (255,255,255,1)
RED = (255,0,0,1)
LIME = (0,255,0,1)
BLUE = (0,0,255,1)
YELLOW = (255,255,0,1)
CYAN = (0,255,255,1)
MAGENTA = (255,0,255,1)
SILVER = (192,192,192,1)
GRAY = (128,128,128,1)
MAROON = (128,0,0,1)
OLIVE = (128,128,0,1)
GREEN = (0,128,0,1)
PURPLE = (128,0,128,1)
TEAL = (0,128,128,1)
NAVY = (0,0,128,1)
ASU_MAROON = (140,29,64,1)
ASU_GOLD = (255,198,39,1)
ALL_COLORS = [BLACK,WHITE,RED,LIME,BLUE,YELLOW,CYAN,MAGENTA,SILVER,GRAY,MAROON,OLIVE,GREEN,PURPLE,TEAL,NAVY,ASU_MAROON,ASU_GOLD]


class Robot:
    def __init__(self, mass=1, pos=(0,0), ori=0):
        self.mass = mass
        # self.max_speed = 15
        self.speed = 20
        self.max_steering_force = 1
        self.max_turn_radians = math.pi/800
        self.friction = .05

        self.body, self.shape = self.create_pymunk_robot(self.mass)
        self.body.position = pos
        self.body.angle = ori
        self.previous_angle = self.body.angle
        self.prior_angular_velocity = 0

        self.sensors, self.sensor_angles, self.sensor_range = self.add_sensors()

    def create_pymunk_robot(self, mass):
        length, width = 20, 30
        moment = pm.moment_for_box(mass, (length,width))
        body = pm.Body(mass, moment)
        corners = [ (-length,-width),
                    (length,-width),
                    (length,width),
                    (-length,width) ]
        shape = pm.Poly(body, corners)
        shape.filter = pm.ShapeFilter(categories=0b1)
        # shape.friction = .5 #seems to have no effect
        shape.color = WHITE #TODO make bounding box invisible somehow
        return body, shape

    def add_sensors(self, sensor_range=150.0):
        self.sensor_range = sensor_range
        sensor_shapes = []
        sensor_end_points = []
        sensor_angles = [66,33,0,-33,-66]
        for a in sensor_angles:
            angle = self.body.angle + math.radians(a)
            v = vector(angle)
            p = v * sensor_range
            sensor_end_points.append(p)
        thickness = 1
        for p in sensor_end_points:
            sensor_shape = pm.Segment(self.body, (0,0), p, thickness)
            sensor_shape.color = BLUE
            sensor_shape.sensor = True
            sensor_shapes.append(sensor_shape)
        return sensor_shapes, sensor_angles, sensor_range


class SimulationEnvironment:
    def __init__(self):
        self.sim_steps = 0 # counter incremented each step for debug
        pg.init()
        self.screen_width, self.screen_height = 1080, 900
        self.screen = pg.display.set_mode((self.screen_width, self.screen_height))
        # self.screen.fill(WHITE)
        pg.display.set_caption("PyGame Display")
        self.clock = pg.time.Clock()

        self.space = pm.Space()
        self.draw_options = DrawOptions(self.screen)

        self.CENTER = (self.screen_width/2, self.screen_height/2)
        NORTH, SOUTH, EAST, WEST = math.pi/2, 3*math.pi/2, 0, math.pi # PLEASE ONLY USE EAST AT THIS TIME
        self.robot = Robot(mass=20, pos=self.CENTER, ori=EAST)

        self.space.add(self.robot.body)
        self.space.add(self.robot.shape)
        for sensor_shape in self.robot.sensors:
            self.space.add(sensor_shape)

        self.wall_shapes = self.assemble_walls(self.screen_width, self.screen_height, 180)
        self.goal_body, self.goal_shape = self.add_goal()
        self.last_goal_position = -1
        self.move_goal()
        self.time_since_collision = 0


    def add_goal(self):
        body = pm.Body(1,1)
        body.position = 1000,80
        radius = 40
        shape = pm.Circle(body, radius)
        shape.color = BLUE
        shape.sensor = True
        self.space.add(body, shape)

        return body, shape

    def move_goal(self):
        offset=60
        i = random.randint(0,5)
        while i == self.last_goal_position:
            i = random.randint(0,5)
        self.last_goal_position = i
        positions = [Vec2d(offset,offset),
                    Vec2d(self.screen_width-300,self.screen_height/2+200),
                    Vec2d(self.screen_width-offset, offset),
                    Vec2d(self.screen_width-offset,400),
                    Vec2d(offset,self.screen_height-offset),
                    Vec2d(700, offset)
        ]

        self.goal_body.position = positions[i]


    def assemble_walls(self, w, h, u):
        wall_shapes = []
        parameter_points = [(0,0),(0,h),(w,h),(w,0),(0,0)]
        inner_wall1 = [(u,u),(u,h-u),(2*u,h-u),(2*u,u),(u,u)]#,(2*u,2*u),(w-u,2*u)]
        inner_wall2 = [(w-u,u),(w-u,2*u),(w-2*u,2*u),(w-2*u,u), (w-u,u)]
        corner_wall = [(w-2*u,h),(w,h-2*u)]
        wall_points_list = [parameter_points, inner_wall1, inner_wall2, corner_wall]
        for wall_points in wall_points_list:
            for i in range(len(wall_points)-1):
                wall_body, wall_shape = self.build_wall(wall_points[i], wall_points[i+1], thickness=8)
                wall_shapes.append(wall_shape)
                self.space.add(wall_body, wall_shape)
        return wall_shapes

    def build_wall(self, point_a, point_b, thickness=5):
        body = pm.Body(body_type=pm.Body.STATIC)
        shape = pm.Segment(body, point_a, point_b, thickness)
        shape.color = GRAY
        return body, shape

    def _draw_everything(self, velocity=False, steering=True):
        self.screen.fill(WHITE)
        self.space.debug_draw(self.draw_options)

        img_pos, img_ori = self.pm2pgP(self.robot.body.position), degrees(self.robot.body.angle)
        self._apply_image_to_robot(img_pos, img_ori)

        pm_botPos = self.pm2pgP(self.robot.body.position)
        if velocity:
            velocity_vector = pm_botPos + self.pm2pgV(self.robot.body.velocity)*5
            velocity_line = pg.draw.line(self.screen, RED, pm_botPos, velocity_vector)
        if steering:
            steering_vector = pm_botPos + self.pm2pgV(self.steering_force)*7.5
            steering_line = pg.draw.line(self.screen, GREEN, pm_botPos, steering_vector)

        pg.display.flip()

    def _apply_image_to_robot(self, pos, angle, damage=False):
        # https://stackoverflow.com/questions/4183208/how-do-i-rotate-an-image-around-its-center-using-pygame User: Rabbid76 blitRotate()
        if damage:
            image = pg.image.load("assets/robot_inverse.png")
        else:
            image = pg.image.load("assets/robot.png")
        originPos = image.get_rect().center
        w, h = image.get_size()
        box = [pg.math.Vector2(p) for p in [(0, 0), (w, 0), (w, -h), (0, -h)]]
        box_rotate = [p.rotate(angle) for p in box]
        min_box = (min(box_rotate, key=lambda p: p[0])[0], min(box_rotate, key=lambda p: p[1])[1])
        max_box = (max(box_rotate, key=lambda p: p[0])[0], max(box_rotate, key=lambda p: p[1])[1])
        pivot = pg.math.Vector2(originPos[0], -originPos[1])
        pivot_rotate = pivot.rotate(angle)
        pivot_move = pivot_rotate - pivot
        origin = (pos[0] - originPos[0] + min_box[0] - pivot_move[0], pos[1] - originPos[1] - max_box[1] + pivot_move[1])
        rotated_image = pg.transform.rotate(image, angle)
        self.screen.blit(rotated_image, origin)

    def _apply_robot_motion(self, steering_direction):
        self.steering_force = steering_direction * self.robot.speed
        self.robot.body.apply_force_at_world_point(self.steering_force*10, self.robot.body.position)

        velocity_mag = la.norm(self.robot.body.velocity)
        if velocity_mag > 0: #apply friction
            friction = GRAVITY * self.robot.mass * self.robot.friction * -self.robot.body.velocity
            self.robot.body.apply_force_at_world_point(friction, self.robot.body.position)
            self.robot.body.angular_velocity = self.robot.body.angular_velocity * .95
            if self.robot.body.angular_velocity  < .00001:
                self.robot.body.angular_velocity = 0

            if velocity_mag > 2.0: #update orientation
                orientation_shift = angle(self.robot.body.velocity) - self.robot.body.angle
                orientation_shift = (orientation_shift + PI) % (2*PI) - PI
                if abs(orientation_shift) > .1:
                    self.robot.body.angle += .02 * np.sign(orientation_shift)
                else:
                    self.robot.body.angle = angle(self.robot.body.velocity)
                self.robot.previous_angle = self.robot.body.angle

            elif velocity_mag < .05: #zero out velocity, may have no effect
                self.robot.body.velocity = (0,0)


    def _detect_collisions(self):
        for wall_shape in self.wall_shapes:
            collisions = wall_shape.shapes_collide(self.robot.shape)
            if(collisions.points):
                img_pos, img_ori = self.pm2pgP(self.robot.body.position), degrees(self.robot.body.angle)
                self._apply_image_to_robot(img_pos, img_ori, damage=True)
                pg.display.flip()
                if not HEADLESS:
                    time.sleep(.5)
                return 1, collisions.points
        return 0, None

    def _check_wall_overlap(self):
        for wall_shape in self.wall_shapes:
            collisions = wall_shape.shapes_collide(self.goal_shape)
            if(collisions.points):
                return True
        return False


    def raycasting(self, print_sensors=False):
        robot_filter = pm.ShapeFilter(mask=pm.ShapeFilter.ALL_MASKS ^ 0b1)
        sensor_end_points=[]
        for a in self.robot.sensor_angles:
            angle = self.robot.body.angle + math.radians(a)
            v = vector(angle)
            p = v * self.robot.sensor_range + self.robot.body.position
            sensor_end_points.append(p)
        segment_queries = []
        for i, p in enumerate(sensor_end_points):
            segment_query = self.space.segment_query_first(self.robot.body.position,p,1,robot_filter)
            if segment_query:
                segment_queries.append(la.norm(segment_query.point - self.robot.body.position))
            else:
                segment_queries.append(self.robot.sensor_range)
        sq = np.array(segment_queries)

        if print_sensors:
            print("%d %d %d %d %d"%(int(sq[0]),int(sq[1]),int(sq[2]),int(sq[3]),int(sq[4])))
        return sq

    def _detect_sensor_collisions(self):
        sensors_triggered = []
        for wall_shape in self.wall_shapes:
            for i, sensor in enumerate(self.robot.sensors):
                collisions = wall_shape.shapes_collide(sensor)
                if(collisions.points):
                    sensors_triggered.append(i)
        for i in range(len(self.robot.sensors)):
            if i in sensors_triggered:
                self.robot.sensors[i].color = RED
            else:
                self.robot.sensors[i].color = BLUE
        return sensors_triggered


    def _reset_robot(self, center=False, collision_points=None):
        previous_angle = vector(self.robot.body.angle)
        previous_position = self.robot.body.position

        if collision_points:
            self.robot.body.angle = angle(collision_points[0].point_a - collision_points[0].point_b)
        else:
            self.robot.body.angle = angle(-previous_angle)

        if center:
            self.robot.body.position = self.CENTER
        else:
            self.robot.body.position = previous_position - (previous_angle * 25)

        self.robot.body.angular_velocity = 0
        self.robot.body.velocity = (0,0)

    def turn_robot_around(self):
        previous_angle = vector(self.robot.body.angle)
        previous_position = self.robot.body.position
        steering_vector = -previous_angle + .01*np.random.randn(2)

        turn_len = random.randint(0,1)
        if turn_len == 0:
            for i in range(180):
                self.step(steering_vector, ignore_collisions=True)
        else:
            for i in range(250):
                self.step(steering_vector, ignore_collisions=True)

        self.robot.body.angular_velocity = 0
        self.robot.body.velocity = (0,0)

    def step(self, steering_direction, ignore_collisions=False):
        pos, ori = self.robot.body.position, self.robot.body.angle
        state = np.array([pos[0], pos[1], ori], dtype=float)
        self._apply_robot_motion(steering_direction)
        collision = None
        if not ignore_collisions:
            collision, collision_points = self._detect_collisions()
            if collision:
                if self.time_since_collision < 10:
                    self._reset_robot(center=True)
                    self.time_since_collision = 0
                else:
                    self._reset_robot(collision_points=collision_points)
                    self.time_since_collision = 0
            else:
                self.time_since_collision += 1
        sensor_readings = self.raycasting(print_sensors=False)
        self._detect_sensor_collisions()
        self._env_step()
        return state, collision, sensor_readings

    def _env_step(self):
        self.space.step(1/50.0)
        self.clock.tick(10000)
        self._draw_everything()


    def pm2pgP(self, pos):
        return Vec2d(pos[0], self.screen_height - pos[1])
    def pg2pmP(self, pos):
        return Vec2d(pos[0], pos[1] - self.screen_height)
    def pm2pgV(self, pos):
        return Vec2d(pos[0], -pos[1])
    def pg2pmV(self, pos):
        pass #TODO if necessary
        # return Vec2d(pos[0], pos[1] - self.screen_height)
    def oangle(self, vector):
        return math.atan2(vector[1], vector[0])
    def ovector(self, angle):
        return math.cos(angle),math.sin(angle)
