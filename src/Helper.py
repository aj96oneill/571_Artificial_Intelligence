import math
import pymunkoptions
pymunkoptions.options["debug"] = False
from pymunk.vec2d import Vec2d

PI = math.pi
PIx2 = PI * 2
GRAVITY = 9.81

def radians(degrees):
    return (2*PI * degrees) / 360
def degrees(radians):
    return (360*radians) / PIx2
def angle(vector):
    return math.atan2(vector[1], vector[0])
def vector(angle):
    return Vec2d(math.cos(angle),math.sin(angle))
