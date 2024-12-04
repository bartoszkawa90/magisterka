import glob
import os
import sys

try:
    sys.path.append(glob.glob('**/*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import random
import time
import math


locations = [carla.Location(x=19.6231, y=-187.691, z=2.45202),
             carla.Location(x=19.606, y=-212.138, z=2.94891)]


def spawn_pedestrian(world, location, rotation):
    blueprint_library = world.get_blueprint_library()
    pedestrian_blueprints = blueprint_library.filter('walker.pedestrian.*')
    pedestrian_bp = random.choice(pedestrian_blueprints)
    
    spawn_transform = carla.Transform(location, rotation)
    pedestrian = world.try_spawn_actor(pedestrian_bp, spawn_transform)
    
    if pedestrian:
        print(f"Pedestrian spawned at location {location}.")
    else:
        print("Failed to spawn pedestrian.")
    return pedestrian

def calculate_direction(start, end):
    direction_vector = carla.Vector3D(
        x=end.x - start.x,
        y=end.y - start.y,
        z=0  # dont need to be used
    )
    # normalize vector length
    magnitude = math.sqrt(direction_vector.x**2 + direction_vector.y**2)
    direction_vector.x /= magnitude
    direction_vector.y /= magnitude
    
    return direction_vector

def move_pedestrian(pedestrian, direction, duration, speed):
    walker_control = carla.WalkerControl()
    walker_control.speed = speed
    walker_control.direction = direction
    
    # how long pedestrian should walk ?
    for _ in range(int(duration)):  
        pedestrian.apply_control(walker_control)
        time.sleep(0.1)

def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    
    pedestrian = spawn_pedestrian(world, locations[0], carla.Rotation())
    
    try:
        while True:
            for location in locations:
                if pedestrian:
                    # find goo direction
                    other_location = [loc for loc in locations if loc != location][0]
                    forward_direction = calculate_direction(location, other_location)
                    # move to target
                    move_pedestrian(pedestrian, forward_direction, duration=150, speed=1.5)
    
    finally:
        if pedestrian:
            pedestrian.destroy()
            print(" --- destroy --- ")
        
if __name__ == "__main__":
    main()
