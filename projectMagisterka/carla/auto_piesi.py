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

import argparse
import random
import time
from datetime import datetime
import math

import threading

" Original "
" Code for one car and multiple pedestrians walking "


# needed variables
spawn_point = carla.Transform(
    carla.Location(x=55.4543, y=-195.095, z=0.5), # (x=72.4543, y=-195.095, z=0.5), # dobry start
    carla.Rotation(pitch=0.0, yaw=-175.0, roll=0.0)
)

car_locations = [
    carla.Location(x=1.6221, y=-198.864, z=6.16348), 
    carla.Location(x=2.57179, y=-137.168, z=6.5291),
    carla.Location(x=83.9912, y=-136.025, z=13.0743),
    carla.Location(x=80.0235, y=-199.607, z=6.1153),
]

human_locations = [carla.Location(x=19.6231, y=-187.691, z=2.45202),
             carla.Location(x=19.606, y=-212.138, z=2.94891)]


# needed functions

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
        
def distance(loc1, loc2):
    return math.sqrt((loc1.x - loc2.x)**2 + (loc1.y - loc2.y)**2)


# main funcs

def pedestrian(world, locations):
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
            
            
            
def car(world, locations):
    vehicle = None

    try:
        # choose vehicle
        blueprint_library = world.get_blueprint_library()
        # car_blueprint = random.choice(blueprint_library.filter('vehicle.*'))
        car_blueprint = blueprint_library.filter('vehicle.*')[1]
        print(f"Selected blueprint: {car_blueprint.id}")

        # spawn the vehicle
        vehicle = world.try_spawn_actor(car_blueprint, spawn_point)
        if not vehicle:
            print("Something went wrong, car did not spawn")
            return
        print(f"Vehicle spawned at: {vehicle.get_transform().location}")

        # camera setup
        spectator = world.get_spectator()

        camera_height = 5.0  
        camera_distance = -10.0  
        camera_pitch = -10.0  

        # slowly move the car
        vehicle.apply_control(carla.VehicleControl(throttle=0.3))

        #  update camera pisition all the time to follow the car
        
        print("camera starts to follow the car")
        iter = 0
        time_stamp = 0
        turn = False
        last_waypoint_timestamp = datetime.now()
        
        while True:
            for target_location in locations:
                while True:
                    # display camera position
                    if iter % 10 ==0:
                        separator_transform = spectator.get_transform()
                        camera_location = separator_transform.location
                        camera_rotation = separator_transform.rotation
                        print(f'current camera location {camera_location} and rotation {camera_rotation}')

                    # update camera
                    if len(sys.argv) > 1:
                        if sys.argv[1] == "follow":
                            
                            vehicle_transform = vehicle.get_transform()
                            vehicle_location = vehicle_transform.location
                            vehicle_rotation = vehicle_transform.rotation
                            
                            camera_location = vehicle_location + carla.Location(
                                x=camera_distance * vehicle_rotation.get_forward_vector().x,
                                y=camera_distance * vehicle_rotation.get_forward_vector().y,
                                z=camera_height
                            )
                            camera_rotation = carla.Rotation(
                                pitch=camera_pitch,
                                yaw=vehicle_rotation.yaw,
                                roll=0.0
                            )
                            spectator.set_transform(carla.Transform(camera_location, camera_rotation))
                    
                    vehicle_location = vehicle.get_location()
                    current_distance = distance(vehicle_location, target_location)

                    if current_distance < 4.0: 
                        break

                    direction_vector = target_location - vehicle_location
                    direction_vector.z = 0 

                    yaw = math.degrees(math.atan2(direction_vector.y, direction_vector.x))

                    control = carla.VehicleControl()
                    control.throttle = 0.5
                    control.steer = (yaw - vehicle.get_transform().rotation.yaw) / 180 
                    vehicle.apply_control(control)

                    time.sleep(0.05)

    finally:
        if vehicle:
            vehicle.destroy()
            print("vehicle was distroyed")




if __name__ == "__main__":
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    
    human_thread = threading.Thread(pedestrian, )