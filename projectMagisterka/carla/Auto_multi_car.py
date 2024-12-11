import glob
import os
import sys
from copy import deepcopy

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


# basic location for camera and reference point for cars
basic_spawn_point = carla.Transform(
    carla.Location(x=55.4543, y=-195.095, z=20.5),
    carla.Rotation(pitch=0.0, yaw=-175.0, roll=0.0)
)


def get_nearby_spawn_points(world, center_location, radius=100):
    """
    Get some spawn points near chosen basic location
    """
    map_spawn_points = world.get_map().get_spawn_points()
    nearby_spawn_points = [
        sp for sp in map_spawn_points
        if sp.location.distance(center_location) <= radius
    ]
    return nearby_spawn_points


def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(20.0)
    world = client.get_world()

    vehicles = []

    try:
        # choose some random car model
        blueprint_library = world.get_blueprint_library()
        car_blueprint = random.choice(blueprint_library.filter('vehicle.*'))
        print(f"Selected blueprint: {car_blueprint.id}")

        # get near spawn points
        spawn_points = get_nearby_spawn_points(world, basic_spawn_point.location, radius=100)

        # randomly shuffle and limit spawn points
        random.shuffle(spawn_points)
        num_spawn_points = min(len(spawn_points), 30)

        # spawn cars
        for i, spawn_point in enumerate(spawn_points[:num_spawn_points]):
            vehicle = world.try_spawn_actor(car_blueprint, spawn_point)
            if vehicle:
                vehicles.append(vehicle)
                print(f"car {i} spawned at {vehicle.get_transform().location}")
            else:
                print(f"car spawned at {i}")

        # camera location set to basic location
        spectator = world.get_spectator()
        spectator.set_transform(basic_spawn_point)
        print(f"Camera moved to basic spawn point: {basic_spawn_point.location}")

        # turn on autopilot for all vehicles
        traffic_manager = client.get_trafficmanager()
        traffic_manager.set_random_device_seed(0)
        random.seed(0)

        for vehicle in vehicles:
            traffic_manager.auto_lane_change(vehicle, True)
            traffic_manager.ignore_lights_percentage(vehicle, 100)
            vehicle.set_autopilot(True)

        while True:
            time.sleep(1)  # keep simulation
            
    except KeyboardInterrupt:
        # destroy all vehicles
        for vehicle in vehicles:
            if vehicle:
                vehicle.destroy()
        print("All vehicles were destroyed.") 
    
    finally:
        # destroy all vehicles
        for vehicle in vehicles:
            if vehicle:
                vehicle.destroy()
        print("All vehicles were destroyed.")


if __name__ == '__main__':
    print('Start')
    main()
    print('End')
