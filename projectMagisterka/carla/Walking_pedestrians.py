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


class ManualPedestrian:
    def __init__(self, client, world, waypoints, walking_speed=1.5):
        """ Spawn pedestrian and create WalkerControl """
        # choose some random pedestrian 
        self.waypoints = waypoints
        blueprint_library = world.get_blueprint_library()
        walker_bp = random.choice(blueprint_library.filter('walker.pedestrian.*'))
        spawn_point = carla.Transform(location=waypoints[0])

        # spawn pedestrian
        self.walker = world.try_spawn_actor(walker_bp, spawn_point)

        # print(f"Spawned pedestrian {self.walker.id} at {spawn_point.location}.")

        self.walker_control = carla.WalkerControl()
        self.walker_control.speed = walking_speed  # pedestrian speed in m/s
        
    def move(self):
        """ Make pedestrian move from spawn to each point in waypoints"""
        for target in self.waypoints:
            while True:
                print(f"Walking toward waypoint at {target}.")
                # calculate direction to the target
                current_location = self.walker.get_location()
                direction = target - current_location
                # print(f'Curretn location {current_location}')
                # print(f'Target {target} ')
                # print(f'Direction {direction} ')

                # count distance
                distance = math.sqrt(direction.x ** 2 + direction.y ** 2 + direction.z ** 2)
                # normalize direction if distance is larger than 2
                if distance > 2.0:
                    direction.x /= distance
                    direction.y /= distance
                    direction.z /= distance

                    self.walker_control.direction = direction
                    self.walker.apply_control(self.walker_control)
                    time.sleep(0.5)
                else:
                    print(f"Reached waypoint at {target}.")
                    break
                
                
class AIPedestrian:
    def __init__(self, world, waypoints):
        #TODO AI pedestrian moves on ist own
        """ Spawn pedestrian and choose controller """
        blueprint_library = world.get_blueprint_library()
        walker_bp = random.choice(blueprint_library.filter('walker.pedestrian.*'))
        walker_controller_bp = blueprint_library.find('controller.ai.walker')

        self.waypoints = waypoints
        spawn_point = waypoints[0]
        self.walker = world.spawn_actor(walker_bp, carla.Transform(location=spawn_point))

        self.walker_controller = world.spawn_actor(walker_controller_bp, carla.Transform(), 
                                                   attach_to=self.walker)
        
    def move(self):
        """ Move pedestrian from each to each location in waypoints """
        self.walker_controller.start()
        for target in self.waypoints:
            self.walker_controller.go_to_location(target)
            while True:
                print(f"Setting destination for AI-controlled walker: {target}")
                current_location = self.walker.get_location()
                distance = current_location.distance(target)
                # print(f'Distance {distance}, location  {current_location}   target {target}')
                if distance < 3.0:
                    break
                time.sleep(0.5)


# points to go by
waypoints = [carla.Location(x=19.6231, y=-187.691, z=0.5),
             carla.Location(x=19.606, y=-212.138, z=0.5)]

waypoints2 = [
    carla.Location(x=-10.978936, y=-186.981277, z=2.835318), 
    carla.Location(x=-9.480467, y=-214.167236, z=3.974376)
]

def main():
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        world = client.get_world()
        
        p1 = AIPedestrian(world, waypoints)
        p2 = ManualPedestrian(client, world, waypoints2)

        # move camera to pedestrian spawn location
        spectator = world.get_spectator()
        spectator.set_transform(carla.Transform(
            location=waypoints[0],
            rotation=carla.Rotation(pitch=0)
        ))

        # move from point to point in infinite loop
        # while True:
            # p2.move()
            # p1.move()
            
        import threading
        threads = [
            threading.Thread(target=p1.move),
            threading.Thread(target=p2.move)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    
    except KeyboardInterrupt:
        print("All waypoints reached.")
        p1.walker.destroy()
        p2.walker.destroy()
    
    finally:
        print("All waypoints reached.")
        p1.walker.destroy()
        p2.walker.destroy()



if __name__ == '__main__':
    main()
