import carla
import time

def main():
    # Define the CARLA client and connect to the simulator
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    # Get the world and set synchronous mode
    world = client.get_world()
    settings = world.get_settings()
    settings.synchronous_mode = True
    world.apply_settings(settings)

    # Define the spawn point
    spawn_point = carla.Transform(
        carla.Location(x=55.4543, y=-195.095, z=0.5),
        carla.Rotation(pitch=0.0, yaw=-175.0, roll=0.0)
    )

    # Get the blueprint library and spawn a vehicle
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)

    # Define the waypoints
    locations = [
        carla.Location(x=1.6221, y=-198.864, z=6.16348),
        carla.Location(x=2.57179, y=-137.168, z=6.5291),
        carla.Location(x=83.9912, y=-136.025, z=13.0743),
        carla.Location(x=80.0235, y=-199.607, z=6.1153),
    ]

    # Use autopilot mode
    vehicle.set_autopilot(False)

    try:
        for target_location in locations:
            # Create a waypoint and move to it
            waypoint = world.get_map().get_waypoint(target_location)
            destination = waypoint.transform.location

            print(f"Driving to: {destination}")
            while True:
                # Compute control to drive towards the destination
                current_location = vehicle.get_location()
                distance = current_location.distance(destination)

                # Break the loop when close to the destination
                if distance < 2.0:
                    print("Reached destination")
                    break

                control = carla.VehicleControl()
                control.throttle = 0.5  # Adjust throttle as needed
                control.steer = 0.0    # Adjust steering based on path logic if required
                vehicle.apply_control(control)

                # Advance the simulation
                world.tick()
                time.sleep(0.05)

    finally:
        print("Destroying actors...")
        vehicle.destroy()
        settings.synchronous_mode = False
        world.apply_settings(settings)
        print("Done.")

if __name__ == '__main__':
    main()
