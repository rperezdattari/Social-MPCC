#!/usr/bin/env python3

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Spawn NPCs into the simulation"""

from random import randrange
import math

import rospy
import pkg_resources

from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped
from nav_msgs.msg import Odometry

import glob
import os
import sys
import time

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import numpy as np
import argparse
import logging
from numpy import random
from random import shuffle


def get_waypoint_after_junction(waypoint):
    # More advanced: get the junction and continue beyond it
    d_waypoints = 1.0
    direction = waypoint.transform.rotation
    junction_ahead = waypoint.next(d_waypoints)[0].get_junction()
    if junction_ahead == None:
        print('Pedestrian Module: Found end of road')
    else:
        print('Pedestrian Module: Found road')

    # Get a tuple of waypoints that describe lanes in the junction
    junction_waypoints = junction_ahead.get_waypoints(carla.LaneType.Driving)
    straight_waypoints = None
    for tuple_waypoints in junction_waypoints:
        if tuple_waypoints[0].transform.location.distance(waypoint.transform.location) < 5.0 and \
                tuple_waypoints[1].transform.rotation == direction:
            straight_waypoints = tuple_waypoints
            break

    if straight_waypoints is not None:
        # Return one waypoint further (that is where the opendrive lane starts)
        return straight_waypoints[1].next(d_waypoints)[0]
    else:
        print('Pedestrian Module: Found end of road')
        return None


# Find the sidewalk segments around a driving location
def find_sidewalk_around(map, location):
    # Get sidewalk next to this waypoint
    sidewalk_waypoint = map.get_waypoint(location, lane_type=carla.LaneType.Sidewalk)

    # Distance between waypoints throughout
    d_waypoints = 1.0
    current_waypoint = sidewalk_waypoint

    # Find waypoints from the vehicle position to the end of the lane
    right_sidewalk = current_waypoint.next_until_lane_end(d_waypoints)

    if right_sidewalk is None:
        print('Pedestrian Module: Failed to find a lane from the given location')
        return None

    # Take the last waypoint and translate to the other side
    # First get the road
    lane_waypoint = right_sidewalk[-1]
    while lane_waypoint.lane_type != carla.LaneType.Driving:
        lane_waypoint = lane_waypoint.get_left_lane()

    # Translate to the other side of the road
    opposite_driving = lane_waypoint.transform.get_right_vector() * -lane_waypoint.lane_width * 1.5 + lane_waypoint.transform.location
    # Then get the sidewalk
    opposite_sidewalk = map.get_waypoint(opposite_driving, lane_type=carla.LaneType.Sidewalk)

    opposite_waypoint = opposite_sidewalk
    left_sidewalk = [opposite_waypoint]
    for i in range(len(right_sidewalk)):
        opposite_waypoint = opposite_waypoint.next(d_waypoints)[0]
        left_sidewalk += [opposite_waypoint]

    # return the waypoints and the end of the lane
    return [right_sidewalk, left_sidewalk, right_sidewalk[-1]]


class PedestrianModule:

    def __init__(self, client,world):
        print('Pedestrian Module: Init')

        parameters = rospy.get_param('/carla/scenario/pedestrians')
        self.walker_count = parameters['spawn_count']
        print(self.walker_count)
        self.sync = False
        self.seed = None
        self.client = client

        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

        self.walkers_list = []
        self.all_id = []
        self.all_actors = []
        self.synchronous_master = False

        self.pedestrian_goals = []

        rospy.on_shutdown(self.on_shutdown)

        self.shutdown_set = False
        random.seed(self.seed if self.seed is not None else int(time.time()))

        # Initialize a subscriber for the reset
        self.reset_sub = rospy.Subscriber("/lmpcc/initialpose", PoseWithCovarianceStamped,
                                          lambda msg: self.reset_callback())

        self.world = world

        self.create()

    def create(self):

        if self.sync:
            settings = self.world.get_settings()
            if not settings.synchronous_mode:
                self.synchronous_master = True
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = 0.05
                self.world.apply_settings(settings)
            else:
                self.synchronous_master = False

        blueprintsWalkers = self.world.get_blueprint_library().filter('walker.pedestrian.*')
        SpawnActor = carla.command.SpawnActor

        #  -------------
        #  Spawn Walkers
        #  -------------
        #  some settings
        self.percentage_running = 0.0  # how many pedestrians will run
        self.percentage_crossing = 90.0  # how many pedestrians will walk through the road

        self.map = self.world.get_map()  # Expensive, should only be called once
        # More complicated: also spawning the ego vehicle randomly!
        # Retrieve points where we can spawn the ego-vehicle
        spawn_world_points = self.world.get_map().get_spawn_points()
        shuffle(spawn_world_points)

        spawn_points = spawn_world_points

        shuffle(spawn_world_points)
        goal_points = spawn_world_points

        # Convert to side-walk position
        walker_speed2 = []

        failures = 0
        self.spawn_points = []
        self.goal_points = []
        batch = []
        walker_speed = []
        for i in range(self.walker_count):
            if len(spawn_points) == 0:
                break

            sidewalk_waypoint = self.map.get_waypoint(spawn_points[0].location, lane_type=carla.LaneType.Sidewalk)
            self.spawn_points.append(sidewalk_waypoint)
            spawn_points.pop(0)

            sidewalk_waypoint = self.map.get_waypoint(goal_points[0].location, lane_type=carla.LaneType.Sidewalk)
            self.goal_points.append(sidewalk_waypoint)
            goal_points.pop(0)

            # 2. we spawn the walker object
            walker_bp = random.choice(blueprintsWalkers)
            # set as not invincible
            if walker_bp.has_attribute('is_invincible'):
                walker_bp.set_attribute('is_invincible', 'false')
            # set the max speed
            if walker_bp.has_attribute('speed'):
                if (random.random() > self.percentage_running):
                    # walking
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[1])
                else:
                    # running
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[2])
            else:
                print("Walker has no speed")
                walker_speed.append(0.0)
            batch.append(SpawnActor(walker_bp, self.spawn_points[-1].transform))

        results = self.client.apply_batch_sync(batch, True)

        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
                failures += 1
            else:
                self.walkers_list.append({"id": results[i].actor_id})
                walker_speed2.append(walker_speed[i])

        # 3. we spawn the walker controller
        batch = []
        walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
        for i in range(len(self.walkers_list)):
            batch.append(SpawnActor(walker_controller_bp, self.goal_points[i].transform, self.walkers_list[i]["id"]))
        results = self.client.apply_batch_sync(batch, True)
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                self.walkers_list[i]["con"] = results[i].actor_id
        # 4. we put altogether the walkers and controllers id to get the objects from their id
        for i in range(len(self.walkers_list)):
            self.all_id.append(self.walkers_list[i]["con"])
            self.all_id.append(self.walkers_list[i]["id"])
        self.all_actors = self.world.get_actors(self.all_id)

        # wait for a tick to ensure client receives the last transform of the walkers we have just created
        if not self.sync or not self.synchronous_master:
            self.world.wait_for_tick()
        else:
            self.world.tick()

        # 5. initialize each controller and set target to walk to (list is [controler, actor, controller, actor ...])
        # set how many pedestrians can cross the road
        self.world.set_pedestrians_cross_factor(100)  # All peds are allowed to cross!

        for i in range(0, len(self.all_id), 2):
            print('start walker')
            self.all_actors[i].start()
            print('set walk to random point')
            try:
                self.all_actors[i].go_to_location(self.goal_points[i])
                print(i)
            except:
                print("failed to set the goal for pedestrian")

        print('Pedestrian Module: Spawned %d/%d walkers, press Ctrl+C to exit.' % (
        len(self.walkers_list), self.walker_count))

    def update(self):
        pass

    def on_shutdown(self):
        self.shutdown_set = True

    def reset_callback(self):
        for i in range(0, len(self.all_id), 2):
            print('set walk to random point')
            try:
                self.all_actors[i].go_to_location(random.choice(self.goal_points))
            except:
                print("Failed to set new Pedestrian goal")
        # self.destroy_actors()
        # self.create()

    def destroy_actors(self):
        # stop walker controllers (list is [controller, actor, controller, actor ...])
        for i in range(0, len(self.all_id), 2):
            self.all_actors[i].stop()

        print('\nPedestrian Module: Destroying %d walkers' % len(self.walkers_list))
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.all_id])
        self.walkers_list = []
        self.all_id = []
        self.all_actors = []
        time.sleep(0.5)


class OtherVehiclesModule:

    def __init__(self, client,world):
        print('OtherVehicles Module: Init')

        parameters = rospy.get_param('/carla/scenario/othervehicles')
        self.walker_count = parameters['spawn_count']
        print(self.walker_count)
        self.sync = False
        self.seed = None
        self.client = client

        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

        self.walkers_list = []
        self.all_id = []
        self.all_actors = []
        self.synchronous_master = False

        self.pedestrian_goals = []

        rospy.on_shutdown(self.on_shutdown)

        self.shutdown_set = False
        random.seed(self.seed if self.seed is not None else int(time.time()))

        # Initialize a subscriber for the reset
        self.reset_sub = rospy.Subscriber("/lmpcc/initialpose", PoseWithCovarianceStamped,
                                          lambda msg: self.reset_callback())
        self.world = world
        self.create()

    def create(self):

        try:
            self.traffic_manager = self.client.get_trafficmanager()
            self.traffic_manager.set_global_distance_to_leading_vehicle(3.0)
            #self.traffic_manager.set_respawn_dormant_vehicles(True)
            #self.traffic_manager.set_boundaries_respawn_dormant_vehicles(25, 700)

            self.port = self.traffic_manager.get_port()

            if self.sync:
                settings = self.world.get_settings()
                self.traffic_manager.set_synchronous_mode(True)
                if not settings.synchronous_mode:
                    self.synchronous_master = True
                    settings.synchronous_mode = True
                    settings.fixed_delta_seconds = 0.05
                    self.world.apply_settings(settings)
                else:
                    self.synchronous_master = False

            blueprintsWalkers = self.world.get_blueprint_library().filter('vehicle.*')
            SpawnActor = carla.command.SpawnActor
            SetAutopilot = carla.command.SetAutopilot
            FutureActor = carla.command.FutureActor
            #  -------------
            #  Spawn Other Vehicles
            #  -------------

            self.map = self.world.get_map()  # Expensive, should only be called once
            # More complicated: also spawning the ego vehicle randomly!
            # Retrieve points where we can spawn the ego-vehicle
            self.map = self.world.get_map()
            spawn_points = self.world.get_map().get_spawn_points()

            # 2. we spawn the walker object
            batch = []
            walker_speed = []
            self.spawn_points = []
            k = 0
            j = 0
            while j < self.walker_count:
                """"""
                if k > len(spawn_points):
                    break

                spawn_point = self.map.get_waypoint(spawn_points[k].location, lane_type=carla.LaneType.Driving)
                k += 1
                self.spawn_points.append(spawn_point)

                walker_bp = random.choice(blueprintsWalkers)
                # set as not invincible
                if walker_bp.has_attribute('is_invincible'):
                    walker_bp.set_attribute('is_invincible', 'false')
                walker_bp.set_attribute('role_name', 'autopilot')

                # batch = [SpawnActor(walker_bp, spawn_point.transform).then(SetAutopilot(FutureActor, True, self.traffic_manager.get_port()))]
                batch.append(SpawnActor(walker_bp, spawn_point.transform).then(
                    SetAutopilot(FutureActor, True, self.port)))

                j += 1

                print("Vehicle Module: Published initial pose ({}, {})".format(spawn_point.transform.location.x,
                                                                               spawn_point.transform.location.y))

            results = self.client.apply_batch_sync(batch, True)
            failures = 0

            for i in range(len(results)):
                if results[i].error:
                    logging.error(results[0].error)
                    failures += 1
                    # self.spawn_points.pop(-1)
                else:
                    self.walkers_list.append({"id": results[i].actor_id})
                    # j += 1

            # 4. we put altogether the walkers and controllers id to get the objects from their id
            for i in range(len(self.walkers_list)):
                self.all_id.append(self.walkers_list[i]["id"])
            self.all_actors = self.world.get_actors(self.all_id)

            # wait for a tick to ensure client receives the last transform of the walkers we have just created
            if not self.sync or not self.synchronous_master:
                self.world.wait_for_tick()
            else:
                self.world.tick()

            print('Other Cars Module: Spawned %d/%d cars, press Ctrl+C to exit.' % (
            len(self.walkers_list), self.walker_count))

        except:
            print("Unexpected error:", sys.exc_info()[0])

    def update(self):
        pass

    def on_shutdown(self):
        self.shutdown_set = True

    def reset_callback(self):
        print("doing nothing")
        self.destroy_actors()
        self.create()

    def destroy_actors(self):

        print('\nOther Cars Module: Destroying %d cars' % len(self.walkers_list))
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.all_id])
        self.walkers_list = []
        self.all_id = []
        self.all_actors = []
        time.sleep(0.5)


class VehicleModule:

    def __init__(self, client,world):

        params = rospy.get_param('/carla/scenario/vehicle')

        # Starting location
        self.start_x_ = params['start_x']
        self.start_y_ = params['start_y']

        # The goal to drive to
        self.goal_x_ = params['goal_x']
        self.goal_y_ = params['goal_y']

        self.run_counter = 0
        self.last_reset_counter = 0

        self.goal_pub_ = rospy.Publisher('/carla/ego_vehicle/goal', PoseStamped, queue_size=1)
        self.state_sub_ = rospy.Subscriber('/carla/ego_vehicle/odometry', Odometry,
                                           lambda msg: self.state_callback(msg))
        # Also a listener to republish the goal
        self.reset_sub = rospy.Subscriber("/carla/ego_vehicle/initialpose", PoseWithCovarianceStamped,
                                          self.reset_callback)

        rospy.sleep(1.0)

        self.world = world
        self.spawn_points = self.world.get_map().get_spawn_points()

        print('Vehicle Module: Init')

    def state_callback(self, msg):
        # If less than 5 meters from the goal
        if abs(msg.pose.pose.position.x - self.goal_x_) < 5 and self.last_reset_counter > 10:

            self.last_reset_counter = 0
            self.run_counter += 1
        else:
            self.last_reset_counter += 1

    def reset_callback(self, msg):
        rospy.sleep(1.0)

        self.goal = random.choice(self.spawn_points)

        while (np.linalg.norm(np.array([self.goal.location.x - msg.pose.pose.position.x,
                                        self.goal.location.y - msg.pose.pose.position.y]))) > 10:
            self.goal = random.choice(self.spawn_points)

        # Publish the goal location
        goal_msg = PoseStamped()
        goal_msg.pose.position.x = self.goal.location.x
        goal_msg.pose.position.y = self.goal.location.y
        goal_msg.header.frame_id = 'map'
        goal_msg.header.stamp = rospy.get_rostime()
        self.goal_pub_.publish(goal_msg)
        print("Vehicle Module: Published goal ({}, {})".format(self.goal.location.x, self.goal.location.y))

    def update(self):
        pass


if __name__ == '__main__':

    host = '127.0.0.1'
    port = 2000

    pedestrian_module = None
    vehicle_module = None

    rospy.init_node("carla_scene", anonymous=True)

    client = carla.Client(host, port)
    client.set_timeout(5.0)

    try:
        world = client.get_world()

        # Start a pedestrian spawner
        #pedestrian_module = PedestrianModule(client,world)

        # Start Other vehicles
        #other_vehicles_module = OtherVehiclesModule(client,world)

        # Start the vehicle monitor module
        vehicle_module = VehicleModule(client,world)

        while not rospy.is_shutdown():
            #pedestrian_module.update()
            #other_vehicles_module.update()
            #vehicle_module.update()

            rospy.spin()

            #world.wait_for_tick()

    finally:
        pedestrian_module.destroy_actors()
        other_vehicles_module.destroy_actors()
