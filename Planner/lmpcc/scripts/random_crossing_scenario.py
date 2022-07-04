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

import argparse
import logging
from numpy import random


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

    def __init__(self, client):
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
        self.reset_sub = rospy.Subscriber("/carla/ego_vehicle/initialpose", PoseWithCovarianceStamped,
                                          lambda msg: self.reset_callback())

        try:
            self.world = self.client.get_world()

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
            self.percentage_running = 0.0      # how many pedestrians will run
            self.percentage_crossing = 90.0     # how many pedestrians will walk through the road

            self.map = self.world.get_map() # Expensive, should only be called once
            # More complicated: also spawning the ego vehicle randomly!
            # Retrieve points where we can spawn the ego-vehicle
            # spawn_points = world.get_map().get_spawn_points()
            # number_of_spawn_points = len(spawn_points)
            # spawn_point = spawn_points[0]
            #waypoints = map.generate_waypoints(10.0)
            # for waypoint in waypoints:
            #     print("type = {}, x = {}, y = {}".format(waypoint.road_id, waypoint.transform.location.x, waypoint.transform.location.y))

            # Get the ego-vehicle
            ego_vehicle = None
            while not ego_vehicle:
                for carla_actor in self.world.get_actors():
                    if carla_actor.type_id.startswith("vehicle"):
                            ego_vehicle = carla_actor

                time.sleep(0.02)

            # Location of the ego-vehicle
            ego_transform = ego_vehicle.get_transform()
            ego_location = ego_transform.location

            # get a location a bit in front
            forward_dist = 10.0
            forward_location = ego_location + ego_transform.get_forward_vector()*forward_dist

            # Waypoint on the lane (and sidewalk) closest to where the ego vehicle is
            closest_waypoint = self.map.get_waypoint(forward_location)
            [self.lane_waypoints_right, self.lane_waypoints_left, end_of_lane] = find_sidewalk_around(self.map, forward_location)

            while True:
                next_lane_waypoint = get_waypoint_after_junction(end_of_lane)
                if next_lane_waypoint is None:
                    break

                [new_lane_waypoints_right, new_lane_waypoints_left, end_of_lane] = find_sidewalk_around(self.map, next_lane_waypoint.transform.location)
                self.lane_waypoints_right += new_lane_waypoints_right
                self.lane_waypoints_left += new_lane_waypoints_left

            self.lane_waypoints = self.lane_waypoints_left + self.lane_waypoints_right

            # 1. take all the random locations to spawn
            self.spawn_points = []
            self.spawned_right = []
            for i in range(self.walker_count):
                # Find a random starting location (50/50 right/left)
                spawn_point = carla.Transform()
                will_spawn_right = randrange(2)
                if will_spawn_right:
                    rand_waypoint_idx = randrange(len(self.lane_waypoints_right))  # world.get_random_location_from_navigation()
                    self.spawned_right.append(True)  # if in the right waypoints
                    loc = self.lane_waypoints_right[rand_waypoint_idx].transform.location
                else:
                    rand_waypoint_idx = randrange(len(self.lane_waypoints_left))  # world.get_random_location_from_navigation()
                    self.spawned_right.append(False)  # if in the right waypoints
                    loc = self.lane_waypoints_left[rand_waypoint_idx].transform.location

                if (loc != None):
                    # print(str(loc))
                    spawn_point.location = loc
                    self.spawn_points.append(spawn_point)

            # 2. we spawn the walker object
            batch = []
            walker_speed = []
            for spawn_point in self.spawn_points:
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
                batch.append(SpawnActor(walker_bp, spawn_point))
            results = self.client.apply_batch_sync(batch, True)
            walker_speed2 = []
            failures = 0
            for i in range(len(results)):
                if results[i].error:
                    logging.error(results[i].error)
                    failures += 1
                else:
                    self.walkers_list.append({"id": results[i].actor_id})
                    walker_speed2.append(walker_speed[i])
            walker_speed = walker_speed2
            #print(failures)
            # Todo: Fix these failures so that the spawned number is correct

            # 3. we spawn the walker controller
            batch = []
            walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
            for i in range(len(self.walkers_list)):
                batch.append(SpawnActor(walker_controller_bp, carla.Transform(), self.walkers_list[i]["id"]))
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
            self.redirect_all_pedestrians(start_pedestrians=True)

            for i in range(0, len(self.all_id), 2):
                self.all_actors[i].set_max_speed(float(walker_speed[int(i/2)]))

            print('Pedestrian Module: Spawned %d/%d walkers, press Ctrl+C to exit.' % (len(self.walkers_list), self.walker_count))

        except:
            print("Unexpected error:", sys.exc_info()[0])

    def update(self):
        pass

    def on_shutdown(self):
        self.shutdown_set = True

    def redirect_all_pedestrians(self, start_pedestrians=False):

        redirect_counter = 0
        j = 0

        for i in range(0, len(self.all_id), 2):

            if start_pedestrians:
                self.all_actors[i].start()

            # Check how far this ped is from its goal location, if too far, do not redirect
            if (not start_pedestrians) and \
                    self.pedestrian_goals[j].distance(self.all_actors[i+1].get_location()) > 50:
                j += 1
                continue

            # set walk to random point (bias on the opposite side to get crossing!)
            # Determine where to spawn based on the cross factor
            cross_realization = randrange(100)
            will_cross = cross_realization < self.percentage_crossing

            # Spawn condition on the right
            if (self.spawned_right[j] and not will_cross) or (not self.spawned_right[j] and will_cross):
                rand_waypoint_idx = randrange(len(self.lane_waypoints_right))
                loc = self.lane_waypoints_right[rand_waypoint_idx].transform.location
            else:
                rand_waypoint_idx = randrange(len(self.lane_waypoints_left))
                loc = self.lane_waypoints_left[rand_waypoint_idx].transform.location

            self.spawned_right[j] = not self.spawned_right[j]

            if start_pedestrians:
                self.pedestrian_goals.append(loc)
            else:
                self.pedestrian_goals[j] = loc  # Save their goal locations

            j += 1
            redirect_counter += 1

            self.all_actors[i].go_to_location(loc)  # world.get_random_location_from_navigation())

        if not start_pedestrians:
            print('Pedestrian Module: Redirected {} pedestrians'.format(redirect_counter))


    def reset_callback(self):
        self.redirect_all_pedestrians()

    def destroy_actors(self):
        rospy.signal_shutdown("")

        if self.sync and self.synchronous_master:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            self.world.apply_settings(settings)

        # stop walker controllers (list is [controller, actor, controller, actor ...])
        for i in range(0, len(self.all_id), 2):
            self.all_actors[i].stop()

        print('\nPedestrian Module: Destroying %d walkers' % len(self.walkers_list))
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.all_id])

        time.sleep(0.5)


class VehicleModule:

    def __init__(self):
        print('Vehicle Module: Init')

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
        self.reset_pub_ = rospy.Publisher('/lmpcc/initialpose', PoseWithCovarianceStamped, queue_size=1)

        # Also a listener to republish the goal
        self.reset_sub = rospy.Subscriber("/carla/ego_vehicle/initialpose", PoseWithCovarianceStamped,
                                          lambda msg: self.reset_callback())

        rospy.sleep(1.0)
        self.reset_callback()

    def state_callback(self, msg):
        # If less than 5 meters from the goal
        if abs(msg.pose.pose.position.x - self.goal_x_) < 5 and self.last_reset_counter > 10:

            # Reset
            print("Vehicle Module: Vehicle reached the goal")
            print("Vehicle Module: Resetting...")

            reset_msg = PoseWithCovarianceStamped()
            reset_msg.pose.pose.position.x = self.start_x_
            reset_msg.pose.pose.position.y = self.start_y_
            reset_msg.header.frame_id = 'map'
            reset_msg.header.stamp = rospy.get_rostime()
            self.reset_pub_.publish(reset_msg)

            self.last_reset_counter = 0
            self.run_counter += 1
        else:
            self.last_reset_counter += 1

    def reset_callback(self):
        # Publish the goal location
        goal_msg = PoseStamped()
        goal_msg.pose.position.x = self.goal_x_
        goal_msg.pose.position.y = self.goal_y_
        goal_msg.header.frame_id = 'map'
        goal_msg.header.stamp = rospy.get_rostime()
        self.goal_pub_.publish(goal_msg)
        print("Vehicle Module: Published goal ({}, {})".format(self.goal_x_, self.goal_y_))

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
        pedestrian_module = PedestrianModule(client)

        # Start the vehicle monitor module
        vehicle_module = VehicleModule()

        while not rospy.is_shutdown():

            pedestrian_module.update()
            vehicle_module.update()

            rospy.spin()

            world.wait_for_tick()

    finally:
        pedestrian_module.destroy_actors()
