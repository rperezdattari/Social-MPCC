"""Spawn NPCs into the simulation"""

import glob
import os
import sys
import csv
import argparse
import logging
import numpy as np
import random
import time
import math
import xml.etree.ElementTree as ET
from multiprocessing import Process
import carla

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

print('Initialization ...')

# Read parameters
tree = ET.parse('crossing_2people.xml')
root = tree.getroot()

Spawnpoint = np.zeros((len(root)-1, 2))
Waypoint = np.zeros((len(root)-1, 6, 2))
Speed = np.zeros((len(root)-1, 1))
Threshold = np.zeros((len(root)-1, 1))

Rate = float(root.find('Distance_check_rate').text)
number = len(root.findall('agent'))

for idx, agent in enumerate(root):
    for idx2, par in enumerate(agent):
        if par.tag == "spawnpoint":
            Spawnpoint[idx][0] = par.attrib.get('x')
            Spawnpoint[idx][1] = par.attrib.get('y')
        
        elif par.tag == "speed":
            Speed[idx] = par.text

        elif par.tag == "threshold":
            Threshold[idx] = par.text
        
        elif par.tag == "waypoint":
            Waypoint[idx][idx2-3][0] = par.attrib.get('x')
            Waypoint[idx][idx2-3][1] = par.attrib.get('y')
        
        else:
            pass

class SpawnWalker:
    def __init__(self):
        self.argparser = argparse.ArgumentParser(description=__doc__)
        self.argparser.add_argument(
            '--host',
            metavar='H',
            default='127.0.0.1',
            help='IP of the host server (default: 127.0.0.1)')
        self.argparser.add_argument(
            '-p', '--port',
            metavar='P',
            default=2000,
            type=int,
            help='TCP port to listen to (default: 2000)')
        self.argparser.add_argument(
            '-w', '--number-of-walkers',
            metavar='W',
            default=number, # Read from xml file
            type=int,
            help='number of walkers (default: 50)')
        self.argparser.add_argument(
            '--filterw',
            metavar='PATTERN',
            default='walker.pedestrian.*',
            help='pedestrians filter (default: "walker.pedestrian.*")')
        self.argparser.add_argument(
            '--tm-port',
            metavar='P',
            default=8000,
            type=int,
            help='port to communicate with TM (default: 8000)')
        self.argparser.add_argument(
            '--sync',
            action='store_true',
            help='Synchronous mode execution')
        self.argparser.add_argument(
            '--hybrid',
            action='store_true',
            help='Enanble')
        self.args = self.argparser.parse_args()
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

        self.client = carla.Client(self.args.host, self.args.port)
        self.client.set_timeout(10.0)
        self.synchronous_master = False

        self.world = self.client.get_world()
        self.blueprintsWalkers = self.world.get_blueprint_library().filter(self.args.filterw)
        
        self.SpawnActor = carla.command.SpawnActor

        if self.args.sync:
            self.settings = self.world.get_settings()
            self.traffic_manager.set_synchronous_mode(True)
            if not self.settings.synchronous_mode:
                self.synchronous_master = True
                self.settings.synchronous_mode = True
                self.settings.fixed_delta_seconds = 0.05
                self.world.apply_settings(self.settings)
            else:
                self.synchronous_master = False

    def Apply_Settings(self):
        self.walkers_list = []
        self.all_id = []
        if not self.args.sync or not self.synchronous_master:
            self.world.wait_for_tick()
        else:
            self.world.tick()
        print('Settings are applied')
        return self.walkers_list, self.all_id

    def SpawnPedestrians(self, walkers_list, all_id, SpawnLocation):
        # 1. Take all the locations to spawn
        self.walkers_list = walkers_list
        self.all_id = all_id
        spawn_points = []
        percentagePedestriansRunning = 0.0
        
        for i in range(self.args.number_of_walkers):
            spawn_point = carla.Transform(carla.Location(x=SpawnLocation[i][0], y=SpawnLocation[i][1]))

            if (spawn_point.location != None):
                spawn_points.append(spawn_point)
        
        # 2. Spawn the walker object
        self.batch = []
        walker_speed = []
        for spawn_point in spawn_points:
            self.walker_bp = random.choice(self.blueprintsWalkers)
            # set as not invincible
            if self.walker_bp.has_attribute('is_invincible'):
                self.walker_bp.set_attribute('is_invincible', 'true')
            # set the max speed
            if self.walker_bp.has_attribute('speed'):
                if (random.random() > percentagePedestriansRunning):
                    # walking
                    walker_speed.append(self.walker_bp.get_attribute('speed').recommended_values[1])
                else:
                    # running
                    walker_speed.append(self.walker_bp.get_attribute('speed').recommended_values[2])
            else:
                print("Walker has no speed")
                walker_speed.append(0.0)
            self.batch.append(self.SpawnActor(self.walker_bp, spawn_point))
                
        results = self.client.apply_batch_sync(self.batch, True)
        walker_speed2 = []
        for i in range(len(results)):
            if results[i].error:
                #print('Pedestrian %d-th is not spawned' %(i+1))
                #print('Location of that pedestrian is:', spawn_points[i].location)
                logging.error(results[i].error)
            else:
                self.walkers_list.append({"id": results[i].actor_id})
                walker_speed2.append(walker_speed[i])
        #print(self.walkers_list)
        self.walker_speed = walker_speed2

        # 3. Spawn the walker controller
        self.batch = []
        self.walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
    
        for i in range(len(self.walkers_list)):
            self.batch.append(self.SpawnActor(self.walker_controller_bp, carla.Transform(), self.walkers_list[i]["id"]))
        results = self.client.apply_batch_sync(self.batch, True)
        
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                self.walkers_list[i]["con"] = results[i].actor_id

        # 4. Put altogether the walkers and controllers id to get the objects from their id
        for i in range(len(self.walkers_list)):
            self.all_id.append(self.walkers_list[i]["con"])
            self.all_id.append(self.walkers_list[i]["id"])
        self.all_actors = self.world.get_actors(self.all_id)
       
        # wait for a tick to ensure client receives the last transform of the walkers we have just created
        if not self.args.sync or not self.synchronous_master:
            self.world.wait_for_tick()
        else:
            self.world.tick()  

        return self.walkers_list, self.all_id, self.all_actors, self.walker_speed 
                
    def PedestrianTarget(self, TargetLocation, all_actors, walker_speed, i, Speed): 
        self.all_actors = all_actors
        self.walker_speed = walker_speed

        # Get the current position
        current_position = np.array([all_actors[i+1].get_location().x, all_actors[i+1].get_location().y])
        
        # Compute the distance from the current position to the next waypoint
        dist = np.linalg.norm(TargetLocation - current_position)

        # Set the destination
        self.all_actors[i].go_to_location(carla.Location(x = TargetLocation[0], y = TargetLocation[1]))
        
        # Set the speed of the walker
        # self.all_actors[i].set_max_speed(float(self.walker_speed[int(i/2)]))
        self.all_actors[i].set_max_speed(float(Speed))

        if not self.args.sync or not self.synchronous_master:
            self.world.wait_for_tick()
        else:
            self.world.tick() 
        
        return self.all_actors, dist, self.walker_speed

    def StartOneWalker(self, all_id, Waypoint, all_actors, walker_speed, Speed, Rate, Threshold):
        TargetLocation = Waypoint[0]
        
        while True:
            for j in range(TargetLocation.shape[0]):
                
                # Print out the next waypoint
                print(TargetLocation[j])
                print(j)
                self.all_actors, dist, self.walker_speed = self.PedestrianTarget(TargetLocation[j], all_actors, walker_speed, 0, Speed[0])
                
                # Condition
                while dist > Threshold[0]:
                    current_position = np.array([all_actors[1].get_location().x, all_actors[1].get_location().y])
                    dist = np.linalg.norm(TargetLocation[j] - current_position)
                    print(dist)
                    time.sleep(1/Rate)
                    if dist > 20:
                        self.all_actors[1].set_location(carla.Location(x = SpawnLocation[0], y = SpawnLocation[1]))
        
        return self.all_actors, self.walker_speed
    
    def StartTwoWalker(self, all_id, Waypoint, all_actors, walker_speed, Speed, Rate, Threshold):
        TargetLocation1 = Waypoint[0]
        TargetLocation2 = Waypoint[1]
        i = 0
        j = 0

        # Print out the next waypoint
        print("First pedestrian:", TargetLocation1[i])
        print("Second pedestrian:", TargetLocation2[j])

        self.all_actors, dist1, self.walker_speed = self.PedestrianTarget(TargetLocation1[i], all_actors, walker_speed, 0, Speed[0])
        self.all_actors, dist2, self.walker_speed = self.PedestrianTarget(TargetLocation2[j], all_actors, walker_speed, 2, Speed[1])
    
        while True:
            if dist1 > Threshold[0]:
                print("First pedestrian:", dist1)
                current_position1 = np.array([all_actors[1].get_location().x, all_actors[1].get_location().y])
                dist1 = np.linalg.norm(TargetLocation1[i] - current_position1)
                time.sleep(1/Rate)
            else:
                i = i + 1
                if i == TargetLocation1.shape[0]:
                    i = 0
                self.all_actors, dist1, self.walker_speed = self.PedestrianTarget(TargetLocation1[i], all_actors, walker_speed, 0, Speed[0])
                
            if dist2 > Threshold[1]:
                print("Second pedestrian:", dist2)
                current_position2 = np.array([all_actors[3].get_location().x, all_actors[3].get_location().y])
                dist2 = np.linalg.norm(TargetLocation2[j] - current_position2)
                time.sleep(1/Rate)
            else:
                j = j + 1
                if j == TargetLocation2.shape[0]:
                    j = 0
                self.all_actors, dist2, self.walker_speed = self.PedestrianTarget(TargetLocation2[j], all_actors, walker_speed, 2, Speed[1])
        
        return self.all_actors, self.walker_speed
    
    def StartNWalker(self, all_id, Waypoint, all_actors, walker_speed, Speed, Rate, Threshold, number):
        # Start all the pedestrians
        dist = np.zeros((number, 1))
        location_idx = np.zeros((number))
        actor_idx = 0
        for i in range(number):
            TargetLocation = Waypoint[i]
            self.all_actors, dist[i], self.walker_speed = self.PedestrianTarget(TargetLocation[int(location_idx[i])], all_actors, walker_speed, actor_idx, Speed[i])
            actor_idx = actor_idx + 2
        print(dist)

        # Distance check for all the pedestrians
        while True:
            actor_idx = 0
            for i in range(number):
                if dist[i] > Threshold[i]:
                    TargetLocation = Waypoint[i]
                    print("Pedestrian %d next waypoint:(%d, %d)" %((i+1), TargetLocation[int(location_idx[i])][0], TargetLocation[int(location_idx[i])][1]))
                    print("Pedestrian %d distance to next waypoint is %f" %((i+1), dist[i]))
                    current_position = np.array([all_actors[actor_idx].get_location().x, all_actors[actor_idx].get_location().y])
                    dist[i] = np.linalg.norm(TargetLocation[int(location_idx[i])] - current_position)
                else:
                    location_idx[i] = location_idx[i] + 1
                    TargetLocation = Waypoint[i]
                    if location_idx[i] == TargetLocation.shape[0]:
                        location_idx[i] = 0
                    self.all_actors, dist[i], self.walker_speed = self.PedestrianTarget(TargetLocation[int(location_idx[i])], all_actors, walker_speed, actor_idx, Speed[i])
                actor_idx = actor_idx + 2
            time.sleep(1/Rate)
        
        return self.all_actors, self.walker_speed
    
    def DestroyActors(self, walkers_list, all_id, all_actors):
        # Stop walker controllers
        # self.all_actors = all_actors

        for i in range(0, len(all_id), 2):
            all_actors[i].stop()

        print('\ndestroying %d walkers' % len(walkers_list))
        self.client.apply_batch([carla.command.DestroyActor(x) for x in all_id])
        
        if not self.args.sync or not self.synchronous_master:
            self.world.wait_for_tick()
        else:
            self.world.tick()

def main():
    try:
        # Run codes only once
        walkers_list, all_id = SpawnWalker().Apply_Settings()
        walkers_list, all_id, all_actors, walker_speed = SpawnWalker().SpawnPedestrians(walkers_list, all_id, Spawnpoint)
        
        # Start the actors
        for i in range(0, len(all_id), 2):
            all_actors[i].start()
        
        # Start the AIcontroller
        if number == 5:
            SpawnWalker().StartOneWalker(all_id, Waypoint, all_actors, walker_speed, Speed, Rate, Threshold)
        
        elif number == 6:
            SpawnWalker().StartTwoWalker(all_id, Waypoint, all_actors, walker_speed, Speed, Rate, Threshold)
        
        else:
            SpawnWalker().StartNWalker(all_id, Waypoint, all_actors, walker_speed, Speed, Rate, Threshold, number)
        
    finally:
        SpawnWalker().DestroyActors(walkers_list, all_id, all_actors)
        #pass

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')