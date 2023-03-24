import carla_msgs.msg
import geometry_msgs.msg as geom_msg
import sensor_msgs.msg as sensor_msg
import carla_msgs.msg as carla_msg
import std_msgs.msg as std_msg
import nav_msgs.msg as nav_msgs
import numpy as np
from numpy import random
import rospy
import time
import carla
from tf.transformations import quaternion_from_euler
import cv2
from scipy.spatial.transform import Rotation
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


class CarlaEnv:
    def __init__(self):
        self.front_img = None
        self.front_img_gray = None
        self.front_segmented = None
        self.front_segmented_single = None
        self.front_depth = None
        self.keyboard_info = None
        self.feedback_received = False
        self.human_reset = False
        self.feedback = 0
        self.hysteresis_thr = 3
        self.hysteresis = 0
        self.collision = False
        self.random_velocity_ref = False
        self.selected_velocity_ref = np.random.uniform(3, 10)
        self.collision_info = []
        self.received_image_counter = 0
        self.last_image_counter = 0
        self.joystick = True
        self.action_before_feedback = 0
        self.velocity = 0
        self.observation_type = 'segmented-gray-depth'
        self.exit_code = 1
        self.velocity_stuck = False
        self.velocity_stuck_counter = 0
        self.position = np.array([0, 0])
        self.orientation = np.array([0, 0, 0, 0])
        self.traffic_lights_status = None
        self.drive = 1
        self.traffic_light_heuristic = False
        self.prev_position = np.array([])

        self.feedback_pub = rospy.Publisher('/feedback', std_msg.Bool, queue_size=10)
        self.done_pub = rospy.Publisher('/done', std_msg.Bool, queue_size=10)
        self.reset_pub = rospy.Publisher('/lmpcc/initialpose', geom_msg.PoseWithCovarianceStamped, queue_size=10)
        self.velocity_ref_pub = rospy.Publisher('rodrigo', std_msg.Float64, queue_size=10)
        self.human_action_pub = rospy.Publisher('human_action', std_msg.Float64, queue_size=10)
        self.network_action_pub = rospy.Publisher('network_action', std_msg.Float64, queue_size=10)
        self.h_counter_pub = rospy.Publisher('h_counter', std_msg.Float64, queue_size=10)
        self.drive_pub = rospy.Publisher('drive', std_msg.Float64, queue_size=10)
        rospy.Subscriber('/carla/ego_vehicle/camera/rgb/front/image_color', sensor_msg.Image, self._callback_front_image,
                         queue_size=10)
        rospy.Subscriber('/lmpcc/feasibility', std_msg.Float64,self._callback_feasibility, queue_size=10)
        rospy.Subscriber('/carla/ego_vehicle/camera/semantic_segmentation/front/image_segmentation', sensor_msg.Image, self._callback_front_segmented,
                         queue_size=10)
        rospy.Subscriber('/carla/ego_vehicle/camera/depth/front/image_depth', sensor_msg.Image, self._callback_front_depth,
                         queue_size=10)
        rospy.Subscriber('/carla/ego_vehicle/waypoints', nav_msgs.Path, self._callback_waypoints,
                         queue_size=10)
        rospy.Subscriber('/carla/ego_vehicle/odometry', nav_msgs.Odometry, self._callback_odometry,
                         queue_size=10)
        rospy.Subscriber('/carla/traffic_lights_info', carla_msgs.msg.CarlaTrafficLightInfoList, self._callback_traffic_lights_info,
                         queue_size=10)
        rospy.Subscriber('/carla/traffic_lights', carla_msgs.msg.CarlaTrafficLightStatusList, self._callback_traffic_lights_status,
                         queue_size=10)

        if self.joystick:
            rospy.Subscriber('/joy', sensor_msg.Joy, self._callback_joy, queue_size=10)
        else:  # use keyboard
            rospy.Subscriber('/cmd_vel', geom_msg.Twist, self._callback_keyboard, queue_size=10)

        rospy.Subscriber('/path_over', std_msg.Bool, self._callback_over, queue_size=10)
        rospy.Subscriber('/carla/ego_vehicle/collision', carla_msg.CarlaCollisionEvent, self._callback_collision, queue_size=10)

        # Get CARLA interface
        host = '127.0.0.1'
        port = 2000
        self.client = carla.Client(host, port)
        self.client.set_timeout(5.0)
        self.world = self.client.get_world()
        self.spawn_points = self.world.get_map().get_spawn_points()
        self.map = self.world.get_map()

        self.n_waypoints = 0
        self.past_feedback_received = False
        self.h_counter = 0
        self.n_infeasible_steps = 0
        self.total_infeasible_steps = 0

        self.is_over = False

        self.total_distance = 0

    def _callback_feasibility(self, data):
        self.exit_code = data.data

    def _callback_over(self, data):
        self.is_over = data.data

    def _callback_waypoints(self, data):
        self.n_waypoints = len(data.poses)
        self.last_waypoint = np.array([data.poses[-5].pose.position.x, data.poses[-5].pose.position.y])

    def _callback_front_image(self, data):
        self.front_img = np.reshape(np.fromstring(data.data, np.uint8), [256, 256, 4])  # RGBA
        self.front_img_gray = cv2.cvtColor(self.front_img, cv2.COLOR_BGR2GRAY)

    def _callback_front_segmented(self, data):
        self.front_segmented = np.reshape(np.fromstring(data.data, np.uint8), [256, 256, 4])  # RGBA
        self.front_segmented_single = self.front_segmented[:, :, 0]
        self.received_image_counter += 1

    def _callback_front_depth(self, data):
        image = np.reshape(np.fromstring(data.data, np.float32), [256, 256])  # Get depth image coded in "32FC1", which in numpy corresponds to np.float32
        # Map values of image between 0 and 255
        image = (image / 1000) * 255
        self.front_depth = image  # Last channel has the combined information

    def _callback_keyboard(self, data):
        self.feedback_received = True
        self.keyboard_info = data

    def _callback_joy(self, data):
        if data.buttons[0] == 1 or data.buttons[3] == 1:
            self.feedback_received = True
        else:
            self.feedback_received = False
            self.past_feedback_received = self.feedback_received

        self.joystick_info = data

    def _callback_collision(self, data):
        self.collision = True
        self.collision_info = data
        if self.collision:
            print('Vehicle Module: Collision detected. Restarting...')

    def _callback_odometry(self, data):
        self.velocity = (data.twist.twist.linear.x / 5) - 1
        self.position = np.array([data.pose.pose.position.x, data.pose.pose.position.y])
        self.orientation = np.array([data.pose.pose.orientation.x, data.pose.pose.orientation.y, data.pose.pose.orientation.z, data.pose.pose.orientation.w])

        if self.prev_position.size == 0:
            self.prev_position = self.position

    def _callback_traffic_lights_info(self, data):
        self.traffic_light_info = data

    def _callback_traffic_lights_status(self, data):
        self.traffic_lights_status = data.traffic_lights

    def _detect_traffic_light(self):
        angle_dist_thr = np.pi / 4
        alpha_width_0 = 3
        alpha_width_1 = 7
        alpha_length_0 = -3
        alpha_length_1 = 25
        active_traffic_lights = []
        # Find active traffic lights
        for traffic_light in self.traffic_light_info.traffic_lights:
            id = traffic_light.id

            # Get positions and angles
            traffic_light_position = np.array([traffic_light.transform.position.x, traffic_light.transform.position.y])
            traffic_light_orientation = np.array([traffic_light.transform.orientation.x,
                                                  traffic_light.transform.orientation.y,
                                                  traffic_light.transform.orientation.z,
                                                  traffic_light.transform.orientation.w])

            orientation_euler = Rotation.from_quat(self.orientation).as_euler('zxy')
            orientation_traffic_light_euler = Rotation.from_quat(traffic_light_orientation).as_euler('zxy')

            # Create car window
            point_base_1 = np.array(
                [self.position[0] + alpha_width_0 * np.cos(orientation_euler[0] + np.pi / 2),
                 self.position[1] + alpha_width_0 * np.sin(orientation_euler[0] + np.pi / 2)])

            point_base_2 = np.array(
                [self.position[0] + alpha_width_1 * np.cos(orientation_euler[0] - np.pi / 2),
                 self.position[1] + alpha_width_1 * np.sin(orientation_euler[0] - np.pi / 2)])

            point_1 = np.array([point_base_1[0] + alpha_length_0 * np.cos(orientation_euler[0]),
                                point_base_1[1] + alpha_length_0 * np.sin(orientation_euler[0])])

            point_2 = np.array([point_base_2[0] + alpha_length_0 * np.cos(orientation_euler[0]),
                                point_base_2[1] + alpha_length_0 * np.sin(orientation_euler[0])])

            point_3 = np.array([point_base_1[0] + alpha_length_1 * np.cos(orientation_euler[0]),
                                point_base_1[1] + alpha_length_1 * np.sin(orientation_euler[0])])

            point_4 = np.array([point_base_2[0] + alpha_length_1 * np.cos(orientation_euler[0]),
                                point_base_2[1] + alpha_length_1 * np.sin(orientation_euler[0])])

            # Check if light in car window
            polygon = Polygon([point_1, point_2, point_4, point_3])
            point = Point(traffic_light_position[0], traffic_light_position[1])
            light_in_window = polygon.contains(point)

            # Align car-traffic lights frames
            orientation_euler[0] += np.pi / 2

            # Compute angle distance
            angle_distance = np.abs(np.abs(orientation_euler[0] - orientation_traffic_light_euler[0]) - np.pi)

            if angle_distance < angle_dist_thr and light_in_window:
                active_traffic_lights.append(id)

        # Get active traffic lights status
        active_traffic_lights_status = []

        for traffic_light_id in active_traffic_lights:
            for traffic_light_status in self.traffic_lights_status:
                if traffic_light_status.id == traffic_light_id:
                    active_traffic_lights_status.append(traffic_light_status.state)

        if len(active_traffic_lights_status) > 1:
            print('Too many traffic lights detected! Continue driving...')
            self.drive = 1
        elif len(active_traffic_lights_status) == 0:
            self.drive = 1  # No traffic light detected, so continue driving
        else:
            if active_traffic_lights_status[0] == 0:  # red light, stop
                self.drive = 0
            else:  # green/yellow light, drive
                self.drive = 1

    def _get_obs(self):
        if self.front_segmented is None:
            print('Front image not received!')
            observation = np.zeros([600, 800, 3])
        else:
            if self.observation_type == 'RGB':
                observation = self.front_img[:, :, :3]
            elif self.observation_type == 'segmented':
                observation = self.front_segmented[:, :, :3]
            elif self.observation_type == 'segmented-gray-depth':
                observation = np.zeros([256, 256, 3])
                observation[:, :, 0] = self.front_segmented_single
                observation[:, :, 1] = self.front_img_gray
                observation[:, :, 2] = self.front_depth

        return [self.front_segmented[:, :, :3], self.velocity, self.drive]

    def get_feedback(self):
        if self.feedback_received or True:
            self.hysteresis = 0
            if self.joystick:
                if self.joystick_info.buttons[3] == 1:
                    self.human_reset = True
                    self.feedback = np.array([0])
                h = np.array([self.joystick_info.axes[1]])
                self.feedback = h
            else:
                if np.abs(self.keyboard_info.linear.x) > 0.0:
                    self.human_reset = True
                    self.feedback = np.array([0])
                self.feedback = np.array([np.sign(self.keyboard_info.linear.z)])
                self.feedback_received = False
        else:
            self.hysteresis += 1
            if self.hysteresis >= self.hysteresis_thr:
                self.feedback = np.array([0])

        if self.human_reset:
            print('Vehicle Module: Human reset. Restarting...')
        return self.feedback

    def reset(self):
        msg = std_msg.Float64()
        msg.data = -1
        self.velocity_ref_pub.publish(msg)

        msg = geom_msg.PoseWithCovarianceStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = 'map'
        # Reset
        print('Vehicle Module: Vehicle reached the goal')
        print('Vehicle Module: Resetting...')
        initial_pose = random.choice(self.spawn_points)

        sidewalk_waypoint = self.map.get_waypoint(initial_pose.location, lane_type=carla.LaneType.Driving)

        print('Vehicle Module: Published initial pose ({}, {})'.format(initial_pose.location.x, initial_pose.location.y))

        msg.pose.pose.position.x = sidewalk_waypoint.transform.location.x
        msg.pose.pose.position.y = sidewalk_waypoint.transform.location.y
        quarternion = quaternion_from_euler(sidewalk_waypoint.transform.rotation.roll*np.pi/180, sidewalk_waypoint.transform.rotation.pitch*np.pi/180, sidewalk_waypoint.transform.rotation.yaw*np.pi/180, 'sxyz')
        msg.pose.pose.orientation.x = quarternion[0]
        msg.pose.pose.orientation.y = quarternion[1]
        msg.pose.pose.orientation.z = quarternion[2]
        msg.pose.pose.orientation.w = quarternion[3]
        print('YAW: ' + str(sidewalk_waypoint.transform.rotation.yaw))
        self.reset_pub.publish(msg)

        # Tell lmpcc to reset with ref -1
        msg = std_msg.Float64()
        msg.data = -1.0  # Reference going from 0 to 8
        self.is_over = False
        self.velocity_ref_pub.publish(msg)
        time.sleep(4.0)

        print('Social MPCC ready!')

        # Random velocity ref for database generation
        if self.random_velocity_ref:
            self.selected_velocity_ref = np.random.uniform(3, 8)

        # Set done variables to False
        self.human_reset = False
        self.collision = False
        self.n_infeasible_steps = 0
        self.total_infeasible_steps = 0
        return self._get_obs()

    def check_if_stuck(self):
        if int(self.exit_code) != 1:
            self.total_infeasible_steps += 1
            self.n_infeasible_steps += 1
            if self.n_infeasible_steps > 20:
                return True
        else:
            self.n_infeasible_steps = 0
            return False

    def check_velocity_stuck(self):
        if self.velocity < -0.95:
            self.velocity_stuck_counter += 1
        else:
            self.velocity_stuck_counter = 0

        if self.velocity_stuck_counter > 600:
            self.velocity_stuck = True
            self.velocity_stuck_counter = 0
            print('Velocity stuck!')
        else:
            self.velocity_stuck = False

    def check_if_over(self):
        return self.is_over
        #return np.linalg.norm(self.last_waypoint-self.position) < 4

    def step(self, velocity_ref, action_human, action_agent):
        if not self.feedback_received:
            self.action_before_feedback = velocity_ref
        else:
            self.h_counter += 1

        #self.stuck = self.check_if_stuck()
        self.check_velocity_stuck()
        self.done_with_path = self.check_if_over()

        # Detect traffic light
        self._detect_traffic_light()

        distance = np.linalg.norm(self.position-self.prev_position)
        if distance < 10.0:
            self.total_distance += distance
        self.prev_position = self.position

        # Check for restart
        if self.human_reset or self.done_with_path or self.collision or self.velocity_stuck:
            done = True
            # Publish accumulated feedback
            msg = std_msg.Float64()
            msg.data = self.h_counter
            self.h_counter_pub.publish(msg)
            self.h_counter = 0
        else:
            done = False

        msg = std_msg.Float64()
        if self.random_velocity_ref:
            msg.data = self.selected_velocity_ref
        else:
            if self.joystick:
                #msg.data = np.clip(velocity_ref * 8, 0, 8)
                msg.data = np.clip((velocity_ref + 1) * 4, 0, 7.9)
            else:
                msg.data = np.clip((velocity_ref + 1) * 4, 0, 7.9)  # Reference going from 0 to 8

        # Syncrhonize with simulator
        while True:
            if self.received_image_counter > self.last_image_counter:
                break

        self.last_image_counter = self.received_image_counter

        if self.drive == 1 or not self.traffic_light_heuristic:
            self.velocity_ref_pub.publish(msg)
        else:
            print('Traffic light stop!')
            msg.data = 0
            self.velocity_ref_pub.publish(msg)

        # Publish human intervention signal
        feedback_msg = std_msg.Bool()
        feedback_msg.data = self.feedback_received
        self.feedback_pub.publish(feedback_msg)

        # Publish action human
        msg = std_msg.Float64()
        msg.data = (action_human + 1) * 4
        self.human_action_pub.publish(msg)

        # Publish action agent
        msg = std_msg.Float64()
        msg.data = (action_agent + 1) * 4
        self.network_action_pub.publish(msg)

        # Publish done signal
        done_msg = std_msg.Bool()
        done_msg.data = done
        self.done_pub.publish(done_msg)

        # Publish drive
        msg = std_msg.Float64()
        msg.data = self.drive
        self.drive_pub.publish(msg)

        return self._get_obs(), {}, done, {}
