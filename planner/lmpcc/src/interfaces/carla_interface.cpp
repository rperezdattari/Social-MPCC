#include "interfaces/carla_interface.h"

CarlaInterface::CarlaInterface(ros::NodeHandle &nh, MPCC *controller, predictive_configuration *config, BaseModel *solver_interface_ptr)
    : Interface(nh, controller, config)
{
    ROS_WARN("Initializing Carla Interface.");

    solver_interface_ptr_ = solver_interface_ptr;

    // Subscribers for sensor data
    state_sub_ = nh.subscribe(config_->robot_state_topic_, 1, &CarlaInterface::StateCallBack, this);
    obstacle_sub_ = nh.subscribe(config_->obs_state_topic_, 1, &CarlaInterface::ObstacleCallBack, this);
    waypoints_sub_ = nh.subscribe(config_->waypoint_topic_, 1, &CarlaInterface::WaypointsCallback, this);
    vehicle_info_sub_ = nh.subscribe("/carla/ego_vehicle/vehicle_info", 1, &CarlaInterface::VehicleInfoCallback, this);

    // Subscribe to initial pose estimate in rviz (set this topic in the rviz config!)
    reset_pose_sub_ = nh.subscribe("/lmpcc/initialpose", 1, &CarlaInterface::ResetCallback, this);

    // Debugging callback for Carla
    carla_status_sub_ = nh.subscribe("/carla/status", 1, &CarlaInterface::carlaStatusCallback, this);

    // Publisher for vehicle command
    command_pub_ = nh.advertise<ackermann_msgs::AckermannDrive>(config_->cmd_, 1);
    reset_carla_pub_ = nh.advertise<geometry_msgs::PoseWithCovarianceStamped>(config_->reset_carla_topic_, 1);
    obstacles_pub_ = nh.advertise<visualization_msgs::MarkerArray>("/obstacles",1);
    // Enable / disable to receive info on the actual vehicle commands after the ackermann node!
    // ackermann_info_sub_ = nh.subscribe("/carla/ego_vehicle/ackermann_control/control_info", 1, &CarlaInterface::AckermannCallback, this);

    // To add for publishing the goal
    goal_pub_ = nh.advertise<geometry_msgs::PoseStamped>(config_->navigation_goal_topic_, 1);
    goal_set_ = false; // Publish when the waypoint publisher is online
    goal_location_ = Eigen::Vector2d(324, -2);

    // Initialize the obstacles variable
    obstacles_.lmpcc_obstacles.resize(config_->max_obstacles_);
    for (int obst_it = 0; obst_it < config_->max_obstacles_; obst_it++)
    {
        obstacles_.lmpcc_obstacles[obst_it].trajectory.poses.resize(solver_interface_ptr_->FORCES_N);
        obstacles_.lmpcc_obstacles[obst_it].major_semiaxis.resize(solver_interface_ptr_->FORCES_N);
        obstacles_.lmpcc_obstacles[obst_it].minor_semiaxis.resize(solver_interface_ptr_->FORCES_N);
        obstacles_.lmpcc_obstacles[obst_it].id = obst_it;
        for (size_t t = 0; t < solver_interface_ptr_->FORCES_N; t++)
        {
            obstacles_.lmpcc_obstacles[obst_it].trajectory.poses[t].pose.position.x = 100; // Lowered...
            obstacles_.lmpcc_obstacles[obst_it].trajectory.poses[t].pose.position.y = 100;
            obstacles_.lmpcc_obstacles[obst_it].trajectory.poses[t].pose.orientation.z = 0;
            obstacles_.lmpcc_obstacles[obst_it].major_semiaxis[t] = 0.1;
            obstacles_.lmpcc_obstacles[obst_it].minor_semiaxis[t] = 0.1;
        }
    }

    ROS_WARN("Carla Interface Initialized");
}

// Callback to get Ego vehicle information
void CarlaInterface::VehicleInfoCallback(const carla_msgs::CarlaEgoVehicleInfo &msg){
    ROS_INFO_STREAM("CarlaInterface::VehicleInfoCallback");
    ROS_INFO_STREAM(msg);
    ego_vehicle_info_ = msg;
}

void CarlaInterface::PublishGoal(){

    // // To add for publishing the goal
    // geometry_msgs::PoseStamped initial_goal;
    // initial_goal.pose.position.x = goal_location_(0);
    // initial_goal.pose.position.y = goal_location_(1);
    // initial_goal.header.frame_id = config_->target_frame_;
    // initial_goal.header.stamp = ros::Time::now();
    // goal_pub_.publish(initial_goal);
}
// Callback for steering angle (carla ego_vehicle status or next control input), can add gnss via navsat launch in robot_localization
// Maybe the virtual wall is related to the spline index limit. Otherwise disable obstacles fully to be sure.

void CarlaInterface::Actuate()
{
    if (config_->activate_debug_output_)
        ROS_WARN("Actuating Vehicle");

    if (config_->activate_debug_output_)
        ROS_WARN_STREAM("COMMAND = \ntheta = " << solver_interface_ptr_->delta(1) <<  "\nv = " << solver_interface_ptr_->v(1)<<  "\na = " << solver_interface_ptr_->a());

    // Send the computed commands to the vehicle (these are already the axis)
    // Ackermann is as if there is a single wheel being controlled at the center of the front axis
    // Note: we using integrate() to compute the setpoint 1/f in the future rather than the integrator time step!
    // Steering (integrated to the next control time step)
    command_msg_.steering_angle = solver_interface_ptr_->delta(1);
    command_msg_.steering_angle_velocity = solver_interface_ptr_->w();

    // Throttle (integrated to the next control time step)
    command_msg_.speed = solver_interface_ptr_->v(1);
    command_msg_.acceleration = solver_interface_ptr_->a();
    command_pub_.publish(command_msg_);

    // We need to update the steering angle since there is no feedback
    solver_interface_ptr_->getState()->set_delta(command_msg_.steering_angle);
}

void CarlaInterface::ActuateBrake(double deceleration)
{
    if (config_->activate_debug_output_)
        ROS_INFO_STREAM("Actuating the Brake: " << deceleration);

    command_msg_.steering_angle = 0.0;
    command_msg_.steering_angle_velocity = 0.0;

    // Go to zero speed with deceleration
    command_msg_.speed = 0.0;
    command_msg_.acceleration = -deceleration;
    command_msg_.jerk = 0.0;

    command_pub_.publish(command_msg_);
}

void CarlaInterface::ResetCallback(const geometry_msgs::PoseWithCovarianceStamped& initial_pose){
    ROS_ERROR_STREAM("Resetting to: " << initial_pose);
    if (external_reset_callback_)
        (controller_->*external_reset_callback_)();
    // solver_interface_ptr_->resetSolver();

    // solver_interface_ptr_->getState()->init();
    ROS_ERROR_STREAM("Resetting to: " << initial_pose);
    Reset(initial_pose.pose.pose);

}

void CarlaInterface::Reset(const geometry_msgs::Pose &position_after_reset)
{

    geometry_msgs::PoseWithCovarianceStamped reset_msg;
    reset_msg.pose.pose = position_after_reset;

    for (int j = 0; j < 10; j++)
    {
        ActuateBrake(0.0);
        ros::Duration(1.0/config_->clock_frequency_).sleep();
    }

    reset_carla_pub_.publish(reset_msg);
    ROS_WARN_STREAM("Resetting to: " << reset_msg);

}

void CarlaInterface::Reset()
{
    geometry_msgs::Pose default_position;
    default_position.position.x = 10;
    default_position.position.y = 0;

    Reset(default_position);
}

// I have to debug the obstacle callback!
/* CALLBACKS */
// Needs to call some function (I need to give a function handle into the init)
void CarlaInterface::ObstacleCallBack(const derived_object_msgs::ObjectArray &received_obstacles) 
{
    // If there were other objects
    if((received_obstacles.objects.size() > 1) && (ego_vehicle_info_.id != 0) ) {

        double use_predictions = 1.0;
        if (std::abs(solver_interface_ptr_->getState()->get_v())<0.1 )
            use_predictions = 0.0;

        // If there are less objects then the maximum configured number of obstacles
        if (received_obstacles.objects.size() < (unsigned int)config_->max_obstacles_) {

            if (config_->activate_debug_output_)
                ROS_INFO_STREAM("Carla Interface: Received " << received_obstacles.objects.size() - 1 << " Obstacles (Less than max_obstacles=" << config_->max_obstacles_ << ")");
            
            // Read all objects that are not the ego-vehicle
            int obst_it = 0;
            for (size_t i = 0; i < received_obstacles.objects.size(); i++) {

                if(received_obstacles.objects[i].id != ego_vehicle_info_.id) {
                    obstacles_.lmpcc_obstacles[obst_it].id = received_obstacles.objects[i].classification;
                    obstacles_.lmpcc_obstacles[obst_it].pose = received_obstacles.objects[i].pose;
                    for (size_t t = 0; t < solver_interface_ptr_->FORCES_N; t++)
                    {
                        obstacles_.lmpcc_obstacles[obst_it].trajectory.poses[t].pose.position.x = received_obstacles.objects[i].pose.position.x + 0.2 * t * received_obstacles.objects[i].twist.linear.x * use_predictions;
                        obstacles_.lmpcc_obstacles[obst_it].trajectory.poses[t].pose.position.y = received_obstacles.objects[i].pose.position.y + 0.2 * t * received_obstacles.objects[i].twist.linear.y * use_predictions;
                        obstacles_.lmpcc_obstacles[obst_it].trajectory.poses[t].pose.orientation = received_obstacles.objects[i].pose.orientation;
                        obstacles_.lmpcc_obstacles[obst_it].major_semiaxis[t] = std::max(received_obstacles.objects[i].shape.dimensions[0],received_obstacles.objects[i].shape.dimensions[1])/2+0.6;
                        obstacles_.lmpcc_obstacles[obst_it].minor_semiaxis[t] = std::min(received_obstacles.objects[i].shape.dimensions[0],received_obstacles.objects[i].shape.dimensions[1])/2+0.4;
                    }
                    obst_it ++;
                }
            }

            obst_it --; // Ego vehicle?
            if(received_obstacles.objects[obst_it].id == ego_vehicle_info_.id)
                obst_it --;


            for (int j = received_obstacles.objects.size() - 1; j < config_->max_obstacles_; j++)
            {
                obstacles_.lmpcc_obstacles[j].id = obstacles_.lmpcc_obstacles[obst_it].id;
                for (size_t t = 0; t < solver_interface_ptr_->FORCES_N; t++)
                {

                    obstacles_.lmpcc_obstacles[j].trajectory.poses[t].pose.position.x = obstacles_.lmpcc_obstacles[obst_it ].trajectory.poses[t].pose.position.x;
                    obstacles_.lmpcc_obstacles[j].trajectory.poses[t].pose.position.y = obstacles_.lmpcc_obstacles[obst_it].trajectory.poses[t].pose.position.y;
                    obstacles_.lmpcc_obstacles[j].trajectory.poses[t].pose.orientation = obstacles_.lmpcc_obstacles[obst_it].trajectory.poses[t].pose.orientation;
                    obstacles_.lmpcc_obstacles[j].major_semiaxis[t] = obstacles_.lmpcc_obstacles[obst_it - 1].major_semiaxis[t];
                    obstacles_.lmpcc_obstacles[j].minor_semiaxis[t] = obstacles_.lmpcc_obstacles[obst_it - 1].minor_semiaxis[t];
                }
            }
        } else {

            double use_predictions = 1.0;
            if (std::abs(solver_interface_ptr_->getState()->get_v())<0.1 )
                use_predictions = 0.0;

            if (config_->activate_debug_output_)
                ROS_INFO_STREAM("Carla Interface: Received " << received_obstacles.objects.size() - 1 << " Obstacles (More than max_obstacles=" << config_->max_obstacles_ << ")");

            // If there are more obstacles than the maximum, resize the obstacles
            if (received_obstacles.objects.size() != obstacles_.lmpcc_obstacles.size()-1)
                obstacles_.lmpcc_obstacles.resize(received_obstacles.objects.size()-1);

            // Save all obstacles that are not the ego-vehicle
            int obst_it = 0;
            for (size_t i = 0; i < received_obstacles.objects.size(); i++)
            {
                if(received_obstacles.objects[i].id != ego_vehicle_info_.id) {
                    obstacles_.lmpcc_obstacles[obst_it].major_semiaxis.resize(solver_interface_ptr_->FORCES_N);
                    obstacles_.lmpcc_obstacles[obst_it].minor_semiaxis.resize(solver_interface_ptr_->FORCES_N);
                    obstacles_.lmpcc_obstacles[obst_it].trajectory.poses.resize(solver_interface_ptr_->FORCES_N);
                    obstacles_.lmpcc_obstacles[obst_it].pose = received_obstacles.objects[i].pose;

                    for (size_t t = 0; t < solver_interface_ptr_->FORCES_N; t++)
                    {
                        obstacles_.lmpcc_obstacles[obst_it].trajectory.poses[t].pose.position.x = received_obstacles.objects[i].pose.position.x + 0.2 * t * received_obstacles.objects[i].twist.linear.x * use_predictions;
                        obstacles_.lmpcc_obstacles[obst_it].trajectory.poses[t].pose.position.y = received_obstacles.objects[i].pose.position.y + 0.2 * t * received_obstacles.objects[i].twist.linear.y * use_predictions;
                        obstacles_.lmpcc_obstacles[obst_it].trajectory.poses[t].pose.orientation = received_obstacles.objects[i].pose.orientation;
                        obstacles_.lmpcc_obstacles[obst_it].id = received_obstacles.objects[i].classification;
                        obstacles_.lmpcc_obstacles[obst_it].major_semiaxis[t] = std::max(received_obstacles.objects[i].shape.dimensions[0],received_obstacles.objects[i].shape.dimensions[1])/2+0.8;
                        obstacles_.lmpcc_obstacles[obst_it].minor_semiaxis[t] = std::min(received_obstacles.objects[i].shape.dimensions[0],received_obstacles.objects[i].shape.dimensions[1])/2+0.4;
                    }
                    obst_it++;
                }
            }
        }

        if(config_->activate_debug_output_)
            ROS_INFO("Carla Interface: Ordering obstacles");

        if (received_obstacles.objects.size()-1 > (unsigned int)config_->max_obstacles_) {
            double Xp, Yp, distance;
            std::vector<double> objectDistances;
            //ROS_INFO_STREAM("COmputing distances");

            for (size_t i = 0; i < obstacles_.lmpcc_obstacles.size(); i++)
            {

                //ROS_INFO_STREAM("transform the pose to base_link in order to calculate the distance to the obstacle");
                transformPose("map", "base_link", obstacles_.lmpcc_obstacles[i].pose);

                //get obstacle coordinates in base_link frame
                Xp = obstacles_.lmpcc_obstacles[i].pose.position.x;
                Yp = obstacles_.lmpcc_obstacles[i].pose.position.y;

                if(std::abs(std::atan2(Yp,Xp))<1.57079632679)
                {

                    //ROS_INFO_STREAM("distance between the Prius and the obstacle");
                    distance = sqrt(pow(Xp, 2) + pow(Yp, 2));

                    //ROS_INFO_STREAM("transform the pose back to planning_frame for further calculations");
                    transformPose("base_link", config_->target_frame_,
                                  obstacles_.lmpcc_obstacles[i].pose);

                    obstacles_.lmpcc_obstacles[i].distance = distance;
                }
                else{
                    obstacles_.lmpcc_obstacles[i].distance = std::numeric_limits<double>::infinity();
                }
            }

            OrderObstacles(obstacles_);
        }
    }
    else{
        if (config_->activate_debug_output_)
            ROS_INFO_STREAM("Carla Interface: Received no obstacles, inserting dummies");

        for (int j = 0; j < config_->max_obstacles_; j++)
        {
            obstacles_.lmpcc_obstacles[j].id = 10+j;
            for (size_t t = 0; t < solver_interface_ptr_->FORCES_N; t++)
            {
                obstacles_.lmpcc_obstacles[j].trajectory.poses[t].pose.position.x = solver_interface_ptr_->getState()->get_x() + 100;
                obstacles_.lmpcc_obstacles[j].trajectory.poses[t].pose.position.y = solver_interface_ptr_->getState()->get_y() + 100;
                obstacles_.lmpcc_obstacles[j].trajectory.poses[t].pose.orientation.z = 0 ;
                obstacles_.lmpcc_obstacles[j].major_semiaxis[t] = 0.1;
                obstacles_.lmpcc_obstacles[j].minor_semiaxis[t] = 0.1;
            }
        }
    }

    if(config_->activate_debug_output_){
        for (size_t i = 0; i < obstacles_.lmpcc_obstacles.size(); i++)
            ROS_WARN_STREAM("Carla Interface: Distance to obstacle: " << obstacles_.lmpcc_obstacles[i].distance);
    }

    plotObstacles(); 

    if (external_obstacle_callback_)
        (controller_->*external_obstacle_callback_)();
}

void CarlaInterface::OrderObstacles(lmpcc_msgs::lmpcc_obstacle_array &ellipses)
{
    // Create vector of obstacles
    if (ellipses.lmpcc_obstacles.size() > 0)
    {
        std::vector<lmpcc_msgs::lmpcc_obstacle> ellipsesVector;
        ellipsesVector = ellipses.lmpcc_obstacles;

        // Sort vector according to distances
        std::sort(ellipsesVector.begin(), ellipsesVector.end(), [](lmpcc_msgs::lmpcc_obstacle const &obst1, lmpcc_msgs::lmpcc_obstacle const &obst2) {
            return (obst1.distance < obst2.distance);
        });

        // Write vector of sorted obstacles to obstacles structure
        ellipses.lmpcc_obstacles = ellipsesVector;
    }
    //ROS_INFO_STREAM(ellipses);
}

void CarlaInterface::StateCallBack(const nav_msgs::Odometry::ConstPtr &msg)
{
    if (config_->activate_debug_output_)
        ROS_INFO("State Callback");

    // Update the frame
    config_->target_frame_ = msg->header.frame_id;

    // Update the states
    solver_interface_ptr_->getState()->set_psi(Helpers::quaternionToAngle(msg->pose.pose));
    solver_interface_ptr_->getState()->set_x(msg->pose.pose.position.x ); // for shifting the current coordinates to the center of mass
    solver_interface_ptr_->getState()->set_y(msg->pose.pose.position.y );

    // Plots a Axis to display the path variable location
    geometry_msgs::TransformStamped transformStamped;
    transformStamped.header.stamp = ros::Time::now();
    transformStamped.header.frame_id = config_->target_frame_;
    transformStamped.child_frame_id = "robot_state";

    transformStamped.transform.translation.x = solver_interface_ptr_->getState()->get_x();
    transformStamped.transform.translation.y = solver_interface_ptr_->getState()->get_y();
    transformStamped.transform.translation.z = 0.0;
    // tf::Quaternion q = tf::createQuaternionFromRPY(0, 0, 0);
    transformStamped.transform.rotation.x = 0;
    transformStamped.transform.rotation.y = 0;
    transformStamped.transform.rotation.z = 0;
    transformStamped.transform.rotation.w = 1;

    state_pub_.sendTransform(transformStamped);

    // Update the velocity
    solver_interface_ptr_->getState()->set_v(std::min(std::sqrt(std::pow(msg->twist.twist.linear.x, 2) + std::pow(msg->twist.twist.linear.y, 2)),7.9));

    // Check whether the waypoints that we are meant to have are ready
    if (!goal_set_)
    {
        if (x_.size() == 0 || Helpers ::dist(Eigen::Vector2d(x_[x_.size() - 1], y_[y_.size() - 1]), goal_location_) > 25)
        {
            PublishGoal();
            goal_set_ = true;
        }
    }

    if (config_->activate_debug_output_)
        solver_interface_ptr_->getState()->print();

    if (external_state_callback_)
        (controller_->*external_state_callback_)();

}

// Callback for ax and ay
void CarlaInterface::AccelerationCallback(const geometry_msgs::AccelWithCovarianceStamped &msg)
{
    if (config_->activate_debug_output_)
        ROS_INFO("Acceleration Callback");

    solver_interface_ptr_->getState()->set_ax(msg.accel.accel.linear.x);
    solver_interface_ptr_->getState()->set_ay(msg.accel.accel.linear.y);
}

void CarlaInterface::AckermannCallback(const carla_ackermann_control::EgoVehicleControlInfo &msg)
{
    ROS_WARN_STREAM("Ackermann acceleration: " << msg.current.accel);
    ROS_WARN_STREAM("Max steering angle: " << msg.restrictions.max_steering_angle);
    ROS_WARN_STREAM("Steering Target: :" << msg.target.steering_angle);
}

void CarlaInterface::WaypointsCallback(const nav_msgs::Path &msg){

    // if (config_->activate_debug_output_)
    ROS_INFO_STREAM("Carla Interface: Received " << std::floor(0.5*msg.poses.size()) << " Waypoints");

    int waypoint_count = 0;
    double L;

    theta_.clear();
    x_.clear();
    y_.clear();

    theta_.push_back(Helpers::quaternionToAngle(msg.poses[0].pose));
    x_.push_back(msg.poses[0].pose.position.x);
    y_.push_back(msg.poses[0].pose.position.y);

    for (size_t i = 1; i < msg.poses.size(); i += 2)
    {
        L = std::sqrt(std::pow(x_[i-1]-msg.poses[i].pose.position.x,2)+std::pow(y_[i-1]-msg.poses[i].pose.position.y,2));

        if (L>config_->min_wapoints_distance_){
            theta_.push_back(Helpers::quaternionToAngle(msg.poses[i].pose));
            x_.push_back(msg.poses[i].pose.position.x);
            y_.push_back(msg.poses[i].pose.position.y);
        }

        waypoint_count++;
    }

    if(waypoint_count < 6){
        for (size_t i = 0; i < 12; i += 2)
        {
            theta_.push_back(solver_interface_ptr_->getState()->get_psi());
            x_.push_back(solver_interface_ptr_->getState()->get_x()+ 10*(i+1));
            y_.push_back(solver_interface_ptr_->getState()->get_y());
        }
    }

    // std::cout << "x" << std::endl;
    // for (int i = 0; i < waypoint_count; i++)
    // {
    //     std::cout << x_[i] << ", ";
    // }
    // std::cout << "y" << std::endl;
    // for (int i = 0; i < waypoint_count; i++)
    // {
    //     std::cout << y_[i] << ", ";
    // }
    // std::cout << "theta" << std::endl;
    // for (int i = 0; i < waypoint_count; i++)
    // {
    //     std::cout << theta_[i] << ", ";
    // }

    if(external_waypoints_callback_)
        (controller_->*external_waypoints_callback_)();
}

// Callback for detecting slow down of the Carla ROS bridge
void CarlaInterface::carlaStatusCallback(const carla_msgs::CarlaStatus & msg){

    static Helpers::Benchmarker carla_benchmarker("Carla Frames");  // Print min/max/avg timing of frames of Carla

    // Not nice, but just to ignore the first few frames, because they are slow
    static bool hasReset;
    if(!hasReset && carla_benchmarker.getTotalRuns() > 200){
        carla_benchmarker.stop();
        carla_benchmarker.reset();
        hasReset = true;
    }


    if(carla_benchmarker.isRunning()){
        double last_duration = carla_benchmarker.stop();
               
        if(config_->activate_debug_output_)
        ROS_INFO_STREAM("Carla Frame Update Took: " << last_duration*1000 << " ms, configured delta: " << msg.fixed_delta_seconds*1000 << " ms");

    }
    
    carla_benchmarker.start();
}

bool CarlaInterface::transformPose(const std::string &from, const std::string &to, geometry_msgs::Pose &pose)
{
    bool transform = false;
    tf::StampedTransform stamped_tf;

    geometry_msgs::PoseStamped stampedPose_in, stampedPose_out;

    stampedPose_in.pose = pose;

    if (std::sqrt(std::pow(pose.orientation.x, 2) + std::pow(pose.orientation.y, 2) + std::pow(pose.orientation.z, 2) + std::pow(pose.orientation.w, 2)) < 1.0 - 1e-9)
    {
        stampedPose_in.pose.orientation.x = 0;
        stampedPose_in.pose.orientation.y = 0;
        stampedPose_in.pose.orientation.z = 0;
        stampedPose_in.pose.orientation.w = 1;
        std::cout << "LMPCC: Quaternion was not normalised properly!" << std::endl;
    }
    stampedPose_in.header.frame_id = from;

    // make sure source and target frame exist
    if (tf_listener_.frameExists(to) && tf_listener_.frameExists(from))
    {
        try
        {
            tf_listener_.transformPose(to, stampedPose_in, stampedPose_out);

            transform = true;
        }
        catch (tf::TransformException &ex)
        {
            ROS_ERROR("MPCC::getTransform: %s", ex.what());
        }
    }
    else
    {
        ROS_WARN("MPCC::getTransform: '%s' or '%s' frame doesn't exist, pass existing frame", from.c_str(), to.c_str());
    }
    pose = stampedPose_out.pose;
    stampedPose_in.pose = stampedPose_out.pose;
    stampedPose_in.header.frame_id = to;

    return transform;
}

void CarlaInterface::plotObstacles(void){
    visualization_msgs::MarkerArray obstacles_list;

    for (size_t obs_id = 0; obs_id < obstacles_.lmpcc_obstacles.size(); obs_id++) // 100 points
    {
        visualization_msgs::Marker obs_shape;

        obs_shape.header.frame_id = config_->target_frame_;
        obs_shape.id = 100+obs_id;

        if(obstacles_.lmpcc_obstacles[obs_id].id == 6){
            obs_shape.type = visualization_msgs::Marker::CUBE;
        }
        else{
            obs_shape.type = visualization_msgs::Marker::SPHERE;
        }
        obs_shape.type = visualization_msgs::Marker::SPHERE;
        obs_shape.scale.x = obstacles_.lmpcc_obstacles[obs_id].major_semiaxis[0]*2;
        obs_shape.scale.y = obstacles_.lmpcc_obstacles[obs_id].minor_semiaxis[0]*2;
        obs_shape.scale.z = 0.2;
        // Line strip is blue
        obs_shape.color.r = 0.2;
        obs_shape.color.g = 0.2;
        obs_shape.color.b = 0.7;
        obs_shape.color.a = 0.8;

        obs_shape.pose = obstacles_.lmpcc_obstacles[obs_id].trajectory.poses[0].pose;

        obstacles_list.markers.push_back(obs_shape);
    }
    unsigned int n = std::min((unsigned int)config_->max_obstacles_, (unsigned int)obstacles_.lmpcc_obstacles.size());

    for (size_t obs_id = 0; obs_id < n; obs_id++) // 100 points
    {
        for (size_t t = 0; t < solver_interface_ptr_->FORCES_N; t++) // 100 points
        {
            visualization_msgs::Marker obs_shape;

            obs_shape.header.frame_id = config_->target_frame_;
            obs_shape.id = 200 + obs_id * solver_interface_ptr_->FORCES_N + t;

            if (obstacles_.lmpcc_obstacles[obs_id].id == 6) {
                obs_shape.type = visualization_msgs::Marker::CUBE;
            } else {
                obs_shape.type = visualization_msgs::Marker::SPHERE;
            }
            obs_shape.type = visualization_msgs::Marker::SPHERE;
            obs_shape.scale.x = obstacles_.lmpcc_obstacles[obs_id].major_semiaxis[0]*2;
            obs_shape.scale.y = obstacles_.lmpcc_obstacles[obs_id].minor_semiaxis[0]*2;
            obs_shape.scale.z = 0.2;

            obs_shape.color.r = 0.8;
            obs_shape.color.g = 0.2;
            obs_shape.color.b = 0.2;
            obs_shape.color.a = 1.0/(1+t);

            obs_shape.pose = obstacles_.lmpcc_obstacles[obs_id].trajectory.poses[t].pose;
            //ROS_INFO_STREAM("obs_id: " << obs_id << " t: " << t);
            obstacles_list.markers.push_back(obs_shape);
        }
    }

    obstacles_pub_.publish(obstacles_list);
}
