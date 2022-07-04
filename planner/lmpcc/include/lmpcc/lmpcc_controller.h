/**
 * @file lmpcc_controller.h
 * @brief Main controller class
 * @version 0.1
 * @date 2022-07-04
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#ifndef LMPCC_LMPCC_H
#define LMPCC_LMPCC_H

// ros includes
#include <pluginlib/class_loader.h>
#include <ros/ros.h>
#include <ros/package.h>
#include <geometry_msgs/AccelWithCovarianceStamped.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseArray.h>
#include <sensor_msgs/JointState.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Joy.h>
#include <tf/tf.h>
#include <std_msgs/Float64.h>
#include <std_msgs/Float32.h>
#include <std_msgs/Float64MultiArray.h>
#include <tf/transform_listener.h>

// eigen includes
#include <Eigen/Eigen>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/LU>

// std includes
#include <iostream>
#include <string>
#include <vector>
#include <math.h>
#include <memory>
#include <algorithm>
#include <limits>

// boost includes
#include <boost/shared_ptr.hpp>
#include <boost/scoped_ptr.hpp>

// Visualization messages
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/InteractiveMarker.h>
#include <visualization_msgs/InteractiveMarkerControl.h>

// yaml parsing
#include <fstream>
#include <yaml-cpp/yaml.h>

// predicitve includes
#include <lmpcc/lmpcc_configuration.h>
#include <lmpcc/control_feedback.h>

// actions, srvs, msgs
#include <actionlib/server/simple_action_server.h>
#include <actionlib/client/simple_action_client.h>
#include <nav_msgs/Path.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/OccupancyGrid.h>


// lmpcc messages
#include <lmpcc_msgs/lmpcc_feedback.h>
#include <lmpcc_msgs/lmpcc_obstacle.h>
#include <lmpcc_msgs/lmpcc_obstacle_array.h>
#include <lmpcc_msgs/IntTrigger.h>
#include <lmpcc_msgs/Control.h>

// Generate solver model
#include <lmpcc/base_model.h>
#include <lmpcc/base_state.h>
#include <lmpcc/base_input.h>
#include <interfaces/carla_interface.h>
#include <lmpcc/reference_path.h>

// Scenario related
#include <scenario/scenario_manager.h>

//Dynamic Reconfigure server
#include <boost/thread/mutex.hpp>
#include <boost/shared_ptr.hpp>
#include <dynamic_reconfigure/server.h>
#include <lmpcc/PredictiveControllerConfig.h>

//TF
#include <tf2_ros/transform_broadcaster.h>

//Joint states
#include <sensor_msgs/JointState.h>

// Visuals
#include <lmpcc_tools/ros_visuals.h>
#include <lmpcc_tools/helpers.h>

//reset msgs
#include <std_srvs/Empty.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <geometry_msgs/PoseWithCovariance.h>
#include <geometry_msgs/Pose.h>
#include <lmpcc/LMPCCReset.h>

typedef double real_t;

class Interface;

class MPCC
{
    /** Managing execution of all classes of predictive control
     * - Handle static and dynamic collision avoidance
     */

public:

    //DYnamic reconfigure server
    boost::shared_ptr<dynamic_reconfigure::Server<lmpcc::PredictiveControllerConfig> > reconfigure_server_;
    boost::recursive_mutex reconfig_mutex_;
    void reconfigureCallback(lmpcc::PredictiveControllerConfig& config, uint32_t level);

    /**
     * @brief MPCC: Default constructor, allocate memory
     */
    MPCC()
    {

    };

    /**
     * @brief ~MPCC: Default distructor, free memory
     */
    ~MPCC();

    /**
     * @brief initialize: Initialize all helper class of predictive control and subscibe joint state and publish controlled joint velocity
     * @return: True with successfully initialize all classes else false
     */
    bool initialize();

    /**
     * @brief Callback for the velocity reference
     * 
     * @param msg 
     */
    void VRefCallBack(const std_msgs::Float64::ConstPtr& msg);

    /**
     * @brief Callback for the joystick
     * 
     * @param msg 
     */
    void JoyCallBack(const sensor_msgs::Joy::ConstPtr& msg);

    /** Update functions when data comes in from the interface */
    void OnReset();
    void OnObstaclesReceived();
    void OnStateReceived();
    void OnWaypointsReceived();

    /** Debugging functionality */
    void checkConstraints();
    void checkCollisionConstraints();

    void mapCallback(const nav_msgs::OccupancyGrid::ConstPtr& occupancy_grid);

    /**
     * @brief getTransform: Find transformation stamed rotation is in the form of quaternion
     * @param from: source frame from find transformation
     * @param to: target frame till find transformation
     * @param stamped_pose: Resultant poseStamed between source and target frame
     * @return: true if transform else false
     */
    bool getTransform(const std::string& from,
                      const std::string& to,
                      geometry_msgs::PoseStamped& stamped_pose
    );

    /**
     * @brief transformStdVectorToEigenVector: tranform std vector to eigen vectors as std vectos are slow to random access
     * @param vector: std vectors want to tranfrom
     * @return Eigen vectors transform from std vectos
     */
    template<typename T>
    static inline Eigen::VectorXd transformStdVectorToEigenVector(const std::vector<T>& vector)
    {
        // resize eigen vector
        Eigen::VectorXd eigen_vector = Eigen::VectorXd(vector.size());

        // convert std to eigen vector
        for (uint32_t i = 0; i < vector.size(); ++i)
        {
            eigen_vector(i) = vector.at(i);
        }

        return eigen_vector;
    }

    /** public data member */
    //Service clients
    ros::ServiceClient reset_simulation_client_, reset_ekf_client_;

    // waypoints subscriber
    ros::Subscriber waypoints_sub_,ped_stop_sub_,v_ref_sub_,joy_sub_;

    ros::Subscriber plan_subs_;

    // Subscriber for point cloud
    ros::Subscriber point_cloud_sub_;

    ros::ServiceServer reset_server_;

    ros::Publisher computation_pub_;
    // ros::Publisher marker_pub_;
    ros::Subscriber obstacles_state_sub_;

    ros::Publisher  joint_state_pub_, close_path_points_pub_, path_is_over_pub_;

    // Reference Path to follow
    ReferencePath reference_path_;

    /* Interface - responsible for interacting with the simulation or robot (i.e., reading sensors, publishing control commands) */
    std::unique_ptr<Interface> system_interface_;

    /* Solver interface - creates an intuitive API for the Forces Pro solver */
    std::unique_ptr<BaseModel> solver_interface_;

    // publish trajectory
    ros::Publisher traj_pub_, pred_traj_pub_, pred_cmd_pub_,cost_pub_,robot_collision_space_pub_,brake_pub_, contour_error_pub_, feedback_pub_, feasibility_pub_;

    //Predicted trajectory
    nav_msgs::Path pred_traj_;
    nav_msgs::Path pred_cmd_;

    //Controller options
    bool enable_output_;
    bool reset_world_;
    bool plan_;
    bool replan_;
    bool debug_;
    bool simulation_mode_;
    bool auto_enable_;
    real_t te_;

    tf2_ros::TransformBroadcaster state_pub_,path_pose_pub_;
    std_msgs::Float64 cost_;
    std_msgs::Float64 brake_;
    double contour_error_;
    double lag_error_;

    int exit_code_;

    // Scenario
    std::vector<std::vector<Eigen::Vector2d>> poses_vec_;
    std::vector<double> orientation_vec_;
    std::vector<std::unique_ptr<ScenarioManager>> scenario_manager_;

    double simulated_velocity_;
    Eigen::Vector4d state_ahead_;

    bool state_received_;

    //Search window parameters for the spline
    double window_size_;
    int n_search_points_;
    bool goal_reached_;
    double x_goal_,y_goal_;

    double prev_x_, prev_y_;

    double minimal_s_;

    //reset simulation msg
    std_srvs::Empty reset_msg_;

private:

    ros::NodeHandle nh;

    tf::TransformListener tf_listener_;

    // Clock frequency
    double clock_frequency_;

    double r_discs_;
    Eigen::VectorXd x_discs_;

    // Timmer
    ros::Timer timer_;

    std::string target_frame_;

    // store pose value for visualize trajectory
    visualization_msgs::MarkerArray traj_marker_array_;

    // Distance between traget frame and tracking frame relative to base link
    std::vector<double> delay_state_;

    double slack_weight_;
    double repulsive_weight_;
    double lateral_weight_;
    double Wcontour_;
    double Wlag_;
    double Ka_,Kalpha_;
    double Kdelta_;
    double reference_velocity_;
    double speed_;
    double velocity_weight_;

    geometry_msgs::Pose reset_pose_;

    //TRajectory execution variables
    double next_point_dist, goal_dist, prev_point_dist;

    visualization_msgs::Marker ellips1;

    // Obstacles
    lmpcc_msgs::lmpcc_obstacle_array obstacles_;
    lmpcc_msgs::lmpcc_obstacle_array obstacles_init_;
    lmpcc_msgs::IntTrigger obstacle_trigger;
    lmpcc_msgs::lmpcc_obstacle dummy_obstacle_;

    // predictive configuration
    boost::shared_ptr<predictive_configuration> controller_config_;

    void getWayPointsCallBack(nav_msgs::Path waypoints);

    void Plan(geometry_msgs::PoseWithCovarianceStamped msg);

    /**
     * @brief spinNode: spin node means ROS is still running
     */
    void spinNode();

    void computeEgoDiscs();
    /**
     * @brief runNode: Continue updating this function depend on clock frequency
     * @param event: Used for computation of duration of first and last event
     */
    void runNode(const ros::TimerEvent& event);

    /**
     * @brief Main control loop, updates FORCES parameters, solves optimization, actuates the system
     * 
     */
    void ControlLoop();
    
    double getVelocityReference(int k);

    /**
     * @brief publishPredictedTrajectory: publish predicted trajectory
     */
    void publishPredictedTrajectory(void);
    
    void publishPredictedOutput(void);

    void publishPredictedCollisionSpace(void);

    void broadcastTF();

    void publishFeedback(int& it, double& time);
    
    void reset_solver();

    void setSolverToCurrentState();

    bool transformPose(const std::string& from, const std::string& to, geometry_msgs::Pose& pose);

    bool ResetCallBack(lmpcc::LMPCCReset::Request  &req, lmpcc::LMPCCReset::Response &res);

    void broadcastPathPose();
    
};

#endif
