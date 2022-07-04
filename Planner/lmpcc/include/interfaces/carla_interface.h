#ifndef CARLA_INTERFACE_H
#define CARLA_INTERFACE_H

#include "interfaces/interface.h"

#include <geometry_msgs/AccelWithCovarianceStamped.h>
#include <geometry_msgs/PoseStamped.h>
#include <sensor_msgs/JointState.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <sensor_msgs/Imu.h>
#include <carla_msgs/CarlaEgoVehicleInfo.h>
#include <tf/transform_listener.h>

#include <lmpcc/LMPCCReset.h>
#include <ackermann_msgs/AckermannDrive.h>
#include <carla_ackermann_control/EgoVehicleControlInfo.h>
//#include <carla_ackermann_msgs/EgoVehicleControlInfo.h>
#include <carla_msgs/CarlaStatus.h>
#include <derived_object_msgs/ObjectArray.h>
#include <tf2_ros/transform_broadcaster.h>
#include <lmpcc_tools/helpers.h>

class CarlaInterface : public Interface
{

public:
    CarlaInterface(ros::NodeHandle &nh, MPCC *controller, predictive_configuration *config, BaseModel *solver_interface_ptr);

private:
    BaseModel *solver_interface_ptr_;

    ros::Subscriber state_sub_, steering_sub_, acceleration_sub_;
    ros::Subscriber obstacle_sub_, waypoints_sub_,vehicle_info_sub_;
    ros::Subscriber reset_pose_sub_;
    ros::Subscriber carla_status_sub_; // For debugging purposes

    // For debugging
    ros::Subscriber ackermann_info_sub_;
    
    ros::Publisher command_pub_;
    ros::Publisher goal_pub_;
    ros::Publisher reset_carla_pub_;
    ros::Publisher obstacles_pub_ ;

    ackermann_msgs::AckermannDrive command_msg_;

    carla_msgs::CarlaEgoVehicleInfo ego_vehicle_info_;

    tf::TransformListener tf_listener_;
    tf2_ros::TransformBroadcaster state_pub_;

    bool goal_set_;
    Eigen::Vector2d goal_location_;

    void OrderObstacles(lmpcc_msgs::lmpcc_obstacle_array &ellipses);

    bool transformPose(const std::string &from, const std::string &to, geometry_msgs::Pose &pose);

public:
    virtual void Actuate() override;
    virtual void ActuateBrake(double deceleration) override;

    virtual void Reset(const geometry_msgs::Pose &position_after_reset) override;
    virtual void Reset();
    
    // Reset from rviz initial pose value
    void ResetCallback(const geometry_msgs::PoseWithCovarianceStamped& initial_pose);

    void StateCallBack(const nav_msgs::Odometry::ConstPtr &msg);
    void AccelerationCallback(const geometry_msgs::AccelWithCovarianceStamped &msg);
    // void SteeringAngleCallback(const carla_msgs::CarlaEgoVehicleControl& msg);

    void PublishGoal();

    //void AckermannCallback(const carla_ackermann_msgs::EgoVehicleControlInfo &msg);
    void AckermannCallback(const carla_ackermann_control::EgoVehicleControlInfo &msg);

    void WaypointsCallback(const nav_msgs::Path &msg);

    void ObstacleCallBack(const derived_object_msgs::ObjectArray &received_obstacles);

    void VehicleInfoCallback(const carla_msgs::CarlaEgoVehicleInfo &msg);

    void carlaStatusCallback(const carla_msgs::CarlaStatus &msg);
    
    void plotObstacles(void);
};

#endif
