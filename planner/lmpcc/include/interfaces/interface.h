

#ifndef INTERFACE_H
#define INTERFACE_H

#include <ros/ros.h>

#include <lmpcc_msgs/lmpcc_obstacle.h>
#include <lmpcc_msgs/lmpcc_obstacle_array.h>

#include <lmpcc/lmpcc_configuration.h>
#include <lmpcc/PredictiveControllerConfig.h>


class MPCC;


class Interface{

public:
    Interface(ros::NodeHandle &nh, MPCC* controller, predictive_configuration *config)
    {
        // Save pointers at init
        config_ = config;
        
        controller_ = controller;

    }

protected:

    // Configuration
    predictive_configuration * config_;

    // Necessary for callbacks
    MPCC * controller_;

    // External callback functions
    void (MPCC::*external_state_callback_)();
    void (MPCC::*external_obstacle_callback_)();
    void (MPCC::*external_waypoints_callback_)();
    void (MPCC::*external_reset_callback_)();

public:

    // The obstacles
    lmpcc_msgs::lmpcc_obstacle_array obstacles_;

    std::vector<double> x_, y_, theta_;

    virtual void Actuate() = 0;
    virtual void ActuateBrake(double deceleration) = 0;

    // Reset specifying a position to reset to (requires implementation in child class)
    virtual void Reset(const geometry_msgs::Pose& position_after_reset){Reset();};
    virtual void Reset() = 0;

    // Retrieve an external obstacle callback function pointer
    template<class T>
    void SetExternalObstacleCallback(void (T::*callback_ptr)())
    {
        external_obstacle_callback_ = callback_ptr;
    };

    template <class T>
    void SetExternalStateCallback(void (T::*callback_ptr)())
    {
        external_state_callback_ = callback_ptr;
    };    

    template <class T>
    void SetExternalWaypointsCallback(void (T::*callback_ptr)())
    {
        external_waypoints_callback_ = callback_ptr;
    }; 
    
    template <class T>
    void SetExternalResetCallback(void (T::*callback_ptr)())
    {
        external_reset_callback_ = callback_ptr;
    };
};

// Necessary to prevent loading the Carla interface before
#include <lmpcc/lmpcc_controller.h> // Before, this was below the class

#endif