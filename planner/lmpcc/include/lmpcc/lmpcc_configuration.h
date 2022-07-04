#ifndef PREDICTIVE_CONFIGURATION_H
#define PREDICTIVE_CONFIGURATION_H

// ros includes
#include<ros/ros.h>

//c++ includes
#include<iostream>
#include<string>
#include<vector>
#include <algorithm>
#include <iomanip>	//print false or true
#include <math.h>

class predictive_configuration
{
    /**
     *  @brief All neccessary configuration parameter of predictive control repository
     *         Read data from parameter server
     *         Updated old data with new data
     *  Note:  All data member name used like xyz_ and all parameter name is normal like xyz.
     */

public:

    /** function member of class **/

    // constructor and distructor
    /**
     * @brief predictive_configuration: defualt constructor of this class
     */
    predictive_configuration();

    /**
      * @brief ~predictive_configuration: defualt distructor of this class
      */
    ~predictive_configuration();

    /**
     * @brief intialize:  check parameter on paramter server and read from there
     * @param node_handle_name: node handler initialize from name, as parameter set inside that name
     * @return true all parameter initialize successfully else false
     */
    bool initialize();  //const std::string& node_handle_name

    /**
     * @brief updateConfiguration: update configuration parameter with new parameter
     * @param new_config: changed configuration parameter
     * @return true all parameter update successfully else false
     */
    bool updateConfiguration(const predictive_configuration& new_config);

    /** data member of class **/
    // DEBUG
    bool activate_debug_output_;
    bool activate_controller_node_output_;
    bool initialize_success_;
    bool sync_mode_;
    bool gazebo_simulation_,simulation_mode_;
    bool auto_enable_plan_;

    /** inputs and output topic definition **/
    std::string cmd_, cmd_sim_;
    std::string robot_state_topic_, reset_topic_;

    // use for finding kinematic chain and urdf model
    std::string robot_;
    std::string robot_base_link_;
    std::string global_path_frame_;  //  End effector of arm
    std::string target_frame_;
    std::string sub_ellipse_topic_;
    std::string obs_state_topic_;
    std::string steering_state_topic_;
    std::string acceleration_state_topic_;
    std::string waypoint_topic_;
    std::string vref_topic_;
    std::string controller_feedback_;
    std::string free_space_topic_;
    std::string reset_carla_topic_;
    std::string navigation_goal_topic_;

    // Visualization Topics
    std::string reference_path_topic_;
    std::string reference_arrows_topic_;
    std::string planned_space_topic_;
    std::string planned_trajectory_topic_;

    // limiting parameter, use to enforce joint to be in limit
    std::vector<std::string> collision_check_obstacles_;

    // Initialize vectors for reference path points
    std::vector<double> ref_x_;
    std::vector<double> ref_y_;
    std::vector<double> ref_theta_;
    double road_width_right_,road_width_left_;

    // Steps of delay on the input signal
    int input_delay_;

    // Variables SET IN LMPCC CONTROLLER (just an efficient way to set important variables)
    size_t N_;
    size_t NVAR_;

    // Static obstacles with JPS
    int occ_width_, occ_height_;
    double occ_res_;

    bool static_obstacles_enabled_;
    std::string occupancy_grid_topic_;

    // Scenario related!
    double eps_t_;
    double r_VRU_;

    bool multithread_scenarios_;
    int sample_count_;
    int batch_count_;
    int removal_count_;
    std::vector<int> indices_to_draw_;
    std::vector<int> discs_to_draw_;
    bool draw_all_scenarios_;
    bool draw_selected_scenarios_;
    bool draw_removed_scenarios_;
    bool draw_ellipsoids_;
    bool draw_constraints_;
    int seed_;
    double r_vehicle_;
    bool truncated_;
    double truncated_radius_;

    bool build_database_;
    int scenario_database_size_;

    // Polygons
    int polygon_checked_constraints_;
    double polygon_range_;
    int inner_approximation_;

    bool iterative_approach_enabled_;
    int max_scenario_iterations_;
    /////////////

    // Numbers of points for spline and clothoid fitting
    int n_points_clothoid_;
    int n_points_spline_;
    double min_wapoints_distance_;
    double epsilon_;
    double slack_weight_;
    double repulsive_weight_;
    double min_velocity_;
    double reference_velocity_;
    double ini_vel_x_;
    // predictive control
    double clock_frequency_;  //hz clock Frequency

    int max_num_iteration_;

    int max_obstacles_;
    int n_discs_;
    double ego_l_;
    double ego_w_;

private:
  /* Retrieve paramater, if it doesn't exist return false */
  template <class T>
  bool retrieveParameter(const ros::NodeHandle &nh, const std::string &name, T &value)
  {

    if (!nh.getParam(name, value))
    {
      ROS_WARN_STREAM(" Parameter " << name << " not set on node " << ros::this_node::getName().c_str());
      return false;
    }
    else
    {
      return true;
    }
  }

  /* Retrieve parameter, if it doesn't exist use the default */
  template <class T>
  void retrieveParameter(const ros::NodeHandle &nh, const std::string &name, T &value, const T &default_value)
  {

    if (!retrieveParameter(nh, name, value))
    {
      ROS_WARN_STREAM(" Setting " << name << " to default value: " << default_value);
      value = default_value;
    }
  }
};

#endif
