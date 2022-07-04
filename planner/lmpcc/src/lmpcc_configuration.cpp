
#include <lmpcc/lmpcc_configuration.h>

predictive_configuration::predictive_configuration()
{

  initialize_success_ = false;
}

predictive_configuration::~predictive_configuration()
{
}

// read predicitve configuration paramter from paramter server
bool predictive_configuration::initialize() //const std::string& node_handle_name
{
  ros::NodeHandle nh_config; //("predictive_config");
  ros::NodeHandle nh;

  if (!nh.getParam("simulation_mode", simulation_mode_))
  {
    ROS_WARN(" Parameter 'simulation_mode' not set on %s node ", ros::this_node::getName().c_str());
    return false;
  }

  if (!nh.getParam("gazebo_simulation", gazebo_simulation_))
  {
    ROS_WARN(" Parameter 'gazebo_simulation' not set on %s node ", ros::this_node::getName().c_str());
    return false;
  }

  if (!nh.getParam ("robot", robot_) )
  {
    ROS_WARN(" Parameter 'robot' not set on %s node " , ros::this_node::getName().c_str());
    return false;
  }

  if (!nh.getParam("sync_mode", sync_mode_))
  {
    ROS_WARN(" Parameter 'sync_mode' not set on %s node ", ros::this_node::getName().c_str());
    return false;
  }  
  
  if (!nh.getParam("auto_enable_plan", auto_enable_plan_))
  {
    ROS_WARN(" Parameter 'auto_enable_plan' not set on %s node ", ros::this_node::getName().c_str());
    return false;
  }

  // read paramter from parameter server if not set than terminate code, as this parameter is essential parameter
  if (!nh.getParam("robot_base_link", robot_base_link_))
  {
    ROS_WARN(" Parameter 'robot_base_link' not set on %s node ", ros::this_node::getName().c_str());
    return false;
  }

  if (!nh.getParam("global_path/frame", global_path_frame_))
  {
    ROS_WARN(" Parameter 'global_path/frame' not set on %s node ", ros::this_node::getName().c_str());
    return false;
  }

  if (!nh.getParam("target_frame", target_frame_))
  {
    ROS_WARN(" Parameter 'target_frame' not set on %s node ", ros::this_node::getName().c_str());
    return false;
  }

  if (!nh_config.getParam("global_path/x", ref_x_))
  {
    ROS_WARN(" Parameter '/global_path/x not set on %s node", ros::this_node::getName().c_str());
    return false;
  }
  if (!nh_config.getParam("global_path/epsilon", epsilon_))
  {
    ROS_WARN(" Parameter '/global_path/epsilon not set on %s node", ros::this_node::getName().c_str());
    return false;
  }
  //
  if (!nh_config.getParam("global_path/y", ref_y_))
  {
    ROS_WARN(" Parameter '/global_path/y not set on %s node", ros::this_node::getName().c_str());
    return false;
  }

  if (!nh_config.getParam("global_path/theta", ref_theta_))
  {
    ROS_WARN(" Parameter '/global_path/theta not set on %s node", ros::this_node::getName().c_str());
    return false;
  }

  if (!nh_config.getParam("global_path/n_points_clothoid", n_points_clothoid_))
  {
    ROS_WARN(" Parameter '/global_path/n_points_clothoid not set on %s node", ros::this_node::getName().c_str());
    return false;
  }

  if (!nh_config.getParam("global_path/n_points_spline", n_points_spline_))
  {
    ROS_WARN(" Parameter '/global_path/n_points_spline not set on %s node", ros::this_node::getName().c_str());
    return false;
  }

  if (!nh_config.getParam("global_path/min_wapoints_distance", min_wapoints_distance_))
  {
    ROS_WARN(" Parameter '/global_path/min_wapoints_distance not set on %s node", ros::this_node::getName().c_str());
    return false;
  }

  /////////////// TOPICS /////////////////////////

  // Publish
  retrieveParameter(nh, "publish/control_command", cmd_);
  retrieveParameter(nh, "publish/feedback", controller_feedback_);
  retrieveParameter(nh, "publish/reset", reset_topic_);
  retrieveParameter(nh, "publish/reset_carla", reset_carla_topic_);
  retrieveParameter(nh, "publish/navigation_goal", navigation_goal_topic_);

  // Subscribe
  retrieveParameter(nh, "subscribe/state", robot_state_topic_);
  retrieveParameter(nh, "subscribe/obstacles", obs_state_topic_);
  retrieveParameter(nh, "subscribe/waypoints", waypoint_topic_);
  retrieveParameter(nh, "subscribe/steering_angle", steering_state_topic_);
  retrieveParameter(nh, "subscribe/acceleration", acceleration_state_topic_);
  retrieveParameter(nh, "subscribe/velocity_reference", vref_topic_);
  retrieveParameter(nh, "subscribe/occupancy_grid", occupancy_grid_topic_);

  // Visualization
  retrieveParameter(nh, "visualization/reference_path", reference_path_topic_);
  retrieveParameter(nh, "visualization/reference_arrows", reference_arrows_topic_);
  retrieveParameter(nh, "visualization/planned_collision_space", planned_space_topic_);
  retrieveParameter(nh, "visualization/planned_trajectory", planned_trajectory_topic_);
  retrieveParameter(nh, "visualization/free_space", free_space_topic_);


  // read parameter from parameter server if not set than terminate code, as this parameter is essential parameter
  if (!nh.getParam("road_width_right", road_width_right_))
  {
    ROS_WARN(" Parameter 'road_width_right' not set on %s node ", ros::this_node::getName().c_str());
    return false;
  }
  if (!nh.getParam("road_width_left", road_width_left_))
  {
    ROS_WARN(" Parameter 'road_width_left' not set on %s node ", ros::this_node::getName().c_str());
    return false;
  }

  if (!nh.getParam("obstacles/max_obstacles", max_obstacles_))
  {
    ROS_WARN(" Parameter 'obstacles/max_obstacles' not set on %s node ", ros::this_node::getName().c_str());
    return false;
  }

  if (!nh.getParam("obstacles/n_discs", n_discs_))
  {
    ROS_WARN(" Parameter 'n_discs' not set on %s node ", ros::this_node::getName().c_str());
    return false;
  }

  if (!nh.getParam("obstacles/ego_l", ego_l_))
  {
    ROS_WARN(" Parameter 'ego_l' not set on %s node ", ros::this_node::getName().c_str());
    return false;
  }

  if (!nh.getParam("obstacles/ego_w", ego_w_))
  {
    ROS_WARN(" Parameter 'ego_w' not set on %s node ", ros::this_node::getName().c_str());
    return false;
  }

  if (!nh.getParam("debug_info_lmpcc", activate_debug_output_))
  {
    ROS_WARN(" Parameter 'debug_info_lmpcc' not set on %s node ", ros::this_node::getName().c_str());
    return false;
  }
  
  if (!nh.getParam("delays/input_delay", input_delay_))
  {
    ROS_WARN(" Parameter 'delays/input_delay' not set on %s node ", ros::this_node::getName().c_str());
    return false;
  }

  if (!nh.getParam("scenarios/enable_static_obstacles", static_obstacles_enabled_))
  {
    ROS_WARN(" Parameter 'scenarios/enable_static_obstacles' not set on %s node ", ros::this_node::getName().c_str());
    return false;
  }

  // Scenario from here
  if (!nh.getParam("scenarios/database/truncated", truncated_))
  {
    ROS_WARN(" Parameter 'scenarios/database/truncated' not set on %s node ", ros::this_node::getName().c_str());
    return false;
  }

  if (!nh.getParam("scenarios/database/truncated_radius", truncated_radius_))
  {
    ROS_WARN(" Parameter 'scenarios/database/truncated_radius' not set on %s node ", ros::this_node::getName().c_str());
    return false;
  }

  /* Scenario related parameters */
  if (!nh.getParam("scenarios/multithread", multithread_scenarios_))
  {
    ROS_WARN(" Parameter 'scenarios/multithread' not set on %s node ", ros::this_node::getName().c_str());
    return false;
  }
  if (!nh.getParam("scenarios/sample_count", sample_count_))
  {
    ROS_WARN(" Parameter 'scenarios/sample_count' not set on %s node ", ros::this_node::getName().c_str());
    return false;
  }
  if (!nh.getParam("scenarios/batch_count", batch_count_))
  {
    ROS_WARN(" Parameter 'scenarios/batch_count' not set on %s node ", ros::this_node::getName().c_str());
    return false;
  }
  if (!nh.getParam("scenarios/removal_count", removal_count_))
  {
    ROS_WARN(" Parameter 'scenarios/removal_count' not set on %s node ", ros::this_node::getName().c_str());
    return false;
  }
  if (!nh.getParam("scenarios/visualisation/indices_to_draw", indices_to_draw_))
  {
    ROS_WARN(" Parameter 'scenarios/visualisation/indices_to_draw' not set on %s node ", ros::this_node::getName().c_str());
    return false;
  }
  if (!nh.getParam("scenarios/visualisation/discs_to_draw", discs_to_draw_))
  {
    ROS_WARN(" Parameter 'scenarios/visualisation/discs_to_draw' not set on %s node ", ros::this_node::getName().c_str());
    return false;
  }
  if (!nh.getParam("scenarios/visualisation/all_scenarios", draw_all_scenarios_))
  {
    ROS_WARN(" Parameter 'scenarios/visualisation/all_scenarios' not set on %s node ", ros::this_node::getName().c_str());
    return false;
  }
  if (!nh.getParam("scenarios/visualisation/selected_scenarios", draw_selected_scenarios_))
  {
    ROS_WARN(" Parameter 'scenarios/visualisation/selected_scenarios' not set on %s node ", ros::this_node::getName().c_str());
    return false;
  }
  if (!nh.getParam("scenarios/visualisation/removed_scenarios", draw_removed_scenarios_))
  {
    ROS_WARN(" Parameter 'scenarios/visualisation/removed_scenarios' not set on %s node ", ros::this_node::getName().c_str());
    return false;
  }
  if (!nh.getParam("scenarios/visualisation/ellipsoids", draw_ellipsoids_))
  {
    ROS_WARN(" Parameter 'scenarios/visualisation/ellipsoids' not set on %s node ", ros::this_node::getName().c_str());
    return false;
  }
  if (!nh.getParam("scenarios/visualisation/constraints", draw_constraints_))
  {
    ROS_WARN(" Parameter 'scenarios/visualisation/constraints' not set on %s node ", ros::this_node::getName().c_str());
    return false;
  }
  if (!nh.getParam("scenarios/database/size", scenario_database_size_))
  {
    ROS_WARN(" Parameter 'scenarios/database/size' not set on %s node ", ros::this_node::getName().c_str());
    return false;
  }
  if (!nh.getParam("scenarios/database/build_database", build_database_))
  {
    ROS_WARN(" Parameter 'scenarios/database/build_database' not set on %s node ", ros::this_node::getName().c_str());
    return false;
  }

  if (!nh.getParam("vru/radius", r_VRU_))
  {
    ROS_WARN(" Parameter 'vru/radius' not set on %s node ", ros::this_node::getName().c_str());
    return false;
  }

  if (!nh.getParam("vru/eps_t", eps_t_))
  {
    ROS_WARN(" Parameter 'vru/eps_t' not set on %s node ", ros::this_node::getName().c_str());
    return false;
  }

  if (!nh.getParam("scenarios/iterative_approach/enabled", iterative_approach_enabled_))
  {
    ROS_WARN(" Parameter 'scenarios/iterative_approach/enabled' not set on %s node ", ros::this_node::getName().c_str());
    return false;
  }

  if (!nh.getParam("scenarios/iterative_approach/max_iterations", max_scenario_iterations_))
  {
    ROS_WARN(" Parameter 'scenarios/iterative_approach/max_iterations' not set on %s node ", ros::this_node::getName().c_str());
    return false;
  }
  if (!nh.getParam("scenarios/polygon/range", polygon_range_))
  {
    ROS_WARN(" Parameter 'scenarios/polygon/range' not set on %s node ", ros::this_node::getName().c_str());
    return false;
  }
  if (!nh.getParam("scenarios/polygon/checked_constraints", polygon_checked_constraints_))
  {
    ROS_WARN(" Parameter 'scenarios/polygon/checked_constraints' not set on %s node ", ros::this_node::getName().c_str());
    return false;
  }
  if (!nh.getParam("scenarios/polygon/inner_approximation", inner_approximation_))
  {
    ROS_WARN(" Parameter 'scenarios/polygon/inner_approximation' not set on %s node ", ros::this_node::getName().c_str());
    return false;
  }

  ROS_INFO("acado configuration parameter");
  // check requested parameter availble on parameter server if not than set default value
  nh.param("clock_frequency", clock_frequency_, double(20.0)); // 25 hz
  nh.param("min_velocity", min_velocity_, double(1.5));            // 0.5 by default

  //nh.param("activate_debug_output", activate_debug_output_, bool(false));                     // debug
  nh.param("activate_controller_node_output", activate_controller_node_output_, bool(false)); // debug
  nh.param("costmap/costmap/width", occ_width_, int(40));
  nh.param("costmap/costmap/height", occ_height_, int(40));
  nh.param("costmap/costmap/resolution", occ_res_, double(0.5));                                         
  nh.param("scenarios/seed", seed_, -1);                                         


  initialize_success_ = true;

  ROS_WARN(" PREDICTIVE PARAMETER INITIALIZED!!");
  return true;
}
