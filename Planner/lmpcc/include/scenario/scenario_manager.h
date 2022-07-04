#ifndef SCENARIO_MANAGER_H
#define SCENARIO_MANAGER_H

#include <ros/ros.h>
#include "lmpcc_configuration.h"
#include "lmpcc_tools/ros_visuals.h"
#include "lmpcc_tools/helpers.h"

#include <lmpcc/base_model.h>

#include "scenario/polygon_constructor.h"
#include "scenario/dynamic_obstacle.h"
#include "scenario/static_obstacles.h"
#include "scenario/safety_certifier.h"

#include <tf/tf.h>

#include <lmpcc_msgs/lmpcc_obstacle.h>
#include <lmpcc_msgs/lmpcc_obstacle_array.h>

// to fix
//#include <FORCESNLPsolver/include/FORCESNLPsolver.h>

#include <vector>
#include <memory>
#include <thread>

#include <Eigen/Eigen>
#include <Eigen/Cholesky>

#define BIG_NUMBER 9999999999

// Struct to keep track of the obstacle related to the closest scenarios
struct DefinedScenario
{

    int obstacle_index;
    int scenario_index;
    double distance;

    // Init for search of the smallest
    void init()
    {
        obstacle_index = -1;
        scenario_index = -1;
        distance = BIG_NUMBER;
    }
};

class ScenarioManager{

public:
    ScenarioManager(ros::NodeHandle &nh, predictive_configuration *config, bool enable_visualization);

private:

    // Received from constructor
    predictive_configuration *config_;
    bool enable_visualization_;

    // The visualisation class
    std::unique_ptr<ROSMarkerPublisher> ros_markers_;

    // Instances for the obstacles
    std::vector<DynamicObstacle> obstacles_;
    StaticObstacles static_obstacles_;

    // Obstacle msgs
    lmpcc_msgs::lmpcc_obstacle_array obstacles_msg_;

    // Threads for multithreading per stage
    std::vector<std::thread> scenario_threads_;

    // Parameters that will be set via config
    u_int S;    // sampling count
    u_int N;    // Prediction horizon
    u_int B;    // Batches

    int R_; // Scenario removal (maximum now)
    int l_;

    // r_obstacle + r_vehicle
    double combined_radius_;

    // Active obstacles
    std::vector<std::vector<int>> active_obstacle_indices_;
    std::vector<int> active_obstacle_count_;

    // Closest R+l scenarios per stage, per vru
    std::vector<std::vector<std::vector<DefinedScenario>>> closest_scenarios_;
    std::vector<std::vector<std::vector<DefinedScenario>>> closest_static_;

    // Vehicle poses when project outside of obstacles
    std::vector<Eigen::Vector2d> projected_poses_;

    // Considered hyperplanes
    std::vector<std::vector<LinearConstraint2D>> possible_constraints_;

    // Classes for computing the minimal polygon
    std::vector<PolygonConstructor> polygon_constructors_;

    // Areas of the regions
    std::vector<double> areas_;

    // Final constraints
    std::vector<LinearConstraint2D> scenario_constraints_;

    // Convert the scenarios to constraints (top level)
    void scenariosToConstraints(int k, const Eigen::Vector2d &pose, double orientation);

    // Helper functions
    void removeScenariosBySorting(std::vector<DefinedScenario> &scenarios_to_sort);
    void sortedInsert(const int &k, const double &new_distance, const int &obstacle_index, const int &scenario_index);
    void sortedInsertStatic(const int &k, const double &new_distance, const int &s);
    void hyperplaneFromScenario(const Eigen::Vector2d &pose, const Eigen::Vector2d &scenario_position, LinearConstraint2D &constraint, int k, const DefinedScenario &scenario, int write_index);

    // Retrieves scenario from obstacle
    Eigen::Vector3d getScenarioLocation(const int &k, const int &obstacle_index, const int &scenario_index);

    // Visualisation methods
    void visualiseEllipsoidConstraints();
    void visualiseConstraints();
    void visualiseScenarios(const std::vector<int> &indices_to_draw);
    void visualiseRemovedScenarios(const std::vector<int> &indices_to_draw);
    void visualiseSelectedScenarios(const std::vector<int> &indices_to_draw);
    void visualisePolygons();
    void visualiseProjectedPosition();

public:
    // Update function
    /* Params are used as EV predictions */
    void update(const std::vector<Eigen::Vector2d> &poses, const std::vector<double>& orientations);

    // Some function that inserts chance constraint parameters
    void insertConstraints(BaseModel *solver_interface, int k, int start_index, int &end_index);

    // Callback for the predictor
    void predictionCallback(const lmpcc_msgs::lmpcc_obstacle_array &predicted_obstacles);
    
    // Callback for the static map
    void mapCallback(const nav_msgs::OccupancyGrid::ConstPtr &occupancy_grid, const Eigen::Vector2d &pose, tf::TransformListener& tf_listener);

    // Call to publish all scenario visuals
    void publishVisuals();

    double getArea(int k){ return areas_[k]; };
};

#endif