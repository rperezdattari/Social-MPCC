#ifndef STATIC_OBSTACLES_H
#define STATIC_OBSTACLES_H

#include <Eigen/Eigen>
#include <ros/ros.h>
#include <iostream>
#include <fstream>
#include <ros/package.h>
#include <string>

#include "scenario/sampler.h"
#include <tf/tf.h>

#include "lmpcc_tools/helpers.h"
#include "lmpcc_msgs/lmpcc_obstacle.h"
#include "lmpcc/lmpcc_configuration.h"
#include <nav_msgs/OccupancyGrid.h>

// Class for processing scenarios of a single obstacle
class StaticObstacles
{

public:
    StaticObstacles(){};

    void initialise(predictive_configuration *config);

private:
    // Configuration
    predictive_configuration *config_;

    nav_msgs::OccupancyGrid::ConstPtr occupancy_grid_;

    // Parameters that will be set via config
    uint N_; // Prediction horizon
    uint B_; // Batches

    int largest_sample_size_;

    // Saved data of the used batches for all stages
    int scenario_count_;

    // Scenario Storage (N stages with S samples of 2D gaussian scenarios)
    std::vector<Eigen::Vector2d> scenarios_;

    // Distances from the vehicle to scenarios
    std::vector<std::vector<double>> distances_;

public:
    // Convert ellipsoidal detections to gaussian scenarios using random numbers (sets scenarios_)
    void mapCallback(const nav_msgs::OccupancyGrid::ConstPtr &occupancy_grid, const Eigen::Vector2d &pose, tf::TransformListener &tf_listener);

    // Compute distances from the vehicle to all obstacles
    void computeDistancesTo(const int &k, const Eigen::Vector2d &pose);

    void occupancyToScenarios(const nav_msgs::OccupancyGrid::ConstPtr &occupancy_grid,
                                           const Eigen::Vector2d &pose, tf::TransformListener &tf_listener);

    // Get the distance at stage k, with index index
    double getDistance(int k, int index);

    // Getters
    Eigen::Vector2d &getScenarioRef(const int &s) { return scenarios_[s]; };
    int getScenarioCount(){return scenario_count_;};
    int getLargestSampleCount() { return largest_sample_size_; };
};

#endif