#ifndef SCENARIO_TYPES_H
#define SCENARIO_TYPES_H

#include <Eigen/Eigen>
#include <ros/ros.h>
#include <iostream>
#include <fstream>
#include <ros/package.h>
#include <string>

#include "scenario/sampler.h"

#include "lmpcc_tools/helpers.h"
#include "lmpcc_msgs/lmpcc_obstacle.h"
#include "lmpcc/lmpcc_configuration.h"

// Class for processing scenarios of a single obstacle
class DynamicObstacle
{

public:
    DynamicObstacle(){};

    void initialise(predictive_configuration *config);

private:

    // Configuration
    predictive_configuration *config_;
    Helpers::RandomGenerator rand_;

    // Obstacle data received
    lmpcc_msgs::lmpcc_obstacle obstacle_msg_;

    // Parameters that will be set via config
    uint N_; // Prediction horizon
    uint B_; // Batches

    int largest_sample_size_;

    // Saved data of the used batches for all stages
    std::vector<int> sample_sizes_;
    std::vector<int> batch_select_;

    // Scenario Storage (N stages with S samples of 2D gaussian scenarios)
    std::vector<std::vector<Eigen::Vector2d>> scenarios_;

    // Distances from the vehicle to scenarios
    std::vector<std::vector<double>> distances_;

public:

    // Convert ellipsoidal detections to gaussian scenarios using random numbers (sets scenarios_)
    void ellipsoidToGaussian(const lmpcc_msgs::lmpcc_obstacle &obstacle_msg);

    // Project the vehicle position outside of an obstacle if inside
    void projectOutwards(const int &k, Eigen::Vector2d &pose) const;

    // Compute distances from the vehicle to all obstacles
    void computeDistancesTo(const int &k, const Eigen::Vector2d &pose);

    // Get the distance at stage k, with index index
    double getDistance(int k, int index);

    // Getters
    Eigen::Vector2d &getScenarioRef(const int &k, const int &index) { return scenarios_[k][index]; }; 
    lmpcc_msgs::lmpcc_obstacle& getPrediction() {return obstacle_msg_;};
    int getPrunedSampleCount(int k) { return sample_sizes_[k]; };
    int getLargestSampleCount() { return largest_sample_size_; };
};


#endif