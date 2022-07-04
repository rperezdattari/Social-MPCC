#include "scenario/static_obstacles.h"

void StaticObstacles::initialise(predictive_configuration *config)
{
    // Save the config pointer
    config_ = config;

    // Configured sizes
    N_ = config_->N_;
    B_ = config_->batch_count_;

    // Allocate space for scenarios
    scenario_count_ = 0;

    // Largest sample size
    double res = config_->occ_res_;
    int width = (int)(config_->occ_width_ / res);
    int height = (int)(config_->occ_height_ / res);

    // Allocate space for the maximal width/height of the occupancy grid
    if (config_->static_obstacles_enabled_)
        largest_sample_size_ = width * height;
    else
        largest_sample_size_ = 0;

    // Resize scenarios
    scenarios_.resize(largest_sample_size_);

    // Resize distances (note distance is per stage!)
    distances_.resize(N_);
    for (uint k = 0; k < N_; k++)
        distances_[k].resize(largest_sample_size_);

    if (config_->static_obstacles_enabled_)
        ROS_WARN("Static obstacles intialised");
}
//void MPCC::mapCallback(const sensor_msgs::PointCloud2::ConstPtr& pcl2ptr_map_ros)
void StaticObstacles::mapCallback(const nav_msgs::OccupancyGrid::ConstPtr &occupancy_grid, const Eigen::Vector2d &pose, tf::TransformListener &tf_listener)
{
    if (!config_->static_obstacles_enabled_)    
        return;

    // Convert the occupied cells to scenarios
    occupancyToScenarios(occupancy_grid, pose, tf_listener);

    if (scenario_count_ == 0) 
    {
        ROS_WARN("Occupancy Grid received is empty, maybe map is too small?");
    }
}

void StaticObstacles::occupancyToScenarios(const nav_msgs::OccupancyGrid::ConstPtr &occupancy_grid,
                                           const Eigen::Vector2d &pose, tf::TransformListener &tf_listener)
{
    scenario_count_ = 0;

    // Retrieve parameters of the map
    double res = config_->occ_res_;
    double width = double(occupancy_grid->info.width);
    double height = double(occupancy_grid->info.height);

    for (size_t x = 0; x < occupancy_grid->info.width; x++)
    {
        for (size_t y = 0; y < occupancy_grid->info.height; y++)
        {

            // If this is free_space, dont add it to the occupancy map
            if (occupancy_grid->data[x + y * occupancy_grid->info.width] == 0)
                continue;

            Eigen::Vector2d rotated_data = pose + Eigen::Vector2d(double(x) - width / 2.0, double(y) - height / 2.0) * res;
            rotated_data += Eigen::Vector2d(0.5, 0.5) * res;

            geometry_msgs::Pose obstacle_pose;
            obstacle_pose.position.x = rotated_data(0);
            obstacle_pose.position.y = rotated_data(1);
            obstacle_pose.orientation = occupancy_grid->info.origin.orientation;

            Helpers::transformPose(tf_listener, "/" + occupancy_grid->header.frame_id, "/map", obstacle_pose);

            scenarios_[scenario_count_](0) = obstacle_pose.position.x;
            scenarios_[scenario_count_](1) = obstacle_pose.position.y;
            scenario_count_++;

        }
    }
}

void StaticObstacles::computeDistancesTo(const int &k, const Eigen::Vector2d &pose)
{
    Eigen::Vector2d diff;

    // Compute the difference vector and its length (This should apply to all circles of the EV...)
    for (int s = 0; s < scenario_count_; s++)
    {
        diff = scenarios_[s] - pose;

        // I could map, vector product, then map ( for now the simple way )
        distances_[k][s] = diff.transpose() * diff;
    }
}

// Return the distance to the scenario or a large number if the index does not exist
double StaticObstacles::getDistance(int k, int index)
{
    if (index < scenario_count_)
        return distances_[k][index];
    else
        return 99999999.0;
};


