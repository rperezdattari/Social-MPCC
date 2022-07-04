#include "scenario/dynamic_obstacle.h"

void DynamicObstacle::initialise(predictive_configuration *config)
{
    // Save the config pointer
    config_ = config;

    // Configured sizes
    N_ = config_->N_;
    B_ = config_->batch_count_;

    // Allocate space for scenarios
    sample_sizes_.resize(N_);
    batch_select_.resize(N_, 0);

    // Largest sample size
    largest_sample_size_ = Sampler::Get().LargestSampleSize();

    // Resize scenarios
    scenarios_.resize(N_);
    for (uint k = 0; k < N_; k++)
        scenarios_[k].resize(largest_sample_size_);

    // Resize distances
    distances_.resize(N_);
    for(uint k = 0; k < N_; k++)
        distances_[k].resize(largest_sample_size_);

    obstacle_msg_.trajectory.poses.resize(N_);
    obstacle_msg_.major_semiaxis.resize(N_);
    obstacle_msg_.minor_semiaxis.resize(N_);
    for(uint k = 0; k < N_; k++){
        obstacle_msg_.trajectory.poses[k].pose.position.x = 50; // Lowered...
        obstacle_msg_.trajectory.poses[k].pose.position.y = 50;
        obstacle_msg_.trajectory.poses[k].pose.orientation.z = 0;
        obstacle_msg_.major_semiaxis[k] = 0.1;
        obstacle_msg_.minor_semiaxis[k] = 0.1;
    }
}


void DynamicObstacle::ellipsoidToGaussian(const lmpcc_msgs::lmpcc_obstacle &obstacle_msg)
{
    // Save the obstacle msg
    obstacle_msg_ = obstacle_msg;

    // Initialize variables
    Eigen::Matrix<double, 2, 2> A, Sigma, R, SVD;
    double theta;

    // Easy reference for the path
    nav_msgs::Path path = obstacle_msg.trajectory;
    // batches indentical per obstacle
    static int batch_static = rand_.Int(B_);

    // For all stages
    for (uint k = 0; k < N_; k++)
    {
        batch_select_[k] = batch_static;

        // Retrieve a pointer to the samples with our random index from the sample library
        std::vector<Eigen::Vector2d>& batch_ref = Sampler::Get().BatchReference(batch_select_[k]);

        // Get the angle of the path
        theta = Helpers::quaternionToAngle(path.poses[k].pose);

        // Get the rotation matrix (CHECK FOR ACTUAL HEADINGS!)
        R = Helpers::rotationMatrixFromHeading(-theta);

        // Convert the semi axes back to gaussians
        SVD << std::pow(obstacle_msg.major_semiaxis[k]/3.0, 2), 0.0,
            0.0, std::pow(obstacle_msg.minor_semiaxis[k]/3.0, 2);

        // Compute Sigma and cholesky decomposition
        Sigma = R * SVD * R.transpose();
        A = Sigma.llt().matrixL();

        // Convert the current random number batch to a gaussian set of scenarios
        for (uint s = 0; s < batch_ref.size(); s++)
        {
            // Adapt the gaussian random number to this sigma and mu
            scenarios_[k][s] = A * batch_ref[s] + Eigen::Vector2d(path.poses[k].pose.position.x, path.poses[k].pose.position.y);
        }

        sample_sizes_[k] = batch_ref.size();
    }
}

// Project in a circle outwards to the radius of the furthest away point
void DynamicObstacle::projectOutwards(const int &k, Eigen::Vector2d &pose) const
{
    //https://math.stackexchange.com/questions/475436/2d-point-projection-on-an-ellipse
    // ELLIPSE
    Eigen::Vector2d mean(obstacle_msg_.trajectory.poses[k].pose.position.x,
                         obstacle_msg_.trajectory.poses[k].pose.position.y);

    double theta = Helpers::quaternionToAngle(obstacle_msg_.trajectory.poses[k].pose);

    // Matrix to rotate back
    Eigen::Matrix<double, 2, 2> R = Helpers::rotationMatrixFromHeading(theta);
    Eigen::Matrix<double, 2, 2> R_reverse = Helpers::rotationMatrixFromHeading(-theta);

    std::vector<int> &extreme_sample_indices = Sampler::Get().ExtremeSampleIndices(batch_select_[k]);

    // To deal with rotations, we first rotate the worst indices back to 0 angle
    Eigen::Vector2d scenario_a = R * (scenarios_[k][extreme_sample_indices[0]] - mean);
    Eigen::Vector2d scenario_b = R * (scenarios_[k][extreme_sample_indices[1]] - mean);

    // Then check the distance to the mean in the large direction
    double b = std::abs(scenario_a(1));
    double a = std::abs(scenario_b(0));

    // Rotate the position of the vehicle as well
    Eigen::Vector2d pose_relative = (pose - mean);
    
    // Check if the pose is inside the ellipse
    double eval_radius = std::pow(pose_relative(0), 2) / std::pow(a, 2) +
                         std::pow(pose_relative(1), 2) / std::pow(b, 2);

    if (eval_radius < 1) /* May also need the radii in some way (now just simple 2).. */
    {
        ROS_WARN("Projecting state!");
        pose_relative = R*(pose - mean);

        // Project to the ellipse boundary
        double theta_ellipse = std::atan2(pose_relative(1), pose_relative(0));
        double k_ellipse = (a * b) /
                   (std::sqrt(b * b * std::pow(std::cos(theta_ellipse), 2) + a * a * std::pow(std::sin(theta_ellipse), 2)));


        k_ellipse *= 1.01 + config_->r_vehicle_ + config_->r_VRU_;

        pose(0) = k_ellipse * std::cos(theta_ellipse);
        pose(1) = k_ellipse * std::sin(theta_ellipse);

        // Rotate back
        pose = R_reverse * pose + mean;
    }

}

void DynamicObstacle::computeDistancesTo(const int &k, const Eigen::Vector2d &pose)
{
    Eigen::Vector2d diff;
    // Compute the difference vector and its length (This should apply to all circles of the EV...)
    for (int s = 0; s < sample_sizes_[k]; s++)
    {
        diff = scenarios_[k][s] - pose;

        // I could map, vector product, then map ( for now the simple way )
        distances_[k][s] = diff.transpose() * diff;
    }
}

// Return the distance to the scenario or a large number if the index does not exist
double DynamicObstacle::getDistance(int k, int index)
{
    if (index < sample_sizes_[k])
        return distances_[k][index];
    else
        return 99999999.0;
};