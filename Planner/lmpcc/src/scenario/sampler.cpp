#include "scenario/sampler.h"


Sampler::Sampler(){}

void Sampler::Init(predictive_configuration *config){

    config_ = config;

    S_ = config_->sample_count_;
    R_ = config_->removal_count_;
    B_ = config_->batch_count_;

    initScenarios();

    // Find the largest sample size
    for (int b = 0; b < B_; b++)
    {
        if ((int)random_numbers_[b].size() > largest_sample_size_)
        {
            largest_sample_size_ = random_numbers_[b].size();
        }
    }

    ROS_WARN("SAMPLE LIBRARY INITIALIZED");
}

std::vector<Eigen::Vector2d> &Sampler::BatchReference(int batch_index)
{
    return random_numbers_[batch_index];
}

std::vector<int> &Sampler::ExtremeSampleIndices(int batch_index)
{
    return extreme_sample_indices_[batch_index];
}

int Sampler::LargestSampleSize() const 
{
    return largest_sample_size_;
}

// Draw random numbers that will function as scenarios
void Sampler::initScenarios()
{
    // Allocate space for the random numbers
    random_numbers_.resize(B_);
    extreme_sample_indices_.resize(B_);

    // Check if create database is true
    if (config_->build_database_)
    {
        // If it is, we will sample new scenarios
        for (int i = 0; i < config_->scenario_database_size_; i++)
        {
            sampleScenarios(i);
        }
    }

    // For all batches that we need
    for (int b = 0; b < B_; b++)
    {

        // Allocate space for this batch
        random_numbers_[b].resize(S_);
        extreme_sample_indices_[b].resize(2);

        // Select a batch at random
        int batch_select = rand_.Int(config_->scenario_database_size_);

        // Read scenarios from that batch
        readScenarios(batch_select, random_numbers_[b], extreme_sample_indices_[b]);
    }
}

void Sampler::sampleScenarios(int index)
{
    // Allocate space for the new samples and the extremest samples
    std::vector<Eigen::Vector2d> samples;
    samples.resize(S_);

    std::vector<int> extreme_sample_indices;
    extreme_sample_indices.resize(2);

    // Initialize variables
    double max_x = 0.0;
    double max_y = 0.0;
    double truncated_cap = 0.0;

    // If we sample a truncated Gaussian, compute the range modifier
    if (config_->truncated_)
    {
        truncated_cap = std::exp(-std::pow(config_->truncated_radius_, 2.0) / 2.0);
    }

    // Draw scenarios
    for (int s = 0; s < S_; s++)
    {

        // Generate uniform random numbers in 2D
        samples[s] = Eigen::Vector2d(rand_.Double(), rand_.Double());

        // In the case of truncated gaussian distribution, we modify the random sampled variables
        if (config_->truncated_)
        {
            samples[s](0) = samples[s](0) * (1.0 - truncated_cap) + truncated_cap;
        }

        // Convert them to a Gaussian
        Helpers::uniformToGaussian2D(samples[s]);
    }

    // Sort the samples on their distance to the mean of the distribution
    std::sort(samples.begin(), samples.end(), [](const Eigen::Vector2d &a, const Eigen::Vector2d &b) {
        return a.squaredNorm() > b.squaredNorm();
    });

    // Go through all samples
    for (int s = 0; s < S_; s++)
    {
        // Check if this is the furthest x point
        double abs_current_x = std::abs(samples[s](0));
        if (abs_current_x > max_x)
        {
            extreme_sample_indices[0] = s;
            max_x = abs_current_x;
        }

        // Check if this is the furthest x point
        double abs_current_y = std::abs(samples[s](1));
        if (abs_current_y > max_y)
        {
            extreme_sample_indices[1] = s;
            max_y = abs_current_y;
        }
    }

    // Prune scenarios! Save the left over sample count
    int batch_prune_count = pruneScenarios(samples, extreme_sample_indices);

    // Resize the samples to those not pruned
    samples.resize(batch_prune_count);

    std::cout << "total removed scenarios : " << S_ - batch_prune_count << "/" << S_ << " (" << (double)(S_ - batch_prune_count) / (double)S_ * 100.0 << ")" << std::endl;

    // Save scenarios
    saveScenarios(samples, extreme_sample_indices, index);
}

// Prune samples for efficiency
int Sampler::pruneScenarios(const std::vector<Eigen::Vector2d> &samples, const std::vector<int> &extreme_sample_indices)
{
    // General idea: sample points in a circle around the distribution, check which ones are never close enough
    // Initialize vectors that we need
    std::vector<double> distances;
    distances.resize(S_);

    // Vector of indices that are close enough at some point
    std::vector<bool> used_indices(S_, false);

    std::vector<int> all_indices;
    all_indices.resize(S_);

    // Idea: Sample a few points around the circle, check if the scenario is used in that case
    for (int point_it = 0; point_it < 50; point_it++)
    {

        double angle = 2.0 * M_PI / 50.0 * (double)point_it; // Space the angle

        // Compute the associated point
        Eigen::Vector2d point(
            std::abs(samples[extreme_sample_indices[0]](0)) * std::cos(angle),
            std::abs(samples[extreme_sample_indices[1]](1)) * std::sin(angle)); // rx * cos(theta) , ry * sin(theta)

        // For all samples, compute the distance
        for (int s = 0; s < S_; s++)
        {
            distances[s] = Helpers::dist(point, samples[s]);
            all_indices[s] = s;
        }

        // Sort to find the smallest distances to our point outside (keep track of indices!)
        std::sort(all_indices.begin(), all_indices.end(), [&distances](const int &a, const int &b) {
            return distances[a] < distances[b];
        });

        // The closest l+R are marked as used
        for (int i = 0; i < config_->polygon_checked_constraints_ + config_->removal_count_; i++)
        {
            used_indices[all_indices[i]] = true;
        }
    }

    // Find the first index marked as used, starting from the end (remember sorted from large distance to small distance to mean)
    for (size_t s = S_ - 1; s >= 0; s--)
    {
        if (used_indices[s] == true)
            return s;
    }
}
// Save a batch of scenarios
void Sampler::saveScenarios(const std::vector<Eigen::Vector2d> &samples, const std::vector<int> &extreme_sample_indices, int batch_index)
{

    // Get the package path
    std::string path = ros::package::getPath("lmpcc");

    // Save based on truncated or not truncated
    if (config_->truncated_)
    {
        path += "/samples/scenario_batch_truncated_at_" + std::to_string((int)config_->truncated_radius_) + "S" + std::to_string(config_->sample_count_) +
                "_R" + std::to_string(config_->removal_count_) + "_" + std::to_string(batch_index) + ".txt";
    }
    else
    {
        path += "/samples/scenario_batch_S" + std::to_string(config_->sample_count_) +
                "_R" + std::to_string(config_->removal_count_) + "_" + std::to_string(batch_index) + ".txt";
    }

    // Setup a file stream
    std::ofstream scenario_file;

    ROS_INFO_STREAM("ScenarioVRU: Saving scenario batch " << batch_index << " to\n"
                                                          << path);
    scenario_file.open(path);

    // Write scenarios to the file
    for (size_t s = 0; s < samples.size(); s++)
    {
        char str[39];

        // File will have two doubles with whitespace per line
        sprintf(str, "%.12f %.12f\n", samples[s](0), samples[s](1));
        scenario_file << str;
    }

    // Add the far index at the end
    char str[39];

    sprintf(str, "%d %d", extreme_sample_indices[0], extreme_sample_indices[1]);
    scenario_file << str;

    // Close the file
    scenario_file.close();
}

void Sampler::readScenarios(int index, std::vector<Eigen::Vector2d> &samples, std::vector<int> &extreme_sample_indices)
{

    // Get the package path
    std::string path = ros::package::getPath("lmpcc");

    if (config_->truncated_)
    {
        path += "/samples/scenario_batch_truncated_at_" + std::to_string((int)config_->truncated_radius_) + "S" + std::to_string(config_->sample_count_) +
                "_R" + std::to_string(config_->removal_count_) + "_" + std::to_string(index) + ".txt";
    }
    else
    {
        path += "/samples/scenario_batch_S" + std::to_string(config_->sample_count_) +
                "_R" + std::to_string(config_->removal_count_) + "_" + std::to_string(index) + ".txt";
    }

    // Start the file reading
    std::ifstream scenario_file(path);

    // Check if the file is okay
    if (!scenario_file.good())
    {
        throw std::runtime_error("Could not read scenario batch from database!");
    }

    // Initialize the reading variables
    double x, y;
    int s = 0;

    while (scenario_file >> x >> y)
    {
        samples[s] = Eigen::Vector2d(x, y);
        s++;
    }

    // Far index is at the end
    extreme_sample_indices[0] = (int)samples[s - 1](0);
    extreme_sample_indices[1] = (int)samples[s - 1](1);

    // s is index but is +1, size is +1, extreme_sample_indices is also -1
    samples.resize(s - 1);
}