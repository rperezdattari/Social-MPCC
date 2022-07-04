

#ifndef SAMPLER_H
#define SAMPLER_H

#include <vector>
#include <Eigen/Eigen>
#include "lmpcc/lmpcc_configuration.h"
#include "lmpcc_tools/helpers.h"
#include <ros/ros.h>

#include <fstream>
#include <ros/package.h>
#include <string>

// #define SAMPLE_LIBRARY Sampler::Get()

/* Singleton Class for generating samples, saving and loading. */
class Sampler{

public:
    Sampler(const Sampler&) = delete;

private:

    // Private constructor!
    Sampler();

    // General initialization function
    void initScenarios();

    // Sample generation and pruning    
    void sampleScenarios(int index);
    int pruneScenarios(const std::vector<Eigen::Vector2d> &samples, const std::vector<int> &far_index);

    // Saving and loading
    void saveScenarios(const std::vector<Eigen::Vector2d> &samples, const std::vector<int> &far_index, int batch_index);
    void readScenarios(int index, std::vector<Eigen::Vector2d> &samples, std::vector<int> &far_index);

    // VARIABLES
    // Configuration
    predictive_configuration *config_;
    Helpers::RandomGenerator rand_;

    int S_; // sampling count
    int B_; // Batches
    int R_; // Scenario removal (maximum now)

    int largest_sample_size_;

    // SAMPLE STORAGE
    // B Batches of S samples with 2D uniform random numbers
    std::vector<std::vector<int>> extreme_sample_indices_;                       // Most extreme samples
    std::vector<std::vector<Eigen::Vector2d>> random_numbers_;      // Actual database

public:

    // Singleton function
    static Sampler& Get(){

        static Sampler instance_;

        return instance_;

    }

    void Init(predictive_configuration *config);

    std::vector<Eigen::Vector2d>& BatchReference(int batch_index);
    std::vector<int>& ExtremeSampleIndices(int batch_index);
    
    int LargestSampleSize() const;
};


#endif