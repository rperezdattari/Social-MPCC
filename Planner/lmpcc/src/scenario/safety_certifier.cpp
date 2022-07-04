#include "scenario/safety_certifier.h"

void SafetyCertifier::init(predictive_configuration *config) //int sample_size, int removal_size, double beta, int max_support)
{

    //epsilon_lut_.resize(config->inner_polygon_approximation_);
    double beta = 1e-6;
    double root;
    double picking_size = config->sample_count_ - config->removal_count_;

    double eps = sigmaToEpsilon(3.0);
    std::cout << "Epsilon for 3 sigma is " << eps << std::endl;

    int k = config->inner_approximation_;
    // Compute epsilon for all possible support subsample sizes
    // for (int k = 0; k < config->inner_polygon_approximation_; k++)
    // {
    root = picking_size - k;

    epsilon_lut_ = 1.0 - std::pow(beta, 1.0 / root) *
                             (1.0 / rootedNChooseK(config->sample_count_, picking_size, root)) *
                             std::pow(1.0 / config->inner_approximation_, 1.0 / root) *
                             (1.0 / rootedNChooseK(picking_size, k, root));

    std::cout << "eps[" << k << "] = " << epsilon_lut_ << std::endl; // Matches matlab!
    // }
}

double SafetyCertifier::rootedNChooseK(double N, double k, double root)
{

    double result = 1.0;
    for(int i = 1; i <= k; i++){
        result *= std::pow((N - (k - i)) / (k - i + 1), 1.0 / root);
    }

    return result;

}

double SafetyCertifier::getReliability(unsigned int k)
{
    return epsilon_lut_;
}

double SafetyCertifier::sigmaToEpsilon(double sigma){

    return 1.0 - std::exp(-sigma * sigma / 2.0);

}


int SafetyCertifier::requestRemovalSize(double epsilon, double beta, int max_support, int sample_size){

    return 0;
    
    // double root;
    // double picking_size;

    // double current_epsilon;
    // int removal_size = -10;

    // while(current_epsilon < epsilon){
    
    //     removal_size += 10;

    //     picking_size = sample_size - removal_size;
    //     root = picking_size - max_support;

    //     current_epsilon = 1.0 - std::pow(beta, 1.0 / root) *
    //                                 (1.0 / rootedNChooseK(sample_size, picking_size, root)) *
    //                                 std::pow(1.0 / max_support, 1.0 / root) *
    //                                 (1.0 / rootedNChooseK(picking_size, max_support, root));
    // }

    // return removal_size - 10;
}
