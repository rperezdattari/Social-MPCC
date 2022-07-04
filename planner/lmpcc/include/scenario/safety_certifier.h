
#ifndef SAFETY_CERTIFIER_H
#define SAFETY_CERTIFIER_H

#include <vector>
#include <cmath>
#include <iostream>
#include "lmpcc_configuration.h"

// For now assuming constant N / R
class SafetyCertifier{

public:
    SafetyCertifier(){};

    void init(predictive_configuration *config);

private:

    //std::vector<double> epsilon_lut_;
    double epsilon_lut_;

    double rootedNChooseK(double N, double k, double root);

    double sigmaToEpsilon(double sigma);


public:

    double getReliability(unsigned int k);

// Epsilon, beta, N to R
    int requestRemovalSize(double epsilon, double beta, int max_support, int sample_size);

    
};



#endif