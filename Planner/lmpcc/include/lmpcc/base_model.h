//
// Created by bdebrito on 16-2-21.
//

#ifndef BASE_MODEL_H
#define BASE_MODEL_H

#include <lmpcc/rk4.hpp>
#include <lmpcc/base_state.h>

class BaseModel{

public:
    BaseModel()
    {

    }

protected:
    double tspan[2], integrator_t[2], integrator_y[4], y0[2];

public:

    unsigned int FORCES_N;       // Horizon length
    unsigned int FORCES_NU;       // Number of control variables
    unsigned int FORCES_NX;       // Differentiable variables
    unsigned int FORCES_TOTAL_V; // Total variable count
    unsigned int FORCES_NPAR;   // Parameters per iteration

    bool enable_scenario_constraints,enable_ellipsoid_constraints, use_sqp_solver;

    virtual 
    void resetSolver() = 0;

    virtual
    void setParameter(unsigned int k, unsigned int index, double value) = 0;

    virtual
    double getParameter(unsigned int k) = 0;

    virtual 
    int solve() = 0;

    virtual
    void printSolveInfo() = 0;

    virtual 
    void insertPredictedTrajectory() = 0;

    virtual 
    void setInitialToState() = 0;

	virtual 
    void setInitialState(int index, double value) = 0;

    virtual
    void resetAtInfeasible(double s) = 0;

    virtual
    void setReinitialize(bool value) = 0;

    virtual
    BaseState* getState() = 0;

    // All Possible Inputs (throw an error if they do not exist)
    virtual double v(){ throw std::runtime_error("BaseModel: Undefined input v accessed");};
    virtual double a() { throw std::runtime_error("BaseModel: Undefined input a accessed"); };
    virtual double j() { throw std::runtime_error("BaseModel: Undefined input j accessed"); };
    virtual double delta() { throw std::runtime_error("BaseModel: Undefined input delta accessed"); };
    virtual double w() { throw std::runtime_error("BaseModel: Undefined input w accessed"); };
    virtual double alpha() { throw std::runtime_error("BaseModel: Undefined input alpha accessed"); };
    virtual double slack() { throw std::runtime_error("BaseModel: Undefined input slack accessed"); };

    virtual double& predicted_v(unsigned int k) { throw std::runtime_error("BaseModel: Undefined input v accessed"); };
    virtual double& predicted_a(unsigned int k) { throw std::runtime_error("BaseModel: Undefined input a accessed"); };
    virtual double& predicted_j(unsigned int k) { throw std::runtime_error("BaseModel: Undefined input j accessed"); };
    virtual double& predicted_delta(unsigned int k) { throw std::runtime_error("BaseModel: Undefined input delta accessed"); };
    virtual double& predicted_w(unsigned int k) { throw std::runtime_error("BaseModel: Undefined input w accessed"); };
    virtual double& predicted_alpha(unsigned int k) { throw std::runtime_error("BaseModel: Undefined input alpha accessed"); };
    virtual double& predicted_slack(unsigned int k) { throw std::runtime_error("BaseModel: Undefined input slack accessed"); };

    // Manditory States
    virtual double &x(unsigned int k) = 0;
    virtual double &y(unsigned int k) = 0;
    virtual double &psi(unsigned int k) = 0;
    virtual double &v(unsigned int k) = 0;

    // Optional States
    virtual double &spline(unsigned int k) { throw std::runtime_error("BaseModel: Undefined state spline accessed"); };
    virtual double &delta(unsigned int k) { throw std::runtime_error("BaseModel: Undefined state delta accessed"); };
    virtual double &ax(unsigned int k) { throw std::runtime_error("BaseModel: Undefined state ax accessed"); };
    virtual double &ay(unsigned int k) { throw std::runtime_error("BaseModel: Undefined state ay accessed"); };
    virtual double &w(unsigned int k) { throw std::runtime_error("BaseModel: Undefined state w accessed"); };
    virtual double &j(unsigned int k) { throw std::runtime_error("BaseModel: Undefined state j accessed"); };
    virtual double &alpha(unsigned int k) { throw std::runtime_error("BaseModel: Undefined state alpha accessed"); };

    // Implements rk4 forward integration of time seconds
    double integrate(double time, double cur_var, double input)
    {
        tspan[0] = 0;
        tspan[1] = time;

        y0[0] = cur_var;
        y0[1] = input;

        auto integration_function = [](double t, double y[], double dydt[]) {
            dydt[0] = y[1];
            dydt[1] = 0.0;
        };

        rk4(integration_function, tspan, y0, 1, 2, integrator_t, integrator_y);

        return integrator_y[2];
    }
};

#endif //BASE_MODEL_H
