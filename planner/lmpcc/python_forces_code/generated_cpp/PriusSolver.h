#ifndef PriusMODEL_H
#define PriusMODEL_H

#include <PriusFORCESNLPsolver/include/PriusFORCESNLPsolver.h>
#include <lmpcc/base_model.h>
#include <lmpcc/base_state.h>
#include <lmpcc/base_input.h>

#define PRIUS(X) (PriusSolver*)&(*X)

extern "C"
{
	extern void PriusFORCESNLPsolver_casadi2forces(PriusFORCESNLPsolver_float *x,  /* primal vars                                         */
										PriusFORCESNLPsolver_float *y,  /* eq. constraint multiplers                           */
										PriusFORCESNLPsolver_float *l,  /* ineq. constraint multipliers                        */
										PriusFORCESNLPsolver_float *p,  /* parameters                                          */
										PriusFORCESNLPsolver_float *f,  /* objective function (scalar)                         */
										PriusFORCESNLPsolver_float *nabla_f, /* gradient of objective function                      */
										PriusFORCESNLPsolver_float *c,	   /* dynamics                                            */
										PriusFORCESNLPsolver_float *nabla_c, /* Jacobian of the dynamics (column major)             */
										PriusFORCESNLPsolver_float *h,	   /* inequality constraints                              */
										PriusFORCESNLPsolver_float *nabla_h, /* Jacobian of inequality constraints (column major)   */
										PriusFORCESNLPsolver_float *hess,	   /* Hessian (column major)                              */
										solver_int32_default stage,	   /* stage number (0 indexed)                            */
										solver_int32_default iteration, /* iteration number of solver                          */
										solver_int32_default threadID /* Id of caller thread 								   */);
	PriusFORCESNLPsolver_extfunc extfunc_eval_prius = &PriusFORCESNLPsolver_casadi2forces;
}


class PriusDynamicsState : public BaseState
{

public:
	PriusDynamicsState(){};

	// Setter functions for variables in the model
	void set_x(double value) override { x = value; };
	void set_y(double value) override { y = value; };
	void set_psi(double value) override { psi = value; };
	void set_v(double value) override { v = value; };
	void set_delta(double value) override { delta = value; };

	// Getter functions for variables in the model
	double& get_x() override { return x; };
	double& get_y() override { return y; };
	double& get_psi() override { return psi; };
	double& get_v() override { return v; };
	double& get_delta() override { return delta; };

	void init()
	{
		x = 0.0;
		y = 0.0;
		psi = 0.0;
		v = 0.0;
		delta = 0.0;
	}

	void print()
	{
		std::cout <<
		"========== State ==========\n" << 
		"x = " << x << "\n" <<
		"y = " << y << "\n" <<
		"psi = " << psi << "\n" <<
		"v = " << v << "\n" <<
		"delta = " << delta << "\n" <<
		"============================\n";
	}
};

class PriusDynamicsModel : public BaseModel
{

public:
	PriusDynamicsState state_;

	PriusFORCESNLPsolver_params forces_params_;
	PriusFORCESNLPsolver_output forces_output_;
	PriusFORCESNLPsolver_info forces_info_;

	PriusDynamicsModel(){
		FORCES_N = 15; // Horizon length
		FORCES_NU = 3; // Number of control variables
		FORCES_NX = 6; // Differentiable variables
		FORCES_TOTAL_V = 9; // Total variable count
		FORCES_NPAR = 73; // Parameters per iteration
		enable_scenario_constraints = false;
		enable_ellipsoid_constraints = true;
		use_sqp_solver = false;
	};

	/* Inputs */
	double& predicted_a(unsigned int k) override { return forces_params_.x0[k * FORCES_TOTAL_V + 0]; };
	double& predicted_w(unsigned int k) override { return forces_params_.x0[k * FORCES_TOTAL_V + 1]; };
	double& predicted_slack(unsigned int k) override { return forces_params_.x0[k * FORCES_TOTAL_V + 2]; };

	double a() override { return forces_output_.x01[0]; };
	double w() override { return forces_output_.x01[1]; };
	double slack() override { return forces_output_.x01[2]; };

	/* States */ 
	double& x(unsigned int k) override { return forces_params_.x0[k * FORCES_TOTAL_V + 3]; };
	double& y(unsigned int k) override { return forces_params_.x0[k * FORCES_TOTAL_V + 4]; };
	double& psi(unsigned int k) override { return forces_params_.x0[k * FORCES_TOTAL_V + 5]; };
	double& v(unsigned int k) override { return forces_params_.x0[k * FORCES_TOTAL_V + 6]; };
	double& delta(unsigned int k) override { return forces_params_.x0[k * FORCES_TOTAL_V + 7]; };
	double& spline(unsigned int k) override { return forces_params_.x0[k * FORCES_TOTAL_V + 8]; };
	
BaseState* getState(){return &state_;};

	// Reset solver variables
	void resetSolver(){
		for (size_t i = 0; i < *(&forces_params_.all_parameters + 1) - forces_params_.all_parameters; i++)
			forces_params_.all_parameters[i] = 0.0;

		for (size_t i = 0; i < *(&forces_params_.xinit + 1) - forces_params_.xinit; i++)
			forces_params_.xinit[i] = 0.0;

		for (size_t i = 0; i < FORCES_N*FORCES_TOTAL_V; i++)
			forces_params_.x0[i] = 0.0;
	}

	// Set a solver parameter at index index of stage k to value
	void setParameter(unsigned int k, unsigned int index, double value){
		forces_params_.all_parameters[k*FORCES_NPAR + index] = value;
	}

	// Solve the optimization
	int solve(){
		return PriusFORCESNLPsolver_solve(&forces_params_, &forces_output_, &forces_info_, stdout, extfunc_eval_prius);
	}

	// Print Solver Info for this Iteration
	void printSolveInfo(){
		ROS_INFO_STREAM("primal objective " << forces_info_.pobj);
		ROS_INFO_STREAM("number of iterations for optimality " << forces_info_.it2opt);
	}

	// Reinitialize the solver (SQP)
	void setReinitialize(bool value){

	}

	void insertPredictedTrajectory()
	{
		for (unsigned int i = 0; i < FORCES_TOTAL_V; i++)
		{
			forces_params_.x0[i + 0 * FORCES_TOTAL_V] = forces_output_.x01[i];
			forces_params_.x0[i + 1 * FORCES_TOTAL_V] = forces_output_.x02[i];
			forces_params_.x0[i + 2 * FORCES_TOTAL_V] = forces_output_.x03[i];
			forces_params_.x0[i + 3 * FORCES_TOTAL_V] = forces_output_.x04[i];
			forces_params_.x0[i + 4 * FORCES_TOTAL_V] = forces_output_.x05[i];
			forces_params_.x0[i + 5 * FORCES_TOTAL_V] = forces_output_.x06[i];
			forces_params_.x0[i + 6 * FORCES_TOTAL_V] = forces_output_.x07[i];
			forces_params_.x0[i + 7 * FORCES_TOTAL_V] = forces_output_.x08[i];
			forces_params_.x0[i + 8 * FORCES_TOTAL_V] = forces_output_.x09[i];
			forces_params_.x0[i + 9 * FORCES_TOTAL_V] = forces_output_.x10[i];
			forces_params_.x0[i + 10 * FORCES_TOTAL_V] = forces_output_.x11[i];
			forces_params_.x0[i + 11 * FORCES_TOTAL_V] = forces_output_.x12[i];
			forces_params_.x0[i + 12 * FORCES_TOTAL_V] = forces_output_.x13[i];
			forces_params_.x0[i + 13 * FORCES_TOTAL_V] = forces_output_.x14[i];
			forces_params_.x0[i + 14 * FORCES_TOTAL_V] = forces_output_.x15[i];
		}
	}

	// Set xinit at index to value
	void setInitialState(int index, double value){
		forces_params_.xinit[index] = value;
	}


	// Set all initial solver values to the current state
	void setInitialToState(){
		x(0) = state_.get_x();
		y(0) = state_.get_y();
		psi(0) = state_.get_psi();
		v(0) = state_.get_v();
		delta(0) = state_.get_delta();
		forces_params_.xinit[0] = state_.get_x();
		forces_params_.xinit[1] = state_.get_y();
		forces_params_.xinit[2] = state_.get_psi();
		forces_params_.xinit[3] = state_.get_v();
		forces_params_.xinit[4] = state_.get_delta();
	}

	// Set solver values to sensor values
	void resetAtInfeasible(double brake){
		for (size_t k = 0; k < FORCES_N; k++){
			x(k) = state_.get_x();
			y(k) = state_.get_y();
			psi(k) = state_.get_psi();
			v(k) = state_.get_v();
			delta(k) = state_.get_delta();
		}
	}
	double getParameter(unsigned int k){
		return forces_params_.all_parameters[k];
	}


};
#endif