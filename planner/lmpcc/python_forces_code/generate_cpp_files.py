import os

def write_model_header(settings, model):

    file_path = os.path.dirname(os.path.abspath(__file__)) + "/generated_cpp/"+ model.system.name +"Solver.h"
    print(file_path)
    header_file = open(file_path, "w")

    header_file.write(\
        "#ifndef "+model.system.name+"MODEL_H\n"
        "#define "+model.system.name+"MODEL_H\n\n"
        "#include <"+model.system.name+"FORCESNLPsolver/include/"+model.system.name+"FORCESNLPsolver.h>\n"
        "#include <lmpcc/base_model.h>\n"
        "#include <lmpcc/base_state.h>\n"
        "#include <lmpcc/base_input.h>\n\n"
        "#define "+model.system.name.upper()+"(X) ("+model.system.name+"Solver*)&(*X)\n\n")

    header_file.write(\
        "extern \"C\"\n"
        "{\n"
        "\textern void "+model.system.name+"FORCESNLPsolver_casadi2forces("+model.system.name+"FORCESNLPsolver_float *x,  /* primal vars                                         */\n"
                                                           "\t\t\t\t\t\t\t\t\t\t"+model.system.name+"FORCESNLPsolver_float *y,  /* eq. constraint multiplers                           */\n"
                                                           "\t\t\t\t\t\t\t\t\t\t"+model.system.name+"FORCESNLPsolver_float *l,  /* ineq. constraint multipliers                        */\n"
                                                           "\t\t\t\t\t\t\t\t\t\t"+model.system.name+"FORCESNLPsolver_float *p,  /* parameters                                          */\n"
                                                           "\t\t\t\t\t\t\t\t\t\t"+model.system.name+"FORCESNLPsolver_float *f,  /* objective function (scalar)                         */\n"
                                                           "\t\t\t\t\t\t\t\t\t\t"+model.system.name+"FORCESNLPsolver_float *nabla_f, /* gradient of objective function                      */\n"
                                                           "\t\t\t\t\t\t\t\t\t\t"+model.system.name+"FORCESNLPsolver_float *c,	   /* dynamics                                            */\n"
                                                           "\t\t\t\t\t\t\t\t\t\t"+model.system.name+"FORCESNLPsolver_float *nabla_c, /* Jacobian of the dynamics (column major)             */\n"
                                                           "\t\t\t\t\t\t\t\t\t\t"+model.system.name+"FORCESNLPsolver_float *h,	   /* inequality constraints                              */\n"
                                                           "\t\t\t\t\t\t\t\t\t\t"+model.system.name+"FORCESNLPsolver_float *nabla_h, /* Jacobian of inequality constraints (column major)   */\n"
                                                           "\t\t\t\t\t\t\t\t\t\t"+model.system.name+"FORCESNLPsolver_float *hess,	   /* Hessian (column major)                              */\n"
                                                           "\t\t\t\t\t\t\t\t\t\tsolver_int32_default stage,	   /* stage number (0 indexed)                            */\n"
                                                           "\t\t\t\t\t\t\t\t\t\tsolver_int32_default iteration, /* iteration number of solver                          */\n"
                                                           "\t\t\t\t\t\t\t\t\t\tsolver_int32_default threadID /* Id of caller thread 								   */);\n"
            "\t"+model.system.name+"FORCESNLPsolver_extfunc extfunc_eval_"+model.system.name.lower()+" = &"+model.system.name+"FORCESNLPsolver_casadi2forces;\n"
        "}\n\n"
    )

    # The dynamic state structure
    header_file.write("\nclass "+model.system.name+"DynamicsState : public BaseState\n"
                      "{\n\n")

    header_file.write("public:\n")

    header_file.write("\t"+model.system.name+"DynamicsState(){};\n\n")

    # Set functions
    header_file.write("\t// Setter functions for variables in the model\n")
    for idx, state in enumerate(model.states):
        if model.states_from_sensor[idx]:
            header_file.write("\tvoid set_"+state+"(double value) override { "+state+" = value; };\n")

    header_file.write("\n\t// Getter functions for variables in the model\n")
    for idx, state in enumerate(model.states):
        if model.states_from_sensor[idx]:
            header_file.write("\tdouble& get_"+state+"() override { return "+state+"; };\n")

    header_file.write("\n\tvoid init()\n"
                      "\t{\n")

    for idx, state in enumerate(model.states):
        if model.states_from_sensor[idx]:
            header_file.write("\t\t" + state + " = 0.0;\n")

    header_file.write("\t}\n")

    header_file.write("\n\tvoid print()\n"
                      "\t{\n"
                      "\t\tstd::cout <<\n"
                      "\t\t\"========== State ==========\\n\" << \n")

    for idx, state in enumerate(model.states):
        if model.states_from_sensor[idx]:
            header_file.write("\t\t\"" + state + " = \" << " + state + " << \"\\n\" <<\n")

    header_file.write("\t\t\"============================\\n\";\n"
                      "\t}\n")

    header_file.write("};\n\n")

    # # The dynamic input structure
    # header_file.write("\nclass "+model.system.name+"DynamicsInput : public BaseInput\n"
    #                   "{\n\n")
    #
    # header_file.write("public:\n")
    #
    # header_file.write("\t"+model.system.name+"DynamicsInput(){};\n\n")
    #
    # for idx, input_var in enumerate(model.inputs):
    #     if model.inputs_to_vehicle[idx]:
    #         header_file.write("\tdouble " + input_var + ";\n")
    #
    # header_file.write("\n\tvoid print()\n"
    #                   "\t{\n"
    #                   "\t\tstd::cout <<\n"
    #                   "\t\t\"========== Input ==========\\n\" << \n")
    #
    # for idx, input_var in enumerate(model.inputs):
    #     if model.inputs_to_vehicle[idx]:
    #         header_file.write("\t\t\"" + input_var + " = \" << " + input_var + " << \"\\n\" <<\n")
    #
    # header_file.write("\t\t\"============================\\n\";\n"
    #                   "\t}\n")
    #
    # header_file.write("};\n\n")

    header_file.write("class "+model.system.name+"DynamicsModel : public BaseModel\n"
                                              "{\n\n")

    header_file.write("public:\n")

    # header_file.write("\t"+model.system.name+"DynamicsInput control_;\n")

    header_file.write("\t"+model.system.name+"DynamicsState state_;\n\n")

    header_file.write(\
        "\t"+model.system.name+"FORCESNLPsolver_params forces_params_;\n" +
        "\t"+model.system.name+"FORCESNLPsolver_output forces_output_;\n" +
        "\t"+model.system.name+"FORCESNLPsolver_info forces_info_;\n\n" +
        "\t"+model.system.name+"DynamicsModel(){\n"
                            "\t\tFORCES_N = " + str(settings.N) + "; // Horizon length\n"
                            "\t\tFORCES_NU = " + str(model.nu) + "; // Number of control variables\n"
                            "\t\tFORCES_NX = " + str(model.nx) + "; // Differentiable variables\n"
                            "\t\tFORCES_TOTAL_V = " + str(model.nvar) + "; // Total variable count\n"
                            "\t\tFORCES_NPAR = " + str(settings.npar) + "; // Parameters per iteration\n"
    )

    if settings.enable_scenario_constraints:
        header_file.write("\t\tenable_scenario_constraints = true;\n")
    else:
        header_file.write("\t\tenable_scenario_constraints = false;\n")

    if settings.enable_ellipsoid_constraints:
        header_file.write("\t\tenable_ellipsoid_constraints = true;\n")
    else:
        header_file.write("\t\tenable_ellipsoid_constraints = false;\n")

    if settings.use_sqp_solver:
        header_file.write("\t\tuse_sqp_solver = true;\n")
    else:
        header_file.write("\t\tuse_sqp_solver = false;\n")

    header_file.write("\t};\n\n"
        "\t/* Inputs */\n")

    for i in range(0, len(model.inputs)):
        header_file.write("\tdouble& predicted_" + model.inputs[i] + \
                          "(unsigned int k) override { return forces_params_.x0[k * FORCES_TOTAL_V + " + str(i) + "]; };\n")

    header_file.write("\n")

    for i in range(0, len(model.inputs)):
        header_file.write("\tdouble " + model.inputs[i] + \
                          "() override { return forces_output_.x01[" + str(i) + "]; };\n")

    header_file.write("\n\t/* States */ \n")

    for i in range(0, len(model.states)):
        header_file.write("\tdouble& " + model.states[i] + \
                          "(unsigned int k) override { return forces_params_.x0[k * FORCES_TOTAL_V + " + str(model.nu + i) + "]; };\n")

    header_file.write("\t\nBaseState* getState(){return &state_;};\n")

    # Todo: Add this for states
    # "	double &j(unsigned int k) { throw std::runtime_error("PriusSolver: You tried to access state variable j, that does not exist in this model!"); };"

    # Reset solver function #
    header_file.write("\n"
                        "\t// Reset solver variables\n"
                        "\tvoid resetSolver(){\n"
                        "\t\tfor (size_t i = 0; i < *(&forces_params_.all_parameters + 1) - forces_params_.all_parameters; i++)\n"
                        "\t\t\tforces_params_.all_parameters[i] = 0.0;\n\n"
                        "\t\tfor (size_t i = 0; i < *(&forces_params_.xinit + 1) - forces_params_.xinit; i++)\n"
                        "\t\t\tforces_params_.xinit[i] = 0.0;\n\n"
                        "\t\tfor (size_t i = 0; i < FORCES_N*FORCES_TOTAL_V; i++)\n"
                        "\t\t\tforces_params_.x0[i] = 0.0;\n"
                        "\t}\n\n")

    # Set Parameter function #
    header_file.write("\t// Set a solver parameter at index index of stage k to value\n"
                      "\tvoid setParameter(unsigned int k, unsigned int index, double value){\n"
                      "\t\tforces_params_.all_parameters[k*FORCES_NPAR + index] = value;\n"
                      "\t}\n\n")

    # Solve function #
    header_file.write("\t// Solve the optimization\n"
                      "\tint solve(){\n"
                        "\t\treturn "+model.system.name+"FORCESNLPsolver_solve(&forces_params_, &forces_output_, &forces_info_, stdout, extfunc_eval_"+model.system.name.lower()+");\n"
                      "\t}\n\n")

    header_file.write("\t// Print Solver Info for this Iteration\n"
                      "\tvoid printSolveInfo(){\n"
                      "\t\tROS_INFO_STREAM(\"primal objective \" << forces_info_.pobj);\n")
    if settings.use_sqp_solver:
        header_file.write("\t\tROS_INFO_STREAM(\"number of iterations for optimality \" << forces_info_.it);\n")
    else:
        header_file.write("\t\tROS_INFO_STREAM(\"number of iterations for optimality \" << forces_info_.it2opt);\n")
    header_file.write("\t}\n\n")

    header_file.write("\t// Reinitialize the solver (SQP)\n"
                      "\tvoid setReinitialize(bool value){\n")
    if settings.use_sqp_solver:
        header_file.write("\t\tforces_params_.reinitialize = value;\n"
                          "\t}\n")
    else:
        header_file.write("\n\t}\n")

    # Insert Predicted Trajectory function #
    header_file.write("\n" + \
    "\tvoid insertPredictedTrajectory()\n" + \
    "\t{\n" + \
    "\t\tfor (unsigned int i = 0; i < FORCES_TOTAL_V; i++)\n" + \
    "\t\t{\n")

    for k in range(0, settings.N):
        if(k >= 9):
            header_file.write("\t\t\tforces_params_.x0[i + " + str(k) + " * FORCES_TOTAL_V] = forces_output_.x" + str(k + 1) + "[i];\n")
        else:
            header_file.write("\t\t\tforces_params_.x0[i + " + str(k) + " * FORCES_TOTAL_V] = forces_output_.x0" + str(k + 1) + "[i];\n")

    header_file.write("\t\t}\n" + \
    "\t}\n")

    # Set initial state function (value based) #
    header_file.write("\n\t// Set xinit at index to value\n"
                      "\tvoid setInitialState(int index, double value){\n"
                      "\t\tforces_params_.xinit[index] = value;\n"
                      "\t}\n\n")

    # Set initial to state function (value based) #
    header_file.write("\n\t// Set all initial solver values to the current state\n"
                      "\tvoid setInitialToState(){\n")

    for idx, state in enumerate(model.states):
        if model.states_from_sensor[idx]:
            header_file.write("\t\t" + state + "(0) = state_.get_" + state + "();\n")

    for idx, state in enumerate(model.states):
        if model.states_from_sensor[idx]:
            header_file.write("\t\tforces_params_.xinit[" + str(idx) + "] = state_.get_" + state + "();\n")

    header_file.write("\t}\n")

    header_file.write("\n\t// Set solver values to sensor values\n"
                      "\tvoid resetAtInfeasible(double brake){\n"
                      "\t\tfor (size_t k = 0; k < FORCES_N; k++){\n")

    for idx, state in enumerate(model.states):
        if model.states_from_sensor[idx]:
            if model.states_from_sensor_at_infeasible[idx]:  # Set to the sensor
                header_file.write("\t\t\t" + state + "(k) = state_.get_" + state + "();\n")
            else:   # Otherwise set to 0.0
                header_file.write("\t\t\t" + state + "(k) = 0.0;\n")

    header_file.write("\t\t}\n"
                      "\t}\n")

    header_file.write("\tdouble getParameter(unsigned int k){\n"
                      "\t\treturn forces_params_.all_parameters[k];\n\t}\n\n")

    header_file.write("\n};\n")

    header_file.write("#endif")

    header_file.close()

