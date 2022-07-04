
// This file containts read parameter from server, callback, call class objects, control all class, objects of all class

#include <lmpcc/lmpcc_controller.h>

// Solvers (NEED TO BE LOADED HERE TO PREVENT DOUBLE DECLARATIONS)
#include <generated_cpp/PriusSolver.h>

MPCC::~MPCC()
{
}

void MPCC::spinNode()
{
    ROS_INFO(" Predictive control node is running, now it's 'Spinning Node'");
    ros::spin();
}

// initialize all helper class of predictive control and subscibe joint state and publish controlled joint velocity
bool MPCC::initialize()
{
    // make sure node is still running
    if (ros::ok())
    {
        // initialize helper classes, make sure pd_config should be initialized first as mother of other class
        controller_config_.reset(new predictive_configuration());
        bool controller_config_success = controller_config_->initialize();

        if (controller_config_success == false)
        {
            ROS_ERROR("MPCC: FAILED TO INITILIZED!!");
            std::cout << "States: \n"
                      << " pd_config: " << std::boolalpha << controller_config_success << "\n"
                      << " pd config init success: " << std::boolalpha << controller_config_->initialize_success_
                      << std::endl;
            return false;
        }

        // Loading Solver
        if (controller_config_->robot_.compare("carla") == 0)
        {

            // The solver interface
            solver_interface_.reset(new PriusDynamicsModel());

            // state_ = &(new PriusDynamicsState())
            system_interface_.reset(new CarlaInterface(nh, this, &(*controller_config_), (BaseModel *)&(*solver_interface_)));
        }

        controller_config_->N_ = solver_interface_->FORCES_N;

        clock_frequency_ = controller_config_->clock_frequency_;

        if (controller_config_->activate_debug_output_)
        {
            ROS_WARN("===== DEBUG INFO ACTIVATED =====");
        }

        /******************************** Subscribers **********************************************************/
        // Subscriber providing information about velocity ref
        v_ref_sub_ = nh.subscribe(controller_config_->vref_topic_, 1, &MPCC::VRefCallBack, this);
        joy_sub_ = nh.subscribe("joy", 1, &MPCC::JoyCallBack, this);

        // Subscriber providing point cloud data
        if (controller_config_->static_obstacles_enabled_)
            point_cloud_sub_ = nh.subscribe(controller_config_->occupancy_grid_topic_, 1, &MPCC::mapCallback, this);

        /******************************** Service Servers **********************************************************/
        // reset_server_ = nh.advertiseService(controller_config_->reset_topic_, &MPCC::ResetCallBack,this);

        /******************************** Publishers **********************************************************/
        computation_pub_ = nh.advertise<std_msgs::Float64>("lmpcc/computation_times", 1);
        pred_traj_pub_ = nh.advertise<nav_msgs::Path>(controller_config_->planned_trajectory_topic_, 1);
        robot_collision_space_pub_ = nh.advertise<visualization_msgs::MarkerArray>(controller_config_->planned_space_topic_, 100);
        feedback_pub_ = nh.advertise<lmpcc::control_feedback>(controller_config_->controller_feedback_, 1); /*"lmpcc/controller_feedback",1);*/
        feasibility_pub_ = nh.advertise<std_msgs::Float64>("lmpcc/feasibility", 1);
        pred_cmd_pub_ = nh.advertise<nav_msgs::Path>("lmpcc/plan_cmd", 1);
        close_path_points_pub_ = nh.advertise<nav_msgs::Path>("close_path_points", 1);
        path_is_over_pub_ = nh.advertise<std_msgs::Bool>("path_over", 10);

        // Timer used for unsynchronous mode
        timer_ = nh.createTimer(ros::Duration(1.0 / clock_frequency_), &MPCC::runNode, this);

        // Initialize trajectory variables
        next_point_dist = 0;
        goal_dist = 0;
        minimal_s_ = 0;
        prev_x_ = 0.0;
        prev_y_ = 0.0;

        prev_point_dist = 0;
        goal_reached_ = false;
        state_received_ = false;

        Sampler::Get().Init(&(*controller_config_));

        pred_traj_.poses.resize(solver_interface_->FORCES_N);
        pred_cmd_.poses.resize(solver_interface_->FORCES_N);
        pred_traj_.header.frame_id = controller_config_->target_frame_;
        for (size_t i = 0; i < solver_interface_->FORCES_N; i++)
        {
            pred_traj_.poses[i].header.frame_id = controller_config_->target_frame_;
        }

        // Initialize the scenario handlers for the discs
        orientation_vec_.resize(solver_interface_->FORCES_N);
        poses_vec_.resize(controller_config_->n_discs_);
        scenario_manager_.resize(controller_config_->n_discs_);
        for (int i = 0; i < controller_config_->n_discs_; i++)
        {
            poses_vec_[i].resize(solver_interface_->FORCES_N);

            scenario_manager_[i].reset(new ScenarioManager(nh, &(*controller_config_), controller_config_->discs_to_draw_[i])); // Incorrect!
        }

        ROS_INFO("initialize state and control weight factors");
        slack_weight_ = controller_config_->slack_weight_;
        repulsive_weight_ = controller_config_->repulsive_weight_;
        reference_velocity_ = controller_config_->reference_velocity_;
        ros::NodeHandle nh_predictive("predictive_controller");

        if (controller_config_->auto_enable_plan_)
        {
            enable_output_ = true;
            plan_ = true;
            auto_enable_ = true;
        }
        else
        {
            enable_output_ = false;
            plan_ = false;
            auto_enable_ = false;
        }

        /////////// --- > Before dynamic reconfigure!
        // Check if all reference vectors are of the same length
        if (!((controller_config_->ref_x_.size() == controller_config_->ref_y_.size()) && (controller_config_->ref_x_.size() == controller_config_->ref_theta_.size()) && (controller_config_->ref_y_.size() == controller_config_->ref_theta_.size())))
        {
            ROS_ERROR("Reference path inputs should be of equal length");
            return false;
        }

        // To be continued

        system_interface_->SetExternalObstacleCallback(&MPCC::OnObstaclesReceived);
        system_interface_->SetExternalStateCallback(&MPCC::OnStateReceived);
        system_interface_->SetExternalWaypointsCallback(&MPCC::OnWaypointsReceived);
        system_interface_->SetExternalResetCallback(&MPCC::OnReset);

        solver_interface_->getState()->init();
        reference_path_.Init(nh, &(*controller_config_));

        simulated_velocity_ = 0.0;

        ROS_INFO("Setting up dynamic_reconfigure server for the parameters");
        reconfigure_server_.reset(new dynamic_reconfigure::Server<lmpcc::PredictiveControllerConfig>(reconfig_mutex_, nh_predictive));
        reconfigure_server_->setCallback(boost::bind(&MPCC::reconfigureCallback, this, _1, _2));

        computeEgoDiscs();

        // Controller options

        replan_ = false;
        debug_ = false;

        // Plot variables
        ellips1.type = visualization_msgs::Marker::CYLINDER;
        ellips1.id = 60;
        ellips1.color.b = 1.0;
        ellips1.color.a = 0.6;
        ellips1.header.frame_id = controller_config_->target_frame_;
        ellips1.ns = "trajectory";
        ellips1.action = visualization_msgs::Marker::ADD;
        ellips1.scale.x = r_discs_ * 2.0;
        ellips1.scale.y = r_discs_ * 2.0;
        ellips1.scale.z = 0.05;

        exit_code_ = 1;

        ROS_WARN("PREDICTIVE CONTROL INTIALIZED!!");
        return true;
    }
    else
    {
        ROS_ERROR("MPCC: Failed to initialize as ROS Node is shutdown");
        return false;
    }
}

void MPCC::mapCallback(const nav_msgs::OccupancyGrid::ConstPtr &occupancy_grid)
{

    // Find the pose of the origin from the occupancy map info
    Eigen::Vector2d pose_eigen(occupancy_grid->info.origin.position.x, occupancy_grid->info.origin.position.y);
    pose_eigen(0) += (double)occupancy_grid->info.width / 2.0 * occupancy_grid->info.resolution;
    pose_eigen(1) += (double)occupancy_grid->info.height / 2.0 * occupancy_grid->info.resolution;

    if (controller_config_->activate_debug_output_)
        ROS_INFO("Map received");

    if (solver_interface_->enable_scenario_constraints)
    {
        for (int i = 0; i < controller_config_->n_discs_; i++)
            scenario_manager_[i]->mapCallback(occupancy_grid, pose_eigen, tf_listener_);
    }
}

void MPCC::computeEgoDiscs()
{
    // Collect parameters for disc representation
    int n_discs = controller_config_->n_discs_;
    double length = controller_config_->ego_l_;
    double width = controller_config_->ego_w_;
    double com_to_back = 2.4; // distance center of mass to back of the car

    // Initialize positions of discs
    x_discs_.resize(n_discs);

    // Compute radius of the discs
    r_discs_ = width / 2;

    // Loop over discs and assign positions, with respect to center of mass
    for (int discs_it = 0; discs_it < n_discs; discs_it++)
    {

        if (n_discs == 1)
        { // if only 1 disc, position in center;
            x_discs_[discs_it] = -com_to_back + length / 2;
        }
        else if (discs_it == 0)
        { // position first disc so it touches the back of the car
            x_discs_[discs_it] = -com_to_back + r_discs_;
        }
        else if (discs_it == n_discs - 1)
        {
            x_discs_[discs_it] = -com_to_back + length - r_discs_;
        }
        else
        {
            x_discs_[discs_it] = -com_to_back + r_discs_ + discs_it * (length - 2 * r_discs_) / (n_discs - 1);
        }
    }

    ROS_WARN_STREAM("Generated " << n_discs << " ego-vehicle discs with radius " << r_discs_);
    ROS_DEBUG_STREAM(x_discs_);
}

void MPCC::reset_solver()
{

    solver_interface_->resetSolver();
}

// update this function 1/clock_frequency
void MPCC::runNode(const ros::TimerEvent &event)
{
    ControlLoop();
}

void multiplyMatrices(double firstMatrix[][2], double secondMatrix[][2], double mult[][2], int rowFirst, int columnFirst, int rowSecond, int columnSecond)
{
    int i, j, k;

    // Initializing elements of matrix mult to 0.
    for (i = 0; i < rowFirst; ++i)
    {
        for (j = 0; j < columnSecond; ++j)
        {
            mult[i][j] = 0;
        }
    }

    // Multiplying matrix firstMatrix and secondMatrix and storing in array mult.
    for (i = 0; i < rowFirst; ++i)
    {
        for (j = 0; j < columnSecond; ++j)
        {
            for (k = 0; k < columnFirst; ++k)
            {
                mult[i][j] += firstMatrix[i][k] * secondMatrix[k][j];
            }
        }
    }
}

void multiplyMatriceArray(double firstMatrix[][2], double array[][1], double mult[][1], int rowFirst, int columnFirst, int rowSecond, int columnSecond)
{
    int i, j, k;

    // Initializing elements of matrix mult to 0.
    for (i = 0; i < rowFirst; ++i)
    {
        for (j = 0; j < columnSecond; ++j)
        {
            mult[i][j] = 0;
        }
    }

    // Multiplying matrix firstMatrix and array and storing in array mult.
    for (i = 0; i < rowFirst; ++i)
    {
        for (j = 0; j < columnSecond; ++j)
        {
            for (k = 0; k < columnFirst; ++k)
            {
                mult[i][j] += firstMatrix[i][k] * array[k][j];
            }
        }
    }
}

void MPCC::checkCollisionConstraints()
{
    ROS_INFO("MPCC::checkCollisionConstraints()");
    double obstacle_rotation[2][2], ab[2][2], obstacle_rotation_transpose[2][2];

    double ab_rot[2][2], obstacle_ellipse_matrix[2][2], disc_to_obstacle[2][1], disc_to_obstacle_transpose[1][2];

    double c_disc_obstacle[2][1];

    double constraint_value[1];

    // Check road cosntraints
    for (size_t N_iter = 0; N_iter < solver_interface_->FORCES_N; N_iter++)
    {

        double x = solver_interface_->x(N_iter);
        double y = solver_interface_->y(N_iter);

        for (int obst_it = 0; obst_it < controller_config_->max_obstacles_; obst_it++)
        {

            double obst_x = system_interface_->obstacles_.lmpcc_obstacles[obst_it].trajectory.poses[N_iter].pose.position.x;
            double obst_y = system_interface_->obstacles_.lmpcc_obstacles[obst_it].trajectory.poses[N_iter].pose.position.y;
            double obst_theta = system_interface_->obstacles_.lmpcc_obstacles[obst_it].trajectory.poses[N_iter].pose.orientation.z;
            double obst_major = system_interface_->obstacles_.lmpcc_obstacles[obst_it].major_semiaxis[N_iter];
            double obst_minor = system_interface_->obstacles_.lmpcc_obstacles[obst_it].minor_semiaxis[N_iter];

            // Compute ellipse matrix
            ab[0][0] = 1 / std::pow((obst_major + r_discs_), 2);
            ab[0][1] = 0;
            ab[1][0] = 0;
            ab[1][1] = 1 / std::pow((obst_minor + r_discs_), 2);

            obstacle_rotation[0][0] = cos(obst_theta);
            obstacle_rotation[0][1] = -sin(obst_theta);
            obstacle_rotation[1][0] = sin(obst_theta);
            obstacle_rotation[1][1] = cos(obst_theta);

            obstacle_rotation_transpose[0][0] = cos(obst_theta);
            obstacle_rotation_transpose[0][1] = sin(obst_theta);
            obstacle_rotation_transpose[1][0] = -sin(obst_theta);
            obstacle_rotation_transpose[1][1] = cos(obst_theta);

            multiplyMatrices(ab, obstacle_rotation, ab_rot, 2, 2, 2, 2);

            multiplyMatrices(obstacle_rotation_transpose, ab_rot, obstacle_ellipse_matrix, 2, 2, 2, 2);

            for (int disc_it = 0; disc_it < controller_config_->n_discs_; disc_it++)
            {
                double disc_x = x_discs_[disc_it];

                // Get and compute the disc position
                std::array<double, 2> disc_pos;
                disc_pos[0] = cos(solver_interface_->psi(0)) * disc_x + x;
                disc_pos[1] = sin(solver_interface_->psi(0)) * disc_x + y;

                // construct the constraint and append it
                disc_to_obstacle[0][0] = disc_pos[0] - obst_x;
                disc_to_obstacle[1][0] = disc_pos[1] - obst_y;
                disc_to_obstacle_transpose[0][0] = disc_pos[0] - obst_x;
                disc_to_obstacle_transpose[0][1] = disc_pos[1] - obst_y;

                multiplyMatriceArray(obstacle_ellipse_matrix, disc_to_obstacle, c_disc_obstacle, 2, 2, 2, 1);
                constraint_value[0] = disc_to_obstacle_transpose[0][0] * c_disc_obstacle[0][0] + disc_to_obstacle_transpose[0][1] * c_disc_obstacle[1][0];

                if (constraint_value[0] < 1.0)
                {
                    ROS_INFO_STREAM("Violating collision constraint between obstacle " << obst_it << " and disc " << disc_it << " for time-step " << N_iter);
                    ROS_INFO_STREAM("Obstacle position" << obst_x << " , " << obst_y << " car position " << x << " , " << y);
                }
            }
        }
    }
    ROS_INFO("MPCC::checkCollisionConstraints()");
}

void MPCC::checkConstraints()
{
    ROS_INFO("MPCC::checkConstraints()");
    // Check road cosntraints
    for (size_t N_iter = 0; N_iter < solver_interface_->FORCES_N; N_iter++)
    {

        double x = solver_interface_->x(N_iter);
        double y = solver_interface_->y(N_iter);
        double s = solver_interface_->spline(N_iter);

        double current_spline_0 = reference_path_.ss_[reference_path_.spline_index_];
        double current_spline_1 = reference_path_.ss_[reference_path_.spline_index_ + 1];
        double current_spline_2 = reference_path_.ss_[reference_path_.spline_index_ + 2];

        double path_x_0 = reference_path_.ref_path_x_.m_a[reference_path_.spline_index_] * std::pow((s - current_spline_0), 3) +
                          reference_path_.ref_path_x_.m_b[reference_path_.spline_index_] * std::pow((s - current_spline_0), 2) +
                          reference_path_.ref_path_x_.m_c[reference_path_.spline_index_] * (s - current_spline_0) +
                          reference_path_.ref_path_x_.m_d[reference_path_.spline_index_];

        double path_y_0 = reference_path_.ref_path_y_.m_a[reference_path_.spline_index_] * std::pow((s - current_spline_0), 3) +
                          reference_path_.ref_path_y_.m_b[reference_path_.spline_index_] * std::pow((s - current_spline_0), 2) +
                          reference_path_.ref_path_y_.m_c[reference_path_.spline_index_] * (s - current_spline_0) +
                          reference_path_.ref_path_y_.m_d[reference_path_.spline_index_];

        double path_dx_0 = reference_path_.ref_path_x_.m_a[reference_path_.spline_index_] * std::pow((s - current_spline_0), 2) +
                           reference_path_.ref_path_x_.m_b[reference_path_.spline_index_] * (s - current_spline_0) +
                           reference_path_.ref_path_x_.m_c[reference_path_.spline_index_];

        double path_dy_0 = reference_path_.ref_path_y_.m_a[reference_path_.spline_index_] * std::pow((s - current_spline_0), 2) +
                           reference_path_.ref_path_y_.m_b[reference_path_.spline_index_] * (s - current_spline_0) +
                           reference_path_.ref_path_y_.m_c[reference_path_.spline_index_];

        double path_x_1 = reference_path_.ref_path_x_.m_a[reference_path_.spline_index_ + 1] * std::pow((s - current_spline_1), 3) +
                          reference_path_.ref_path_x_.m_b[reference_path_.spline_index_ + 1] * std::pow((s - current_spline_1), 2) +
                          reference_path_.ref_path_x_.m_c[reference_path_.spline_index_ + 1] * (s - current_spline_1) +
                          reference_path_.ref_path_x_.m_d[reference_path_.spline_index_ + 1];

        double path_y_1 = reference_path_.ref_path_y_.m_a[reference_path_.spline_index_ + 1] * std::pow((s - current_spline_1), 3) +
                          reference_path_.ref_path_y_.m_b[reference_path_.spline_index_ + 1] * std::pow((s - current_spline_1), 2) +
                          reference_path_.ref_path_y_.m_c[reference_path_.spline_index_ + 1] * (s - current_spline_1) +
                          reference_path_.ref_path_y_.m_d[reference_path_.spline_index_ + 1];

        double path_dx_1 = reference_path_.ref_path_x_.m_a[reference_path_.spline_index_ + 1] * std::pow((s - current_spline_1), 2) +
                           reference_path_.ref_path_x_.m_b[reference_path_.spline_index_ + 1] * (s - current_spline_1) +
                           reference_path_.ref_path_x_.m_c[reference_path_.spline_index_ + 1];

        double path_dy_1 = reference_path_.ref_path_y_.m_a[reference_path_.spline_index_ + 1] * std::pow((s - current_spline_1), 2) +
                           reference_path_.ref_path_y_.m_b[reference_path_.spline_index_ + 1] * (s - current_spline_1) +
                           reference_path_.ref_path_y_.m_c[reference_path_.spline_index_ + 1];

        double path_x_2 = reference_path_.ref_path_x_.m_a[reference_path_.spline_index_ + 2] * std::pow((s - current_spline_2), 3) +
                          reference_path_.ref_path_x_.m_b[reference_path_.spline_index_ + 2] * std::pow((s - current_spline_2), 2) +
                          reference_path_.ref_path_x_.m_c[reference_path_.spline_index_ + 2] * (s - current_spline_2) +
                          reference_path_.ref_path_x_.m_d[reference_path_.spline_index_ + 2];

        double path_y_2 = reference_path_.ref_path_y_.m_a[reference_path_.spline_index_ + 2] * std::pow((s - current_spline_2), 3) +
                          reference_path_.ref_path_y_.m_b[reference_path_.spline_index_ + 2] * std::pow((s - current_spline_2), 2) +
                          reference_path_.ref_path_y_.m_c[reference_path_.spline_index_ + 2] * (s - current_spline_2) +
                          reference_path_.ref_path_y_.m_d[reference_path_.spline_index_ + 2];

        double path_dx_2 = reference_path_.ref_path_x_.m_a[reference_path_.spline_index_ + 2] * std::pow((s - current_spline_2), 2) +
                           reference_path_.ref_path_x_.m_b[reference_path_.spline_index_ + 2] * (s - current_spline_2) +
                           reference_path_.ref_path_x_.m_c[reference_path_.spline_index_ + 2];

        double path_dy_2 = reference_path_.ref_path_y_.m_a[reference_path_.spline_index_ + 2] * std::pow((s - current_spline_2), 2) +
                           reference_path_.ref_path_y_.m_b[reference_path_.spline_index_ + 2] * (s - current_spline_2) +
                           reference_path_.ref_path_y_.m_c[reference_path_.spline_index_ + 2];

        // Derive contouring and lagging errors
        double param_lambda = 1 / (1 + exp((s - current_spline_1 + 0.02) / 0.1));
        double param_lambda2 = 1 / (1 + exp((s - current_spline_2 + 0.02) / 0.1));

        double path_x = param_lambda * path_x_0 + param_lambda2 * (1 - param_lambda) * path_x_1 + (1 - param_lambda2) * path_x_2;
        double path_y = param_lambda * path_y_0 + param_lambda2 * (1 - param_lambda) * path_y_1 + (1 - param_lambda2) * path_y_2;
        double path_dx = param_lambda * path_dx_0 + param_lambda2 * (1 - param_lambda) * path_dx_1 + (1 - param_lambda2) * path_dx_2;
        double path_dy = param_lambda * path_dy_0 + param_lambda2 * (1 - param_lambda) * path_dy_1 + (1 - param_lambda2) * path_dy_2;

        double path_norm = std::sqrt(std::pow(path_dx, 2) + std::pow(path_dy, 2));
        double path_dx_normalized = path_dx / path_norm;
        double path_dy_normalized = path_dy / path_norm;

        for (int disc_it = 0; disc_it < controller_config_->n_discs_; disc_it++)
        {
            double disc_x = x_discs_[disc_it];

            std::array<double, 2> disc_pos;
            disc_pos[0] = cos(solver_interface_->psi(0)) * disc_x + x;
            disc_pos[1] = sin(solver_interface_->psi(0)) * disc_x + y;

            double road_boundary = -path_dy_normalized * (disc_pos[0] - path_x) + path_dx_normalized * (disc_pos[1] - path_y);
            ROS_INFO_STREAM("Countour error: " << road_boundary);
            double left_boundary = controller_config_->road_width_left_ - controller_config_->ego_w_ / 2.0;
            if (road_boundary - left_boundary > 0)
            {
                ROS_ERROR_STREAM("Left boundary not respected: " << N_iter << " Contour error: " << road_boundary);
            }
            double right_boundary = controller_config_->road_width_right_ - controller_config_->ego_w_ / 2.0;
            if (-road_boundary - right_boundary > 0)
            {
                ROS_ERROR_STREAM("Right boundary not respected: " << N_iter << " Contour error: " << road_boundary);
            }
        }
    }
    ROS_INFO("MPCC::checkConstraints()");
}

void MPCC::ControlLoop()
{
    static Helpers::Benchmarker control_benchmarker("Control loop");

    unsigned int N_iter;
    exit_code_ = 0;

    control_benchmarker.start();

    if (plan_ && (reference_path_.waypoints_size_ > 0))
    {

        // ROS_INFO("Initialise the initial solver states to the current state");
        solver_interface_->setInitialToState();

        // ROS_INFO("Check if the end of the current trajectory is reached (changed to smin! instead of xinit[5]");
        if (reference_path_.EndOfCurrentSpline(minimal_s_))
        {

            // If this is the end of the trajectory
            if (reference_path_.ReachedEnd())
            {

                ROS_DEBUG_STREAM("GOAL REACHED");

                goal_reached_ = true;
            }
            else
            {
                reference_path_.spline_index_++;
            }
        }

        // if(exit_code_ ==1)
        // minimal_s_ = solver_interface_->spline(0);

        reference_path_.UpdateClosestPoint(&(*solver_interface_), minimal_s_, window_size_, n_search_points_);

        bool over = reference_path_.ReachedEnd();

        if (over)
        {
            std_msgs::Bool msg;
            msg.data = over;
            path_is_over_pub_.publish(msg);
            enable_output_ = false;
            plan_ = false;
        }

        solver_interface_->setInitialState(5, minimal_s_);
        solver_interface_->spline(0) = minimal_s_;

        // CONSTRUCT A VECTOR OF POSES
        for (size_t k = 0; k < solver_interface_->FORCES_N; k++)
        {
            int k_actual = k == solver_interface_->FORCES_N - 1 ? k - 1 : k;
            orientation_vec_[k] = solver_interface_->psi(k_actual + 1);

            // Construct disc poses
            for (int i = 0; i < controller_config_->n_discs_; i++)
            {
                poses_vec_[i][k] = Eigen::Vector2d(
                    solver_interface_->x(k_actual + 1) + (x_discs_[i]) * std::cos(solver_interface_->psi(k_actual + 1)),
                    solver_interface_->y(k_actual + 1) + (x_discs_[i]) * std::sin(solver_interface_->psi(k_actual + 1)));
            }
        }

        if (solver_interface_->enable_scenario_constraints)
        {
            // Update the scenarios
            for (int i = 0; i < controller_config_->n_discs_; i++)
            {
                scenario_manager_[i]->update(poses_vec_[i], orientation_vec_);
            }
        }

        if (controller_config_->activate_debug_output_)
            ROS_INFO("LMPCC: Setting Parameters");

        for (N_iter = 0; N_iter < solver_interface_->FORCES_N; N_iter++)
        {
            // int k = N_iter*solver_interface_->FORCES_NPAR;

            solver_interface_->setParameter(N_iter, 0, reference_path_.ref_path_x_.m_a[reference_path_.spline_index_]); // spline coefficients
            solver_interface_->setParameter(N_iter, 1, reference_path_.ref_path_x_.m_b[reference_path_.spline_index_]);
            solver_interface_->setParameter(N_iter, 2, reference_path_.ref_path_x_.m_c[reference_path_.spline_index_]); // spline coefficients
            solver_interface_->setParameter(N_iter, 3, reference_path_.ref_path_x_.m_d[reference_path_.spline_index_]);
            solver_interface_->setParameter(N_iter, 4, reference_path_.ref_path_y_.m_a[reference_path_.spline_index_]); // spline coefficients
            solver_interface_->setParameter(N_iter, 5, reference_path_.ref_path_y_.m_b[reference_path_.spline_index_]);
            solver_interface_->setParameter(N_iter, 6, reference_path_.ref_path_y_.m_c[reference_path_.spline_index_]); // spline coefficients
            solver_interface_->setParameter(N_iter, 7, reference_path_.ref_path_y_.m_d[reference_path_.spline_index_]);

            solver_interface_->setParameter(N_iter, 8, reference_path_.ref_path_x_.m_a[reference_path_.spline_index_ + 1]); // spline coefficients
            solver_interface_->setParameter(N_iter, 9, reference_path_.ref_path_x_.m_b[reference_path_.spline_index_ + 1]);
            solver_interface_->setParameter(N_iter, 10, reference_path_.ref_path_x_.m_c[reference_path_.spline_index_ + 1]); // spline coefficients
            solver_interface_->setParameter(N_iter, 11, reference_path_.ref_path_x_.m_d[reference_path_.spline_index_ + 1]);
            solver_interface_->setParameter(N_iter, 12, reference_path_.ref_path_y_.m_a[reference_path_.spline_index_ + 1]); // spline coefficients
            solver_interface_->setParameter(N_iter, 13, reference_path_.ref_path_y_.m_b[reference_path_.spline_index_ + 1]);
            solver_interface_->setParameter(N_iter, 14, reference_path_.ref_path_y_.m_c[reference_path_.spline_index_ + 1]); // spline coefficients
            solver_interface_->setParameter(N_iter, 15, reference_path_.ref_path_y_.m_d[reference_path_.spline_index_ + 1]);

            solver_interface_->setParameter(N_iter, 16, reference_path_.ref_path_x_.m_a[reference_path_.spline_index_ + 2]); // spline coefficients
            solver_interface_->setParameter(N_iter, 17, reference_path_.ref_path_x_.m_b[reference_path_.spline_index_ + 2]);
            solver_interface_->setParameter(N_iter, 18, reference_path_.ref_path_x_.m_c[reference_path_.spline_index_ + 2]); // spline coefficients
            solver_interface_->setParameter(N_iter, 19, reference_path_.ref_path_x_.m_d[reference_path_.spline_index_ + 2]);
            solver_interface_->setParameter(N_iter, 20, reference_path_.ref_path_y_.m_a[reference_path_.spline_index_ + 2]); // spline coefficients
            solver_interface_->setParameter(N_iter, 21, reference_path_.ref_path_y_.m_b[reference_path_.spline_index_ + 2]);
            solver_interface_->setParameter(N_iter, 22, reference_path_.ref_path_y_.m_c[reference_path_.spline_index_ + 2]); // spline coefficients
            solver_interface_->setParameter(N_iter, 23, reference_path_.ref_path_y_.m_d[reference_path_.spline_index_ + 2]);

            solver_interface_->setParameter(N_iter, 24, reference_path_.ss_[reference_path_.spline_index_]);            // s1
            solver_interface_->setParameter(N_iter, 25, reference_path_.ss_[reference_path_.spline_index_ + 1]);        // s2
            solver_interface_->setParameter(N_iter, 26, reference_path_.ss_[reference_path_.spline_index_ + 2]);        // s2
            solver_interface_->setParameter(N_iter, 27, reference_path_.ss_[reference_path_.spline_index_ + 1] + 0.02); // d

            solver_interface_->setParameter(N_iter, 28, Wcontour_);
            solver_interface_->setParameter(N_iter, 29, Wlag_);   // weight lag error
            solver_interface_->setParameter(N_iter, 30, Ka_);     // weight acceleration
            solver_interface_->setParameter(N_iter, 31, Kdelta_); // weight delta
            solver_interface_->setParameter(N_iter, 32, reference_velocity_);
            solver_interface_->setParameter(N_iter, 33, slack_weight_);     // slack weight
            solver_interface_->setParameter(N_iter, 34, repulsive_weight_); // repulsive weight
            solver_interface_->setParameter(N_iter, 35, velocity_weight_);  // repulsive weight
            solver_interface_->setParameter(N_iter, 36, lateral_weight_);   // repulsive weight

            solver_interface_->setParameter(N_iter, 37, r_discs_);    // radius of the disks
            solver_interface_->setParameter(N_iter, 38, x_discs_[0]); // position of the car discs
            solver_interface_->setParameter(N_iter, 39, x_discs_[1]); // position of the car discs
            solver_interface_->setParameter(N_iter, 40, x_discs_[2]); // position of the car discs

            // Define an offset index
            int start_offset = 40; //+ controller_config_->max_obstacles_ * 5;
            int after_offset = 0;  // use + 1 from this to get the next index
            // ROS_WARN_STREAM("x_discs_[0]" << x_discs_[0] << "x_discs_[1]" << x_discs_[1] << "x_discs_[2]" << x_discs_[2]);

            if (solver_interface_->enable_ellipsoid_constraints)
            {
                for (int obst_it = 0; obst_it < controller_config_->max_obstacles_; obst_it++)
                {
                    solver_interface_->setParameter(N_iter, 41 + obst_it * 5, system_interface_->obstacles_.lmpcc_obstacles[obst_it].trajectory.poses[N_iter].pose.position.x);
                    solver_interface_->setParameter(N_iter, 42 + obst_it * 5, system_interface_->obstacles_.lmpcc_obstacles[obst_it].trajectory.poses[N_iter].pose.position.y);
                    solver_interface_->setParameter(N_iter, 43 + obst_it * 5, Helpers::quaternionToAngle(system_interface_->obstacles_.lmpcc_obstacles[obst_it].trajectory.poses[N_iter].pose));
                    solver_interface_->setParameter(N_iter, 44 + obst_it * 5, system_interface_->obstacles_.lmpcc_obstacles[obst_it].major_semiaxis[N_iter]);
                    solver_interface_->setParameter(N_iter, 45 + obst_it * 5, system_interface_->obstacles_.lmpcc_obstacles[obst_it].minor_semiaxis[N_iter]);
                }
                after_offset += controller_config_->max_obstacles_ * 5 + start_offset;
            }
            if (solver_interface_->enable_scenario_constraints)
            {
                // Insert scenario constraints
                for (int i = 0; i < controller_config_->n_discs_; i++)
                {
                    scenario_manager_[i]->insertConstraints(&(*solver_interface_), N_iter, start_offset, after_offset);
                    start_offset = after_offset;
                }
            }
            // ROS_WARN_STREAM("after_offset + 1: " << after_offset + 1);
            solver_interface_->setParameter(N_iter, after_offset + 1, controller_config_->road_width_left_ - controller_config_->ego_w_ / 2.0);
            solver_interface_->setParameter(N_iter, after_offset + 2, controller_config_->road_width_right_ - controller_config_->ego_w_ / 2.0);
        }

        if (controller_config_->activate_debug_output_)
            ROS_INFO("LMPCC: Solving Optimization");

        broadcastPathPose();
        // @Bruno: This uses the shifted plan to initialize -> setting it to false reuses the data is had before.
        // solver_interface_->setReinitialize(true);
        exit_code_ = solver_interface_->solve();

        // If the solver gives an error
        if (exit_code_ != 1)
        {
            checkConstraints();
            checkCollisionConstraints();
            std::cout << "========== State ==========\n"
                      << "x = " << solver_interface_->x(0) << "\n"
                      << "y = " << solver_interface_->y(0) << "\n"
                      << "psi = " << solver_interface_->psi(0) << "\n"
                      << "v = " << solver_interface_->v(0) << "\n"
                      << "delta = " << solver_interface_->delta(0) << "\n"
                      << "slack = " << solver_interface_->slack() << "\n"
                      << "spline = " << solver_interface_->spline(0) << "\n"
                      << "a = " << solver_interface_->a() << "\n"
                      << "w = " << solver_interface_->w() << "\n"
                      << "============================\n";

            ROS_WARN_STREAM("Solver parameters \n");
            for (size_t i = 0; i < solver_interface_->FORCES_NPAR; i++)
            {
                ROS_WARN_STREAM("Parameter " << i << " : " << solver_interface_->getParameter(i) << "\n");
            }

            setSolverToCurrentState();
            // Initialize the spline (and find the closest point to the vehicle)
            // reference_path_.InitializeClosestPoint(&(*solver_interface_));

            if (exit_code_ == -7)
            {
                ROS_ERROR("OPTIMISATION FAILED DUE TO WRONG DATA INSERTION\nCHECK YOUR PARAMETERS!");
            }
            else if (exit_code_ == -6)
            {

                ROS_ERROR("OPTIMISATION WAS INFEASIBLE!");
            }
            else if (exit_code_ == 0)
            {
                ROS_ERROR("MAXIMUM NUMBER OF INTERATIONS!");
            }
            system_interface_->ActuateBrake(1.5);
        }
        else
        {
            if (controller_config_->activate_debug_output_)
                ROS_INFO("Inserting the predicted trajectory for the next solver iteration");

            solver_interface_->insertPredictedTrajectory();

            // publish the control command
            if (enable_output_)
                system_interface_->Actuate();
            else
                system_interface_->ActuateBrake(2.5);

            if (controller_config_->activate_debug_output_)
                ROS_INFO("publish the predicted collision space");
        }

        publishPredictedCollisionSpace();

        std_msgs::Float64 feasibility_msg;
        feasibility_msg.data = exit_code_;
        publishPredictedOutput();
        feasibility_pub_.publish(feasibility_msg);
    }
    else
    {
        system_interface_->ActuateBrake(2.5);
    }

    // std_msgs::Float64 compute_msg;
    // compute_msg.data = control_benchmarker.getDuration();
    // computation_pub_.publish(compute_msg);
    broadcastPathPose();
    reference_path_.plotRoad();
    reference_path_.PublishSpline();
    control_benchmarker.stop();
}

void MPCC::setSolverToCurrentState()
{

    // for (int i = 0; i < solver_interface_->FORCES_N * solver_interface_->FORCES_TOTAL_V; i++)
    // {
    //     solver_interface_->forces_params_.x0[i] = 0.0;
    // }
    if (controller_config_->activate_debug_output_)
        ROS_INFO("Optimization was infeasible, initializing solver variables with the measured state");

    solver_interface_->resetSolver();
    solver_interface_->resetAtInfeasible(minimal_s_);
}

double MPCC::getVelocityReference(int k)
{
    double min_velocity = controller_config_->min_velocity_; // reference_velocity_; (to disable!)

    double max_area = std::pow(controller_config_->polygon_range_ * 2.0, 2.0);
    double min_area = max_area;

    for (int i = 0; i < controller_config_->n_discs_; i++)
    {
        double cur_area = scenario_manager_[i]->getArea(k);
        if (cur_area < min_area)
        {
            min_area = cur_area;
        }
    }

    double ratio = min_area / max_area;

    return min_velocity + (reference_velocity_ - min_velocity) * ratio;
}

bool MPCC::ResetCallBack(lmpcc::LMPCCReset::Request &req, lmpcc::LMPCCReset::Response &res)
{

    ROS_INFO("ResetCallBack");

    OnReset();
    // plan_ = false;
    // enable_output_ = false;
    // state_received_ = false;

    // reset_solver();

    // // for (int j = 0; j < 10; j++)
    // // {
    // //     system_interface_->ActuateBrake(0.0);
    // //     ros::Duration(0.1).sleep();
    // // }

    // system_interface_->Reset(reset_pose_);

    // solver_interface_->getState()->init();
    // ros::Duration(1.0).sleep();

    // goal_reached_ = false;

    // if (controller_config_->sync_mode_)
    //     timer_.start();

    // enable_output_ = true;

    return true;
}

void MPCC::OnReset()
{

    ros::Duration(2.0).sleep();

    if (controller_config_->activate_debug_output_)
        ROS_INFO("MPCC::OnReset()");

    // Disable planning / output
    plan_ = false;
    enable_output_ = false;

    // Stop the timer
    if (controller_config_->sync_mode_)
        timer_.stop();

    // Reset solver variables
    reset_solver();

    // Reinitializes the spline (Note: closest point on the spline will be found in the state callback when state_received == false)
    reference_path_.InitPath();
    minimal_s_ = 0;

    // std::cout << "calling system interface " << std::endl;

    // Initialize the spline (and find the closest point to the vehicle)
    reference_path_.InitializeClosestPoint(&(*solver_interface_));

    // Initialize the state of the solver
    // solver_interface_->spline(0) = 0.0;
    // solver_interface_->setInitialState(5, 0);
    solver_interface_->getState()->init();

    // ros::Duration(1.0).sleep();

    // Will trigger a state update before the controller starts
    state_received_ = false;

    //
    goal_reached_ = false;

    ROS_ERROR("eXIT OnReset");
    setSolverToCurrentState();

    // Start the timer again
    if (controller_config_->sync_mode_)
        timer_.start();
}

void MPCC::reconfigureCallback(lmpcc::PredictiveControllerConfig &config, uint32_t level)
{

    ROS_INFO("reconfigure callback!");

    // Read parameters
    Wcontour_ = config.Wcontour;
    Wlag_ = config.Wlag;
    Ka_ = config.Ka;
    Kalpha_ = config.Kalpha;
    Kdelta_ = config.Kdelta;

    lateral_weight_ = config.Wlateral;
    velocity_weight_ = config.Kv;

    slack_weight_ = config.Wslack;
    repulsive_weight_ = config.Wrepulsive;

    reference_velocity_ = config.vRef;
    simulated_velocity_ = config.ini_v0;

    // reset world
    reset_world_ = config.reset_world;
    reset_pose_.position.x = config.reset_x;
    reset_pose_.position.y = config.reset_y;

    if (reset_world_)
    {
        OnReset();
        config.reset_world = false;
        // timer_.stop();
        // config.reset_world = false;
        // config.plan = false;
        // config.enable_output = false;

        // reset_solver();

        // goal_reached_ = false;
        // reset_world_ = false;

        // state_received_ = false;

        // system_interface_->Reset(reset_pose_);

        // config.plan = true;
        // config.enable_output = true;

        // timer_.start();
    }

    // Search window parameters
    window_size_ = config.window_size;
    n_search_points_ = config.n_search_points;

    if (auto_enable_)
    {
        ROS_WARN("Auto Planning...");
        config.plan = true;
        config.enable_output = true;
        auto_enable_ = false;
    }

    if (reference_path_.waypoints_size_ > 1)
    {

        // If we auto enabled planning, set the reconfigure to be true

        // Take the plan values from the reconfigure
        ROS_WARN("Planning...");
        plan_ = config.plan;
        enable_output_ = config.enable_output;
    }
    else
    {
        ROS_WARN("No waypoints were provided...");

        config.plan = false;
        config.enable_output = false;
        plan_ = false;
        enable_output_ = false;
    }

    if (plan_ && !reset_world_)
    {
        ROS_INFO("Resetting solver");

        // Reset the solver
        reset_solver();
        setSolverToCurrentState(); // Instead wait for the state callback to fire!

        // Finds the closest points to the current vehicle state
        // reference_path_.InitializeClosestPoint(&(*solver_interface_));

        goal_reached_ = false;
        timer_.start();
    }
    else if (!plan_)
    {
        timer_.stop();
    }
}

void MPCC::JoyCallBack(const sensor_msgs::Joy::ConstPtr &msg)
{

    if (msg->buttons[2] == 1.0)
    {
        plan_ = false;
        enable_output_ = false;
    }
    if (msg->buttons[1] == 1.0)
    {
        plan_ = true;
        enable_output_ = true;
    }
}

void MPCC::VRefCallBack(const std_msgs::Float64::ConstPtr &msg)
{

    reference_velocity_ = msg->data;
    ROS_DEBUG_STREAM("Received velocity reference: " << reference_velocity_);

    reference_velocity_ = std::min(std::max(reference_velocity_, 0.0), 10.0);
}

// On Received functions are executed after the callback is received at the interface
void MPCC::OnObstaclesReceived()
{
    if (controller_config_->activate_debug_output_)
        ROS_INFO("MPCC: OnObstaclesReceived()");

    if (solver_interface_->enable_scenario_constraints)
    {
        for (int i = 0; i < controller_config_->n_discs_; i++)
            scenario_manager_[i]->predictionCallback(system_interface_->obstacles_);
    }
}

void MPCC::OnStateReceived()
{
    if (controller_config_->activate_debug_output_)
        ROS_INFO("MPCC: OnStateReceived()");

    // Check if the state jumped
    bool state_jumped = (Helpers::dist(Eigen::Vector2d(solver_interface_->x(0), solver_interface_->y(0)), Eigen::Vector2d(prev_x_, prev_y_)) > 5.0);

    prev_x_ = solver_interface_->x(0);
    prev_y_ = solver_interface_->y(0);

    // If the first state was received or the state jumped
    if (!state_received_ || state_jumped)
    {
        if (controller_config_->activate_debug_output_)
            ROS_WARN("MPCC: Jump in state detected, recomputing path!");

        state_received_ = true;

        // Initialize the spline (and find the closest point to the vehicle)
        reference_path_.InitializeClosestPoint(&(*solver_interface_));

        // Reset the solver to use the current state
        setSolverToCurrentState();

        if (controller_config_->auto_enable_plan_)
        {
            plan_ = true;
            enable_output_ = true;
        }
    }
}

void MPCC::OnWaypointsReceived()
{
    // Initialize the reference path with the received parameters at the interface
    // broadcastPathPose();
    reference_path_.InitPath(system_interface_->x_, system_interface_->y_, system_interface_->theta_);

    // Reenable planning and the output
    plan_ = true;
    enable_output_ = true;
}

void MPCC::publishPredictedTrajectory(void)
{
    for (size_t i = 0; i < solver_interface_->FORCES_N; i++)
    {
        pred_traj_.poses[i].pose.position.x = solver_interface_->x(i);
        pred_traj_.poses[i].pose.position.y = solver_interface_->y(i);
        pred_traj_.poses[i].pose.orientation.z = solver_interface_->psi(i);
    }

    pred_traj_pub_.publish(pred_traj_);
}

void MPCC::publishPredictedOutput(void)
{

    for (size_t i = 0; i < solver_interface_->FORCES_N; i++)
    {
        if (exit_code_ == 1)
            pred_cmd_.poses[i].pose.position.x = solver_interface_->v(i);
        else
            pred_cmd_.poses[i].pose.position.x = 0.0;
    }

    pred_cmd_pub_.publish(pred_cmd_);
}

void MPCC::publishPredictedCollisionSpace(void)
{
    visualization_msgs::MarkerArray collision_space;
    for (int k = 0; k < controller_config_->n_discs_; k++)
    {

        for (size_t i = 0; i < solver_interface_->FORCES_N; i++)
        {
            ellips1.id = 60 + i + k * solver_interface_->FORCES_N;

            ellips1.pose.position.x = poses_vec_[k][i](0);
            ellips1.pose.position.y = poses_vec_[k][i](1);
            ellips1.pose.position.z = 0.3;
            ellips1.pose.orientation.x = 0;
            ellips1.pose.orientation.y = 0;
            ellips1.pose.orientation.z = 0;
            ellips1.pose.orientation.w = 1;
            if (exit_code_ == 1)
            {
                ellips1.color.b = 1.0;
                ellips1.color.g = 0.0;
                ellips1.color.r = 0.0;
            }
            else
            {
                ellips1.color.b = 0.0;
                ellips1.color.g = 0.0;
                ellips1.color.r = 1.0;
            }
            collision_space.markers.push_back(ellips1);
        }
    }
    robot_collision_space_pub_.publish(collision_space);
}

// Utils

bool MPCC::transformPose(const std::string &from, const std::string &to, geometry_msgs::Pose &pose)
{
    bool transform = false;
    tf::StampedTransform stamped_tf;
    // ROS_DEBUG_STREAM("Transforming from :" << from << " to: " << to);
    geometry_msgs::PoseStamped stampedPose_in, stampedPose_out;
    // std::cout << "from " << from << " to " << to << ", x = " << pose.position.x << ", y = " << pose.position.y << std::endl;
    stampedPose_in.pose = pose;
    // std::cout << " value: " << std::sqrt(std::pow(pose.orientation.x, 2.0) + std::pow(pose.orientation.y, 2.0) + std::pow(pose.orientation.z, 2.0) + std::pow(pose.orientation.w, 2.0)) << std::endl;
    if (std::sqrt(std::pow(pose.orientation.x, 2) + std::pow(pose.orientation.y, 2) + std::pow(pose.orientation.z, 2) + std::pow(pose.orientation.w, 2)) < 1.0 - 1e-9)
    {
        stampedPose_in.pose.orientation.x = 0;
        stampedPose_in.pose.orientation.y = 0;
        stampedPose_in.pose.orientation.z = 0;
        stampedPose_in.pose.orientation.w = 1;
        // std::cout << "LMPCC: Quaternion was not normalised properly!" << std::endl;
    }
    //    stampedPose_in.header.stamp = ros::Time::now();
    stampedPose_in.header.frame_id = from;

    // make sure source and target frame exist
    if (tf_listener_.frameExists(to) && tf_listener_.frameExists(from))
    {
        try
        {
            // std::cout << "in transform try " << std::endl;
            // find transforamtion between souce and target frame
            // tf_listener_.waitForTransform(from, to, ros::Time(0), ros::Duration(0.02));
            tf_listener_.transformPose(to, stampedPose_in, stampedPose_out);

            transform = true;
        }
        catch (tf::TransformException &ex)
        {
            ROS_ERROR("MPCC::getTransform: %s", ex.what());
        }
    }
    else
    {
        ROS_WARN("MPCC::getTransform: '%s' or '%s' frame doesn't exist, pass existing frame", from.c_str(), to.c_str());
    }
    pose = stampedPose_out.pose;
    stampedPose_in.pose = stampedPose_out.pose;
    stampedPose_in.header.frame_id = to;

    return transform;
}

void MPCC::broadcastPathPose()
{
    // Plots a Axis to display the path variable location
    geometry_msgs::TransformStamped transformStamped;
    transformStamped.header.stamp = ros::Time::now();
    transformStamped.header.frame_id = controller_config_->target_frame_;
    transformStamped.child_frame_id = "path";

    transformStamped.transform.translation.x = reference_path_.ref_path_x_(solver_interface_->spline(0));
    transformStamped.transform.translation.y = reference_path_.ref_path_y_(solver_interface_->spline(0));
    transformStamped.transform.translation.z = 0.0;
    // tf::Quaternion q = tf::createQuaternionFromRPY(0, 0, 0);
    transformStamped.transform.rotation.x = 0;
    transformStamped.transform.rotation.y = 0;
    transformStamped.transform.rotation.z = 0;
    transformStamped.transform.rotation.w = 1;

    path_pose_pub_.sendTransform(transformStamped);

    nav_msgs::Path close_waypoints;
    geometry_msgs::PoseStamped initial, end;

    close_waypoints.header.stamp = ros::Time::now();
    close_waypoints.header.frame_id = controller_config_->target_frame_;

    initial.pose.position.x = reference_path_.ref_path_x_(solver_interface_->spline(0));
    initial.pose.position.y = reference_path_.ref_path_y_(solver_interface_->spline(0));
    end.pose.position.x = reference_path_.ref_path_x_(solver_interface_->spline(0) + 5);
    end.pose.position.y = reference_path_.ref_path_y_(solver_interface_->spline(0) + 5);

    close_waypoints.poses.push_back(initial);
    close_waypoints.poses.push_back(end);

    close_path_points_pub_.publish(close_waypoints);
}

void MPCC::broadcastTF()
{
    // Only used for perfect state simulation
    geometry_msgs::TransformStamped transformStamped;
    transformStamped.header.stamp = ros::Time::now();
    transformStamped.header.frame_id = controller_config_->target_frame_;
    transformStamped.child_frame_id = controller_config_->robot_base_link_;

    transformStamped.transform.translation.x = solver_interface_->getState()->get_x();
    transformStamped.transform.translation.y = solver_interface_->getState()->get_y();
    transformStamped.transform.translation.z = 0.0;

    tf::Quaternion q = tf::createQuaternionFromRPY(0, 0, solver_interface_->getState()->get_psi());
    transformStamped.transform.rotation.x = q.x();
    transformStamped.transform.rotation.y = q.y();
    transformStamped.transform.rotation.z = q.z();
    transformStamped.transform.rotation.w = q.w();

    state_pub_.sendTransform(transformStamped);

    sensor_msgs::JointState empty;
    empty.position.resize(7);
    empty.name = {"rear_right_wheel_joint", "rear_left_wheel_joint", "front_right_wheel_joint", "front_left_wheel_joint", "front_right_steer_joint", "front_left_steer_joint", "steering_joint"};
    empty.header.stamp = ros::Time::now();
    joint_state_pub_.publish(empty);
}
