#include "scenario/scenario_manager.h"

ScenarioManager::ScenarioManager(ros::NodeHandle &nh, predictive_configuration *config, bool enable_visualization = false)
    : config_(config), enable_visualization_(enable_visualization)
{
    ROS_WARN("INITIALIZING SCENARIO MANAGER");
    // Initialise the visualisation
    ros_markers_.reset(new ROSMarkerPublisher(nh, "scenario_constraints/markers", config_->target_frame_, 1800));

    // Save some useful variables
    S = config_->sample_count_;
    B = config_->batch_count_;
    N = config_->N_;

    R_ = config_->removal_count_;
    l_ = config_->polygon_checked_constraints_;

    // Resize over the horizon
    scenario_constraints_.resize(N);
    projected_poses_.resize(N);
    scenario_threads_.resize(N);

    // Selection vectors
    closest_scenarios_.resize(N);
    for (uint k = 0; k < N; k++){
        closest_scenarios_[k].resize(config_->max_obstacles_);

        for (int v = 0; v < config_->max_obstacles_; v++)
            closest_scenarios_[k][v].resize(l_ + R_);
    }

    // Selection vectors
    closest_static_.resize(N);
    for (uint k = 0; k < N; k++)
    {
        closest_static_[k].resize(1);
        closest_static_[k][0].resize(l_);
    }

    int obstacle_size = config_->max_obstacles_;
    obstacles_.resize(obstacle_size);
    
    
    obstacles_msg_.lmpcc_obstacles.resize(obstacle_size);
    
    // Initialise in case of disabled
    for (int obst_it = 0; obst_it < config_->max_obstacles_; obst_it++)
    {
        obstacles_msg_.lmpcc_obstacles[obst_it].trajectory.poses.resize(N);
        obstacles_msg_.lmpcc_obstacles[obst_it].major_semiaxis.resize(N);
        obstacles_msg_.lmpcc_obstacles[obst_it].minor_semiaxis.resize(N);
        obstacles_msg_.lmpcc_obstacles[obst_it].id = obst_it;
        for (size_t t = 0; t < N; t++)
        {
            obstacles_msg_.lmpcc_obstacles[obst_it].trajectory.poses[t].pose.position.x = 50; // Lowered...
            obstacles_msg_.lmpcc_obstacles[obst_it].trajectory.poses[t].pose.position.y = 50;
            obstacles_msg_.lmpcc_obstacles[obst_it].trajectory.poses[t].pose.orientation.z = 0;
            obstacles_msg_.lmpcc_obstacles[obst_it].major_semiaxis[t] = 0.1;
            obstacles_msg_.lmpcc_obstacles[obst_it].minor_semiaxis[t] = 0.1;
        }
    }

    for(int o = 0; o < obstacle_size; o++)
        obstacles_[o].initialise(config);

    static_obstacles_.initialise(config);

    active_obstacle_count_.resize(N, 0);
    active_obstacle_indices_.resize(N);
    for (uint k = 0; k < N; k++)
        active_obstacle_indices_[k].resize(obstacle_size);

    // Initialise the possible constraints
    possible_constraints_.resize(N);
    for(size_t k = 0; k < N; k++){

        // Possible constraints is just 1D per stage but for all vrus concatenated
        possible_constraints_[k].resize(l_ * obstacle_size + l_);

        for (size_t i = 0; i < possible_constraints_[k].size(); i++)
        {
            possible_constraints_[k][i].A_ = Eigen::Matrix<double, 1, 2>();
            possible_constraints_[k][i].b_ = Eigen::Matrix<double, 1, 1>();
        }
    }

    // Initialize polygon constructors
    for(u_int k = 0; k < N; k++)
        polygon_constructors_.emplace_back(config_->polygon_range_, l_, config_->inner_approximation_);

    areas_.resize(N);
    for(u_int k = 0; k < N; k++)
        areas_[k] = 0.0;

    // Variable initialisation
    combined_radius_ = config_->r_VRU_ + config_->ego_w_ / 2.0;

    ROS_INFO("SCENARIO MANAGER INITIALIZED");
}

void ScenarioManager::update(const std::vector<Eigen::Vector2d> &poses, const std::vector<double>& orientations) //const FORCESNLPsolver_params &forces_params)
{
    if (config_->activate_debug_output_)
        ROS_INFO("Scenario Manager: Constructing Constraints");

    // static Helpers::Benchmarker full_benchmarker("ScenarioManager::update()");

    // if (obstacles_msg_.lmpcc_obstacles.size() == 0)
    // {
    //     ROS_WARN("No obstacles received yet!");
    //     return;
    // }
    // full_benchmarker.start();

    if (config_->activate_debug_output_)
        ROS_INFO("Scenario Manager:     -> Converting Scenarios to Constraints");

    if (config_->multithread_scenarios_)
    {
        // multithread
        for(size_t k = 0; k < N; k++){
            // Create a thread for stage k
            scenario_threads_[k] = std::thread(&ScenarioManager::scenariosToConstraints, this, k, poses[k], orientations[k]);
        }
        // Wait for all threads to finish
        for (auto &t : scenario_threads_)
            t.join();
    }
    else
    {
        //single thread
        for(size_t k = 0; k < N; k++){
            scenariosToConstraints(k, poses[k], orientations[k]);
        }
    }

    if (config_->activate_debug_output_)
        ROS_INFO("Scenario Manager:     -> Done.");

    // full_benchmarker.stop();

    if(enable_visualization_){

        // Visualize the ellipsoid constraint
        if (config_->draw_ellipsoids_)
            visualiseEllipsoidConstraints();

        // Visualise scenarios
        if (config_->draw_all_scenarios_)
            visualiseScenarios(config_->indices_to_draw_);

        // IS BROKEN ATM!
        if (config_->draw_selected_scenarios_)
            visualiseSelectedScenarios(config_->indices_to_draw_);

        if (config_->draw_removed_scenarios_)
            visualiseRemovedScenarios(config_->indices_to_draw_);

        // visualiseProjectedPosition();
        if (config_->draw_constraints_){
            // Draw the polygons using the lines from the constructors
            visualisePolygons();
        }

        publishVisuals();
    }
}

// Constructs N x m linear constraints for all stages and the given scenarios_ (result = scenario_constraints_)
void ScenarioManager::scenariosToConstraints(int k, const Eigen::Vector2d &vehicle_pose, double orientation)
{

    // Make a local copy that changes
    Eigen::Vector2d pose = vehicle_pose;

    //------------------- Verify the relevance of obstacles ----------------------------------//
    active_obstacle_count_[k] = 0;

    // Check all obstacles
    for (uint v = 0; v < config_->max_obstacles_; v++)
    {
        // Get the position at this stage
        lmpcc_msgs::lmpcc_obstacle cur_obstacle_msg = obstacles_[v].getPrediction();

        Eigen::Vector2d obst(cur_obstacle_msg.trajectory.poses[k].pose.position.x,
                             cur_obstacle_msg.trajectory.poses[k].pose.position.y);
        // If this obstacle is relevant
        if (Helpers::dist(obst, vehicle_pose) <= 10.0)
        {
            // Save that this obstacle is active for this stage, increase the count
            active_obstacle_indices_[k][active_obstacle_count_[k]] = v;
            active_obstacle_count_[k]++;
        }
    }

    //--------------------- Compute the distances from scenario to vehicle ----------------------//
    // For all active obstacles
    for (int v = 0; v < active_obstacle_count_[k]; v++)
    {
        // Get the index of this obstacle
        int obst = active_obstacle_indices_[k][v];

        // Project the pose outwards from this VRU if necessary (vehicle is inside the scenarios)
        obstacles_[obst].projectOutwards(k, pose); // Check if this is okay!!

        // Compute for all scenarios the length of the vector from the vehicle to the VRU
        obstacles_[obst].computeDistancesTo(k, pose);
    }

    // Save the projected pose
    projected_poses_[k] = pose;

    //--------------------- Find closest scenarios ----------------------//
    // For all active obstacles
    for (int v = 0; v < active_obstacle_count_[k]; v++)
    {
        // Get the index of this obstacle
        int obst = active_obstacle_indices_[k][v];

        // Initialise at high / non-occupied values
        for (int i = 0; i < l_ + R_; i++)
            closest_scenarios_[k][obst][i].init(); // Structure that identifies an obstacle, its sample index and the corresponding distance

        // Initialise the minimum distance
        double min_dist = BIG_NUMBER;

        for (int s = 0; s < obstacles_[obst].getPrunedSampleCount(k); s++) // CHANGED FROM S_pruned
        {
            if (obstacles_msg_.lmpcc_obstacles.size() == 0)
            {
                ROS_WARN("No obstacles received!");
                return;
            }

            // If this scenario is closer than the R+l th of this VRU, save it
            if(obstacles_[obst].getDistance(k, s) < min_dist){

                sortedInsert(k, obstacles_[obst].getDistance(k, s), obst, s);

                min_dist = closest_scenarios_[k][obst][l_ + R_ - 1].distance;
            }
        }

        // Sort on distance to the mean!
        removeScenariosBySorting(closest_scenarios_[k][obst]);

        //--------------------- Construct constraints for all non-removed points ----------------------//
        // For all l considered scenarios
        for (int i = 0; i < l_; i++)
        {
            // Create a hyperplane
            hyperplaneFromScenario(
                vehicle_pose,
                obstacles_[obst].getScenarioRef(k, closest_scenarios_[k][obst][R_ + i].scenario_index),
                possible_constraints_[k][v * l_ + i], /* saved here */
                k,
                closest_scenarios_[k][obst][R_ + i],
                0);
        }        
    }

    //--------------------- Handle static obstacles ----------------------------//
    // Filtering if necessary
    if (static_obstacles_.getScenarioCount() > 0 && config_->static_obstacles_enabled_)
    {
        if (static_obstacles_.getScenarioCount() > l_){

            static_obstacles_.computeDistancesTo(k, vehicle_pose);

            // Init
            for (int i = 0; i < l_; i++){
                closest_static_[k][0][i].init();
            }
            // Initialise the minimum distance
            double min_dist = BIG_NUMBER;

            for (int s = 0; s < static_obstacles_.getScenarioCount(); s++) // CHANGED FROM S_pruned
            {

                // If this scenario is closer than the R+l th of this VRU, save it
                if (static_obstacles_.getDistance(k, s) < min_dist)
                {

                    sortedInsertStatic(k, static_obstacles_.getDistance(k, s), s);

                    min_dist = closest_static_[k][0][l_ - 1].distance;
                }
            }
        }else{
            // Simply load it
            for(int i = 0; i < static_obstacles_.getScenarioCount(); i++){
                closest_static_[k][0][i].distance = static_obstacles_.getDistance(k, i); // Distance is not set correctly, but it doesn't matter
                closest_static_[k][0][i].scenario_index = i;
            }
        }

        for (size_t i = 0; i < closest_static_[k][0].size(); i++)
        {
            // Create a hyperplane
            hyperplaneFromScenario(
                vehicle_pose,
                static_obstacles_.getScenarioRef(closest_static_[k][0][i].scenario_index),
                possible_constraints_[k][active_obstacle_count_[k] * l_ + i], /* saved here */
                k,
                closest_static_[k][0][i],
                0);
        }
    }

    //--------------------- Construct the minimal polygon ----------------------//
    // Define outputs
    std::vector<LinearConstraint2D> polygon_constraints_out = {};
    std::vector<int> polygon_indices = {};

    polygon_constraints_out.clear();
    polygon_indices.clear();

    // Insert data for the search
    polygon_constructors_[k].insertData(
        &possible_constraints_[k],
        vehicle_pose,
        orientation,
        active_obstacle_count_[k] * l_ + std::min(static_obstacles_.getScenarioCount(), l_));

    // Find the minimal polygon
    polygon_constructors_[k].findPolygon(polygon_constraints_out, polygon_indices);

    // If our specified edge size maximum is surpassed, give an error output
    if(polygon_constraints_out.size() > 20 + 4)
        ROS_ERROR_STREAM("[Stage " << k << "] Found a polygon with " << polygon_constraints_out.size() << " edges, exceeding the limit of 24!");

    areas_[k] = polygon_constructors_[k].getArea();

    //--------------------- Save the constraints ----------------------//

    // Initialise the stage constraints to use
    LinearConstraint2D &stage_constraint = scenario_constraints_[k];

    // Reset A and b, to take the shape of the constraint size
    stage_constraint.A_ = Eigen::MatrixXd::Zero(polygon_constraints_out.size(), 2);
    stage_constraint.b_ = Eigen::VectorXd::Zero(polygon_constraints_out.size());
    //stage_constraint = LinearConstraint2D(stage_constraint.A_, stage_constraint.b_);

    // For the received number of hyperplanes
    for (u_int i = 0; i < polygon_constraints_out.size(); i++)
    {
        // Insert the hyperplanes
        stage_constraint.A_.block<1, 2>(i, 0) = polygon_constraints_out[i].A_;
        stage_constraint.b_(i) = polygon_constraints_out[i].b_(0);
    }
}

void ScenarioManager::sortedInsert(const int &k, const double &new_distance, const int &v, const int &s)
{
    // Find out where the new one goes
    // Start from the back (highest), iterate to the newest (front)
    for (int j = closest_scenarios_[k][v].size() - 1; j >= 0; j--)
    {
        // If we are already at the front or this one is not lower than the next one
        if (j == 0 || new_distance >= closest_scenarios_[k][v][j - 1].distance)
        {

            // Start at the the second to last index
            for (int z = closest_scenarios_[k][v].size() - 2; z >= j; z--)
                closest_scenarios_[k][v][z + 1] = closest_scenarios_[k][v][z];

            // Insert the new one
            closest_scenarios_[k][v][j].distance = new_distance;
            closest_scenarios_[k][v][j].obstacle_index = v;
            closest_scenarios_[k][v][j].scenario_index = s;

            break;
        }
    }
}

void ScenarioManager::sortedInsertStatic(const int &k, const double &new_distance, const int &s)
{
    // Find out where the new one goes
    // Start from the back (highest), iterate to the newest (front)
    for (int j = closest_static_[k][0].size() - 1; j >= 0; j--)
    {
        // If we are already at the front or this one is not lower than the next one
        if (j == 0 || new_distance >= closest_static_[k][0][j - 1].distance)
        {

            // Start at the the second to last index
            for (int z = closest_static_[k][0].size() - 2; z >= j; z--)
                closest_static_[k][0][z + 1] = closest_static_[k][0][z];

            // Insert the new one
            closest_static_[k][0][j].distance = new_distance;
            closest_static_[k][0][j].obstacle_index = 0;
            closest_static_[k][0][j].scenario_index = s;

            break;
        }
    }
}


void ScenarioManager::removeScenariosBySorting(std::vector<DefinedScenario>& scenarios_to_sort){

    // Removal based on distance to VRU mean!
    std::sort(scenarios_to_sort.begin(), scenarios_to_sort.end(), [&](const DefinedScenario &a, const DefinedScenario &b) {
        return a.scenario_index < b.scenario_index;
    });

    // scenarios_to_sort = [R closest to mean | l]
    
}

// Check if distances aren't already computed!
void ScenarioManager::hyperplaneFromScenario(const Eigen::Vector2d &pose, const Eigen::Vector2d &scenario_position, LinearConstraint2D &constraint,
                                             int k, const DefinedScenario &scenario, int write_index){

    // Normalized vector from the scenario to the EV
    Eigen::Vector2d v = scenario_position - pose;
    Eigen::Vector2d v_normalised = (v) / std::sqrt(v.transpose() * v);

    // Set a row of the constraints
    constraint.A_.block<1, 2>(write_index, 0) = v_normalised.transpose();

    // Find b using A^T * x, with x the point on the constraining circle towards the EV
    constraint.b_(write_index) = constraint.A_.block<1, 2>(write_index, 0) * (scenario_position - v_normalised * combined_radius_);
    //std::cout << "Constraint: " << stage_constraint.A_(a, 0) << " * x + " << stage_constraint.A_(a, 1) << " * y <= " << stage_constraint.b_(a) << std::endl;
}


// Insert chance constraints into the optimisation
void ScenarioManager::insertConstraints(BaseModel *solver_interface, int k, int start_index, int &end_index)
{

    // Insert constraints A_l * x <= b_l
    LinearConstraint2D &constraint = scenario_constraints_[k];

    // Insert all active constraints in this stage
    for(int l = 0; l < constraint.A_.rows(); l++){

        // Define the offset in index, 3 parameters per constraint, max_..._polygon constraints per polygon
        int lin_offset = start_index + l*3;

        solver_interface->setParameter(k, lin_offset + 1, constraint.A_(l, 0));
        solver_interface->setParameter(k, lin_offset + 2, constraint.A_(l, 1));
        solver_interface->setParameter(k, lin_offset + 3, constraint.b_(l));

    }

    // Insert dummies on other spots (relative to vehicle position)
    for(int l = constraint.A_.rows(); l < 20 + 4; l++){
        int lin_offset = start_index + l * 3;

        solver_interface->setParameter(k, lin_offset + 1, 1.0);
        solver_interface->setParameter(k, lin_offset + 2, 0.0);
        solver_interface->setParameter(k, lin_offset + 3, solver_interface->x(k) + 100.0);
    }
    end_index = start_index + (20 + 4) * 3;
}

// Callback from a prediction module
void ScenarioManager::predictionCallback(const lmpcc_msgs::lmpcc_obstacle_array &predicted_obstacles)
{
    if (config_->activate_debug_output_)
        ROS_INFO("Scenario Manager: Data Received");

    // std::cout << "scenario" << std::endl;
    // std::cout << "x = " << predicted_obstacles.lmpcc_obstacles[0].trajectory.poses[0].pose.position.x << std::endl;
    // std::cout << "y = " << predicted_obstacles.lmpcc_obstacles[0].trajectory.poses[0].pose.position.y << std::endl;
    /*if(predicted_obstacles.lmpcc_obstacles.size() > 1){
        ROS_WARN("More than 1 obstacle received, scenario only uses the first obstacle!");
    }*/

    obstacles_msg_ = predicted_obstacles;
    // Generate scenarios from the obstacles (this way n_obstacles has to be the obstacle size at all times!)
    for (int i = 0; i < config_->max_obstacles_; i++) 
    {
        // int obstacle_id = 0; //predicted_obstacles.lmpcc_obstacles[i].id;

        obstacles_[i].ellipsoidToGaussian(predicted_obstacles.lmpcc_obstacles[i]); // Should be connected to an obstacle ID... (id is available as float64)
    }


    if (config_->activate_debug_output_)
        ROS_INFO("Scenario Manager: Data Preprocessed");
}

void ScenarioManager::mapCallback(const nav_msgs::OccupancyGrid::ConstPtr &occupancy_grid, const Eigen::Vector2d &pose, tf::TransformListener &tf_listener)
{
    static_obstacles_.mapCallback(occupancy_grid, pose, tf_listener);
}

//-------------------------------- VISUALIZATION -----------------------------------//
// Visualise the predictions
void ScenarioManager::visualiseEllipsoidConstraints()
{
    ROSPointMarker &ellipsoid = ros_markers_->getNewPointMarker("CYLINDER");

    for(size_t v = 0; v < obstacles_msg_.lmpcc_obstacles.size(); v++){

        lmpcc_msgs::lmpcc_obstacle& obstacle = obstacles_[v].getPrediction();
        
        for (uint k = 0; k < config_->indices_to_draw_.size(); k++)
        {
            for (uint sigma = 2; sigma < 3; sigma++) // Data is 3 sigma already
            {

                ellipsoid.setColor((double)k / (double)config_->indices_to_draw_.size() / 4.0);

                const int &index = config_->indices_to_draw_[k];

                // Set the dimensions (constraint is + r on both axes) // 
                ellipsoid.setScale(
                    2 * (sigma + 1) * (obstacle.major_semiaxis[index]),
                    2 * (sigma + 1) * (obstacle.minor_semiaxis[index]),
                    0.01);

                ellipsoid.setOrientation(obstacle.trajectory.poses[index].pose.orientation);

                // Draw the ellipsoid
                ellipsoid.addPointMarker(obstacle.trajectory.poses[index].pose);
            }
        }
    }
}

// Visualise the predictions
void ScenarioManager::visualiseScenarios(const std::vector<int> &indices_to_draw)
{
    ROSMultiplePointMarker &scenario_points = ros_markers_->getNewMultiplePointMarker();
    scenario_points.setScale(0.2, 0.2, 0.2);
    scenario_points.setColor(0, 0, 0);
    scenario_points.setOrientation(0);

    for (uint k = 0; k < indices_to_draw.size(); k++)
    {
        const int &index = indices_to_draw[k];

        for (int v = 0; v < active_obstacle_count_[index]; v++)
        {
            // Get the index of this obstacle
            int obst = active_obstacle_indices_[index][v];

            for (int s = 0; s < obstacles_[obst].getPrunedSampleCount(index); s++)
                scenario_points.addPoint(getScenarioLocation(index, obst, s));
        }

        
    }

    for (int s = 0; s < static_obstacles_.getScenarioCount(); s++)
    {

        Eigen::Vector2d &scenario_ref = static_obstacles_.getScenarioRef(s);
            scenario_points.addPoint(Eigen::Vector3d(scenario_ref(0), scenario_ref(1), 0.2));
    }
    scenario_points.finishPoints();
}

void ScenarioManager::visualiseRemovedScenarios(const std::vector<int> &indices_to_draw){


    ROSMultiplePointMarker &removed_scenarios_ = ros_markers_->getNewMultiplePointMarker();
    removed_scenarios_.setScale(0.15, 0.15, 0.15);
    removed_scenarios_.setColor(0, 0.7, 0);

    for (size_t k = 0; k < indices_to_draw.size(); k++)
    {
        const int &index = indices_to_draw[k];

        // Draw removed and selected scenarios
        for (int i = 0; i < R_; i++)
        {
            for (int v = 0; v < active_obstacle_count_[index]; v++)
            {
                // Get the index of this obstacle
                int obst = active_obstacle_indices_[index][v];

                Eigen::Vector3d scenario_location = getScenarioLocation(
                    index,
                    closest_scenarios_[index][obst][i].obstacle_index,
                    closest_scenarios_[index][obst][i].scenario_index);

                // Draw a point
                removed_scenarios_.addPoint(scenario_location);
            }
        }
    }
    
    removed_scenarios_.finishPoints();
}


void ScenarioManager::visualiseSelectedScenarios(const std::vector<int> &indices_to_draw)
{

    ROSPointMarker &selected_scenario_points = ros_markers_->getNewPointMarker("CUBE");
    selected_scenario_points.setScale(0.1, 0.1, 0.1);
    selected_scenario_points.setColor(1, 1, 0);

    ROSPointMarker &selected_scenario_circles = ros_markers_->getNewPointMarker("CYLINDER");
    selected_scenario_circles.setScale(2 * (config_->r_VRU_), 2 * (config_->r_VRU_), 0.01); // + config_->ego_w_ / 2.0
    selected_scenario_circles.setColor(1, 1, 0, 0.05);


    for (uint k = 0; k < indices_to_draw.size(); k++)
    {
        const int &index = indices_to_draw[k];
        selected_scenario_points.setColor((double)k / (double)indices_to_draw.size() / 4.0);
        selected_scenario_circles.setColor((double)k / (double)indices_to_draw.size() / 4.0, 0.1);

        // Get lines in the polygon
        std::vector<GraphLine> &polygon_lines_ = polygon_constructors_[index].getGraphLines();

        // Draw removed and selected scenarios
        for (size_t i = 0; i < 24; i++)
        {
            
            int line_index = polygon_lines_[i].line_index % (l_ + 4);

            // std::cout << polygon_lines_[i].line_index << std::endl;
            if (i >= polygon_lines_.size() || line_index >= l_ || polygon_lines_[i].line_index == 0)
            {
                Eigen::Vector3d dummy(50, 50, 0);
                // Draw dummy
                selected_scenario_points.addPointMarker(dummy);
                selected_scenario_circles.addPointMarker(dummy);
            }
            else
            {
                int obstacle_index = std::floor(polygon_lines_[i].line_index / (l_ + 4));
                
                Eigen::Vector3d scenario_location;

                if(obstacle_index < active_obstacle_count_[k]){

                    // Dynamic obstacle
                    obstacle_index = active_obstacle_indices_[index][obstacle_index];       
                    
                    scenario_location = getScenarioLocation(
                        index,
                        closest_scenarios_[index][obstacle_index][R_ + line_index].obstacle_index,
                        closest_scenarios_[index][obstacle_index][R_ + line_index].scenario_index);
                }else{
                    // Static obstacle
                    scenario_location(0) = static_obstacles_.getScenarioRef(index)(0);
                    scenario_location(1) = static_obstacles_.getScenarioRef(index)(1);
                }

                // Draw a yellow point
                scenario_location(2) = 0.5;
                selected_scenario_points.addPointMarker(scenario_location);

                // Draw a circle around it
                if (config_->r_VRU_ > 0.0)
                    selected_scenario_circles.addPointMarker(scenario_location);
            }
        }
    }
}

void ScenarioManager::visualisePolygons(){

    ROSLine &line = ros_markers_->getNewLine();
    // line.setLifetime(1.0/25.0);

    geometry_msgs::Point p1, p2;
    p1.z = 0.2; p2.z = 0.2; // 2D points
    // How to deal with multiple discs?!
    for (uint k = 0; k < config_->indices_to_draw_.size(); k++)
    {
        const int &index = config_->indices_to_draw_[k];

        std::vector<GraphLine>& polygon_lines_ = polygon_constructors_[index].getGraphLines();

        int draw_size = 24;//config_->inner_approximation_ + 4;

        // Draw the same number of lines each loop to prevent weard lines
        for(int i = 0; i < draw_size; i++){

            int polygon_index = (i < (int)polygon_lines_.size()) ? i : 0;

            // Get a line drawer and set properties
            line.setScale(0.08, 0.08);
            double smooth_colors = 4.0;
            line.setColor((double)k / config_->indices_to_draw_.size() / smooth_colors);

            // if (polygon_index == 0)
            //     line.setColor(0, 0, 1);

            p1.x = polygon_lines_[polygon_index].point_a(0);
            p1.y = polygon_lines_[polygon_index].point_a(1);

            p2.x = polygon_lines_[polygon_index].point_b(0);
            p2.y = polygon_lines_[polygon_index].point_b(1);

            line.addLine(p1, p2);
        }

    }
}

void ScenarioManager::visualiseProjectedPosition(){

    ROSPointMarker &selected_scenario_circles = ros_markers_->getNewPointMarker("CYLINDER");
    selected_scenario_circles.setScale(1.0, 1.0, 0.01); // + config_->ego_w_ / 2.0
    selected_scenario_circles.setColor(1, 1, 1, 0.7);

    for (uint k = 0; k < config_->indices_to_draw_.size(); k++)
    {
        const int &index = config_->indices_to_draw_[k];

        Eigen::Vector3d scenario_location(projected_poses_[index](0), projected_poses_[index](1), 0.2);

        selected_scenario_circles.addPointMarker(scenario_location);
    }
}

Eigen::Vector3d ScenarioManager::getScenarioLocation(const int &k, const int &obstacle_index, const int &scenario_index)
{

    Eigen::Vector2d &loc = obstacles_[obstacle_index].getScenarioRef(k, scenario_index);
    return Eigen::Vector3d(loc(0), loc(1), 0.2);
}

void ScenarioManager::publishVisuals()
{
    // Draws all the markers added to ros_markers_ and resets
    ros_markers_->publish();
}