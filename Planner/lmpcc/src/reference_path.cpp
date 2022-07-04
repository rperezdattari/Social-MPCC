#include "lmpcc/reference_path.h"

void ReferencePath::Init(ros::NodeHandle& nh, predictive_configuration *config)
{

    ROS_WARN("Initializing Reference Path");

    // Save the config
    config_ = config;

    // Initialize publishers
    // Spline
    spline_pub_ = nh.advertise<nav_msgs::Path>(config_->reference_path_topic_, 1);

    // Reference Path
    ros_markers_.reset(new ROSMarkerPublisher(nh, config_->reference_arrows_topic_.c_str(), config_->target_frame_, 1000));

    //Road
    road_pub_ = nh.advertise<visualization_msgs::MarkerArray>("road", 10);

    marker_pub_ = nh.advertise<visualization_msgs::MarkerArray>("road_limits", 10);

    // MPCC path variables
    x_.resize(config_->ref_x_.size());
    y_.resize(config_->ref_y_.size());
    theta_.resize(config_->ref_theta_.size());
    waypoints_size_ = config_->ref_x_.size();

    // Read the reference from the config
    ReadReferencePath();

    // Initialize the path from file
    InitPath();

    ROS_WARN("Reference Path Initialized");
}

/* Initialize a path from x,y,theta in the class */
void ReferencePath::InitPath(){

    spline_index_ = 0;

    // Construct a spline through these points
    ConstructReferencePath(x_, y_, theta_);

    // Visualizes the given reference points (for debug mostly)
    PublishReferencePath();

    // Visualize the fitted spline
    PublishSpline();
}

/* Initialize a path from x,y,theta given */
void ReferencePath::InitPath(const std::vector<double> &x, const std::vector<double> &y, const std::vector<double> &theta)
{
    ROS_WARN("Received Reference Path");

    if(x.size()>3){

        // Save x,y, theta
        x_ = x;
        y_ = y;
        theta_ = theta;
        waypoints_size_ = x_.size();
        ROS_INFO_STREAM("ReferencePath::InitPath: Received " << std::floor(x.size()) << " Waypoints");
        // Initialize using these
        InitPath();
    }

}

// Restructure initpath to incorporate a callback!

void ReferencePath::ReadReferencePath()
{
    ROS_WARN("Reading Reference Path");

    geometry_msgs::Pose pose;
    tf2::Quaternion myQuaternion;

    // Iterate over the reference points given
    for (size_t ref_point_it = 0; ref_point_it < config_->ref_x_.size(); ref_point_it++)
    {
        // Create a pose at each position
        pose.position.x = config_->ref_x_.at(ref_point_it);
        pose.position.y = config_->ref_y_.at(ref_point_it);

        // Find the orientation as quaternion
        tf::Quaternion q = tf::createQuaternionFromRPY(0, 0, (double)config_->ref_theta_[ref_point_it]);
        pose.orientation.x = q.x();
        pose.orientation.y = q.y();
        pose.orientation.z = q.z();
        pose.orientation.w = q.w();

        // // Convert from global_frame to planning frame
        // if (config_->global_path_frame_.compare(config_->target_frame_) != 0)
        //     transformPose(config_->global_path_frame_, config_->target_frame_, pose);

        x_[ref_point_it] = pose.position.x;
        y_[ref_point_it] = pose.position.y;
        theta_[ref_point_it] = Helpers::quaternionToAngle(pose.orientation);
    }
}

void ReferencePath::ConstructReferencePath(const std::vector<double>& x, const std::vector<double>& y, const std::vector<double>& theta)
{

    double L;
    int j=0;

    ROS_INFO("ConstructReferencePath...");

    std::vector<double> X_all, Y_all, S_all;
    double total_length_= 0;

    S_all.push_back(0);
    X_all.push_back(x[0]);
    Y_all.push_back(y[0]);

    if(config_->activate_debug_output_)
        ROS_INFO("Generating path...");

    for (size_t i = 1; i < x.size(); i++)
    {
        L = std::sqrt(std::pow(x[i]-x[j],2)+std::pow(y[i]-y[j],2));

        if(L>config_->min_wapoints_distance_){
            j = i;
            total_length_ += L;
            X_all.push_back(x[i]);
            Y_all.push_back(y[i]);
            S_all.push_back(total_length_);
        }
    }


    ref_path_x_.set_points(S_all, X_all);
    ref_path_y_.set_points(S_all, Y_all);

    config_->n_points_spline_ = S_all.size();
    dist_spline_pts_ = total_length_ / (config_->n_points_spline_ );

    if(config_->activate_debug_output_)
        ROS_INFO_STREAM("dist_spline_pts_: " << dist_spline_pts_);

    unsigned int n_pts = S_all.size();
    ss_.resize(n_pts);
    xx_.resize(n_pts);
    yy_.resize(n_pts);

    for (size_t i = 0; i < n_pts; i++)
    {
        if(config_->activate_debug_output_){
            ROS_INFO_STREAM("ss: " << S_all[i]);
            ROS_INFO_STREAM("xx: " << X_all[i]);
            ROS_INFO_STREAM("yy: " << Y_all[i]);
        }
        ss_[i] = S_all[i];
        xx_[i] = X_all[i];
        yy_[i] = Y_all[i];
    }

    ROS_INFO("Path generated");
}

// Recursive clashes with no return type now...
int ReferencePath::RecursiveClosestPointSearch(BaseModel *solver_interface_ptr, unsigned int cur_traj_i, double &s_guess, double window, int n_tries)
{


    if(ss_.size()>0) {

        double s_min = ss_[cur_traj_i] - MAX_STEP_BACK_TOLERANCE;
        double s_max = ss_[cur_traj_i + 1] + MAX_STEP_BACK_TOLERANCE;

        double lower = std::max(s_min, s_guess - window);
        double upper = std::min(s_max, s_guess + window);

        double s_i = upper;
        double spline_pos_x_i, spline_pos_y_i;
        double dist_i, min_dist;

        //First, try the furthest point in our search window. This is the reference that must be beat.
        double s_best = s_i;
        spline_pos_x_i = ref_path_x_(s_i);
        spline_pos_y_i = ref_path_y_(s_i);

        min_dist = std::sqrt((spline_pos_x_i - solver_interface_ptr->getState()->get_x()) *
                             (spline_pos_x_i - solver_interface_ptr->getState()->get_x()) +
                             (spline_pos_y_i - solver_interface_ptr->getState()->get_y()) *
                             (spline_pos_y_i - solver_interface_ptr->getState()->get_y()));

        //Compute the step size.
        //Divide by minus one. If you want to go from 1 to 3 (distance two) with three steps, step size must be (3-1)/2=1 to go 1,2,3.
        double step_size = (upper - lower) / (n_tries - 1);
        for (s_i = lower; s_i < upper; s_i += step_size) {
            // Get the current spline position
            spline_pos_x_i = ref_path_x_(s_i);
            spline_pos_y_i = ref_path_y_(s_i);

            // Compute the distance
            dist_i = std::sqrt((spline_pos_x_i - solver_interface_ptr->getState()->get_x()) *
                               (spline_pos_x_i - solver_interface_ptr->getState()->get_x()) +
                               (spline_pos_y_i - solver_interface_ptr->getState()->get_y()) *
                               (spline_pos_y_i - solver_interface_ptr->getState()->get_y()));

            // Save it if it is the smallest
            if (dist_i < min_dist) {
                min_dist = dist_i;
                s_best = s_i;
            }
        }

        // Save the previous best s
        double previous_guess = s_guess;
        s_guess = s_best;

        int next_traj = cur_traj_i;

        // If the smallest distance is the lower bound of the window
        if (s_best == lower && lower != previous_guess) {
            //If we hit the low point of the window, and that low point was the end of this spline segment, try one segment higher!
            if (lower == s_min && cur_traj_i > 0) {
                next_traj--;
            }
            return RecursiveClosestPointSearch(solver_interface_ptr, next_traj, s_guess, window, n_tries);
        }

        if (s_best == upper && upper != previous_guess) {
            //If we hit the high point of the window, and that high point was the end of this spline segment, try one segment higher!
            if (upper == s_max && cur_traj_i < ss_.size() - 2) {
                next_traj++;
            }
            return RecursiveClosestPointSearch(solver_interface_ptr, next_traj, s_guess, window, n_tries);
        }
    }

    return cur_traj_i;
}

void ReferencePath::UpdateClosestPoint(BaseModel *solver_interface_ptr, double &s_guess, double window, int n_tries)
{

    spline_index_ = RecursiveClosestPointSearch(solver_interface_ptr, spline_index_, s_guess, window, n_tries);
    PublishCurrentSplineIndex();

}

void ReferencePath::InitializeClosestPoint(BaseModel* solver_interface_ptr)
{

    Eigen::Vector2d current_pose(solver_interface_ptr->getState()->get_x(), solver_interface_ptr->getState()->get_y());
    Eigen::Vector2d trajectory_pose;

    // Print the distance to the current trajectory index for feedback
    // trajectory_pose = Eigen::Vector2d(ref_path_x_(ss_[spline_index_]), ref_path_y_(ss_[spline_index_]));
    // std::cout << "Distance to current spline point: " << Helpers::dist(current_pose, trajectory_pose) << std::endl;
    // ROS_INFO_STREAM("Current Pose: " << current_pose);

    double smallest_dist = 9999999.0;
    double current_dist;
    int best_i = -1;
    for (int i = 0; i < (int)ss_.size(); i++)
    {
        trajectory_pose = Eigen::Vector2d(ref_path_x_(ss_[i]), ref_path_y_(ss_[i]));
        // ROS_INFO_STREAM("trajectory_pose: " << current_pose);
        current_dist = Helpers::dist(current_pose, trajectory_pose);

        if (current_dist < smallest_dist)
        {
            smallest_dist = current_dist;
            best_i = i;
        }
    }

    if (best_i == -1)
        ROS_ERROR("Initial spline search failed: No point was found!");

    // If it succeeded return our best index
    if(best_i == -1)
        spline_index_ = std::max(0,int(ss_.size()-1));
    else
        spline_index_ = best_i;

    // Visualizes the given reference points (for debug mostly)
    PublishReferencePath();

    // Visualize the fitted spline
    PublishSpline();

    plotRoad();
}

bool ReferencePath::EndOfCurrentSpline(double index){
    if(ss_.size() > spline_index_ +1)
        return index > ss_[spline_index_ + 1];
    else
        return true;
}

void ReferencePath::UpdateWaypoint(BaseModel* solver_interface_ptr){
    Eigen::Vector2d current_pose(solver_interface_ptr->getState()->get_x(), solver_interface_ptr->getState()->get_y());
    Eigen::Vector2d current_goal(x_[spline_index_], y_[spline_index_]);

    if((current_pose-current_goal).norm() < config_->epsilon_) {
        spline_index_ += 1;
    }

    unsigned int s = ss_.size()-1;
    spline_index_ = std::min(spline_index_,s);
}

bool ReferencePath::ReachedEnd()
{
    return spline_index_ + 2 >= ss_.size();
}

void ReferencePath::PublishReferencePath()
{
    ROSPointMarker &arrow = ros_markers_->getNewPointMarker("ARROW");
    ROSLine &line = ros_markers_->getNewLine();
    line.setColor(0.0, 1.0, 0.0);
    line.setScale(0.05);
    line.setOrientation(0.0);
    arrow.setScale(0.5, 0.1, 0.1);

    geometry_msgs::Point prev_point;

    for (unsigned int i = 0; i < config_->ref_x_.size(); i++)
    {
        // Draw the constraint as a line
        geometry_msgs::Point p;
        p.x = config_->ref_x_[i];
        p.y = config_->ref_y_[i];
        p.z = 0.2;

        if (i > 0)
            line.addLine(prev_point, p);

        prev_point = p;

        arrow.setOrientation(config_->ref_theta_[i]);
        arrow.addPointMarker(p);
    }

    ros_markers_->publish();
}

void ReferencePath::PublishSpline()
{
    // Plot 100 points
    spline_msg_.poses.resize(50);

    spline_msg_.header.stamp = ros::Time::now();
    spline_msg_.header.frame_id = config_->target_frame_;
    if(spline_msg_.poses.size() != (unsigned int)config_->n_points_spline_)
        spline_msg_.poses.resize(config_->n_points_spline_);

    for (int i = 0; i < spline_msg_.poses.size(); i++)
    {
        spline_msg_.poses[i].pose.position.x = ref_path_x_(i+ss_[std::min(std::max(int(spline_index_),0),int(ss_.size()-1))]);
        spline_msg_.poses[i].pose.position.y = ref_path_y_(i+ss_[std::min(std::max(int(spline_index_),0),int(ss_.size()-1))]);
        spline_msg_.poses[i].pose.position.z = 0.2;
    }

    spline_pub_.publish(spline_msg_);
}

void ReferencePath::PublishCurrentSplineIndex(){

    // Use to debug spline init
    ROSPointMarker &cube = ros_markers_->getNewPointMarker("Cube");
    cube.setScale(0.5, 0.5, 0.5);
    cube.setOrientation(config_->ref_theta_[spline_index_]);

    geometry_msgs::Point p;
    p.z = 0.2;

    if(xx_.size()>0){

    p.x = xx_[spline_index_];
    p.y = yy_[spline_index_];
    cube.addPointMarker(p);

    cube.setColor(0, 0.8, 0);
    p.x = xx_[spline_index_ + 1];
    p.y = yy_[spline_index_ + 1];
    cube.addPointMarker(p);

    ros_markers_->publish();
    }
}

void ReferencePath::PublishRoadBoundaries(BaseModel* solver_interface_ptr)
{

    visualization_msgs::Marker line_strip;
    visualization_msgs::MarkerArray line_list;
    line_strip.header.frame_id = config_->target_frame_;
    line_strip.id = 1;

    line_strip.type = visualization_msgs::Marker::LINE_STRIP;

    // LINE_STRIP/LINE_LIST markers use only the x component of scale, for the line width
    line_strip.scale.x = 0.2;
    line_strip.scale.y = 0.2;

    // Line strip is blue
    line_strip.color.r = 1.0;
    line_strip.color.a = 1.0;

    // Compute contour and lag error to publish
    geometry_msgs::Point prev_left, prev_right;

    for (size_t i = 0; i < solver_interface_ptr->FORCES_N; i++)
    {
        double x1_path, y1_path, dx1_path, dy1_path;
        double x2_path, y2_path, dx2_path, dy2_path;
        double x_path, y_path, dx_path, dy_path;

        double cur_s = solver_interface_ptr->spline(i);
        // double cur_x = forces_params.x0[3 + (i) * FORCES_TOTAL_V];
        // double cur_y = forces_params.x0[4 + (i) * FORCES_TOTAL_V];

        double d = ss_[spline_index_ + 1] + 0.02;
        double lambda = 1.0 / (1.0 + std::exp((cur_s - d) / 0.1));

        // // Cubic spline cost on x^th stage
        x1_path = ref_path_x_.m_a[spline_index_] * std::pow((cur_s - ss_[spline_index_]), 3.0) +
                  ref_path_x_.m_b[spline_index_] * std::pow((cur_s - ss_[spline_index_]), 2.0) +
                  ref_path_x_.m_c[spline_index_] * (cur_s - ss_[spline_index_]) +
                  ref_path_x_.m_d[spline_index_];

        y1_path = ref_path_y_.m_a[spline_index_] * std::pow((cur_s - ss_[spline_index_]), 3.0) +
                  ref_path_y_.m_b[spline_index_] * std::pow((cur_s - ss_[spline_index_]), 2.0) +
                  ref_path_y_.m_c[spline_index_] * (cur_s - ss_[spline_index_]) +
                  ref_path_y_.m_d[spline_index_];

        // Derivatives
        dx1_path = 3 * ref_path_x_.m_a[spline_index_] * std::pow((cur_s - ss_[spline_index_]), 2.0) +
                   2 * ref_path_x_.m_b[spline_index_] * (cur_s - ss_[spline_index_]) +
                   ref_path_x_.m_c[spline_index_];

        dy1_path = 3 * ref_path_y_.m_a[spline_index_] * std::pow((cur_s - ss_[spline_index_]), 2.0) +
                   2 * ref_path_y_.m_b[spline_index_] * (cur_s - ss_[spline_index_]) +
                   ref_path_y_.m_c[spline_index_];

        // Cubic spline cost on x^th stage
        x2_path = ref_path_x_.m_a[spline_index_ + 1] * std::pow((cur_s - ss_[spline_index_ + 1]), 3.0) +
                  ref_path_x_.m_b[spline_index_ + 1] * std::pow((cur_s - ss_[spline_index_ + 1]), 2.0) +
                  ref_path_x_.m_c[spline_index_ + 1] * (cur_s - ss_[spline_index_ + 1]) +
                  ref_path_x_.m_d[spline_index_ + 1];

        y2_path = ref_path_y_.m_a[spline_index_ + 1] * std::pow((cur_s - ss_[spline_index_ + 1]), 3.0) +
                  ref_path_y_.m_b[spline_index_ + 1] * std::pow((cur_s - ss_[spline_index_ + 1]), 2.0) +
                  ref_path_y_.m_c[spline_index_ + 1] * (cur_s - ss_[spline_index_ + 1]) +
                  ref_path_y_.m_d[spline_index_ + 1];

        // Derivatives
        dx2_path = 3 * ref_path_x_.m_a[spline_index_ + 1] * std::pow((cur_s - ss_[spline_index_ + 1]), 2.0) +
                   2 * ref_path_x_.m_b[spline_index_ + 1] * (cur_s - ss_[spline_index_ + 1]) +
                   ref_path_x_.m_c[spline_index_ + 1];

        dy2_path = 3 * ref_path_y_.m_a[spline_index_ + 1] * std::pow((cur_s - ss_[spline_index_ + 1]), 2.0) +
                   2 * ref_path_y_.m_b[spline_index_ + 1] * (cur_s - ss_[spline_index_ + 1]) +
                   ref_path_y_.m_c[spline_index_ + 1];

        x_path = lambda * x1_path + (1.0 - lambda) * x2_path;
        y_path = lambda * y1_path + (1.0 - lambda) * y2_path;
        dx_path = lambda * dx1_path + (1.0 - lambda) * dx2_path;
        dy_path = lambda * dy1_path + (1.0 - lambda) * dy2_path;

        Eigen::Vector2d path_point = Eigen::Vector2d(x_path, y_path);
        Eigen::Vector2d dpath = Eigen::Vector2d(-dy_path, dx_path);
        Eigen::Vector2d boundary_left = path_point + dpath * config_->road_width_left_;
        Eigen::Vector2d boundary_right = path_point - dpath * config_->road_width_right_;

        geometry_msgs::Point left, right;
        left.x = boundary_left(0);
        left.y = boundary_left(1);
        left.z = 0.2;

        right.x = boundary_right(0);
        right.y = boundary_right(1);
        right.z = 0.2;

        if (i > 0)
        {
            line_strip.points.push_back(prev_left);
            line_strip.points.push_back(left);
            line_list.markers.push_back(line_strip);
            line_strip.points.pop_back();
            line_strip.points.pop_back();
            line_strip.id++;

            line_strip.points.push_back(prev_right);
            line_strip.points.push_back(right);
            line_list.markers.push_back(line_strip);
            line_strip.points.pop_back();
            line_strip.points.pop_back();
            line_strip.id++;
        }

        prev_left = left;
        prev_right = right;
    }

    road_pub_.publish(line_list);
}

void ReferencePath::plotRoad(void)
{
    visualization_msgs::MarkerArray line_list;
    double theta, dx, dy;
    for (size_t i = 1; i < spline_msg_.poses.size()-1; i++) // 100 points
    {
        visualization_msgs::Marker line_strip_left,line_strip_right;

        line_strip_left.header.frame_id = config_->target_frame_;
        line_strip_left.id = 2*i;

        line_strip_left.type = visualization_msgs::Marker::LINE_STRIP;

        // Left Road Limit
        line_strip_left.scale.x = 0.5;
        line_strip_left.scale.y = 0.5;

        // Line strip is blue
        line_strip_left.color.b = 1.0;
        line_strip_left.color.a = 1.0;

        dx = spline_msg_.poses[i].pose.position.x - spline_msg_.poses[i - 1].pose.position.x ;
        dy = spline_msg_.poses[i].pose.position.y - spline_msg_.poses[i - 1].pose.position.y;

        theta = std::atan2(dy, dx);

        geometry_msgs::Pose pose;

        pose.position.x = (config_->road_width_left_ + line_strip_left.scale.x / 2.0) * -sin(theta);
        pose.position.y = (config_->road_width_left_ + line_strip_left.scale.x / 2.0) * cos(theta);

        geometry_msgs::Point p;
        p.x = spline_msg_.poses[i - 1].pose.position.x + pose.position.x;
        p.y = spline_msg_.poses[i - 1].pose.position.y + pose.position.y;
        p.z = 0.2;  //z a little bit above ground to draw it above the pointcloud.

        line_strip_left.points.push_back(p);

        p.x = spline_msg_.poses[i].pose.position.x + pose.position.x;
        p.y = spline_msg_.poses[i].pose.position.y + pose.position.y;
        p.z = 0.2;  //z a little bit above ground to draw it above the pointcloud.

        line_strip_left.points.push_back(p);

        line_list.markers.push_back(line_strip_left);

        // Right Road Limit
        // Left Road Limit
        line_strip_right.scale.x = 0.5;
        line_strip_right.scale.y = 0.5;
        line_strip_right.color.b = 1.0;
        line_strip_right.color.a = 1.0;
        line_strip_right.id = 2*i+1;
        line_strip_right.header.frame_id = config_->target_frame_;
        line_strip_right.type = visualization_msgs::Marker::LINE_STRIP;

        pose.position.x = (-config_->road_width_right_  - line_strip_right.scale.x / 2.0) * -sin(theta);
        pose.position.y = (-config_->road_width_right_  - line_strip_right.scale.x / 2.0) * cos(theta);

        p.x = spline_msg_.poses[i-1].pose.position.x + pose.position.x;
        p.y = spline_msg_.poses[i-1].pose.position.y + pose.position.y;
        p.z = 0.2;  //z a little bit above ground to draw it above the pointcloud.

        line_strip_right.points.push_back(p);

        p.x = spline_msg_.poses[i].pose.position.x + pose.position.x;
        p.y = spline_msg_.poses[i].pose.position.y + pose.position.y;
        p.z = 0.2;  //z a little bit above ground to draw it above the pointcloud.

        line_strip_right.points.push_back(p);

        line_list.markers.push_back(line_strip_right);

    }

    marker_pub_.publish(line_list);

}