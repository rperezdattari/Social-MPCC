/**
 * @file reference_path.h
 * @brief Class responsible for the reference path used in Model Predictive Contouring Control (MPCC)
 * @version 0.1
 * @date 2022-07-04
 * 
 * @copyright Copyright (c) 2022
 * 
 */
#ifndef REFERENCE_PATH_H
#define REFERENCE_PATH_H

#include <lmpcc/lmpcc_configuration.h>
#include <lmpcc/PredictiveControllerConfig.h>
#include <lmpcc_tools/ros_visuals.h>
#include <visualization_msgs/MarkerArray.h>
#include <lmpcc_tools/helpers.h>
#include <std_msgs/Bool.h>
#include <lmpcc/base_model.h>

#include <nav_msgs/Path.h>

#include <Eigen/Eigen>

//splines
#include <tkspline/spline.h>
#include <lmpcc/Clothoid.h>

#include <vector>

//Whens earching for the closest point on the path, this variable indicates the distance that the algorithm searches behind the current spline point.
#define MAX_STEP_BACK_TOLERANCE 0.1f

class ReferencePath{

public:


private:

    predictive_configuration * config_;

    ros::Publisher spline_pub_, road_pub_, marker_pub_;

    std::unique_ptr<ROSMarkerPublisher> ros_markers_;

    void ReadReferencePath();
    void ConstructReferencePath(const std::vector<double>& x, const std::vector<double>& y, const std::vector<double>& theta);

    int RecursiveClosestPointSearch(BaseModel *solver_interface_ptr, unsigned int cur_traj_i, double &s_guess, double window, int n_tries);

public:
    // Current spline index
    unsigned int spline_index_;
    unsigned int waypoints_size_;

    // Waypoints x, y, theta
    std::vector<double> x_, y_, theta_;

    // Output splines
    tk::spline ref_path_x_, ref_path_y_;

    // Spline s, x, y
    std::vector<double> ss_, xx_, yy_;

    double dist_spline_pts_;

    nav_msgs::Path spline_msg_;

    void Init(ros::NodeHandle &nh, predictive_configuration *config);
    void InitPath();
    void InitPath(const std::vector<double> &x, const std::vector<double> &y, const std::vector<double> &theta);

    void UpdateClosestPoint(BaseModel *solver_interface_ptr, double &s_guess, double window, int n_tries);

    void InitializeClosestPoint(BaseModel *solver_interface_ptr);

    bool EndOfCurrentSpline(double index);

    void UpdateWaypoint(BaseModel *solver_interface_ptr);

    bool ReachedEnd();

    void PublishReferencePath();
    void PublishSpline();
    void PublishCurrentSplineIndex();

    void PublishRoadBoundaries(BaseModel* solver_interface_ptr);
    void plotRoad();
};
#endif