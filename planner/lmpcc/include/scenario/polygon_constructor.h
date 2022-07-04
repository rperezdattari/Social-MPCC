#ifndef POLYGON_CONSTRUCTOR_H
#define POLYGON_CONSTRUCTOR_H

#include <Eigen/Dense>
// #include "ros_visuals.h"

#include "lmpcc_tools/helpers.h"

#define RANGE 1.0

// FSM for polygon search //0       1         2     3   4       5           6
enum class PolygonState {START, RIGHT_START, UP, LEFT, DOWN, RIGHT_FINISH, FINISH};

struct DoubleOrInfinite{
    double value_;
    bool is_infinite_;
};

struct GraphLine{
    int line_index;
    LinearConstraint2D constraint;
    Eigen::Vector2d point_a;
    Eigen::Vector2d point_b;
    double dist;
};

/* Construct a polygon with minimal edges given a set of hyperplanes */
class PolygonConstructor{

public:
    /** 
     * @brief Assumes hyperplanes are sorted on distance to vehicle
     */
    PolygonConstructor(double range, double n_constraints, int n_inner_constraints);

private:

    // Debug
    std::vector<LinearConstraint2D>* hyperplanes_;
    std::vector<LinearConstraint2D> range_constraints_;
    std::vector<GraphLine> graph_lines_;
    PolygonState state_;

    LinearConstraint2D starting_line_;
    Eigen::Vector2d vehicle_pose_;

    int n_constraints_;
    int polygon_edge_count_;
    double range_;
    int n_inner_constraints_;

    double orientation_;

    Eigen::MatrixXd A_range_;
    Eigen::VectorXd b_range_;
    Eigen::MatrixXd intersection_matrix_;

    int search_direction_;
    int current_index_, previous_index_;
    DoubleOrInfinite current_direction_;
    Eigen::Vector2d previous_x_;

    // Only sign of direction is used and rarely
    //std::vector<double> directions_;

    bool findIntersect(const LinearConstraint2D& a, const LinearConstraint2D b, Eigen::Vector2d& intersect_out);
    DoubleOrInfinite computeDirection(const LinearConstraint2D& constraint);
    LinearConstraint2D& getConstraint(int index);

    void innerApproximate(std::vector<GraphLine>& graph_lines);


public:
    void insertData(std::vector<LinearConstraint2D> *hyperplanes, const Eigen::Vector2d &vehicle_pose, double orientation, int n_constraints);

    void findPolygon(std::vector<LinearConstraint2D>& constraints_out, std::vector<int>& indices_out);

    double getArea();

    // Getters
    std::vector<GraphLine>& getGraphLines(){ return graph_lines_;};
    int getPolygonEdgeCount(){ return polygon_edge_count_;};
    void setConstraintCount(int count){n_constraints_ = count;};

};



#endif