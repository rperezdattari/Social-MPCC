#include "scenario/polygon_constructor.h"

PolygonConstructor::PolygonConstructor(double range, double n_constraints, int n_inner_constraints)
{
    range_ = range;
    n_constraints_ =  n_constraints;
    n_inner_constraints_ = n_inner_constraints;

    // Do initialisation / allocation a priori
    starting_line_.A_ = Eigen::Matrix<double, 1, 2>();
    starting_line_.A_ << 0.0, 1.0;

    starting_line_.b_ = Eigen::Matrix<double, 1, 1>();

    intersection_matrix_ = Eigen::Matrix<double, 2, 2>();

    // Eigen::MatrixXd A_range = Eigen::MatrixXd::Zero(4, 2);
    // A_range << Eigen::MatrixXd::Identity(2, 2), -Eigen::MatrixXd::Identity(2, 2);
    A_range_ = Eigen::MatrixXd::Zero(4, 2);

    b_range_ = Eigen::VectorXd::Zero(4);

    range_constraints_.resize(4);
}

void PolygonConstructor::insertData(std::vector<LinearConstraint2D> *hyperplanes, const Eigen::Vector2d &vehicle_pose, double orientation, int n_constraints)
{
    // Save input data
    hyperplanes_ = hyperplanes;
    vehicle_pose_ = vehicle_pose;
    orientation_ = orientation;
    n_constraints_ = n_constraints;

    Eigen::MatrixXd rotation_matrix = Helpers::rotationMatrixFromHeading(orientation);
    A_range_ << rotation_matrix * Eigen::MatrixXd::Identity(2, 2), -rotation_matrix * Eigen::MatrixXd::Identity(2, 2);

    for (u_int i = 0; i < 4; i++)
    {
        range_constraints_[i] = LinearConstraint2D(
            Eigen::Matrix<double, 1, 2>(A_range_.block<1, 2>(i, 0)),
            Eigen::Matrix<double, 1, 1>());
    }

    // Setup range constraints (a box with dimensions range_)
    // b_range_ << vehicle_pose_(0) + range_, vehicle_pose_(1) + range_, -vehicle_pose_(0) + range_, -vehicle_pose_(1) + range_;
    b_range_ << vehicle_pose_(0), vehicle_pose_(1), -vehicle_pose_(0), -vehicle_pose_(1);

    b_range_.block<2, 1>(0, 0) = rotation_matrix * b_range_.block<2, 1>(0, 0) + Eigen::Vector2d(range_, range_);
    b_range_.block<2, 1>(2, 0) = rotation_matrix * b_range_.block<2, 1>(2, 0) + Eigen::Vector2d(range_, range_);

    for (u_int i = 0; i < 4; i++)
        range_constraints_[i].b_(0) = b_range_(i); 

    /* Define the initial line */
    // Line dropping down from the vehicle (this is going to the right atm!)
    starting_line_.b_(0) = vehicle_pose_(1);

    previous_x_ = vehicle_pose_;

    current_index_ = n_constraints_ + 4;
    previous_index_ = current_index_;

    current_direction_ = computeDirection(getConstraint(current_index_));
    state_ = PolygonState::START;
    //std::cout << "initial direction: " << current_direction_.value_ << ", inf: " << current_direction_.is_infinite_ << std::endl;

    graph_lines_.clear();
}

// Switch between range and scenario constraints
LinearConstraint2D& PolygonConstructor::getConstraint(int index){

    if(index < n_constraints_)
        return (*hyperplanes_)[index];
    else if(index < n_constraints_ + 4)
        return range_constraints_[index - n_constraints_];
    else
        return starting_line_;
}

DoubleOrInfinite PolygonConstructor::computeDirection(const LinearConstraint2D &constraint)
{

    // Check if this is a line in y
    if(std::abs(constraint.A_(0)) < 1e-9){ // was (1) here
        // If it is, set the direction to "infinite".
        return {9999999999.0, true};
    }
    //return {-constraint.A_(0) / constraint.A_(1), false};
    return {constraint.A_(1) / constraint.A_(0), false};
}

// Find an intersection using 2D matrix inversion
bool PolygonConstructor::findIntersect(const LinearConstraint2D &a, const LinearConstraint2D b, Eigen::Vector2d& intersect_out){

    double det_denom = a.A_(0, 0) * b.A_(0, 1) - a.A_(0, 1) * b.A_(0, 0);

    if (abs(det_denom) < 1e-9)
    {
        // Lines are parallell, there are no intersects
        return false;
    }

    intersection_matrix_ << b.A_(0, 1), -a.A_(0, 1), -b.A_(0, 0), a.A_(0, 0);
    intersect_out = (1.0 / det_denom) * intersection_matrix_ * Eigen::Vector2d(a.b_(0), b.b_(0));

    return true;
}

void PolygonConstructor::findPolygon(std::vector<LinearConstraint2D> &constraints_out, std::vector<int> &indices_out)
{
    // Set the search direction (left or right) based on the y of the initial starting location

    // If we initialised with a scenario, save it
    std::vector<int> result_indices;
    result_indices.push_back(current_index_);

    DoubleOrInfinite new_direction;
    Eigen::Vector2d best_x, x_intersect;
    int best_index;
    bool first_run;

    // Save for loop for now
    for(u_int i = 0; i < 1000; i++){
        best_index = -1;
        first_run = true;
        // std::cout << "state: " << (int)state_ << std::endl;
        // std::cout << "current index: " << current_index_ << std::endl;
        // std::cout << "current_direction: " << current_direction_.value_ << std::endl;

        // Find all intersections with the current line
        for (size_t j = 0; j < n_constraints_ + 4; j++)
        {

            // Don't check the current constraint
            if (j == current_index_)
                continue;

            // Find the intersection of these lines
            bool do_intersect = findIntersect(getConstraint(current_index_), getConstraint(j), x_intersect);

            // Ensure they intersect, in range and not with the line it came from
            if (!do_intersect || x_intersect == previous_x_)
                continue;

            // Now depending on our current search state, we search for different conditions
            switch (state_)
            {
                // If this is the initial line
                case PolygonState::START:
                case PolygonState::DOWN:
                    // Set the initial value very small
                    if(first_run){
                        best_x = {-999999999.0, 0.0};
                        first_run = false;
                    }

                    // Find the maximum x that is smaller than the previous (vehicle) position
                    if(x_intersect(0) >= previous_x_(0))
                        continue;

                    if(x_intersect(0) > best_x(0)){
                        best_index = j;
                        best_x = x_intersect;
                    }

                    break;

                case PolygonState::RIGHT_START: 
                case PolygonState::RIGHT_FINISH:

                    // Set the initial value very small
                    if (first_run)
                    {
                        best_x = {0.0, -999999999.0};
                        first_run = false;
                    }
                    // Find the maximum x that is smaller than the previous (vehicle) position
                    if (x_intersect(1) >= previous_x_(1))
                        continue;

                    if (x_intersect(1) > best_x(1))
                    {
                        best_index = j;
                        best_x = x_intersect;
                    }

                    break;

                case PolygonState::UP:

                    // Set the initial value very small
                    if (first_run)
                    {
                        best_x = {999999999.0, 0.0};
                        first_run = false;
                    }

                    // Find the maximum x that is smaller than the previous (vehicle) position
                    if (x_intersect(0) <= previous_x_(0))
                        continue;

                    if (x_intersect(0) < best_x(0))
                    {
                        best_index = j;
                        best_x = x_intersect;
                    }

                    break;

                case PolygonState::LEFT:

                    // Set the initial value very small
                    if (first_run)
                    {
                        best_x = {0.0, 999999999.0};
                        first_run = false;
                    }

                    // Find the maximum x that is smaller than the previous (vehicle) position
                    if (x_intersect(1) <= previous_x_(1))
                        continue;

                    if (x_intersect(1) < best_x(1))
                    {
                        best_index = j;
                        best_x = x_intersect;
                    }

                    break;

                default:
                    ROS_ERROR_STREAM("Polygon Constructor: Undefined state in polygon search, state: " << (int)state_);
                    break;
            }
        }

        // Make sure an index was found
        if (best_index == -1)
        {
            throw std::runtime_error("Polygon Constructor: Polygon Construction Failed: No Intersection Found!");
        }
        
        // ADD TO GRAPH LINES DATA
        // If this is the second line, extend its intersection to the start of the line (not in the middle!) for visualisation
        if (state_ == PolygonState::START){

            // graph_lines_.push_back({current_index_, getConstraint(current_index_), previous_x_, best_x, Helpers::dist(previous_x_, best_x)});
        }
        // If this is not the first line, safe the line
        if (state_ != PolygonState::START){

            graph_lines_.push_back({current_index_, getConstraint(current_index_), previous_x_, best_x, Helpers::dist(previous_x_, best_x)}); //({helpers::dist(previous_x_, best_x)});
        }
        


        // STOP CONDITION
        // If we return to a visited line while in the right_finish state, STOP
        if((state_ == PolygonState::RIGHT_FINISH || state_ == PolygonState::DOWN) && std::find(result_indices.begin(), result_indices.end(), best_index) != result_indices.end()){
            // Set the start of the first line (it was unknown at the start!)
            graph_lines_[0].point_a = best_x; // 0! Wait this doesn't always work? CHECK THIS
            break;
        }

        // STUFF WE ALWAYS NEED TO DO
        // Save the new variables
        result_indices.push_back(best_index);
       
        previous_index_ = current_index_;
        current_index_ = best_index;
        new_direction = computeDirection(getConstraint(best_index));

        // TRANSITIONS
        switch(state_){
            case PolygonState::START:
                // Always transition
                state_ = PolygonState::RIGHT_START;
                
                break;

            case PolygonState::RIGHT_START:
                // If infinite then this is a line going upwards
                if(new_direction.is_infinite_){
                    state_ = PolygonState::UP;
                // Otherwise if the line is pointing down more, flip the search direction
                }
                else if (new_direction.value_ < current_direction_.value_){ /* is shite */
                    state_ = PolygonState::LEFT;
                }

                break;

            case PolygonState::UP:
                // Always transition (assumption: only one line in vertical)
                state_ = PolygonState::LEFT;
                break;

            case PolygonState::LEFT:
                // If infinite then this is a line going upwards
                if (new_direction.is_infinite_){
                    state_ = PolygonState::DOWN;
                    // Otherwise if the line is pointing down more, flip the search direction
                }
                else if (new_direction.value_ < current_direction_.value_){
                    state_ = PolygonState::RIGHT_FINISH;
                }

                break;

            case PolygonState::DOWN:
                // Always transition
                state_ = PolygonState::RIGHT_FINISH;
                break;

            // No case right_finish, since it ends when the stop command is issued.
            case PolygonState::RIGHT_FINISH:
                break;

            default:
                ROS_ERROR_STREAM("Polygon Constructor: Unknown state in transitioning of polygon.");
                break;
        }

        // Save current variables
        current_direction_ = new_direction;
        previous_x_ = best_x;

    }

    //innerApproximate(graph_lines_);
    int range_constraints = 0;
    // first index is the start line
    for (int i = 1; i < graph_lines_.size(); i++)
    {
        if (graph_lines_[i].line_index >= n_constraints_)
            range_constraints++;
    }

    polygon_edge_count_ = graph_lines_.size() - range_constraints;

    for(int i = 0; i < graph_lines_.size(); i++){
        if (graph_lines_[i].line_index < n_constraints_)
        {
            constraints_out.push_back((*hyperplanes_)[graph_lines_[i].line_index]);
            indices_out.push_back(graph_lines_[i].line_index);
        }else
        {
            constraints_out.push_back(range_constraints_[graph_lines_[i].line_index - n_constraints_]);
            indices_out.push_back(graph_lines_[i].line_index);
        }
        
    }

    // std::cout << "Polygon edge count" << ": " << polygon_edge_count_ << std::endl;
    // std::cout << "graph_lines_.size(): " << graph_lines_.size() << std::endl;
    // std::cout << "constraint size: " << constraints_out.size() << std::endl;
}

// Using shoelace algorithm for simple polygon
double PolygonConstructor::getArea(){

    double area = 0.0;
    int n = graph_lines_.size();
    for(int i = 0; i < n - 1; i++){
        area += graph_lines_[i].point_a(0) * graph_lines_[i + 1].point_a(1);
        area -= graph_lines_[i].point_a(1) * graph_lines_[i + 1].point_a(0);
    }

    area += graph_lines_[n - 1].point_a(0) * graph_lines_[0].point_a(1);
    area -= graph_lines_[n - 1].point_a(1) * graph_lines_[0].point_a(0);
    
    return 0.5*std::abs(area);
}

void PolygonConstructor::innerApproximate(std::vector<GraphLine>& graph_lines){

    // // Count range constraints to know how much real constraints we have beforehand
    // int range_constraints = 0;
    // // // first index is the start line
    // for (int i = 1; i < graph_lines.size(); i++)
    // {
    //     if (graph_lines[i].line_index >= n_constraints_)
    //         range_constraints++;
    // }

    // polygon_edge_count_ = graph_lines.size() - range_constraints;
    // std::cout << "Edges: " << graph_lines.size() - range_constraints << std::endl;

    //     // As long as we have too much real constraints
    //     while (graph_lines.size() > n_inner_constraints_ + range_constraints)
    // {

    //     // Find the smallest distance
    //     double smallest = 99999999.0;
    //     int index = -1;

    //     // first index is the start line
    //     for (int i = 1; i < graph_lines.size(); i++)
    //     {
    //         // Cannot be range constraints
    //         if (graph_lines[i].line_index >= n_constraints_)
    //             continue;

    //         // Or lines that are connected to 2 range constraints
    //         if (graph_lines[(i + graph_lines.size() - 1) % graph_lines.size()].line_index >= n_constraints_ && graph_lines[(i + 1) % graph_lines.size()].line_index >= n_constraints_)
    //             continue;

    //         // If this is the smallest, save it
    //         if (graph_lines[i].dist < smallest)
    //         {
    //             smallest = graph_lines[i].dist;
    //             index = i;
    //         }
    //     }

    //     // If there is no valid line to remove, return!
    //     if(index == -1)
    //         return;

    //     // The next line is longer -> connect to the previous line
    //     // Exact condition: if(previous line is not a range constraint and (the next constraint is a range constraint or the next constraint is longer))
    //     if (graph_lines[(index + graph_lines.size() - 1) % graph_lines.size()].line_index < n_constraints_ &&
    //         (graph_lines[(index + 1) % graph_lines.size()].line_index >= n_constraints_ ||
    //          graph_lines[(index + 1) % graph_lines.size()].dist > graph_lines[(index + graph_lines.size() - 1) % graph_lines.size()].dist))

    //     {
    //         //std::cout << "the next line is longer\n";
    //         // Connect the start point of the previous line to the second point of the selected line
    //         LinearConstraint2D inner_constraint;
    //         Helpers::constraintFromPoints(graph_lines[(index + graph_lines.size() - 1) % graph_lines.size()].point_a, graph_lines[index].point_b, inner_constraint);

    //         // Remove the previous_line and the selected line (set the selected line for ease)
    //         graph_lines[index].line_index = 0; // Arbitrary but has to be low
    //         graph_lines[index].constraint = inner_constraint;

    //         graph_lines[index].point_a = graph_lines[(index + graph_lines.size() - 1) % graph_lines.size()].point_a;
    //         //graph_lines[i].point_b =  the same
    //         graph_lines[index].dist = Helpers::dist(graph_lines[(index + graph_lines.size() - 1) % graph_lines.size()].point_a, graph_lines[index].point_b);
    //         //std::cout << "Erasing line at " << ((index + graph_lines.size() - 1) % graph_lines.size()) << "\n ";

    //         // Remove the previous line
    //         graph_lines.erase(graph_lines.begin() + ((index + graph_lines.size() - 1) % graph_lines.size()));
    //     }
    //     // If the other possibility is also a range constraint, something is wrong
    //     else if (graph_lines[(index + 1) % graph_lines.size()].line_index >= n_constraints_)
    //     {
    //         std::cout << "---\nSmallest: " << index << " with distance " << smallest << std::endl;
    //         std::cout << "Previous: " << graph_lines[(index + graph_lines.size() - 1) % graph_lines.size()].line_index << " with distance " << graph_lines[(index + graph_lines.size() - 1) % graph_lines.size()].dist << std::endl;
    //         std::cout << "Next: " << graph_lines[(index + 1) % graph_lines.size()].line_index << " with distance " << graph_lines[(index + 1) % graph_lines.size()].dist << std::endl;
    //         throw std::runtime_error("Inner approximation tried to remove a range constraint!");
    //     }
    //     else
    //     {

    //         // the previous line is longer -> connect to the next line
    //         LinearConstraint2D inner_constraint;
    //         Helpers::constraintFromPoints(graph_lines[index].point_a, graph_lines[(index + 1) % graph_lines.size()].point_b, inner_constraint);
    //         //std::cout << "Constraint created\n";
    //         // Remove the previous_line and the selected line (set the selected line for ease)
    //         graph_lines[index].line_index = 0; // Arbitrary but has to be low
    //         graph_lines[index].constraint = inner_constraint;
    //         //graph_lines[i].point_a = graph_lines[(index + graph_lines.size() - 1) % graph_lines.size()].point_a; the same
    //         //std::cout << "Setting next point\n";

    //         graph_lines[index].point_b = graph_lines[(index + 1) % graph_lines.size()].point_b;
    //         graph_lines[index].dist = Helpers::dist(graph_lines[index].point_a, graph_lines[(index + 1) % graph_lines.size()].point_b);

    //         // Remove the next line
    //         //std::cout << "Erasing line at " << ((index + 1) % graph_lines.size()) << "\n ";

    //         graph_lines.erase(graph_lines.begin() + ((index + 1) % graph_lines.size()));
    //     }
    // }
}

// Initialise the search variables (some caviat for lines in y where the direction reverses)
// best_index = -1;
// best_x = Eigen::Vector2d();
// best_x << 9999999999.0 * search_direction, 9999999999.0 * search_direction;
// if (current_direction_.is_infinite_)
//     best_x *= -1.0;

// // Start the search over all other constraints
// for (u_int j = 0; j < n_constraints_ + 4; j++)
// {

//     // Don't check the current constraint
//     if (j == current_index_)
//         continue;

//     // Find the intersection of these lines
//     bool do_intersect = findIntersect(getConstraint(current_index_), getConstraint(j), x_intersect);

//     // if(j == 51){
//     // std::cout << "[no intersect: " << !do_intersect << " | range x: " << (abs(-x_intersect(0) + vehicle_pose_(0))) <<
//     // " | range y: " << (abs(-x_intersect(1) + vehicle_pose_(1)))
//     // << " | prev intersect: " << (x_intersect == previous_x_) << "]" << std::endl;
//     // std::cout << "intersect: " << x_intersect(0) << ", " << x_intersect(1) << std::endl;
//     // }
//     //|| abs(vehicle_pose_(0) - x_intersect(0)) > range_ + 1e-3 || abs(vehicle_pose_(1) -x_intersect(1)) > range_ + 1e-3
//     // Ensure they intersect, in range and not with the line it came from
//     if (!do_intersect || x_intersect == previous_x_) // || abs(vehicle_pose_(0) - x_intersect(0)) > 2*range_ || abs(vehicle_pose_(1) - x_intersect(1)) > 2*range_)
//         continue;
//     /*std::cout << "Searching with direction " << search_direction << ", index " << j << ", intersect\n"
//                       << x_intersect << std::endl;*/

//     /** Not the most elegant algorithm, but functional */
//     // If moving to the right
//     if (search_direction == 1)
//     {

//         // If we are searching on a line in y
//         if (current_direction_.is_infinite_)
//         {
//             // Search for a lower y
//             if (x_intersect(1) > previous_x_(1))
//                 continue;

//             // Was - < -
//             if (-x_intersect(1) + previous_x_(1) < -best_x(1) + previous_x_(1))
//             {
//                 best_x = x_intersect;
//                 best_index = j;
//             }
//         }
//         else
//         {
//             // Ignore points on the left
//             if (x_intersect(0) < previous_x_(0))
//                 continue;

//             // Otherwise find the next point based on the x
//             if (x_intersect(0) < best_x(0))
//             {
//                 best_x = x_intersect;
//                 best_index = j;
//             }
//         }
//     }
//     else
//     {

//         if (current_direction_.is_infinite_)
//         {
//             // Search for a lower y
//             if (x_intersect(1) < previous_x_(1))
//             {
//                 // std::cout << "[" << j << "]" << "failure at infinite" << std::endl;
//                 continue;
//             }
//             // Was - < -
//             if (previous_x_(1) - x_intersect(1) > previous_x_(1) - best_x(1))
//             {
//                 best_x = x_intersect;
//                 best_index = j;
//             }
//         }
//         else
//         {
//             // We are searching towards the right
//             if (x_intersect(0) > previous_x_(0))
//             {
//                 // std::cout << "[" << j << "]"
//                 //           << "failure else if" << std::endl;
//                 continue;
//             }

//             // Otherwise find the next point based on the x
//             if (x_intersect(0) > best_x(0))
//             {
//                 best_x = x_intersect;
//                 best_index = j;
//             }
//         }
//     }
//     // std::cout << "x= (" << best_x(0) << ", " << best_x(1) << "), intersect was (" << x_intersect(0) << ", " << x_intersect(1) << ")" << std::endl;
// }

// // Check for direction reversals
// // If the line is in y, the direction has to flip
// if(new_direction.is_infinite_){
//     search_direction *= -1;
//     //std::cout << "Direction Flipped (INF)\n";
// }
// // Or if the sign of the direction flips with some exceptions (coming from a line in y)
// else if (new_direction.value_ < 0 && current_direction_.value_ / new_direction.value_ <= 0.0 && !current_direction_.is_infinite_)
// {
//     search_direction *= -1;
//     //std::cout << "Direction Flipped\n";
// }