#ifndef ROS_VISUALS_H
#define ROS_VISUALS_H

#include <ros/ros.h>
#include <Eigen/Core>
#include <tf/tf.h>

// Visualization messages
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <string>

class ROSMarker;
class ROSLine;
class ROSPointMarker;
class ROSMultiplePointMarker;

class ROSMarkerPublisher{
    
public:
    ROSMarkerPublisher(ros::NodeHandle& nh, const char* topic_name, const std::string& frame_id, int max_size);
    ~ROSMarkerPublisher();

private:
    
    // One publisher
    ros::Publisher pub_;
    
    // One marker list
    visualization_msgs::MarkerArray marker_list_;
    visualization_msgs::MarkerArray prev_marker_list_;
    
    // a set of ros_markers
    std::vector< std::unique_ptr<ROSMarker> > ros_markers_;
    
    std::string frame_id_;
    int id_;
    int max_size_;
    
public:

    void add(const visualization_msgs::Marker& marker);
    
    ROSLine& getNewLine();
    ROSPointMarker& getNewPointMarker(const std::string& marker_type);
    ROSMultiplePointMarker &getNewMultiplePointMarker(); //const std::string &marker_type);
    //ROSEllipse& getNewEllipse();

    void publish();
    
    int getID();
    
    std::string getFrameID() const;
    
};

class ROSMarker{
    
public:
    
    ROSMarker(ROSMarkerPublisher* ros_publisher, const std::string& frame_id);
    // int start_id_;
    // int end_id_;
    // int cur_id_;

protected:
    visualization_msgs::Marker marker_;
    
    ROSMarkerPublisher* ros_publisher_;

    
    geometry_msgs::Point vecToPoint(const Eigen::Vector3d& v);
    void getColorFromRange(double ratio, double &red, double &green, double &blue);
    
public:
    
    void setColor(double r, double g, double b);
    void setColor(double r, double g, double b, double alpha);
    void setColor(double ratio); // Color rainbow wise
    void setColor(double ratio, double alpha);
    void setScale(double x, double y, double z);
    void setScale(double x, double y);
    void setScale(double x);
    void setOrientation(double theta);
    void setOrientation(const geometry_msgs::Quaternion& msg);
    void setOrientation(const tf::Quaternion& q);
    void setLifetime(double lifetime);

    void setActionDelete();
    // void setIDRange(int range);
    // void resetID(); // Internal id count

    void stamp();
};

class ROSLine : public ROSMarker{
    
public:
    ROSLine(ROSMarkerPublisher* ros_publisher, const std::string& frame_id);
    
    void addLine(const Eigen::Vector3d& p1, const Eigen::Vector3d& p2);
    void addLine(const geometry_msgs::Point& p1, const geometry_msgs::Point& p2);
    
};

class ROSPointMarker : public ROSMarker{
    
public:
    ROSPointMarker(ROSMarkerPublisher* ros_publisher, const std::string& frame_id, const std::string& marker_type);
    
    void addPointMarker(const Eigen::Vector3d& p1);
    void addPointMarker(const geometry_msgs::Point& p1);
    void addPointMarker(const geometry_msgs::Pose& pose);

private:
    std::string marker_type_;

    uint getMarkerType(const std::string &marker_type);
};

class ROSMultiplePointMarker : public ROSMarker
{

public:
    ROSMultiplePointMarker(ROSMarkerPublisher *ros_publisher, const std::string &frame_id);

    void addPoint(const Eigen::Vector3d& p1);
    void addPoint(const geometry_msgs::Point& p1);
    void addPoint(const geometry_msgs::Pose& pose);

    void finishPoints();

private:
};

#endif