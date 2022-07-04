#include "lmpcc_tools/ros_visuals.h"


ROSMarkerPublisher::ROSMarkerPublisher(ros::NodeHandle& nh, const char* topic_name, const std::string& frame_id, int max_size)
    :frame_id_(frame_id), max_size_(max_size)
{
    
    pub_ = nh.advertise<visualization_msgs::MarkerArray>(topic_name, 3);
    id_ = 0;

    // We allocate space for the markers in initialization 
    marker_list_.markers.resize(max_size);
    prev_marker_list_.markers.resize(max_size);

    for (int i = 0; i < max_size; i++)
    {
        marker_list_.markers[i].id = i;
        marker_list_.markers[i].action = visualization_msgs::Marker::DELETE;
        marker_list_.markers[i].header.frame_id = frame_id_;
    }
}


void ROSMarkerPublisher::add(const visualization_msgs::Marker& marker){
    
    if(marker.id >= max_size_)
        ROS_ERROR_STREAM("ROS VISUALS: Marker ID " << marker.id << " exceeded max size of " << max_size_ << "!");

    //marker_list_.markers.push_back(marker);
    marker_list_.markers[marker.id] = marker;
}

ROSLine& ROSMarkerPublisher::getNewLine(){
    
    // Create a line
    ros_markers_.emplace_back(new ROSLine(this, frame_id_));

    // Return a pointer
    return (* ((ROSLine*)(&(*ros_markers_.back()))));
    
}

ROSPointMarker& ROSMarkerPublisher::getNewPointMarker(const std::string& marker_type)
{

    //std::unique_ptr<ROSMarker> ros_point(new ROSPointMarker(this, frame_id_, marker_type));

    // Create a point marker
    ros_markers_.emplace_back(new ROSPointMarker(this, frame_id_, marker_type));

    // Return a pointer
    return (*((ROSPointMarker*)(&(*ros_markers_.back()))));
}

ROSMultiplePointMarker& ROSMarkerPublisher::getNewMultiplePointMarker()
{

   // std::unique_ptr<ROSMarker> ros_points(new ROSMultiplePointMarker(this, frame_id_));

    // Create a point marker
   ros_markers_.emplace_back(new ROSMultiplePointMarker(this, frame_id_));

   // Return a pointer
   return (*((ROSMultiplePointMarker *)(&(*ros_markers_.back()))));
}

void ROSMarkerPublisher::publish(){

    // all ids without markers are deleted!
    for(int i = id_; i < max_size_; i++){
        marker_list_.markers[i].action = visualization_msgs::Marker::DELETE;
    }

    if (id_ == 0)
        return;

    // Add new markers
    for(std::unique_ptr<ROSMarker>& marker: ros_markers_){
        marker->stamp();
    }

    // Save the markers in case that we need to remove them afterwards!
    prev_marker_list_ = marker_list_;

    pub_.publish(marker_list_);

    id_ = 0;
}

ROSMarkerPublisher::~ROSMarkerPublisher()
{
    // for (visualization_msgs::Marker &marker : prev_marker_list_.markers)
    // {
    //     marker.action = visualization_msgs::Marker::DELETE;
    // }

    // pub_.publish(prev_marker_list_);
}

int ROSMarkerPublisher::getID(){
    
    int cur_id = id_;
    id_++;
    
    return cur_id;
    
}

    
ROSMarker::ROSMarker(ROSMarkerPublisher* ros_publisher, const std::string& frame_id)
: ros_publisher_(ros_publisher)
{
    marker_.header.frame_id = frame_id;
}
    
void ROSMarker::stamp(){
    marker_.header.stamp = ros::Time::now();
}

void ROSMarker::setColor(double r, double g, double b){
        
    setColor(r, g, b, 1.0);
    
}

void ROSMarker::setColor(double r, double g, double b, double alpha){
        
    marker_.color.r = r;
    marker_.color.g = g;
    marker_.color.b = b;
    marker_.color.a = alpha;
    
}

void ROSMarker::setColor(double ratio){
    double red, green, blue;
    
    getColorFromRange(ratio, red, green, blue);
    setColor(red, green, blue);
}

void ROSMarker::setColor(double ratio, double alpha)
{
    double red, green, blue;

    getColorFromRange(ratio, red, green, blue);
    setColor(red, green, blue, alpha);
}

void ROSMarker::setScale(double x, double y, double z){
        
    marker_.scale.x = x;
    marker_.scale.y = y;
    marker_.scale.z = z;
    
}

void ROSMarker::setScale(double x, double y)
{

    marker_.scale.x = x;
    marker_.scale.y = y;
}

void ROSMarker::setScale(double x)
{
    marker_.scale.x = x;
}

void ROSMarker::setOrientation(double theta)
{

    tf::Quaternion q = tf::createQuaternionFromRPY(0, 0, theta);
    setOrientation(q);
}

void ROSMarker::setOrientation(const geometry_msgs::Quaternion& msg)
{

    tf::Quaternion q;
    tf::quaternionMsgToTF(msg, q);
    setOrientation(q);
}

void ROSMarker::setOrientation(const tf::Quaternion& q)
{
    marker_.pose.orientation.x = q.x();
    marker_.pose.orientation.y = q.y();
    marker_.pose.orientation.z = q.z();
    marker_.pose.orientation.w = q.w();
}

void ROSMarker::setLifetime(double lifetime){
    
    marker_.lifetime = ros::Duration(lifetime);
}

void ROSMarker::setActionDelete()
{

    marker_.action = visualization_msgs::Marker::DELETE;
}

geometry_msgs::Point ROSMarker::vecToPoint(const Eigen::Vector3d& v){
    geometry_msgs::Point p;
    p.x = v(0); p.y = v(1); p.z = v(2);
    return p;
}

void ROSMarker::getColorFromRange(double ratio, double &red, double &green, double &blue)
{
    //we want to normalize ratio so that it fits in to 6 regions
    //where each region is 256 units long
    int normalized = int(ratio * 256 * 6);

    //find the distance to the start of the closest region
    int x = normalized % 256;

    switch (normalized / 256)
    {
    case 0:
        red = 255;
        green = x;
        blue = 0;
        break; //red
    case 1:
        red = 255 - x;
        green = 255;
        blue = 0;
        break; //yellow
    case 2:
        red = 0;
        green = 255;
        blue = x;
        break; //green
    case 3:
        red = 0;
        green = 255 - x;
        blue = 255;
        break; //cyan
    case 4:
        red = x;
        green = 0;
        blue = 255;
        break; //blue
    case 5:
        red = 255;
        green = 0;
        blue = 255 - x;
        break; //magenta
    }

    red /= 256.0;
    green /= 256.0;
    blue /= 256.0;

    
}

ROSLine::ROSLine(ROSMarkerPublisher* ros_publisher, const std::string& frame_id)
: ROSMarker(ros_publisher, frame_id)
{

    marker_.type = visualization_msgs::Marker::LINE_LIST;

    // LINE_STRIP/LINE_LIST markers use only the x component of scale, for the line width
    marker_.scale.x = 0.5;
    // marker_.scale.y = 0.5;

    // Line strip is red
    setColor(1, 0, 0);
    
    marker_.pose.orientation.x = 0.0;
    marker_.pose.orientation.y = 0.0;
    marker_.pose.orientation.z = 0.0;
    marker_.pose.orientation.w = 1.0;
}
    
void ROSLine::addLine(const Eigen::Vector3d& p1, const Eigen::Vector3d& p2){
    
    addLine(vecToPoint(p1), vecToPoint(p2));
    
}

void ROSLine::addLine(const geometry_msgs::Point& p1, const geometry_msgs::Point& p2){
    
    // Request an ID
    marker_.id = ros_publisher_->getID();
    
    // Add the points
    marker_.points.push_back(p1);
    marker_.points.push_back(p2);
    
    // Add line to the publisher
    ros_publisher_->add(marker_);
    
    // Clear points
    marker_.points.clear();
    
}

ROSPointMarker::ROSPointMarker(ROSMarkerPublisher *ros_publisher, const std::string &frame_id, const std::string& marker_type)
    : ROSMarker(ros_publisher, frame_id), marker_type_(marker_type)
{

    marker_.type = getMarkerType(marker_type_);

    // LINE_STRIP/LINE_LIST markers use only the x component of scale, for the line width
    marker_.scale.x = 0.5;
    marker_.scale.y = 0.5;

    setColor(1, 0, 0);
    
    marker_.pose.orientation.x = 0.0;
    marker_.pose.orientation.y = 0.0;
    marker_.pose.orientation.z = 0.0;
    marker_.pose.orientation.w = 1.0;
}

void ROSPointMarker::addPointMarker(const geometry_msgs::Pose& pose)
{
    // Request an ID
    marker_.id = ros_publisher_->getID();

    marker_.pose = pose;

    // Add line to the publisher
    ros_publisher_->add(marker_);
}

void ROSPointMarker::addPointMarker(const Eigen::Vector3d &p1)
{

    addPointMarker(vecToPoint(p1));
}

void ROSPointMarker::addPointMarker(const geometry_msgs::Point &p1)
{

    // Request an ID
    marker_.id = ros_publisher_->getID();
    
    marker_.pose.position = p1;
    
    // Add line to the publisher
    ros_publisher_->add(marker_);
}

uint ROSPointMarker::getMarkerType(const std::string &marker_type)
{

    if(marker_type == "CUBE")
        return visualization_msgs::Marker::CUBE;
    if(marker_type == "SPHERE")
        return visualization_msgs::Marker::SPHERE;
    if(marker_type == "POINTS")
        return visualization_msgs::Marker::POINTS;
    if(marker_type == "CYLINDER")
        return visualization_msgs::Marker::CYLINDER;

    return visualization_msgs::Marker::CUBE;
    
}

ROSMultiplePointMarker::ROSMultiplePointMarker(ROSMarkerPublisher *ros_publisher, const std::string &frame_id)
    : ROSMarker(ros_publisher, frame_id)
{

    marker_.type = visualization_msgs::Marker::POINTS;


    // LINE_STRIP/LINE_LIST markers use only the x component of scale, for the line width
    marker_.scale.x = 0.5;
    marker_.scale.y = 0.5;

    setColor(0, 1, 0);

    marker_.pose.orientation.x = 0.0;
    marker_.pose.orientation.y = 0.0;
    marker_.pose.orientation.z = 0.0;
    marker_.pose.orientation.w = 1.0;
}

void ROSMultiplePointMarker::addPoint(const geometry_msgs::Point &p1)
{
    marker_.points.push_back(p1);
}

void ROSMultiplePointMarker::addPoint(const Eigen::Vector3d &p1)
{
    geometry_msgs::Point result;
    result.x = p1(0);
    result.y = p1(1);
    result.z = p1(2);

    addPoint(result);
}

void ROSMultiplePointMarker::addPoint(const geometry_msgs::Pose &pose)
{
    geometry_msgs::Point result;
    result.x = pose.position.x;
    result.y = pose.position.y;
    result.z = pose.position.z;

    addPoint(result);
}

void ROSMultiplePointMarker::finishPoints(){

    marker_.id = ros_publisher_->getID();
    ros_publisher_->add(marker_);
}