#pragma once

#ifndef HELPERS_H
#define HELPERS_H

#include <random>
#include <Eigen/Eigen>
#include <chrono>
#include <string>
#include <tf/tf.h>
#include <tf/transform_listener.h>

#include <std_msgs/Float64MultiArray.h>

#include <geometry_msgs/Pose.h>

#include "lmpcc_tools/ros_visuals.h"

// Copied from decomp_util!
typedef double decimal_t;

///Pre-allocated std::vector for Eigen using vec_E
template <typename T>
using vec_E = std::vector<T, Eigen::aligned_allocator<T>>;
///Eigen 1D float vector
template <int N>
using Vecf = Eigen::Matrix<decimal_t, N, 1>;
///Eigen 1D int vector
template <int N>
using Veci = Eigen::Matrix<int, N, 1>;
///MxN Eigen matrix
template <int M, int N>
using Matf = Eigen::Matrix<decimal_t, M, N>;
///MxN Eigen matrix with M unknown
template <int N>
using MatDNf = Eigen::Matrix<decimal_t, Eigen::Dynamic, N>;
///Vector of Eigen 1D float vector
template <int N>
using vec_Vecf = vec_E<Vecf<N>>;
///Vector of Eigen 1D int vector
template <int N>
using vec_Veci = vec_E<Veci<N>>;

///Eigen 1D float vector of size 2
typedef Vecf<2> Vec2f;
///Eigen 1D int vector of size 2
typedef Veci<2> Vec2i;
///Eigen 1D float vector of size 3
typedef Vecf<3> Vec3f;
///Eigen 1D int vector of size 3
typedef Veci<3> Vec3i;
///Eigen 1D float vector of size 4
typedef Vecf<4> Vec4f;
///Column vector in float of size 6
typedef Vecf<6> Vec6f;

///Vector of type Vec2f.
typedef vec_E<Vec2f> vec_Vec2f;
///Vector of type Vec2i.
typedef vec_E<Vec2i> vec_Vec2i;
///Vector of type Vec3f.
typedef vec_E<Vec3f> vec_Vec3f;
///Vector of type Vec3i.
typedef vec_E<Vec3i> vec_Vec3i;

///2x2 Matrix in float
typedef Matf<2, 2> Mat2f;
///3x3 Matrix in float
typedef Matf<3, 3> Mat3f;
///4x4 Matrix in float
typedef Matf<4, 4> Mat4f;
///6x6 Matrix in float
typedef Matf<6, 6> Mat6f;

///Dynamic Nx1 Eigen float vector
typedef Vecf<Eigen::Dynamic> VecDf;
///Nx2 Eigen float matrix
typedef MatDNf<2> MatD2f;
///Nx3 Eigen float matrix
typedef MatDNf<3> MatD3f;
///Dynamic MxN Eigen float matrix
typedef Matf<Eigen::Dynamic, Eigen::Dynamic> MatDf;

/// Hyperplane class
template <int Dim>
struct Hyperplane
{
	Hyperplane()
	{
	}
	Hyperplane(const Vecf<Dim> &p, const Vecf<Dim> &n) : p_(p), n_(n)
	{
	}

	/// Calculate the signed distance from point
	decimal_t signed_dist(const Vecf<Dim> &pt) const
	{
		return n_.dot(pt - p_);
	}

	/// Calculate the distance from point
	decimal_t dist(const Vecf<Dim> &pt) const
	{
		return std::abs(signed_dist(pt));
	}

	/// Point on the plane
	Vecf<Dim> p_;
	/// Normal of the plane, directional
	Vecf<Dim> n_;
};

/// Hyperplane2D: first is the point on the hyperplane, second is the normal
typedef Hyperplane<2> Hyperplane2D;
/// Hyperplane3D: first is the point on the hyperplane, second is the normal
typedef Hyperplane<3> Hyperplane3D;

// COPIED FROM DECOMP_UTIL!
///[A, b] for \f$Ax < b\f$
template <int Dim>
struct LinearConstraint
{
	/// Null constructor
	LinearConstraint()
	{
	}
	/// Construct from \f$A, b\f$ directly, s.t \f$Ax < b\f$
	LinearConstraint(const MatDNf<Dim> &A, const VecDf &b) : A_(A), b_(b)
	{
	}
	/**
   * @brief Construct from a inside point and hyperplane array
   * @param p0 point that is inside
   * @param vs hyperplane array, normal should go outside
   */
	LinearConstraint(const Vecf<Dim> p0, const vec_E<Hyperplane<Dim>> &vs)
	{
		const unsigned int size = vs.size();
		MatDNf<Dim> A(size, Dim);
		VecDf b(size);

		for (unsigned int i = 0; i < size; i++)
		{
			auto n = vs[i].n_;
			decimal_t c = vs[i].p_.dot(n);
			if (n.dot(p0) - c > 0)
			{
				n = -n;
				c = -c;
			}
			A.row(i) = n;
			b(i) = c;
		}

		A_ = A;
		b_ = b;
	}

	/// Check if the point is inside polyhedron using linear constraint
	bool inside(const Vecf<Dim> &pt)
	{
		VecDf d = A_ * pt - b_;
		for (unsigned int i = 0; i < d.rows(); i++)
		{
			if (d(i) > 0)
				return false;
		}
		return true;
	}

	/// Get \f$A\f$ matrix
	MatDNf<Dim> A() const
	{
		return A_;
	}

	/// Get \f$b\f$ matrix
	VecDf b() const
	{
		return b_;
	}

	MatDNf<Dim> A_;
	VecDf b_;
};

/// LinearConstraint 2D
typedef LinearConstraint<2> LinearConstraint2D;
/// LinearConstraint 3D
typedef LinearConstraint<3> LinearConstraint3D;

namespace Helpers{

	// Class for generating random ints/doubles
	class RandomGenerator
	{
	public:
		double Double()
		{
			return (double)rand() / RAND_MAX; //(double)distribution_(random_engine_) / (double)std::numeric_limits<uint32_t>::max();
		}

		int Int(int max)
		{
			return rand() % max;
		}
		//static std::mt19937 random_engine_;
		//static std::uniform_int_distribution<std::mt19937::result_type> distribution_;
	};

	inline Eigen::Matrix2d rotationMatrixFromHeading(double heading){

		Eigen::Matrix2d result;
		result << std::cos(heading), std::sin(heading),
				 -std::sin(heading), std::cos(heading);

		return result;
	}

	inline double dist(const Eigen::Vector2d& one, const Eigen::Vector2d& two){
		return (two - one).norm();
	}

	inline double quaternionToAngle(const geometry_msgs::Pose &pose)
	{
		double ysqr = pose.orientation.y * pose.orientation.y;
		double t3 = +2.0 * (pose.orientation.w * pose.orientation.z + pose.orientation.x * pose.orientation.y);
		double t4 = +1.0 - 2.0 * (ysqr + pose.orientation.z * pose.orientation.z);

		return atan2(t3, t4);
	}

	inline double quaternionToAngle(geometry_msgs::Quaternion q){

		double ysqr, t3, t4;
		
		// Convert from quaternion to RPY
		ysqr = q.y * q.y;
		t3 = +2.0 * (q.w *q.z + q.x *q.y);
		t4 = +1.0 - 2.0 * (ysqr + q.z * q.z);
		return std::atan2(t3, t4);
	}

	inline void uniformToGaussian2D(Eigen::Vector2d &uniform_variables)
	{

		// Temporarily safe the first variable
		double temp_u1 = uniform_variables(0);

		// Convert the uniform variables to gaussian via Box-Muller
		uniform_variables(0) = std::sqrt(-2 * std::log(temp_u1)) * std::cos(2 * M_PI * uniform_variables(1));
		uniform_variables(1) = std::sqrt(-2 * std::log(temp_u1)) * std::sin(2 * M_PI * uniform_variables(1));
	}

	inline bool transformPose(tf::TransformListener& tf_listener_, const std::string &from, const std::string &to, geometry_msgs::Pose &pose)
	{
		bool transform = false;
		tf::StampedTransform stamped_tf;
		
		//ROS_DEBUG_STREAM("Transforming from :" << from << " to: " << to);
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
			std::cout << "LMPCC: Quaternion was not normalised properly!" << std::endl;
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
				tf_listener_.waitForTransform(from, to, ros::Time(0), ros::Duration(0.02));
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
			if (!tf_listener_.frameExists(to)){
				ROS_WARN("%s doesn't exist", to.c_str());
			}
			if (!tf_listener_.frameExists(from))
			{
				ROS_WARN("%s doesn't exist", from.c_str());
			}
		}
		pose = stampedPose_out.pose;
		stampedPose_in.pose = stampedPose_out.pose;
		stampedPose_in.header.frame_id = to;

		return transform;
	}

	inline void drawPoint(ROSMarkerPublisher& ros_markers, const Eigen::Vector2d& point){
		// Get a line drawer and set properties
		ROSPointMarker &point_marker = ros_markers.getNewPointMarker("CUBE");
		point_marker.setColor(0, 0, 1);
		point_marker.setScale(0.2, 0.2, 0.2);

		point_marker.addPointMarker(Eigen::Vector3d(point(0), point(1), 0.2));
	}

	inline void drawLine(ROSMarkerPublisher &ros_markers, const LinearConstraint2D &constraint, int r, int g, int b, double intensity)
	{
		// Get a line drawer and set properties
		ROSLine &line = ros_markers.getNewLine();
		line.setScale(0.1, 0.1);
		double line_length = 100.0;
		line.setLifetime(1.0/20.0);
		line.setColor((double)r * intensity, (double)g * intensity, (double)b * intensity);

		// Loop through the columns of the constraints
		for (int i = 0; i < constraint.b_.rows(); i++)
		{

			// Constraint in z
			if (std::abs(constraint.A_(i, 0)) < 0.01 && std::abs(constraint.A_(i, 1)) < 0.01)
			{
				ROS_WARN("Invalid constraint ignored during visualisation!");
			}

			geometry_msgs::Point p1, p2;
			// Debug!
			double z = 0.2;
			if (r == 1)
				z = 0.3;
			// If we cant draw in one direction, draw in the other
			if (std::abs(constraint.A_(i, 0)) < 0.01)
			{
				p1.x = -line_length;
				p1.y = (constraint.b_(i) + constraint.A_(i, 0) * line_length) / constraint.A_(i, 1);
				p1.z = z;

				p2.x = line_length;
				p2.y = (constraint.b_(i) - constraint.A_(i, 0) * line_length) / constraint.A_(i, 1);
				p2.z = z;
			}
			else
			{

				// Draw the constraint as a line
				p1.y = -line_length;
				p1.x = (constraint.b_(i) + constraint.A_(i, 1) * line_length) / constraint.A_(i, 0);
				p1.z = z;

				p2.y = line_length;
				p2.x = (constraint.b_(i) - constraint.A_(i, 1) * line_length) / constraint.A_(i, 0);
				p2.z = z;
			}

			line.addLine(p1, p2);
		}
	};

	inline void drawLinearConstraints(ROSMarkerPublisher& ros_markers, const std::vector<LinearConstraint2D>& constraints, int r = 0, int g = 1, int b = 0){

		for(u_int k = 0; k < constraints.size(); k++){

			double intensity = std::atan(((double)k + constraints.size() * 0.5) / ((double)constraints.size() * 1.5));

			drawLine(ros_markers, constraints[k], r, g, b, intensity);
		}

	};

	inline void drawLinearConstraints(ROSMarkerPublisher& ros_markers, const std::vector<LinearConstraint2D>& constraints, const std::vector<int>& indices,
									int r = 0, int g = 1, int b = 0){


		for (size_t k = 0; k < indices.size(); k++)
		{
			const int &index = indices[k];

			const LinearConstraint2D &constraint = constraints[index];

			// Chance the color for every polygon
			double intensity = std::atan(((double)index + indices.size() * 0.5) / ((double)indices.size() * 1.5));

			drawLine(ros_markers, constraint, r, g, b, intensity);
		}
	};

	// Use as static to print average run time
	class Benchmarker
	{

	public:
		Benchmarker(const std::string &name, bool record_duration = false)
		{

			name_ = name;
			record_duration_ = record_duration;
			running_ = false;
		}

		// Simpler
		Benchmarker()
		{
		}

		void initialize(const std::string &name, bool record_duration = false)
		{
			name_ = name;
			record_duration_ = record_duration;
		}

		// Print results on destruct
		~Benchmarker()
		{

			double average_run_time = total_duration_ / ((double)total_runs_) * 1000.0;

			std::cout << "Timing Results for [" << name_ << "]\n";
			std::cout << "Average: " << average_run_time << " ms\n";
			std::cout << "Min: " << min_duration_ * 1000.0 << " ms\n";
			std::cout << "Max: " << max_duration_ * 1000.0 << " ms\n";
		}

		void start()
		{
			running_ = true;
			start_time_ = std::chrono::system_clock::now();

		}

		double stop()
		{ 
			if(!running_)
				return 0.0;

			auto end_time = std::chrono::system_clock::now();
			std::chrono::duration<double> current_duration = end_time - start_time_;

			if (record_duration_)
				duration_list_.push_back(current_duration.count() * 1000.0); // in ms

			if (current_duration.count() < min_duration_)
				min_duration_ = current_duration.count();

			if (current_duration.count() > max_duration_)
				max_duration_ = current_duration.count();

			total_duration_ += current_duration.count();
			total_runs_++;
			running_ = false;

			return current_duration.count();
		}

		void dataToMessage(std_msgs::Float64MultiArray &msg)
		{

			msg.data.resize(duration_list_.size());

			for (size_t i = 0; i < duration_list_.size(); i++)
				msg.data[i] = duration_list_[i];
		}

		void reset(){
			total_runs_ = 0;
			total_duration_ = 0.0;
			max_duration_ = -1.0;
			min_duration_ = 99999.0;
		}

		bool isRunning(){return running_;};

		int getTotalRuns(){return total_runs_;};

	private:
		std::chrono::system_clock::time_point start_time_;

		double total_duration_ = 0.0;
		double max_duration_ = -1.0;
		double min_duration_ = 99999.0;

		int total_runs_ = 0;

		std::string name_;
		bool record_duration_;
		std::vector<double> duration_list_;
		bool running_ = false;
	};
};

#endif