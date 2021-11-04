/**
* @file fuser.h
* @brief ROS wrapper for CLEAR fusion 
* @author Kaveh Fathian <kavehfathian@gmail.com>
* @date October 2021
*/

#pragma once

#include "clear/PairwiseMatcher.hpp"
#include "clear/MultiwayMatcher.hpp"

#include <vector>
#include <cmath>
#include <random>
#include <deque>
#include <utility>
#include <cstdint>
#include <string>
#include <fstream>
#include <algorithm>
#include <iterator>
#include <numeric>
#include <assert.h>
#include <pthread.h>

// ROS includes
#include <ros/ros.h>
#include <ros/time.h>
#include <tf2_eigen/tf2_eigen.h>
#include <ros/console.h>
#include <std_msgs/MultiArrayDimension.h>
#include <sensor_msgs/LaserScan.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>
#include <geometry_msgs/Point32.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseArray.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <nav_msgs/Path.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PoseWithCovariance.h>


#include <Eigen/Dense>
#include <Eigen/Core>

class Fuser{
public:
	Fuser(const ros::NodeHandle& nh_, const ros::NodeHandle& nhp_);
	~Fuser();


private:
	int num_robots = 0; // number of all robots
	double dist_tol	= 1; // tolerance distance for matching (only object < dist_tol can match)

private:
	ros::NodeHandle nh, nhp;
	ros::Timer tim_fuser;
	double fuser_dt; /// Period of fuser runs and visualization
	std::string frame_id;
	
	// Subscribers (for each robot)
	std::vector<ros::Subscriber> sub_r_landmarks;

	// CLEAR Publishers
	ros::Publisher pub_fused_landmarks; // publish fused landmarks
	
	// vars used in CLEAR 
	std::vector<std::vector<Eigen::Vector3f>> map;
	std::vector<uint> SubmapSizes;
	std::vector<uint> CumSum; 
	Eigen::MatrixXf A; 
	std::vector<int> assignments;
	std::vector<uint> fusedCounts;

	// CLEAR Algorthim
	PairwiseMatcher* Pairmatcher;
	MultiwayMatcher* Multimatcher;
	bool filter_submaps();
	void pairwise_match();
	void multiway_match();

	// callbacks
	void landmarks_cb(const sensor_msgs::PointCloud2ConstPtr& msg, const uint i);
	void timer_cb(const ros::TimerEvent& event);
};