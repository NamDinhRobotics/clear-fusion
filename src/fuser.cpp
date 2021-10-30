/**
* @file fuser.cpp
* @brief ROS wrapper for CLEAR fusion 
* @author Kaveh Fathian <kavehfathian@gmail.com>
* @date October 2021
*/

#include "fuser.h"

#include <sys/types.h>
#include <sys/stat.h>
#include <numeric>
#include <iostream>
#include <fstream>
#include <string>
#include <cassert>

#include <sensor_msgs/point_cloud_conversion.h>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/transforms.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>


Fuser::Fuser(const ros::NodeHandle& nh_, const ros::NodeHandle& nhp_) 
: nh(nh_), nhp(nhp_) 
{
	nhp.getParam("num_robots", num_robots);
	ROS_INFO_STREAM("num_robots: " << num_robots);
	if (num_robots==0) return;
	// CLEAR params
	nhp.getParam("dist_tol", dist_tol); 
	ROS_INFO_STREAM("dist_tol: " << dist_tol);

	Pairmatcher = new PairwiseMatcher(dist_tol);
	Multimatcher = new MultiwayMatcher();
	Multimatcher->set_verbose(false);

	// initialize subscribers for all robots
	sub_r_landmarks.resize(num_robots);
	map.resize(num_robots);
	SubmapSizes.resize(num_robots);

	for (size_t i; i<num_robots; ++i) {
		boost::function<void(const sensor_msgs::PointCloud2ConstPtr& msg)> cb =
			[=] (const sensor_msgs::PointCloud2ConstPtr& msg) {landmarks_cb(msg, i);};
		sub_r_landmarks[i] = nh.subscribe("/robot"+std::to_string(i+1)+"/landmarks_aligned", 1, cb);
		
	}

	// initialize publishers
	pub_fused_landmarks = nh.advertise<sensor_msgs::PointCloud2>("/landmarks_fused", 1);

	nhp.param<double>("fuser_dt", fuser_dt, 1);
	tim_fuser = nh.createTimer(ros::Duration(fuser_dt), &Fuser::timer_cb, this);

	ROS_INFO_STREAM("CLEAR node initialized.");
}

Fuser::~Fuser() {}


// ----------------------------------------------------------------------------
// ROS Callbacks
// ----------------------------------------------------------------------------

void Fuser::landmarks_cb(const sensor_msgs::PointCloud2ConstPtr& msg, const uint i) 
{
	// convert to pcl point cloud
	pcl::PointCloud<pcl::PointXYZ>::Ptr ptcloud (new pcl::PointCloud<pcl::PointXYZ>()); 
	pcl::fromROSMsg(*msg, *ptcloud);

	std::vector<Eigen::Vector3f> submap; // robot landmarks
	
	for (uint i=0; i < ptcloud->points.size(); ++i) {
		pcl::PointXYZ pt = ptcloud->points[i];
		Eigen::Vector3f ept;
		ept << pt.x, pt.y, pt.z;
		submap.emplace_back(ept);
	}

	map[i] = submap; // all robot landmarks
	SubmapSizes[i] = submap.size();
}


// ----------------------------------------------------------------------------

void Fuser::timer_cb(const ros::TimerEvent& event)
{	
	// for (size_t i=0; i<num_robots; ++i){
	// 	ROS_WARN_STREAM("SubmapSizes["<<i<<"]: " << SubmapSizes[i]);
	// }

	// make sure we have received data from all robots
	for (size_t i=0; i<num_robots; ++i){
		if (SubmapSizes[i] < 1) return;
	}

	// // make sure we have received enough data
	// uint nonemptymaps = 0; 
	// for (size_t i=0; i<num_robots; ++i){
	// 	if (SubmapSizes[i] > 0) nonemptymaps ++;
	// }
	// if (nonemptymaps < 2) return;


	// return if no data
	int sum = std::accumulate(SubmapSizes.begin(), SubmapSizes.end(), 0);
	if (sum <1) return;

	// associate landmarks between robot pairs based on proximity	
	Fuser::pairwise_match();

	// run CLEAR to make matches cycle consistent and fuse them
	Fuser::multiway_match();

	// total number of objects after fusion
	uint m = fusedCounts.size();

	std::vector<Eigen::Vector3f> fusedPositions;
	std::vector<float> myFusedCounts(m,0.0);

	for (uint i=0; i<m; ++i) {
		fusedPositions.push_back(Eigen::Vector3f::Zero());
	}

	for (uint i = 0; i<num_robots; ++i)
	{
		std::vector<Eigen::Vector3f> submap = map[i];
		// uint start = 0;
		// if(i > 0){
		// 	start = CumSum[i-1]; // the global index of the first tree in this submap
		// }
		uint start = CumSum[i]; // the global index of the first object in this submap
		for (uint k = 0; k < submap.size(); ++k){
			// recover the corresponding tree in the universe
			int index = assignments[start + k];
			// assert(index >= 0 && index < m);
			fusedPositions[index] = fusedPositions[index] + submap[k];
			myFusedCounts[index] += 1.0;
		}
	}

	// ROS_WARN_STREAM("Fused counts");
	// for (uint i = 0; i < m; ++i){
	// 	ROS_WARN_STREAM(myFusedCounts[i]);
	// }

	for (uint i = 0; i < m; ++i) {
		fusedPositions[i] = fusedPositions[i] / myFusedCounts[i];
	}

	// Publish
	sensor_msgs::PointCloud fused_landmark_cloud;
	fused_landmark_cloud.header.stamp = ros::Time::now();
	fused_landmark_cloud.header.frame_id = "R1";
	for (uint i = 0; i < m; ++i){
		geometry_msgs::Point32 p;
		p.x = fusedPositions[i](0);
		p.y = fusedPositions[i](1);
		p.z = fusedPositions[i](2);
		fused_landmark_cloud.points.push_back(p);
	}

	sensor_msgs::PointCloud2 fused_set_msg;
    sensor_msgs::convertPointCloudToPointCloud2(fused_landmark_cloud, fused_set_msg);
	fused_set_msg.header.frame_id = "R1";

	pub_fused_landmarks.publish(fused_set_msg);
}

// ----------------------------------------------------------------------------
void Fuser::pairwise_match() 
{
	// ROS_INFO_STREAM("CLEAR: Pairwise matching...");
	
	// cumulative sum
	CumSum.resize(map.size()+1, 0);
	std::partial_sum(SubmapSizes.begin(), SubmapSizes.end(), CumSum.begin()+1);

	// total number of objects in all submaps
	uint totalSum = CumSum.back();

	// initialize adjacency matrix for CLEAR
	A = Eigen::MatrixXf::Zero(totalSum, totalSum);

    // reset runtime
    double pairwise_matching_time = 0;
    double pairwise_matching_count = 0;
	// Try to match each pair of submaps
	for (uint j = 0; j < num_robots; ++j){
		for (uint i = 0; i < j ; ++i){
			std::vector<Eigen::Vector3f> pc1 = map[i];
			std::vector<Eigen::Vector3f> pc2 = map[j];
			bool success;
			double start_time = ros::Time::now().toSec();

			success = Pairmatcher->nn_match(pc1, pc2);
			pairwise_matching_time += ros::Time::now().toSec() - start_time;
			pairwise_matching_count += 1.0;

			if (success){
				uint size1 = pc1.size();
				uint size2 = pc2.size();
				uint start1 = CumSum[i];
				uint start2 = CumSum[j];
				Eigen::MatrixXf P12;
				Pairmatcher->get_permutation_matrix(P12);
				A.block(start1, start2, size1, size2) = P12;
				A.block(start2, start1, size2, size1) = P12.transpose();
			}
		}
	}

	// ROS_WARN_STREAM("A:" << A);

	// ROS_INFO_STREAM("CLEAR: Average pairwise matching elapsed time = " << pairwise_matching_time / pairwise_matching_count << "sec.");
}

// ----------------------------------------------------------------------------

void Fuser::multiway_match() 
{
	// ROS_INFO_STREAM("CLEAR multiway matching...");

	double start_time = ros::Time::now().toSec();

	// for (uint i=0; i<SubmapSizes.size(); ++i) {
	// 	ROS_INFO_STREAM("SubmapSizes[i]:" << SubmapSizes[i]);
	// }

	if (SubmapSizes.empty()) return; // if no data 

	Multimatcher->initialize(A, SubmapSizes);
	Multimatcher->CLEAR();
	Multimatcher->get_assignments(assignments);
	Multimatcher->get_fused_counts(fusedCounts);

	ROS_WARN_STREAM("CLEAR multiway fusion returned " << fusedCounts.size() 
		<< " objects in " << ros::Time::now().toSec() - start_time << " sec.");

	// ROS_INFO_STREAM("CLEAR multiway match complete in " << ros::Time::now().toSec() - start_time << " sec.");
}




