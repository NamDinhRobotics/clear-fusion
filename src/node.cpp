/**
* @file node.cpp
* @brief ROS entry point for CLEAR fusion 
* @author Kaveh Fathian <kavehfathian@gmail.com>
* @date October 2021
*/

#include <ros/ros.h>
#include "fuser.h"

int main(int argc, char **argv) 
{
	ros::init(argc, argv, "clear_fusion");
	ros::NodeHandle nhtopics("");
    ros::NodeHandle nhparams("~");
	Fuser fuser(nhtopics, nhparams);

	ros::spin();

	return 0;
}
