# CLEAR object map fusion 

ROS package object map fusion of multiple robots.
The node "fuser" uses the CLEAR algorithm to jointly observed objects across robots and produce a consistent global map.

Demo video with 3 robots: https://youtu.be/v5jIy5QEPiY

## Expected topics
The node "fuser" will subscribe to topics named as
"robot1/landmarks_aligned"
"robot2/landmarks_aligned"
"robot3/landmarks_aligned"
in the case of 3 robots, for example. 

The topic "landmarks_aligned" is expected to be of the sensor_msgs::PointCloud2 type, which is the object/landmark map of a robot in a common frame. 
The x-y-z position of each point indicates the location of the object in the common frame.

The node will publish the topic
"/landmarks_fused"
which is the fused object map of all robots in the common frame. 


## Launch file and parameters
See "fuser.launch" in the launch folder.
See "fuser_demo.launc" for a demo on UCSD bags.

Number of robots must be provided to the node using the (private) ros param "num_robots". Otherwise, it will take the default value of 0.

The tolerance distance for matching objects (using the nearest neighbor search)
used by the CLEAR algorithm can also be set using the parame "dist_tol", which is defaulted to 1.


