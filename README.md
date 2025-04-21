# UWB SAR   

0. Copy the _ROS2 bag file_ from:
```
https://sutdapac-my.sharepoint.com/:f:/g/personal/gihan_appuhamilage_mymail_sutd_edu_sg/EkzMG0b83f1Ni5dh-KxZ6U4BH8bj5rxQERRDwAQnGY82Mg?e=o2BNrD
```
1. Open two terminals: T#1, T#2  
  
**T#1**  
  
2. Clone the repo.  
3. Build the workspace.  
```
colcon build
```
3. Source the ws.
```
source install/setup.bash
```
4. Run the SAR generator:
```
ros2 run uwb_sar sar_node
```

**T#2**  
  
1. Play the _ROS2 bag file_:
```
ros2 bag play rosbag2_2025_aria
```
2. Play till the end. It will generate two `.mat` files in the `src`.  
3. Once the files are generated, evaluate the images using:
```
ros2 run uwb_sar fm_node
```


## NOTES:  

The `results_loops_1` contains feature matching figures related to the Fig. 14 (left) of the paper.  
The `results_loops_2` contains feature matching figures related to the Fig. 14 (right) of the paper.  

Contact `gihan_appuhamilage@mymail.sutd.edu.sg` for troubleshooting.
