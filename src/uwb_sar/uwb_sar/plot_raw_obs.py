## This program is used to plot the raw observations from the radars and the odometry data.
## Works alongside a ros node that publishes odometry and trilateration data from the radars.
## Run either the trilaterator and the odometry node or the simulation node to get the data.
## or run the rosbag file that contains the data.
from lp_slam.src import ekf_funcs_lp as ekf
from matplotlib import pyplot as plt
import time
import numpy as np
import pickle
import math
import pdb
import math
import scipy.io

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from nav_msgs.msg import Odometry
from scipy.spatial.transform import Rotation as rot

np.random.seed(10) # for reproducibility
plt.ion()

class plot_obs(Node):
    def __init__(self, topic_name_odom, topic_name_odom_filtered, topic_name_tri_L, topic_name_tri_R):
        super().__init__('Raw_Observations') # name of the node
        self.del_l = 0 # robot linera displacement
        self.del_th = 0 # robot rotation
        self.prev_odom = np.zeros(3, dtype=np.float32)
        self.tmp_l_obs = [] # temporary left observations >> range and bearing
        self.tmp_l_obs_xy = [] # temporary left observations >> x and y
        self.tmp_l_obs_count = 0 # temporary left observations count
        self.tmp_r_obs = [] # temporary right observations >> range and bearing
        self.tmp_r_obs_xy = [] # temporary right observations >> x and y
        self.tmp_r_obs_count = 0 # temporary right observations count
        
        #**********************************************************************************************
        # Plot
        self.fig, self.ax = plt.subplots() 
        self.ax.set_aspect('equal', adjustable='box')
        self.fig.suptitle('raw observations')
        plt.ylabel("y [mm]")
        plt.xlabel("x [mm]")
        plt.grid()
        plt.show()

        self.odom_plt, = plt.plot([], [], linewidth=1, color='b') # plot the odometry
        self.fig.canvas.draw() 
        self.odom_filt_plt, = plt.plot([], [], linewidth=1, color='r') # plot the filtered odometry
        self.fig.canvas.draw()
        self.l_obs_plt = plt.scatter([], [], s=30, c='g', marker='*', label='LHS raw_observations') # plot the left observations
        self.fig.canvas.draw() 
        self.r_obs_plt = plt.scatter([], [], s=30, c='g', marker='*', label='RHS raw_observations') # plot the right observations
        self.fig.canvas.draw() 
        self.fig.canvas.flush_events() # update the plot
        
        self.odom_filt_x = 0
        self.odom_filt_y = 0
        self.odom_x_data, self.odom_y_data = [] , [] # odometry data
        self.odom_filt_x_data, self.odom_filt_y_data = [] , [] # filtered odometry data

        self.l_obs_data = [] # left observations data >> range and bearing
        self.l_obs_xy = [] # left observations data >> x and y
        self.count_l_obs = [] # left observations count

        self.r_obs_data = [] # right observations data >> range and bearing
        self.count_r_obs = [] # right observations count
        self.r_obs_xy = [] # right observations data >> x and y
        # self.xlims = [0, 450]
        # self.ylims = [-200, 200]

        self.id = 0 # index for the data.

        #**********************************************************************************************
        # Save data from the odometry
        self.subscription = self.create_subscription(
            Odometry,
            topic_name_odom,
            self.update_del_odom,
            1
        )
        self.subscription

        # Save data from the odometry_filtered
        self.subscription = self.create_subscription(
            Odometry,
            topic_name_odom_filtered,
            self.update_filt_odom,
            1
        )
        self.subscription

        # Save left trilateration data
        self.subscription = self.create_subscription(
            Float32MultiArray,
            topic_name_tri_L,
            self.update_l_obs,
            1
        )
        self.subscription

        # Save right trilateration data
        self.subscription = self.create_subscription(
            Float32MultiArray,
            topic_name_tri_R,
            self.update_r_obs,
            1
        )
        self.subscription

    # Update the plot of the odometry
    def update_plot_odom(self): 
        self.odom_plt.set_data(self.odom_x_data, self.odom_y_data)
        self.odom_filt_plt.set_data(self.odom_filt_x_data, self.odom_filt_y_data)
        return self.ax
    
    # Update the raw observations
    def update_obs(self):
        self.l_obs_data = self.l_obs_data + self.tmp_l_obs # concatenate lists
        self.count_l_obs.append(self.tmp_l_obs_count)
        self.l_obs_xy = self.l_obs_xy + self.tmp_l_obs_xy # concatenate lists

        self.r_obs_data = self.r_obs_data + self.tmp_r_obs # concatenate lists
        self.count_r_obs.append(self.tmp_r_obs_count)
        self.r_obs_xy = self.r_obs_xy + self.tmp_r_obs_xy # concatenate lists
        return 0
    
    def update_plot_obs(self):
        # print('l_obs_xy: ', self.l_obs_xy)
        # print('r_obs_xy: ', self.r_obs_xy)
        if self.l_obs_xy:
            self.l_obs_plt.set_offsets(self.l_obs_xy)
        if self.r_obs_xy:
            self.r_obs_plt.set_offsets(self.r_obs_xy)
        # self.l_obs_plt.set_offsets([[1,2], [3,2]])
        # self.r_obs_plt.set_offsets([[-2,1], [-2,3]])
        return self.ax
    
    def rth2xy(self, pose, r, th):
        x = pose[0] + r * math.cos(th + pose[2])
        y = pose[1] + r * math.sin(th + pose[2])
        return [x, y]

    #**********************************************************************************************
    # Callback functions for raw data from the radars
    # Update the left observations
    def update_l_obs(self, msg):
        obs = [element for element in msg.data if element != 0]
        L = len(obs)
        tmp_obs = []
        tmp_obs_xy = []
        for i in range(0, L, 2):
            tmp_obs.append([obs[i], obs[i+1]])
            tmp_obs_xy.append(self.rth2xy(self.prev_odom, obs[i], obs[i+1]))  # the prev_odom is updated to the current odom pose in the previous step
        self.tmp_l_obs = tmp_obs
        self.tmp_l_obs_xy = tmp_obs_xy
        # print('tmp_l_obs_xy: ', self.tmp_l_obs_xy)
        self.tmp_l_obs_count = L/2
        return 0
    
    # Update the the right observations
    def update_r_obs(self, msg):
        obs = [element for element in msg.data if element != 0]
        L = len(obs)
        tmp_obs = []
        tmp_obs_xy = []
        for i in range(0, L, 2):
            tmp_obs.append([obs[i], obs[i+1]])
            tmp_obs_xy.append(self.rth2xy(self.prev_odom, obs[i], obs[i+1]))  # the prev_odom is updated to the current odom pose in the previous step
        self.tmp_r_obs = tmp_obs
        self.tmp_r_obs_xy = tmp_obs_xy
        # print('tmp_r_obs_xy: ', self.tmp_r_obs_xy)
        self.tmp_r_obs_count = L/2
        return 0
    
    # Callback functions odometry data
    def update_del_odom(self, msg):
        prev_x = self.prev_odom[0]
        prev_y = self.prev_odom[1]
        prev_theta = self.prev_odom[2]

        cur_x = msg.pose.pose.position.x * 1000
        cur_y = msg.pose.pose.position.y * 1000
        cur_theta = rot.from_quat([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]).as_euler('zyx')[0]

        del_x = cur_x - prev_x
        del_y = cur_y - prev_y
        del_l = del_x**2 + del_y**2
        if del_l >= 20 or abs(cur_theta - prev_theta) >= 2e-3:
            self.prev_odom = [cur_x, cur_y, cur_theta]
            self.odom_x_data.append(cur_x)
            self.odom_y_data.append(cur_y)
            self.odom_filt_x_data.append(self.odom_filt_x)
            self.odom_filt_y_data.append(self.odom_filt_y)
            self.update_obs()
            if self.id%2 == 0:
                self.update_plot_odom()
                self.update_plot_obs()
                self.ax.relim()
                self.ax.autoscale_view()
                
                # xlim = self.ax.get_xlim()
                # ylim = self.ax.get_ylim()
                self.ax.set_xlim(min(self.odom_x_data) - 2500, max(self.odom_x_data) + 2500)
                self.ax.set_ylim(min(self.odom_y_data) - 2500, max(self.odom_y_data) + 2500)

                self.fig.canvas.flush_events()
                print('Updated plot...')
            self.id += 1
            
            print('id: ', self.id)
    
    def update_filt_odom(self, msg):
        self.odom_filt_x = msg.pose.pose.position.x * 1000
        self.odom_filt_y = msg.pose.pose.position.y * 1000
        # cur_theta = rot.from_quat([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]).as_euler('zyx')[0]
        return 0

def main():
    rclpy.init()
    plot_obs_obj = plot_obs('odom', '/odometry/filtered', 'LeftObs/range_bear', 'RightObs/range_bear')
    rclpy.spin(plot_obs_obj)
    pdb.set_trace()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
    
