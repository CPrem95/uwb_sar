from optparse import OptionParser
import rclpy
from rclpy.node import Node
from scipy.signal import savgol_filter
from scipy.signal import find_peaks
from matplotlib import pyplot as plt
from std_msgs.msg import Float32MultiArray
import numpy as np
import math
import time
import matplotlib.pyplot as plt
import multiprocessing as mp
from multiprocessing import Queue
from nav_msgs.msg import Odometry
from scipy.spatial.transform import Rotation as rot
import logging
import threading
from uwb_sar.src import sar_funcs as sar
from scipy.io import savemat
import cv2

# Parallel processing
from numba import njit, prange
from joblib import Parallel, delayed
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import shared_memory

plt.ion()
# Define ANSI escape codes for colors
class Colors:
    BOLD = '\033[1m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'

# **********************************************************************************************
def find_nearest_val2(val_lst, x, delta_x):
        ind = int(x // delta_x)
        rem = x % delta_x
        val1 = val_lst[ind]
        val2 = val_lst[ind + 1]
        
        val = val1 + (val2 - val1) * rem/delta_x

        return val

def update_img(id, pixels, img_dist_x, img_dist_y, odom, obs_radar, img, samp_size):
        p_y = pixels[0][id]
        p_x = pixels[1][id]
        pix_x = img_dist_x[0, p_x]
        pix_y = img_dist_y[p_y, 0]

        d_sq = (pix_x - odom[0])**2 + (pix_y - odom[1])**2

        val = find_nearest_val2(obs_radar, np.sqrt(d_sq), samp_size)
        img[p_y, p_x] += val

def execute_task_parallel2(img_dist_x, img_dist_y, odom, obs_radar, img, res, samp_size, half_fov, sar_orig_x, sar_end_y):
    start_time = time.time()
    pixels = sar.extract_pixels(img, res, odom, 400, 3000, half_fov, img_dist_x, img_dist_y, sar_orig_x, sar_end_y)
    print('Time taken to extract pixels: ', time.time() - start_time)
    pix_count = len(pixels[0])
    return Parallel(n_jobs=10)(delayed(update_img)(id, pixels, img_dist_x, img_dist_y, odom, obs_radar, img, samp_size) for id in range(pix_count))

class SAR(Node):
    def __init__(self, topic_name_1, topic_name_2, topic_name_3, topic_name_4):
        super().__init__('sar_gen')

        # PARAMETERS
        self.radar_r = 180
        self.d = 210
        self.mpp = 5e-3
        self.mph = 6e-3
        self.gamma = (math.pi - math.radians(30))/2
        self.D_1 = []
        self.D_2 = []
        self.samp_size = 920/143
        self.n_samp = 1000
        self.max_qsize = 10000
        self.winsize = 600
        self.half_fov = math.radians(30)

        # VARIABLES
        self.moving = False
        self.odom_init = False
        self.queue_id = 0
        self.cur_odom = [0, 0, 0]
        self.prev_odom = [0, 0, 0]

        self.cur_obs_radar_1 = np.zeros(self.n_samp)
        self.cur_obs_radar_2 = np.zeros(self.n_samp)
        self.cur_obs_radar_3 = np.zeros(self.n_samp)
        self.cur_obs_radar_4 = np.zeros(self.n_samp)

        self.mem_obs_radar_1 = np.zeros([self.max_qsize, self.n_samp])
        self.mem_obs_radar_2 = np.zeros([self.max_qsize, self.n_samp])
        self.mem_obs_radar_3 = np.zeros([self.max_qsize, self.n_samp])
        self.mem_obs_radar_4 = np.zeros([self.max_qsize, self.n_samp])
        self.mem_odom = np.zeros([self.max_qsize, 4])

        self.dists = np.arange(0, self.n_samp) * self.samp_size
        
        # Parameters for the SAR image
        self.res = 5 # Resolution in mm
        self.radar_range = [300, 2500] # Radar range in mm
        self.radar_azimuth = [math.radians(-30), math.radians(30)] # Radar azimuth in degrees

        self.sar_area_x = 7000 # SAR area in mm # 4500
        self.sar_area_y = 3000*2 # SAR area in mm

        self.sar_orig_x = 0 # SAR origin in mm: 3000/res = 600
        self.sar_orig_y = 3000 # SAR origin in mm: 2500/res = 500
        self.sar_end_y = self.sar_area_y - self.sar_orig_y
        # ind = cart2index(-6.1, -5.1, ori_x, end_y, res)
        # print(ind)
        self.pixels_x = int(self.sar_area_x/self.res)
        self.pixels_y = int(self.sar_area_y/self.res)

        print('resolution: ', self.res)
        print('all pixels_x: ', self.pixels_x)
        print('all pixels_y: ', self.pixels_y)

        # Define the range for x and y
        x = np.arange(-self.sar_orig_x + self.res/2, self.sar_area_x - self.sar_orig_x + self.res/2, self.res)
        y = np.arange(self.sar_area_y - self.sar_orig_y - self.res/2, - self.sar_orig_y - self.res/2, - self.res)

        # Create the meshgrid
        self.img_dist_X, self.img_dist_Y = np.meshgrid(x, y)

        radar_img_dist = np.zeros((2, 5, 10), dtype=np.float32) # Distance matrix: 2 for [x and y]

        self.radar_img_size = (self.pixels_y, self.pixels_x)
        # Initialize the raw images with zeros
        self.radar_img = np.zeros(self.radar_img_size, dtype=np.float32)
        self.radar_img_dist = np.zeros((2, self.pixels_y, self.pixels_x), dtype=np.float32) # Distance matrix: 2 for [x and y
        self.radar_img_dist = self.find_dist_matrix(self.radar_img_dist, self.pixels_x, self.pixels_y, self.res)

        Fs = 23.328e9 # Sampling frequency
        fc = 7.29e9
        BW = 1.4e9
        frac_bw = BW/fc
        PRF = 14e6
        VTX = 0.6
        self.uwb_t, self.uwb_pulse = sar.generate_uwb_pulse(Fs, fc, frac_bw, PRF, VTX)

        # self.ln, = plt.plot([], [], linewidth=0.5, color='b')

        
        # pipes for the raw plotter
        self.parent_pipe_obs_radar1, child_pipe_obs_radar1 = mp.Pipe() # parent (plot), child (plotter)
        self.parent_pipe_obs_radar2, child_pipe_obs_radar2 = mp.Pipe()
        self.parent_pipe_obs_radar3, child_pipe_obs_radar3 = mp.Pipe()
        self.parent_pipe_obs_radar4, child_pipe_obs_radar4 = mp.Pipe()

        self.send_obs_radar1 = self.parent_pipe_obs_radar1.send
        self.send_obs_radar2 = self.parent_pipe_obs_radar2.send
        self.send_obs_radar3 = self.parent_pipe_obs_radar3.send
        self.send_obs_radar4 = self.parent_pipe_obs_radar4.send
        
        # Queue for the SAR
        self.obs_radar1_queue = Queue(self.max_qsize)
        self.obs_radar2_queue = Queue(self.max_qsize)
        self.obs_radar3_queue = Queue(self.max_qsize)
        self.obs_radar4_queue = Queue(self.max_qsize)
        self.odom_queue = Queue(self.max_qsize)
        
        self.subscription = self.create_subscription(
            Float32MultiArray,
            topic_name_1,
            self.obs_radar_1,
            1
        )
        self.subscription

        self.subscription = self.create_subscription(
            Float32MultiArray,
            topic_name_2,
            self.obs_radar_2,
            1
        )
        self.subscription

        self.subscription = self.create_subscription(
            Float32MultiArray,
            topic_name_3,
            self.obs_radar_3,
            1
        )

        self.subscription = self.create_subscription(
            Float32MultiArray,
            topic_name_4,
            self.obs_radar_4,
            1
        )

        self.subscription = self.create_subscription(
            Odometry,
            'odom',
            self.odom_callback,
            1
        )
        self.subscription

        # Communication between the threads and the processes
        self.parent_pipe_radar_img, child_pipe_radar_img = mp.Pipe()
        self.send_radar_img = self.parent_pipe_radar_img.send
        # **********************************************************************************************
        # Start the SAR processes >>> MULTI-PROCESSING
        self.sar_process = mp.Process(target=self.SAR_generation, 
                                       args=(
                                            self.obs_radar1_queue,
                                            self.obs_radar2_queue,
                                            self.obs_radar3_queue,
                                            self.obs_radar4_queue,
                                            self.odom_queue),
                                       daemon=False)
        self.sar_process.start()

        # Start the SAR plotter process
        update_process = mp.Process(target=self.update_figure, args=(child_pipe_radar_img,))
        update_process.start()

    def update_figure(self, pipe):
        # If image hasn't been initialized, use imshow
        self.sar_fig_1 = plt.figure()
        self.sar_ax1 = self.sar_fig_1.add_subplot(121)
        self.sar_ax2 = self.sar_fig_1.add_subplot(122)
        self.sar_fig_2 = plt.figure()
        self.sar_ax3 = self.sar_fig_2.add_subplot(111)

        self.sar_ax1.set_aspect('equal', adjustable='box')
        self.sar_ax1.set_xlabel(f'Pixels_x [1pixel = {self.res}mm]')
        self.sar_ax1.set_ylabel(f'Pixels_y [1pixel = {self.res}mm]')
        self.sar_ax1.set_title('SAR output')

        self.sar_ax2.set_aspect('equal', adjustable='box')
        self.sar_ax2.set_xlabel(f'Pixels_x [1pixel = {self.res}mm]')
        self.sar_ax2.set_ylabel(f'Pixels_y [1pixel = {self.res}mm]')
        self.sar_ax2.set_title('Positive image')

        self.sar_ax3.set_aspect('equal', adjustable='box')
        self.sar_ax3.set_xlabel(f'Pixels_x [1pixel = {self.res}mm]')
        self.sar_ax3.set_ylabel(f'Pixels_y [1pixel = {self.res}mm]')
        self.sar_ax3.set_title('Blurred image')
        self.radar_img_obj_1 = self.sar_ax1.imshow(np.zeros(self.radar_img_size), cmap='jet', animated=True)
        self.radar_img_obj_2 = self.sar_ax2.imshow(np.zeros(self.radar_img_size), cmap='jet', animated=True)
        self.radar_img_obj_3 = self.sar_ax3.imshow(np.zeros(self.radar_img_size), cmap='jet', animated=True)

        self.sar_fig_1.canvas.draw()
        self.sar_fig_1.canvas.flush_events()
        self.sar_fig_2.canvas.draw()
        self.sar_fig_2.canvas.flush_events()
        while True:
            if pipe.poll():
                radar_img = pipe.recv()
                abs_sar = np.abs(radar_img)
                self.radar_img_obj_1.set_data(abs_sar)
                self.radar_img_obj_1.set_clim(vmin=np.min(abs_sar), vmax=np.max(abs_sar))
                positive_sar = radar_img + abs_sar
                self.radar_img_obj_2.set_data(abs_sar)
                self.radar_img_obj_2.set_clim(vmin=np.min(abs_sar), vmax=np.max(abs_sar))
                positive_sar_blur= cv2.GaussianBlur(positive_sar, (0, 0), 2)
                
                self.radar_img_obj_3.set_data(positive_sar_blur)
                self.radar_img_obj_3.set_clim(vmin=np.min(positive_sar_blur), vmax=np.max(positive_sar_blur))

                self.sar_fig_1.canvas.draw()
                self.sar_fig_1.canvas.flush_events()
                self.sar_fig_2.canvas.draw()
                self.sar_fig_2.canvas.flush_events()
                        
    """Callback function for the odometry"""
    def odom_callback(self, msg):
        cur_x = msg.pose.pose.position.x * 1000 # Convert to mm
        cur_y = msg.pose.pose.position.y * 1000 # Convert to mm``
        cur_theta = rot.from_quat([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]).as_euler('zyx')[0]
        # cur_theta = cur_theta + math.pi/2
        if not self.odom_init:
            self.odom_init = True
            self.init_odom = [0, 0, 0, 0] # x, y, theta, displacement
            # self.init_odom = [cur_x, cur_y, cur_theta, 0] # x, y, theta, displacement
            # self.cur_odom = [cur_x, cur_y, cur_theta, 0]
            # self.get_logger().info(f"{Colors.YELLOW}Initial odometry: {self.init_odom}{Colors.RESET}")
        else:
            self.cur_odom = [cur_x - self.init_odom[0], cur_y - self.init_odom[1], cur_theta - self.init_odom[2], 0]
            # self.get_logger().info(f"{Colors.YELLOW}Current odometry: {[cur_x, cur_y, cur_theta]}{Colors.RESET}")
            # self.get_logger().info(f"{Colors.YELLOW}Modified odometry: {self.cur_odom}{Colors.RESET}")

            self.displacement = np.hypot(self.cur_odom[0] - self.prev_odom[0], self.cur_odom[1] - self.prev_odom[1])
            if self.displacement >= 3:
                self.cur_odom[3] = self.displacement
                self.prev_odom = self.cur_odom
                self.odom_queue.put(self.cur_odom)
                self.obs_radar1_queue.put(self.cur_obs_radar_1)
                # self.obs_radar2_queue.put(self.cur_obs_radar_2)
                # self.obs_radar3_queue.put(self.cur_obs_radar_3)
                self.obs_radar4_queue.put(self.cur_obs_radar_4)
                self.queue_id += 1
                self.get_logger().info(f"{Colors.YELLOW}Added to the queue id: {self.queue_id}{Colors.RESET}")


        vx = msg.twist.twist.linear.x
        az = msg.twist.twist.angular.z
        
        if vx != 0 or az != 0:
            self.moving = True
            self.get_logger().info(f"{Colors.GREEN}Moving{Colors.RESET}")
        else:
            self.moving = False
            self.get_logger().info(f"{Colors.RED}Not Moving{Colors.RESET}")
            # print(f"{Colors.RED}Not Moving{Colors.RESET}")

    def obs_radar_1(self, msg):
        series = msg.data
        # series = np.abs(series)
        series[0] = 0
        if self.moving:
            # self.send_obs_radar1(series)
            self.cur_obs_radar_1 = series
    
    def obs_radar_2(self, msg):
        series = msg.data
        # series = np.abs(series)
        series[0] = 0
        if self.moving:
            # self.send_obs_radar2(series)
            self.cur_obs_radar_2 = series
    
    def obs_radar_3(self, msg):
        series = msg.data
        # series = np.abs(series)
        series[0] = 0
        if self.moving:
            # self.send_obs_radar3(series)
            self.cur_obs_radar_3 = series

    def obs_radar_4(self, msg):
        series = msg.data
        # series = np.abs(series)
        series[0] = 0
        if self.moving:
            # self.send_obs_radar4(series)
            self.cur_obs_radar_4 = series

    """For the lambda function in the subscription"""
    def obs_radar(self, msg, obs_var):
        if self.moving:
            obs_var = msg.data
            obs_var[0] = 0
        else:
            obs_var = None
    
    # @staticmethod
    # @njit(parallel=True)
    def SAR_generation(self, obs_radar1_queue, obs_radar2_queue, obs_radar3_queue, obs_radar4_queue, odom_queue):
            # self.init_accumulate()
        odoms = []
        obs_radar1s = []
        obs_radar2s = []
        obs_radar3s = []
        obs_radar4s = []

        step_id = 0
        self.sar_odom = [0, 0, 0]
        fov_angs = [math.radians(60), math.radians(120)] # radar 1
        # fov_angs = [math.radians(15), math.radians(75)] # radar2
        # fov_angs = [math.radians(60 - 180), math.radians(120 - 180)] # radar3
        # fov_angs = [math.radians(15- 90), math.radians(75 - 90)] # radar4

        try:
            while True:
                q_size = odom_queue.qsize()
                if q_size > 0:
                    step_id += 1
                    print('processed_id: ', step_id)

                    odom = odom_queue.get()
                    obs_radar1 = obs_radar1_queue.get()
                    # obs_radar2 = obs_radar2_queue.get()
                    # obs_radar3 = obs_radar3_queue.get()
                    obs_radar4 = obs_radar4_queue.get()
                    
                    '''
                    Save the data of the radar observations and the odometry
                    '''
                    # self.mem_odom[step_id] = odom
                    # self.mem_obs_radar_1[step_id] = obs_radar1
                    # self.mem_obs_radar_2[step_id] = obs_radar2
                    # self.mem_obs_radar_3[step_id] = obs_radar3
                    # self.mem_obs_radar_4[step_id] = obs_radar4

                    # if step_id > 150:
                    #     vis = False
                    # else:
                    #     vis = False
                    
                    rc_obs = sar.pulse_compression(obs_radar1, self.uwb_pulse, False) # Range compressed observation
                    # rc_obs = obs_radar1
                    self.add_sar_radar_1(odom, rc_obs, self.img_dist_X, self.img_dist_Y, self.radar_img, self.res) 
                    rc_obs = sar.pulse_compression(obs_radar4, self.uwb_pulse, False) # Range compressed observation
                    # rc_obs = obs_radar4
                    self.add_sar_radar_2(odom, rc_obs, self.img_dist_X, self.img_dist_Y, self.radar_img, self.res)
                    print("Radar observations added at odom: ", odom)
                    self.get_logger().info(f"{Colors.BLUE}remaining qsize: {q_size -1}{Colors.RESET}")

                    if step_id % 10 == 0: # Update the figure every 1 step
                        self.send_radar_img(self.radar_img)

                    if step_id == 1590: # Save the data of the radar observations and the odometry
                        savemat('sar_data_init.mat', {'obs_radar_1': self.mem_obs_radar_1, 'obs_radar_2': self.mem_obs_radar_2, 'obs_radar_3': self.mem_obs_radar_3, 'obs_radar_4': self.mem_obs_radar_4, 'odom': self.mem_odom, 'img': self.radar_img})

                    if step_id == 1810:
                        self.radar_img = np.zeros(self.radar_img_size, dtype=np.float32)

                    if step_id == 3390:
                        savemat('sar_data_ret.mat', {'obs_radar_1': self.mem_obs_radar_1, 'obs_radar_2': self.mem_obs_radar_2, 'obs_radar_3': self.mem_obs_radar_3, 'obs_radar_4': self.mem_obs_radar_4, 'odom': self.mem_odom, 'img': self.radar_img})
                        raise KeyboardInterrupt

        except KeyboardInterrupt:
            self.get_logger().info(f"{Colors.RED}Keyboard Interrupt{Colors.RESET}")
            savemat('sar_data.mat', {'obs_radar_1': self.mem_obs_radar_1, 'obs_radar_2': self.mem_obs_radar_2, 'obs_radar_3': self.mem_obs_radar_3, 'obs_radar_4': self.mem_obs_radar_4, 'odom': self.mem_odom, 'img': self.radar_img})
            pos_img = self.radar_img + np.abs(self.radar_img)
            self.pos_sar_fig = plt.figure()
            self.pos_sar_ax = self.pos_sar_fig.add_subplot(111)
            self.pos_sar_ax.set_aspect('equal', adjustable='box')
            self.pos_sar_ax.set_xlabel(f'Pixels_x [1pixel = {self.res}mm]')
            self.pos_sar_ax.set_ylabel(f'Pixels_y [1pixel = {self.res}mm]')
            self.pos_sar_ax.set_title('Positive SAR output')
            self.pos_sar_img_obj = self.pos_sar_ax.imshow(pos_img, cmap='jet', animated=True)
            self.pos_sar_fig.canvas.draw()
            self.pos_sar_fig.canvas.flush_events()

            blurred = cv2.GaussianBlur(pos_img, (7, 7), 2) 
            self.blur_sar_fig = plt.figure()
            self.blur_sar_ax = self.blur_sar_fig.add_subplot(111)
            self.blur_sar_ax.set_aspect('equal', adjustable='box')
            self.blur_sar_ax.set_xlabel(f'Pixels_x [1pixel = {self.res}mm]')
            self.blur_sar_ax.set_ylabel(f'Pixels_y [1pixel = {self.res}mm]')
            self.blur_sar_ax.set_title('Blurred SAR output')
            self.blur_sar_img_obj = self.blur_sar_ax.imshow(blurred, cmap='jet', animated=True)
            self.blur_sar_fig.canvas.draw()
            self.blur_sar_fig.canvas.flush_events()

            time.sleep(1000)
            pass

    def find_dist_matrix(self, img_dist, pixels_x, pixels_y, res):
        for i in range(0, pixels_y):
            pix_y = res*(pixels_y - i) - res/2
            for j in range(0, pixels_x):
                pix_x = j*res + res/2
                img_dist[0, i, j] = pix_x
                img_dist[1, i, j] = pix_y
        return img_dist

    # Extract pixels using OpenCV methods
    def add_sar_radar_1(self, odom, obs_radar, img_dist_x, img_dist_y, img, res):
        start_time = time.time()

        # Extract pixel coordinates & sensor base
        pixels, sensor_base = sar.extract_pixels_radar_1(img, res, odom, 400, 3000, self.half_fov, img_dist_x, img_dist_y, self.sar_orig_x, self.sar_end_y, self.radar_r)
        # print('Time taken to extract pixels:', time.time() - start_time)

        # Convert pixel indices to real-world coordinates (vectorized)
        p_y, p_x = pixels
        pix_x = img_dist_x[0, p_x]  # Vectorized lookup
        pix_y = img_dist_y[p_y, 0]  # Vectorized lookup

        # Compute distance squared (vectorized)
        d_sqr = np.hypot(pix_x - sensor_base[0], pix_y - sensor_base[1])

        # Vectorized lookup for nearest values
        val = np.interp(d_sqr, np.arange(len(obs_radar)) * self.samp_size, obs_radar)

        # Vectorized image update
        img[p_y, p_x] += val

        print('Time taken to add SAR:', time.time() - start_time)
        return img
    
    def add_sar_radar_2(self, odom, obs_radar, img_dist_x, img_dist_y, img, res):
        start_time = time.time()

        # Extract pixel coordinates & sensor base
        pixels, sensor_base = sar.extract_pixels_radar_2(img, res, odom, 400, 3000, self.half_fov, img_dist_x, img_dist_y, self.sar_orig_x, self.sar_end_y, self.radar_r)
        # print('Time taken to extract pixels:', time.time() - start_time)

        # Convert pixel indices to real-world coordinates (vectorized)
        p_y, p_x = pixels
        pix_x = img_dist_x[0, p_x]  # Vectorized lookup
        pix_y = img_dist_y[p_y, 0]  # Vectorized lookup

        # Compute distance squared (vectorized)
        d_sqr = np.hypot(pix_x - sensor_base[0], pix_y - sensor_base[1])

        # Vectorized lookup for nearest values using numpy interpolation
        val = np.interp(d_sqr, np.arange(len(obs_radar)) * self.samp_size, obs_radar)

        # Vectorized image update
        img[p_y, p_x] += val

        print('Time taken to add SAR:', time.time() - start_time)
        return img
      

# **********************************************************************************************
# MAIN FUNCTION
def main(args=None):
    rclpy.init(args=args)
    logging.info(f"{Colors.RED}Here we go!!!{Colors.RESET}")  
    sar_gen = SAR('/UWBradar0/readings',
                    '/UWBradar1/readings',
                    '/UWBradar2/readings',
                    '/UWBradar3/readings'
                    )
    sar_gen.get_logger().info(f"{Colors.GREEN}{Colors.BOLD}Bringup the robot...{Colors.RESET}")
    try:
        rclpy.spin(sar_gen)
    except KeyboardInterrupt:
        sar_gen.get_logger().info(f"{Colors.RED}Keyboard Interrupt{Colors.RESET}")
        
    sar_gen.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    try:
        main()
    except:
        logging.error('Error in the main function')
        rclpy.shutdown()
        pass
