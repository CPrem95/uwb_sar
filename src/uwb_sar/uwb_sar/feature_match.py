from optparse import OptionParser
import rclpy
from rclpy.node import Node
from scipy.signal import savgol_filter
from scipy.signal import find_peaks
import matplotlib
from matplotlib import pyplot as plt
from std_msgs.msg import Float32MultiArray
import numpy as np
import math
import time
import matplotlib.pyplot as plt
from nav_msgs.msg import Odometry
from scipy.spatial.transform import Rotation as rot
import logging
from scipy.io import savemat, loadmat
import cv2
import gc
import threading
import traceback

'''Parallel processing'''
# from numba import njit, prange
# from joblib import Parallel, delayed
# import threading
import multiprocessing as mp
from multiprocessing import Queue
from geometry_msgs.msg import Pose2D
import time
import matplotlib

plt.ion()
# Disable all matplotlib plots
matplotlib.use('Agg')  # Use a non-interactive backend
# Define ANSI escape codes for colors
class Colors:
    BOLD = '\033[1m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'

# **********************************************************************************************
class Matcher(Node):
    def __init__(self):
        super().__init__('feature_match')

        # Parameters
        self.k_L = 0.8 # Lowe's ratio test
        self.ransac_thresh = 15.0 # RANSAC threshold
        self.n_thresh = 10 # Threshold for number of matches 
        self.scale_limits = [0.85, 1.15] # Scale limits
        self.tx_tolerance = 5 # Translation tolerance [in pixels]
        self.ty_tolerance = 5 # Translation tolerance [in pixels]
        self.rot_tolerance = 5 # Rotation tolerance [in degrees]

        # Variables
        self.sar_img_names_1 = ['sar_data_init.mat'] # images in region 1
        self.sar_img_names_2 = ['sar_data_ret.mat'] # images in region 2
        self.sar_imgs_1 = []
        self.sar_imgs_2 = []
        n_imgs = len(self.sar_img_names_1) +1

        # Create a figure and axes for displaying images
        self.img_fig, self.axes = plt.subplots(2, n_imgs, figsize=(10, 55))
        self.sar_ax1 = [self.axes[0, i] for i in range(n_imgs)]
        self.sar_ax2 = [self.axes[1, i] for i in range(n_imgs)]
        self.img_fig.canvas.set_window_title('SAR Images')
        # Create a figure and axes for displaying descriptors
        self.feature_figs = []
        self.feature_axes = []
        for i, title in enumerate(["SIFT Features", "SURF Features", "BRISK Features", "AKAZE Features", "ORB Features"]):
            # Create a figure and axes for each match type
            fig, ax = plt.subplots(1, 2, figsize=(30, 10))
            fig.canvas.set_window_title(title)
            ax[0].set_title(title + " 1", fontsize=15)
            ax[1].set_title(title + " 2", fontsize=15)
            self.feature_figs.append(fig)
            self.feature_axes.append(ax)
        # Create a figure and axes for displaying matches
        self.match_figs = []
        self.match_axes = []
        for i, title in enumerate(["SIFT Matches", "SURF Matches", "BRISK Matches", "AKAZE Matches","ORB Matches"]):
            # Create a figure and axes for each match type
            fig, ax = plt.subplots(1, 1, figsize=(30, 10))
            fig.canvas.set_window_title(title)
            ax.set_title(title, fontsize=15)
            self.match_figs.append(fig)
            self.match_axes.append(ax)

        # Create separate figures and axes for displaying filtered matches
        self.filter_figs = []
        self.filter_axes = []
        for i, title in enumerate(["Filtered SIFT Matches", "Filtered SURF Matches", "Filtered BRISK Matches", "Filtered AKAZE Matches", "Filtered ORB Matches"]):
            fig, ax = plt.subplots(1, 1, figsize=(30, 10))
            fig.canvas.set_window_title(title)
            ax.set_title(title, fontsize=15)
            self.filter_figs.append(fig)
            self.filter_axes.append(ax)

        # Initialize lists to store keypoints and descriptors    
        self.all_sift_kp1 = []
        self.all_sift_des1 = []
        self.all_sift_kp2 = []
        self.all_sift_des2 = []
        self.all_orb_kp1 = []
        self.all_orb_des1 = []
        self.all_orb_kp2 = []
        self.all_orb_des2 = []
        self.all_surf_kp1 = []
        self.all_surf_des1 = []
        self.all_surf_kp2 = []
        self.all_surf_des2 = []
        self.all_akaze_kp1 = []
        self.all_akaze_des1 = []
        self.all_akaze_kp2 = []
        self.all_akaze_des2 = []
        self.all_brisk_kp1 = []
        self.all_brisk_des1 = []
        self.all_brisk_kp2 = []
        self.all_brisk_des2 = []
        
        # Initialize lists to store matches
        self.good_sift = []
        self.good_surf = []
        self.good_orb = []
        self.good_akaze = []
        self.good_brisk = []

        self.run()
    
    def SIFT_extract_and_display_keypoints(self, cropped_sar_1, cropped_sar_2, ax1, ax2, fig, plot_data=True):
        start_time = time.time()
        # Initialize SIFT detector
        sift = cv2.SIFT_create(nfeatures=200, contrastThreshold=0.015)
        # Detect keypoints and compute descriptors for cropped_sar_1
        keypoints_1, descriptors_1 = sift.detectAndCompute(cropped_sar_1, None)
        # Detect keypoints and compute descriptors for cropped_sar_2
        keypoints_2, descriptors_2 = sift.detectAndCompute(cropped_sar_2, None)
        elapsed_time = time.time() - start_time
        print(f"Elapsed time for SIFT keypoint detection and descriptor computation: {elapsed_time:.8f} seconds")

        if plot_data:
            # Draw keypoints on the images
            img1 = cv2.drawKeypoints(cropped_sar_1, keypoints_1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            img2 = cv2.drawKeypoints(cropped_sar_2, keypoints_2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            # Display the images with keypoints
            ax1.imshow(img1)
            ax2.imshow(img2)
            fig.canvas.draw()
            fig.canvas.flush_events()

        return keypoints_1, descriptors_1, keypoints_2, descriptors_2
    
    def ORB_extract_and_display_keypoints(self, cropped_sar_1, cropped_sar_2, ax1, ax2, fig, plot_data=True):
        start_time = time.time()
        # Initialize ORB detector
        orb = cv2.ORB_create(nfeatures=200, fastThreshold=15, WTA_K=4)
        # Detect keypoints and compute descriptors for cropped_sar_1
        keypoints_1, descriptors_1 = orb.detectAndCompute(cropped_sar_1, None)
        # Detect keypoints and compute descriptors for cropped_sar_2
        keypoints_2, descriptors_2 = orb.detectAndCompute(cropped_sar_2, None)
        elapsed_time = time.time() - start_time
        print(f"Elapsed time for ORB keypoint detection and descriptor computation: {elapsed_time:.8f} seconds")

        if plot_data:
            # Draw keypoints on the images
            img1 = cv2.drawKeypoints(cropped_sar_1, keypoints_1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            img2 = cv2.drawKeypoints(cropped_sar_2, keypoints_2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            # Display the images with keypoints
            ax1.imshow(img1)
            ax2.imshow(img2)
            fig.canvas.draw()
            fig.canvas.flush_events()

        return keypoints_1, descriptors_1, keypoints_2, descriptors_2
    
    def SURF_extract_and_display_keypoints(self, cropped_sar_1, cropped_sar_2, ax1, ax2, fig, plot_data=True):
        start_time = time.time()
        # Initialize SURF detector
        surf = cv2.xfeatures2d.SURF_create(hessianThreshold=200, nOctaves=4, nOctaveLayers=4)
        # Detect keypoints and compute descriptors for cropped_sar_1
        keypoints_1, descriptors_1 = surf.detectAndCompute(cropped_sar_1, None)
        # Detect keypoints and compute descriptors for cropped_sar_2
        keypoints_2, descriptors_2 = surf.detectAndCompute(cropped_sar_2, None)
        elapsed_time = time.time() - start_time
        print(f"Elapsed time for SURF keypoint detection and descriptor computation: {elapsed_time:.8f} seconds")

        # Sort keypoints by response and keep top 200
        def select_top_keypoints(keypoints, descriptors, max_keypoints=200):
            if len(keypoints) > max_keypoints:
                keypoints, descriptors = zip(*sorted(zip(keypoints, descriptors), key=lambda x: x[0].response, reverse=True)[:max_keypoints])
                descriptors = np.array(descriptors)
            else:
                descriptors = np.array(descriptors)
            return keypoints, descriptors

        keypoints_1, descriptors_1 = select_top_keypoints(keypoints_1, descriptors_1)
        keypoints_2, descriptors_2 = select_top_keypoints(keypoints_2, descriptors_2)

        if plot_data:
            # Draw keypoints on the images
            img1 = cv2.drawKeypoints(cropped_sar_1, keypoints_1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            img2 = cv2.drawKeypoints(cropped_sar_2, keypoints_2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            # Display the images with keypoints
            ax1.imshow(img1)
            ax2.imshow(img2)
            fig.canvas.draw()
            fig.canvas.flush_events()

        return keypoints_1, descriptors_1, keypoints_2, descriptors_2

    def AKAZE_extract_and_display_keypoints(self, cropped_sar_1, cropped_sar_2, ax1, ax2, fig, plot_data=True):
        start_time = time.time()
        # Initialize AKAZE detector
        akaze = cv2.AKAZE_create(threshold= 0.0005, max_points=200, descriptor_type=cv2.AKAZE_DESCRIPTOR_KAZE) # cv2.AKAZE_DESCRIPTOR_MLDB [Change matching distance accordingly]
        # Detect keypoints and compute descriptors for cropped_sar_1
        keypoints_1, descriptors_1 = akaze.detectAndCompute(cropped_sar_1, None)
        # Detect keypoints and compute descriptors for cropped_sar_2
        keypoints_2, descriptors_2 = akaze.detectAndCompute(cropped_sar_2, None)
        elapsed_time = time.time() - start_time
        print(f"Elapsed time for AKAZE keypoint detection and descriptor computation: {elapsed_time:.8f} seconds")

        if plot_data:
            # Draw keypoints on the images
            img1 = cv2.drawKeypoints(cropped_sar_1, keypoints_1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            img2 = cv2.drawKeypoints(cropped_sar_2, keypoints_2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            # Display the images with keypoints
            ax1.imshow(img1)
            ax2.imshow(img2)
            fig.canvas.draw()
            fig.canvas.flush_events()

        return keypoints_1, descriptors_1, keypoints_2, descriptors_2

    def BRISK_extract_and_display_keypoints(self, cropped_sar_1, cropped_sar_2, ax1, ax2, fig, plot_data=True):
        start_time = time.time()    
        # Initialize BRISK detector
        brisk = cv2.BRISK_create(thresh = 15)
        # Detect keypoints and compute descriptors for cropped_sar_1
        keypoints_1, descriptors_1 = brisk.detectAndCompute(cropped_sar_1, None)
        # Detect keypoints and compute descriptors for cropped_sar_2
        keypoints_2, descriptors_2 = brisk.detectAndCompute(cropped_sar_2, None)
        elapsed_time = time.time() - start_time
        print(f"Elapsed time for BRISK keypoint detection and descriptor computation: {elapsed_time:.8f} seconds")

        # Sort keypoints by response and keep top 500
        def select_top_keypoints(keypoints, descriptors, max_keypoints=200):
            if len(keypoints) > max_keypoints:
                keypoints, descriptors = zip(*sorted(zip(keypoints, descriptors), key=lambda x: x[0].response, reverse=True)[:max_keypoints])
                descriptors = np.array(descriptors)
            else:
                descriptors = np.array(descriptors)
            return keypoints, descriptors

        keypoints_1, descriptors_1 = select_top_keypoints(keypoints_1, descriptors_1)
        keypoints_2, descriptors_2 = select_top_keypoints(keypoints_2, descriptors_2)

        if plot_data:
            # Draw keypoints on the images
            img1 = cv2.drawKeypoints(cropped_sar_1, keypoints_1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            img2 = cv2.drawKeypoints(cropped_sar_2, keypoints_2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            # Display the images with keypoints
            ax1.imshow(img1)
            ax2.imshow(img2)
            fig.canvas.draw()
            fig.canvas.flush_events()

        return keypoints_1, descriptors_1, keypoints_2, descriptors_2
    
    def load_sar_images(self, lim_1=(0, None), lim_2=(0, None)):
        for i in range(len(self.sar_img_names_1)):
            # Load the SAR images
            sar_img_1 = loadmat(self.sar_img_names_1[i])['img']  # Replace 'data' with the actual key in the .mat file
            sar_img_2 = loadmat(self.sar_img_names_2[i])['img']  # Replace 'data' with the actual key in the .mat file

            sar_img_1 = sar_img_1[:, lim_1[0]:lim_1[1]]
            sar_img_2 = sar_img_2[:, lim_2[0]:lim_2[1]]

            abs_sar_1 = np.abs(sar_img_1)
            abs_sar_2 = np.abs(sar_img_2)

            positive_sar_1 = sar_img_1 + abs_sar_1
            positive_sar_2 = sar_img_2 + abs_sar_2

            sar_img_1 = cv2.GaussianBlur(positive_sar_1, (0, 0), 2)
            sar_img_2 = cv2.GaussianBlur(positive_sar_2, (0, 0), 2)

            # Normalize the images
            sar_img_1 = cv2.normalize(sar_img_1, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
            sar_img_2 = cv2.normalize(sar_img_2, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
            
            # Append the images to the lists
            self.sar_imgs_1.append(sar_img_1)
            self.sar_imgs_2.append(sar_img_2)

            # Display the loaded SAR images
            self.sar_ax1[i].imshow(sar_img_1, cmap='jet', animated=True)
            self.sar_ax2[i].imshow(sar_img_2, cmap='jet', animated=True)
            self.img_fig.canvas.draw()
            self.img_fig.canvas.flush_events()

    def gen_kps_and_descriptors(self):
        for i in range(len(self.sar_img_names_1)):
            # Retrieve the SAR images
            sar_img_1 = self.sar_imgs_1[i]
            sar_img_2 = self.sar_imgs_2[i]

            # Extract and display keypoints using SIFT
            sift_kp1, sift_des1, sift_kp2, sift_des2 = self.SIFT_extract_and_display_keypoints(sar_img_1, sar_img_2, self.feature_axes[0][0], self.feature_axes[0][1], self.feature_figs[0])
            self.all_sift_kp1.append(sift_kp1)
            self.all_sift_des1.append(sift_des1)
            self.all_sift_kp2.append(sift_kp2)
            self.all_sift_des2.append(sift_des2)

            # Extract and display keypoints using SURF
            surf_kp1, surf_des1, surf_kp2, surf_des2 = self.SURF_extract_and_display_keypoints(sar_img_1, sar_img_2, self.feature_axes[1][0], self.feature_axes[1][1], self.feature_figs[1])
            self.all_surf_kp1.append(surf_kp1)
            self.all_surf_des1.append(surf_des1)
            self.all_surf_kp2.append(surf_kp2)
            self.all_surf_des2.append(surf_des2)

            # Extract and display keypoints using BRISK
            brisk_kp1, brisk_des1, brisk_kp2, brisk_des2 = self.BRISK_extract_and_display_keypoints(sar_img_1, sar_img_2, self.feature_axes[2][0], self.feature_axes[2][1], self.feature_figs[2])
            self.all_brisk_kp1.append(brisk_kp1)
            self.all_brisk_des1.append(brisk_des1)
            self.all_brisk_kp2.append(brisk_kp2)
            self.all_brisk_des2.append(brisk_des2)

            # Extract and display keypoints using AKAZE
            akaze_kp1, akaze_des1, akaze_kp2, akaze_des2 = self.AKAZE_extract_and_display_keypoints(sar_img_1, sar_img_2, self.feature_axes[3][0], self.feature_axes[3][1], self.feature_figs[3])
            self.all_akaze_kp1.append(akaze_kp1)
            self.all_akaze_des1.append(akaze_des1)
            self.all_akaze_kp2.append(akaze_kp2)
            self.all_akaze_des2.append(akaze_des2)

            # Extract and display keypoints using ORB
            orb_kp1, orb_des1, orb_kp2, orb_des2 = self.ORB_extract_and_display_keypoints(sar_img_1, sar_img_2, self.feature_axes[4][0], self.feature_axes[4][1], self.feature_figs[4])
            self.all_orb_kp1.append(orb_kp1)
            self.all_orb_des1.append(orb_des1)
            self.all_orb_kp2.append(orb_kp2)
            self.all_orb_des2.append(orb_des2)

        return True
    
    # Match between images of region 1 and region 2
    def kp_matcher_1(self, img_ind_1, img_ind_2, type='sift', plot_data=True):
        img1 = self.sar_imgs_1[img_ind_1]
        img2 = self.sar_imgs_2[img_ind_2]
        if type == 'sift':
            kp1 = self.all_sift_kp1[img_ind_1]
            des1 = self.all_sift_des1[img_ind_1]
            kp2 = self.all_sift_kp2[img_ind_2]
            des2 = self.all_sift_des2[img_ind_2]
            
            self.good_sift = []
            print('Matching SIFT features...')
            self.Knn_matcher(img1, img2, kp1, des1, kp2, des2, cv2.NORM_L2, self.good_sift, self.match_axes[0], self.match_figs[0], plot_data)

        elif type == 'surf':
            kp1 = self.all_surf_kp1[img_ind_1]
            des1 = self.all_surf_des1[img_ind_1]
            kp2 = self.all_surf_kp2[img_ind_2]
            des2 = self.all_surf_des2[img_ind_2]

            self.good_surf = []
            print('Matching SURF features...')
            self.Knn_matcher(img1, img2, kp1, des1, kp2, des2, cv2.NORM_L2, self.good_surf, self.match_axes[1], self.match_figs[1], plot_data)

        elif type == 'brisk':
            kp1 = self.all_brisk_kp1[img_ind_1]
            des1 = self.all_brisk_des1[img_ind_1]
            kp2 = self.all_brisk_kp2[img_ind_2]
            des2 = self.all_brisk_des2[img_ind_2]

            self.good_brisk = []
            print('Matching BRISK features...')
            self.Knn_matcher(img1, img2, kp1, des1, kp2, des2, cv2.NORM_HAMMING, self.good_brisk, self.match_axes[2], self.match_figs[2], plot_data) # cv2.NORM_HAMMING for BRISK descriptors
        
        elif type == 'akaze':
            kp1 = self.all_akaze_kp1[img_ind_1]
            des1 = self.all_akaze_des1[img_ind_1]
            kp2 = self.all_akaze_kp2[img_ind_2]
            des2 = self.all_akaze_des2[img_ind_2]

            self.good_akaze = []
            print('Matching AKAZE features...')
            self.Knn_matcher(img1, img2, kp1, des1, kp2, des2, cv2.NORM_L2, self.good_akaze, self.match_axes[3], self.match_figs[3], plot_data) # cv2.NORM_L2 for AKAZE_KAZE descriptors & cv2.NORM_HAMMING for AKAZE_MLDB descriptors

        elif type == 'orb':
            kp1 = self.all_orb_kp1[img_ind_1]
            des1 = self.all_orb_des1[img_ind_1]
            kp2 = self.all_orb_kp2[img_ind_2]
            des2 = self.all_orb_des2[img_ind_2]

            self.good_orb = []
            print('Matching ORB features...')
            self.Knn_matcher(img1, img2, kp1, des1, kp2, des2, cv2.NORM_HAMMING2, self.good_orb, self.match_axes[4], self.match_figs[4], plot_data)

    def Knn_matcher(self,img1, img2, kp1, des1, kp2, des2, norm_type, good_matches, ax, fig, plot_data):
        # BFMatcher with default params
        bf = cv2.BFMatcher(normType=norm_type)
        start_time = time.time()
        matches = bf.knnMatch(des1, des2, k=2)
        # Apply ratio test
        for m, n in matches:
            if m.distance < self.k_L * n.distance:
                good_matches.append([m])
        elapsed_time = time.time() - start_time
        print(f"Elapsed time for matching: {elapsed_time:.8f} seconds")
        if plot_data:
            # cv.drawMatchesKnn expects list of lists as matches.
            img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good_matches,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            ax.imshow(img3)
            fig.canvas.draw()
            fig.canvas.flush_events()

    # RANSAC filtering between images of region 1 and region 2
    def ransac_filter_1(self, img_ind_1, img_ind_2, type = 'sift', ransac_threshold = 25.0, plot_data=True):
        # Load the SAR images
        img1 = self.sar_imgs_1[img_ind_1]
        img2 = self.sar_imgs_2[img_ind_2]
        if type == 'sift':
            kp1 = self.all_sift_kp1[img_ind_1]
            kp2 = self.all_sift_kp2[img_ind_2]
        elif type == 'surf':
            kp1 = self.all_surf_kp1[img_ind_1]
            kp2 = self.all_surf_kp2[img_ind_2]
        elif type == 'orb':
            kp1 = self.all_orb_kp1[img_ind_1]
            kp2 = self.all_orb_kp2[img_ind_2]
        elif type == 'akaze':
            kp1 = self.all_akaze_kp1[img_ind_1]
            kp2 = self.all_akaze_kp2[img_ind_2]
        elif type == 'brisk':
            kp1 = self.all_brisk_kp1[img_ind_1]
            kp2 = self.all_brisk_kp2[img_ind_2]
        else:
            raise ValueError("Invalid type. Choose 'sift', 'surf', or 'orb'.")

        # Use the good matches from the previous step
        matches = {
            'sift': self.good_sift,
            'surf': self.good_surf,
            'orb': self.good_orb,
            'akaze': self.good_akaze,
            'brisk': self.good_brisk
        }.get(type, [])   

        print(f"\n{Colors.RED}Descriptor: {type}{Colors.RESET}")
        print(f"Number of keypoints in image 1 for {type}: {len(kp1)}")
        print(f"Number of keypoints in image 2 for {type}: {len(kp2)}")
        print(f"{Colors.YELLOW}Number of total matches for {type}: {len(matches)}{Colors.RESET}")

        # Check if there are enough matches
        if len(matches) > 5:         
            # Convert keypoints to numpy arrays
            src_pts = np.float32([kp1[m[0].queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m[0].trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            # Find homography using RANSAC
            # M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_threshold)
            M, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=ransac_threshold)
            print(f"Homography matrix for {type}:\n{M}")
            # Extract rotation angle and scale from the affine transformation matrix
            if M is not None:
                scale = np.sqrt(M[0, 0]**2 + M[0, 1]**2)
                rotation_angle = np.arctan2(M[0, 1], M[0, 0]) * (180.0 / np.pi)
                tx = M[0, 2]
                ty = M[1, 2]
                print(f"Scale: {scale}, Rotation Angle: {rotation_angle} degrees")
            
            # Filter matches based on the mask
            matches_masked = [m[0] for i, m in enumerate(matches) if mask[i]]

            print(f"{Colors.GREEN}Number of good matches after RANSAC filtering for {type}: {len(matches_masked)}{Colors.RESET}")

            # Draw the matches
            if plot_data:
                plt_ind = {'sift': 0, 'surf': 1, 'brisk': 2, 'akaze': 3, 'orb': 4}.get(type, 0)
                img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches_masked, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS, singlePointColor=(0, 255, 255), matchColor=(255, 255, 0))
                self.filter_axes[plt_ind].imshow(img3)
                self.filter_figs[plt_ind].canvas.draw()
                self.filter_figs[plt_ind].canvas.flush_events()
        else:
            logging.warning(f"Not enough matches found for {type}.")
            matches_masked = []
            tx = 0
            ty = 0
            scale = 0
            rotation_angle = 0
        return matches_masked, scale, tx, ty, rotation_angle

    def eval_loops(self, limits=(0, None)):
        ind1 = 0
        ind2 = 0
        # Load SAR images
        self.load_sar_images(limits, limits)
        # Generate keypoints and descriptors
        self.gen_kps_and_descriptors()

        # Match keypoints and descriptors
        self.kp_matcher_1(ind1, ind2, type='sift') # kp_matcher_2(ind1, ind2, type='sift', region=2) # needs region parameter
        self.kp_matcher_1(ind1, ind2, type='surf')
        self.kp_matcher_1(ind1, ind2, type='brisk')
        self.kp_matcher_1(ind1, ind2, type='akaze')
        self.kp_matcher_1(ind1, ind2, type='orb')

        self.ransac_filter_1(ind1, ind2, type='sift', ransac_threshold=self.ransac_thresh) # self.ransac_filter_2(ind1, ind2, type='sift', ransac_threshold=20.0)
        self.ransac_filter_1(ind1, ind2, type='surf', ransac_threshold=self.ransac_thresh)
        self.ransac_filter_1(ind1, ind2, type='brisk', ransac_threshold=self.ransac_thresh)
        self.ransac_filter_1(ind1, ind2, type='akaze', ransac_threshold=self.ransac_thresh)
        self.ransac_filter_1(ind1, ind2, type='orb', ransac_threshold=self.ransac_thresh)

    def eval_non_loops(self, limits_1, limits_2):
        ind1 = 0
        ind2 = 0
        # Load SAR images
        self.load_sar_images(limits_1, limits_2)
        # Generate keypoints and descriptors
        self.gen_kps_and_descriptors()

        # Match keypoints and descriptors
        self.kp_matcher_1(ind1, ind2, type='sift') # kp_matcher_2(ind1, ind2, type='sift', region=2) # needs region parameter
        self.kp_matcher_1(ind1, ind2, type='surf')
        self.kp_matcher_1(ind1, ind2, type='brisk')
        self.kp_matcher_1(ind1, ind2, type='akaze')
        self.kp_matcher_1(ind1, ind2, type='orb')

        self.ransac_filter_1(ind1, ind2, type='sift', ransac_threshold=self.ransac_thresh) # self.ransac_filter_2(ind1, ind2, type='sift', ransac_threshold=20.0)
        self.ransac_filter_1(ind1, ind2, type='surf', ransac_threshold=self.ransac_thresh)
        self.ransac_filter_1(ind1, ind2, type='brisk', ransac_threshold=self.ransac_thresh)
        self.ransac_filter_1(ind1, ind2, type='akaze', ransac_threshold=self.ransac_thresh)
        self.ransac_filter_1(ind1, ind2, type='orb', ransac_threshold=self.ransac_thresh)

    def eval_non_loops_akaze_orb(self, limits_1, limits_2):
        ind1 = 0
        ind2 = 0
        # Load SAR images
        self.load_sar_images(limits_1, limits_2)
        # Generate keypoints and descriptors
        self.gen_kps_and_descriptors()

        # Match keypoints and descriptors
        self.kp_matcher_1(ind1, ind2, type='akaze')
        self.kp_matcher_1(ind1, ind2, type='orb')

        matches_akaze, scale_akaze, tx_akaze, ty_akaze, rot_akaze = self.ransac_filter_1(ind1, ind2, type='akaze', ransac_threshold=self.ransac_thresh)
        matches_orb, scale_orb, tx_orb, ty_orb, rot_orb  = self.ransac_filter_1(ind1, ind2, type='orb', ransac_threshold=self.ransac_thresh)

        return [len(matches_akaze), scale_akaze, tx_akaze, ty_akaze, rot_akaze], [len(matches_orb), scale_orb, tx_orb, ty_orb, rot_orb]
    
    def gen_loop_matrix(self, all_n_akaze, all_scale_akaze, all_tx_akaze, all_ty_akaze, all_rot_akaze, all_n_orb, all_scale_orb, all_tx_orb, all_ty_orb, all_rot_orb):
        # Initialize the loop matrix
        self.loop_matrix = np.zeros((len(all_n_akaze), len(all_n_akaze[0])), dtype=float)
        for i in range(len(all_n_akaze)):
            for j in range(len(all_n_akaze[0])):
                if all_n_akaze[i][j] > self.n_thresh and all_n_orb[i][j] > self.n_thresh:
                    s_akaze = all_scale_akaze[i][j]
                    s_orb = all_scale_orb[i][j]
                    # Check if the scale and translation values are within the specified thresholds
                    if all_in_region([s_akaze, s_orb], self.scale_limits[0], self.scale_limits[1]):
                        tx_akaze = all_tx_akaze[i][j]
                        ty_akaze = all_ty_akaze[i][j]
                        tx_orb = all_tx_orb[i][j]
                        ty_orb = all_ty_orb[i][j]
                        rot_akaze = all_rot_akaze[i][j]
                        rot_orb = all_rot_orb[i][j]
                        # Check if the translation values are within the specified thresholds
                        if within_same_region([tx_akaze, tx_orb], self.tx_tolerance):
                            if within_same_region([ty_akaze, ty_orb], self.ty_tolerance):
                                if within_same_region([rot_akaze, rot_orb], self.rot_tolerance):
                                    # If all conditions are met, add the values to the loop matrix
                                    self.loop_matrix[i][j] = 1
                                    print(f"Loop detected between region D{i+1} and region R{j+1}")
        
        print(f"{Colors.BOLD}{Colors.RED}Loop matrix:\n{self.loop_matrix}{Colors.RESET}")
        return self.loop_matrix

    def reset_memory(self):
        # Reset the memory to free up space
        self.sar_imgs_1.clear()
        self.sar_imgs_2.clear()

        self.all_sift_kp1.clear()
        self.all_sift_des1.clear()
        self.all_sift_kp2.clear()
        self.all_sift_des2.clear()
        self.all_orb_kp1.clear()
        self.all_orb_des1.clear()
        self.all_orb_kp2.clear()
        self.all_orb_des2.clear()
        self.all_surf_kp1.clear()
        self.all_surf_des1.clear()
        self.all_surf_kp2.clear()
        self.all_surf_des2.clear()
        self.all_akaze_kp1.clear()
        self.all_akaze_des1.clear()
        self.all_akaze_kp2.clear()
        self.all_akaze_des2.clear()
        self.all_brisk_kp1.clear()
        self.all_brisk_des1.clear()
        self.all_brisk_kp2.clear()
        self.all_brisk_des2.clear()

        self.good_sift.clear()
        self.good_surf.clear()
        self.good_orb.clear()
        self.good_akaze.clear()
        self.good_brisk.clear()

    def run(self):
        """"""
        """LOOPS"""
        """"""
        # print(f"{Colors.BOLD}{Colors.RED}\n\nLOOP D1 >>> R1 ------------------------------------------------------------------------{Colors.RESET}")
        # self.eval_loops([0, 400])
        # self.reset_memory()
        # print(f"{Colors.BOLD}{Colors.RED}\n\nLOOP D2 >>> R2 ------------------------------------------------------------------------{Colors.RESET}")
        # self.eval_loops([200, 600])
        # self.reset_memory()
        # print(f"{Colors.BOLD}{Colors.RED}\n\nLOOP D3 >>> R3 ------------------------------------------------------------------------{Colors.RESET}")
        # self.eval_loops([400, 800])
        # self.reset_memory()
        # print(f"{Colors.BOLD}{Colors.RED}\n\nLOOP D4 >>> R4 ------------------------------------------------------------------------{Colors.RESET}")
        # self.eval_loops([600, 1000])
        # self.reset_memory()
        # print(f"{Colors.BOLD}{Colors.RED}\n\nLOOP D5 >>> R5 ------------------------------------------------------------------------{Colors.RESET}")
        # self.eval_loops([800, 1200])
        # self.reset_memory()
        # print(f"{Colors.BOLD}{Colors.RED}\n\nLOOP D6 >>> R6 ------------------------------------------------------------------------{Colors.RESET}")
        # self.eval_loops([1000, 1400])
        # self.reset_memory()

        # print(f"{Colors.BOLD}{Colors.RED}\n\nLOOP Dc >>> Rc  -----------------------------------------------------------------------{Colors.RESET}")
        # self.eval_loops([1100, 1400])
        # self.reset_memory()

        """"""
        """NON-LOOPS"""
        """"""
        # print(f"{Colors.BOLD}{Colors.RED}\n\nD1 >>> R3 ------------------------------------------------------------------------{Colors.RESET}")
        # self.eval_non_loops([0, 400], [400, 800])
        # self.reset_memory()
        # print(f"{Colors.BOLD}{Colors.RED}\n\nD2 >>> R4 ------------------------------------------------------------------------{Colors.RESET}")
        # self.eval_non_loops([200, 600], [600, 1000])
        # self.reset_memory()
        # print(f"{Colors.BOLD}{Colors.RED}\n\nD3 >>> R5 ------------------------------------------------------------------------{Colors.RESET}")
        # self.eval_non_loops([400, 800], [800, 1200])
        # self.reset_memory()
        # print(f"{Colors.BOLD}{Colors.RED}\n\nD4 >>> R6 ------------------------------------------------------------------------{Colors.RESET}")
        # self.eval_non_loops([600, 1000], [1000, 1400])
        # self.reset_memory()
        # print(f"{Colors.BOLD}{Colors.RED}\n\nD1 >>> R4 ------------------------------------------------------------------------{Colors.RESET}")
        # self.eval_non_loops([0, 400], [600, 1000])
        # self.reset_memory()
        # print(f"{Colors.BOLD}{Colors.RED}\n\nD2 >>> R5 ------------------------------------------------------------------------{Colors.RESET}")
        # self.eval_non_loops([200, 600], [800, 1200])
        # self.reset_memory()
        # print(f"{Colors.BOLD}{Colors.RED}\n\nD3 >>> R6 ------------------------------------------------------------------------{Colors.RESET}")
        # self.eval_non_loops([400, 800], [1000, 1400])
        # self.reset_memory()
        
        """"""
        """Near-LOOPS"""
        """"""
        all_n_akaze = []
        all_scale_akaze = []
        all_tx_akaze = []
        all_ty_akaze = []
        all_rot_akaze = []
        all_n_orb = []
        all_scale_orb = []
        all_tx_orb = []
        all_ty_orb = []
        all_rot_orb = []

        D_regions = [(0, 400), (200, 600), (400, 800), (600, 1000), (800, 1200), (1000, 1400)]
        R_regions = [(0, 400), (200, 600), (400, 800), (600, 1000), (800, 1200), (1000, 1400)] # should be like: 0, 200, 400, 600, 800, 1000, 1200, 1400
        # R_regions = [(0, 400), (100, 500), (200, 600), (300, 700), (400, 800), (500, 900), (600, 1000), (700, 1100), (800, 1200), (900, 1300), (1000, 1400)]
        R_length = len(R_regions)
        n_akaze = [0 for _ in range(R_length)]
        scale_akaze = [0 for _ in range(R_length)]
        tx_akaze = [0 for _ in range(R_length)]
        ty_akaze = [0 for _ in range(R_length)]
        rot_akaze = [0 for _ in range(R_length)]
        n_orb = [0 for _ in range(R_length)]
        scale_orb = [0 for _ in range(R_length)]
        tx_orb = [0 for _ in range(R_length)]
        ty_orb = [0 for _ in range(R_length)]
        rot_orb = [0 for _ in range(R_length)]

        R_delta = R_regions[1][0] - R_regions[0][0]
        D_delta = D_regions[1][0] - D_regions[0][0]

        for d_start, d_end in D_regions:
            for r_start, r_end in R_regions:
                print(f"{Colors.BOLD}{Colors.RED}\n\nD{d_start // D_delta + 1} >>> R{r_start // R_delta + 1} ------------------------------------------------------------------------{Colors.RESET}")
                akaze, orb = self.eval_non_loops_akaze_orb([d_start, d_end], [r_start, r_end])
                n_akaze[r_start // R_delta], scale_akaze[r_start // R_delta], tx_akaze[r_start // R_delta], ty_akaze[r_start // R_delta], rot_akaze[r_start // R_delta] = akaze
                n_orb[r_start // R_delta], scale_orb[r_start // R_delta], tx_orb[r_start // R_delta], ty_orb[r_start // R_delta], rot_orb[r_start // R_delta] = orb

                if False:
                    # Save all filtered figures to PNG files
                    for i, fig in enumerate(self.filter_figs[-2:]):
                        fig.savefig(f"filter_fig_D{d_start // D_delta + 1}_R{r_start // R_delta + 1}_{i}.png")
            
                self.reset_memory()

            print(f"n_akaze = {n_akaze}")
            print(f"scale_akaze = {scale_akaze}")
            print(f"tx_akaze = {tx_akaze}")
            print(f"ty_akaze = {ty_akaze}")
            print(f"rot_akaze = {rot_akaze}")
            print(f"n_orb = {n_orb}")
            print(f"scale_orb = {scale_orb}")
            print(f"tx_orb = {tx_orb}")
            print(f"ty_orb = {ty_orb}")
            print(f"rot_orb = {rot_orb}")

            all_n_akaze.append(n_akaze.copy())
            all_scale_akaze.append(scale_akaze.copy())
            all_tx_akaze.append(tx_akaze.copy())
            all_ty_akaze.append(ty_akaze.copy())
            all_rot_akaze.append(rot_akaze.copy())
            all_n_orb.append(n_orb.copy())
            all_scale_orb.append(scale_orb.copy())
            all_tx_orb.append(tx_orb.copy())
            all_ty_orb.append(ty_orb.copy())
            all_rot_orb.append(rot_orb.copy())
        
        print("\n\n\n")
        print(f"all_n_akaze = {all_n_akaze}")
        print(f"all_scale_akaze = {all_scale_akaze}")
        print(f"all_tx_akaze = {all_tx_akaze}")
        print(f"all_ty_akaze = {all_ty_akaze}")
        print(f"all_rot_akaze = {all_rot_akaze}")
        print(f"all_n_orb = {all_n_orb}")
        print(f"all_scale_orb = {all_scale_orb}")
        print(f"all_tx_orb = {all_tx_orb}")
        print(f"all_ty_orb = {all_ty_orb}")
        print(f"all_rot_orb = {all_rot_orb}")
        """DATASET #1"""
        all_n_akaze = [[28, 6, 5, 4, 4, 0], [5, 16, 4, 0, 3, 0], [6, 2, 51, 11, 4, 6], [7, 4, 6, 42, 12, 8], [5, 3, 3, 10, 42, 5], [4, 4, 0, 0, 7, 20]]
        all_scale_akaze = [[1.0001753378306222, 1.0036124162171693, 1.1559131522882715, 1.049491956207417, 1.0159310502496866, 0], [0.9743065133112139, 1.0040871943606313, 1.0249854971372434, 0, 0.10658523534307174, 0], [0.9870169447750803, 2.7005569975306862, 1.0001052776520252, 1.1085852368919535, 0.919740435391719, 0.014027028588146207], [0.00619872881411888, 0.013120556259601159, 1.1087909101927655, 0.9992437334333351, 0.9908961511939033, 0.00271964378927176], [0.971324545741301, 1.2043683172031832, 0.9123108700537502, 1.008424344527058, 0.9984876127344069, 1.0832670559200104], [0.27254067460909454, 0.05638716567348145, 0, 0, 0.9946202108920135, 1.004182813824189]]
        all_tx_akaze = [[5.023627551510377, 40.570989976948404, -340.41432282926735, 375.5585524808522, 234.39190966863274, 0], [-55.14361865994861, 3.067640529058401, -204.43179691515664, 0, 271.1937871245828, 0], [-153.26365396711185, -189.6347536029473, 7.804591761727184, -220.6659575950068, -25.125040419442175, 90.48620085598611], [228.9932564804879, 215.57107487303253, 197.66128767552954, 6.829942754591061, 358.1981214973484, 89.2732269932805], [434.3774315074665, 533.1150844301062, 53.11358925253563, 368.5404934908108, 1.7408893832970231, -296.4624213058988], [37.82585356301127, 140.87337376587323, 0, 0, 196.91346453538452, -5.674706196366181]]
        all_ty_akaze = [[-7.330438969727919, -18.08246364357391, 54.68764041067111, 1338.6594578677252, 1320.2702567350555, 0], [38.84586215422871, -6.9546782788866555, -7.90724240072518, 0, 318.2577063252076, 0], [30.494531294933736, -478.34708961909814, -6.005597247670048, -37.82727833662936, 34.77146779943883, 902.8021196256751], [820.8691745015796, 843.9973343774508, -35.07394979362428, -2.3107657348961674, 1285.42995642271, 907.3017772690922], [1229.501637075135, 309.85174752112687, 5.684352294801325, 1291.5551649374672, -0.1561636446846293, 77.56378891601153], [313.7405429625154, 289.5345331571121, 0, 0, 1.1273208530290277, -1.039232054504864]]
        all_rot_akaze = [[-0.6246974276198478, -2.889838696818224, 20.592669465438036, 173.8818118787165, 169.9048049895915, 0], [6.679171358356738, -0.282109934230875, -0.1313796121889936, 0, -109.76267800225422, 0], [9.587201038530067, -15.0660541124871, -0.8062681014816127, -0.6761503213181306, -4.442955250683907, -21.11043018379962], [-61.21419867313682, 26.423320989291433, -0.2163382054601896, -0.6189803730234602, 179.2523657768901, 60.16258189598218], [-171.45396513838327, -128.7415164720715, -2.7135336804869916, 179.95692778086928, -0.21922807299066013, 15.50976880315981], [118.56143790797798, 102.89794033635843, 0, 0, 0.16374134872096924, 0.22129122123062114]]
        all_n_orb = [[35, 13, 7, 6, 10, 9], [14, 18, 8, 8, 10, 12], [7, 6, 22, 10, 10, 17], [4, 4, 9, 23, 12, 7], [7, 6, 7, 11, 11, 5], [5, 5, 7, 9, 4, 23]]
        all_scale_orb = [[1.000644320878523, 1.0464974384149552, 1.1311284973622913, 1.2142471174659009, 1.2317991103413357, 0.03621033880882365], [0.15528400017384275, 0.9921967189927007, 0.2860146681617643, 0.04807691236151543, 1.1361078613068127, 0.06495785041275348], [1.782301350566515, 1.1767034987805247, 1.0104561588646386, 1.004669764525826, 0.0, 0.0004547270552424107], [0.05168501999239127, 0.0002267690048184252, 0.0016882956305546116, 1.0030887519430467, 1.0906670522958146, 0.9400321739929236], [0.0, 0.005512366862599394, 0.0, 0.9772435010155994, 1.0018339053902967, 0.03355234289012471], [0.004783719596598565, 0.01744070549404064, 0.02982979185675725, 0.015905336619202712, 0.019923903702271924, 0.991303997162536]]
        all_tx_orb = [[5.455675603722185, -193.65664807992425, -334.2526755884771, 227.76346479457044, -351.15294616827674, 100.1687427195799], [297.7344548661999, 10.663019141703444, 5.1833230206123675, 28.17742399445171, 678.7063665004183, 70.75592231102999], [-436.99967397504264, 478.15266219171497, 1.0050868829239046, -194.9136759551831, 292.0, 91.84428299855655], [365.3234466599052, 131.00708049046182, 131.0750343299049, 4.9019590195132325, 327.6466934511755, 226.24518682472637], [253.44000244140625, 153.03562435017847, 130.0, 375.5586798852395, 5.6867927879148334, 287.04360359833976], [348.5390384083093, 96.38876477629219, 263.88450803005594, 51.51806432625239, 121.3294340448204, -1.3660888547141232]]
        all_ty_orb = [[-9.482603552036815, -34.01570950313419, 79.00929661026385, -161.6368063143963, 45.84263011380019, 930.1547773780875], [388.35018708272634, -4.080101153653843, 289.3581239927046, 826.7613674092544, 1273.3091026700438, 923.0049385323764], [348.82883102780005, -8.35183454935223, -6.341669652662611, -6.832122768917041, 900.0, 900.3162028329132], [254.88743796707521, 190.9477219390976, 281.99425510416125, -3.7206611826021847, 1387.2186978628713, 500.89220836353127], [348.4800109863281, 267.0614607364975, 281.0, 1275.1050745289367, -1.2719020720606944, 929.2357085375344], [273.8319762888129, 960.4781572207421, 276.32585559573846, 305.57783633142606, 363.0384601608753, 4.682353549008602]]
        all_rot_orb = [[-0.6574040160139217, -2.461206275275544, 21.213629026538307, -25.82858057380214, 21.633260274662014, 179.70750128834788], [-135.92343412128463, -1.5068609164299902, 38.53632921916987, 7.112556793268216, -170.40374391179142, 99.82272483302295], [75.21245281194436, -37.911162093173424, -0.1243818322178884, -0.5414391753932266, -0.0, 126.00074911854574], [-85.44476383929073, -132.835596121785, -153.83834896151765, -0.5052702551533232, 176.00255036515404, -26.77607048491172], [-0.0, 4.6216436291213645, 0.0, -178.16552724392167, -0.5725200742089236, 63.72181707299453], [107.6721673075189, -79.77117965918173, -9.326989659102637, 72.1645842170859, 95.63361872935904, 0.17680014787879714]]
        '''
        Resulting Loop Matrix:
        [[1. 0. 0. 0. 0. 0.]
        [0. 1. 0. 0. 0. 0.]
        [0. 0. 1. 0. 0. 0.]
        [0. 0. 0. 1. 0. 0.]
        [0. 0. 0. 0. 1. 0.]
        [0. 0. 0. 0. 0. 1.]]
        '''
        
        """DATASET #2"""
        # all_n_akaze = [[28, 19, 6, 5, 5, 4, 4, 4, 4, 5, 0], [5, 9, 16, 10, 4, 2, 0, 3, 3, 3, 0], [6, 5, 2, 18, 51, 30, 11, 6, 4, 7, 6], [7, 4, 4, 5, 6, 22, 42, 22, 12, 6, 8], [5, 3, 3, 0, 3, 0, 10, 23, 42, 16, 5], [4, 0, 4, 4, 0, 3, 0, 2, 7, 10, 20]]
        # all_scale_akaze = [[1.0001753378306222, 0.9981959776964353, 1.0036124162171693, 1.0990642482932256, 1.1559131522882715, 1.0339098710977346, 1.049491956207417, 0.32251081900888323, 1.0159310502496866, 1.0995307277270956, 0], [0.9743065133112139, 1.0015957610420112, 1.0040871943606313, 1.0047907604311246, 1.0249854971372434, 6.515211595737906, 0, 0.06874813818952673, 0.10658523534307174, 1.0016882711416981, 0], [0.9870169447750803, 0.452844218648653, 2.7005569975306862, 0.9857921213873596, 1.0001052776520252, 1.0019302254768487, 1.1085852368919535, 0.0006883098309508729, 0.919740435391719, 0.21912919563368372, 0.014027028588146207], [0.00619872881411888, 1.0461999674738003, 0.013120556259601159, 0.011250618932643508, 1.1087909101927655, 1.0022612235729207, 0.9992437334333351, 0.9996175502669855, 0.9908961511939033, 0.0, 0.00271964378927176], [0.971324545741301, 0.369267019228851, 1.2043683172031832, 0, 0.9123108700537502, 0, 1.008424344527058, 0.9987980689640676, 0.9984876127344069, 0.9983939087764989, 1.0832670559200104], [0.27254067460909454, 0, 0.05638716567348145, 0.9026796681862241, 0, 0.9155166856062016, 0, 0.16680457294821197, 0.9946202108920135, 0.995501520791609, 1.004182813824189]]
        # all_tx_akaze = [[5.023627551510377, -94.06076241993014, 40.570989976948404, -224.92663715804258, -340.41432282926735, 194.73748206184956, 375.5585524808522, 195.31875735289447, 234.39190966863274, 357.0620979701114, 0], [-55.14361865994861, 105.11410268860688, 3.067640529058401, -94.6476048742212, -204.43179691515664, -34.599958949721135, 0, 205.51945970271314, 271.1937871245828, 585.0354677190558, 0], [-153.26365396711185, 521.6985553042949, -189.6347536029473, 108.39545152706278, 7.804591761727184, -95.11941038740247, -220.6659575950068, 274.73407471170657, -25.125040419442175, 210.2431419885772, 90.48620085598611], [228.9932564804879, 105.94991478075308, 215.57107487303253, 345.5939998948379, 197.66128767552954, 106.53619626441146, 6.829942754591061, -97.54495338082766, 358.1981214973484, 82.42233276367188, 89.2732269932805], [434.3774315074665, 56.062972006267515, 533.1150844301062, 0, 53.11358925253563, 0, 368.5404934908108, 102.78107246222558, 1.7408893832970231, -102.78858550999466, -296.4624213058988], [37.82585356301127, 0, 140.87337376587323, 363.5085624729989, 0, 204.14416610858245, 0, 218.14263204137316, 196.91346453538452, 94.0098363098593, -5.674706196366181]]
        # all_ty_akaze = [[-7.330438969727919, -6.096420510787986, -18.08246364357391, 79.6404972305621, 54.68764041067111, -55.504321193824, 1338.6594578677252, 682.9949613911866, 1320.2702567350555, 1367.9593487029856, 0], [38.84586215422871, -5.0336536270525345, -6.9546782788866555, -5.189399669437405, -7.90724240072518, -1826.4258962428614, 0, 1005.0712469882459, 318.2577063252076, 1177.1876188416234, 0], [30.494531294933736, 744.2130356053548, -478.34708961909814, 0.5858645589468616, -6.005597247670048, -5.79520336833277, -37.82727833662936, 1005.2311704765349, 34.77146779943883, 807.9206228495959, 902.8021196256751], [820.8691745015796, -73.71970918897574, 843.9973343774508, 846.1279391665862, -35.07394979362428, -3.9307463290859626, -2.3107657348961674, -0.866667238362397, 1285.42995642271, 463.5520935058594, 907.3017772690922], [1229.501637075135, 1126.3725943915713, 309.85174752112687, 0, 5.684352294801325, 0, 1291.5551649374672, -0.011457644526434052, -0.1561636446846293, 1.0519422168789812, 77.56378891601153], [313.7405429625154, 0, 289.5345331571121, 1213.46076719686, 0, 1284.1185055778894, 0, 412.69764303871096, 1.1273208530290277, 0.6979528870605383, -1.039232054504864]]
        # all_rot_akaze = [[-0.6246974276198478, -0.6270331453744747, -2.889838696818224, 21.237111096188485, 20.592669465438036, -18.092916033390843, 173.8818118787165, -4.287788482924431, 169.9048049895915, -178.6559475085042, 0], [6.679171358356738, -0.4846124109712491, -0.282109934230875, -0.5420802827598593, -0.1313796121889936, -30.928725128909313, 0, 77.9890255655526, -109.76267800225422, -161.609405272529, 0], [9.587201038530067, -94.96020336297192, -15.0660541124871, -0.25906726543601566, -0.8062681014816127, -0.49415481412882745, -0.6761503213181306, -6.688186132919087, -4.442955250683907, -41.91125416632239, -21.11043018379962], [-61.21419867313682, -11.44013307257825, 26.423320989291433, -21.959207779557758, -0.2163382054601896, -0.5124327819935643, -0.6189803730234602, -0.30725799834779405, 179.2523657768901, 0.0, 60.16258189598218], [-171.45396513838327, 157.2074326535915, -128.7415164720715, 0, -2.7135336804869916, 0, 179.95692778086928, -0.35038701402885747, -0.21922807299066013, 0.33934073379899493, 15.50976880315981], [118.56143790797798, 0, 102.89794033635843, -173.1837019759087, 0, 166.24997581368325, 0, 146.12336172879438, 0.16374134872096924, 0.37300217980438916, 0.22129122123062114]]
        # all_n_orb = [[35, 33, 13, 5, 7, 8, 6, 8, 10, 9, 9], [14, 14, 18, 8, 8, 9, 8, 6, 10, 11, 12], [7, 6, 6, 19, 22, 14, 10, 4, 10, 28, 17], [4, 6, 4, 12, 9, 17, 23, 15, 12, 14, 7], [7, 5, 6, 7, 7, 11, 11, 10, 11, 8, 5], [5, 6, 5, 6, 7, 4, 9, 6, 4, 7, 23]]
        # all_scale_orb = [[1.000644320878523, 1.0036809225014083, 1.0464974384149552, 1.3366950769079675, 1.1311284973622913, 0.4221297401238725, 1.2142471174659009, 0.09069762056971342, 1.2317991103413357, 0.0, 0.03621033880882365], [0.15528400017384275, 1.002201775905638, 0.9921967189927007, 0.999412882597484, 0.2860146681617643, 0.4033716293923764, 0.04807691236151543, 1.3140412582898373, 1.1361078613068127, 0.31217357482582275, 0.06495785041275348], [1.782301350566515, 0.09235923383997834, 1.1767034987805247, 1.0014469495888216, 1.0104561588646386, 1.0362042353444134, 1.004669764525826, 0.1459248732248257, 0.0, 0.001044209383603572, 0.0004547270552424107], [0.05168501999239127, 0.04831063099517577, 0.0002267690048184252, 0.006358656114912514, 0.0016882956305546116, 1.0019728169813749, 1.0030887519430467, 1.0002412801585063, 1.0906670522958146, 0.005854529936881372, 0.9400321739929236], [0.0, 0.9372705360281667, 0.005512366862599394, 1.039634665041048, 0.0, 0.12325604120948098, 0.9772435010155994, 1.0010065242924444, 1.0018339053902967, 0.9710092173420017, 0.03355234289012471], [0.004783719596598565, 0.006352228551656701, 0.01744070549404064, 0.9404411134773216, 0.02982979185675725, 0.02759948632466764, 0.015905336619202712, 0.021825370610340847, 0.019923903702271924, 0.5486198230547938, 0.991303997162536]]
        # all_tx_orb = [[5.455675603722185, -95.36837648746575, -193.65664807992425, -278.7459172408014, -334.2526755884771, 503.358965542861, 227.76346479457044, 204.71799381043547, -351.15294616827674, 192.0, 100.1687427195799], [297.7344548661999, 104.14920641570555, 10.663019141703444, -94.29107245617423, 5.1833230206123675, 490.67663806692144, 28.17742399445171, 588.1697002675543, 678.7063665004183, 48.37056547295563, 70.75592231102999], [-436.99967397504264, 93.68972595751967, 478.15266219171497, 103.39776812598436, 1.0050868829239046, -105.97170806650524, -194.9136759551831, 330.2791974317151, 292.0, 192.15806534885252, 91.84428299855655], [365.3234466599052, 136.2721563003248, 131.00708049046182, 342.83637406754866, 131.0750343299049, 105.25335039179667, 4.9019590195132325, -98.55974439785966, 327.6466934511755, 189.37083089061997, 226.24518682472637], [253.44000244140625, 407.258992951253, 153.03562435017847, -20.87480445581148, 130.0, 376.9313358838034, 375.5586798852395, 264.58078332468676, 5.6867927879148334, -86.82483304066216, 287.04360359833976], [348.5390384083093, 129.05937215878498, 96.38876477629219, -5.065480750879835, 263.88450803005594, 151.59869525872745, 51.51806432625239, 220.71798047774791, 121.3294340448204, 570.2707218575965, -1.3660888547141232]]
        # all_ty_orb = [[-9.482603552036815, -10.001043303714532, -34.01570950313419, -17.282287898270436, 79.00929661026385, 862.1303996642642, -161.6368063143963, 387.3267186803857, 45.84263011380019, 900.0, 930.1547773780875], [388.35018708272634, -6.756870676372058, -4.080101153653843, -4.381415935146127, 289.3581239927046, 840.4511174308726, 826.7613674092544, 1319.5539866140446, 1273.3091026700438, 887.8587079322598, 923.0049385323764], [348.82883102780005, 376.03569457426715, -8.35183454935223, -4.5524395540497595, -6.341669652662611, -14.088138211665898, -6.832122768917041, 326.6272185888764, 900.0, 899.2482637759516, 900.3162028329132], [254.88743796707521, 333.99132886893733, 190.9477219390976, 842.5904694227414, 281.99425510416125, -2.6130633784451316, -3.7206611826021847, -1.1063993874368556, 1387.2186978628713, 897.2387095231347, 500.89220836353127], [348.4800109863281, 1253.9132402085795, 267.0614607364975, -57.345798870109604, 281.0, 906.3329411989911, 1275.1050745289367, 1288.5866328374118, -1.2719020720606944, 10.303434224941604, 929.2357085375344], [273.8319762888129, 377.33727862897183, 960.4781572207421, 86.70364196608513, 276.32585559573846, 281.2988281714686, 305.57783633142606, 365.78406563433623, 363.0384601608753, 246.54470550034037, 4.682353549008602]]
        # all_rot_orb = [[-0.6574040160139217, -0.6466175628123384, -2.461206275275544, 16.099368864842067, 21.213629026538307, -105.69518125897787, -25.82858057380214, -166.0550730034646, 21.633260274662014, -0.0, 179.70750128834788], [-135.92343412128463, -0.46140483597444204, -1.5068609164299902, -0.4944971337011639, 38.53632921916987, -100.15462002003365, 7.112556793268216, -168.4118345431734, -170.40374391179142, 49.759774974674656, 99.82272483302295], [75.21245281194436, 69.1268213013017, -37.911162093173424, -0.3512800743872508, -0.1243818322178884, -0.046591358069100974, -0.5414391753932266, 25.956015519530105, -0.0, -39.52634055981602, 126.00074911854574], [-85.44476383929073, 6.777386130073949, -132.835596121785, 120.57890435972378, -153.83834896151765, -0.5311135186277911, -0.5052702551533232, -0.22415395810191477, 176.00255036515404, 44.650395845528834, -26.77607048491172], [-0.0, 179.3103683785483, 4.6216436291213645, 16.633806020210404, 0.0, -106.6007034885541, -178.16552724392167, 179.53940013220762, -0.5725200742089236, -1.1554525917837601, 63.72181707299453], [107.6721673075189, -49.17657848308696, -79.77117965918173, 22.142230680332652, -9.326989659102637, -7.673019198198, 72.1645842170859, 101.7627598885787, 95.63361872935904, -95.1173514024562, 0.17680014787879714]]
        '''
        Resulting Loop matrix:
        [[1. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
        [0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
        [0. 0. 0. 1. 1. 0. 0. 0. 0. 0. 0.]
        [0. 0. 0. 0. 0. 1. 1. 1. 0. 0. 0.]
        [0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
        [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]]
        '''
        self.gen_loop_matrix(all_n_akaze, all_scale_akaze, all_tx_akaze, all_ty_akaze, all_rot_akaze, all_n_orb, all_scale_orb, all_tx_orb, all_ty_orb, all_rot_orb)



        """"""
        """SAVE IMAGES"""
        """"""
        # # Save all generated figures to PNG files
        # for i, fig in enumerate(self.feature_figs):
        #     fig.savefig(f"feature_fig_{i}.png")
        # for i, fig in enumerate(self.match_figs):
        #     fig.savefig(f"match_fig_{i}.png")
        # for i, fig in enumerate(self.filter_figs):
        #     fig.savefig(f"filter_fig_{i}.png")
        # self.img_fig.savefig("sar_images.png")

def within_same_region(numbers, tolerance):
    # e.g. numbers = [1, 2, 3, 4, 5]
    # tolerance = 2
    # Check if all numbers are within the same region
    # by checking if the difference between the maximum and minimum
    # values is less than or equal to 2 * tolerance
    return max(numbers) - min(numbers) <= 2 * tolerance

def all_in_region(numbers, a, b, inclusive=True):
    # e.g. numbers = [1, 2, 3, 4, 5]
    # a = 1
    # b = 5
    # Check if all numbers are within the range [a, b]
    if inclusive:
        return all(a <= x <= b for x in numbers)
    else:
        return all(a < x < b for x in numbers)
# **********************************************************************************************
# MAIN FUNCTION
def main(args=None):
    rclpy.init(args=args)
    logging.info(f"{Colors.RED}Here we go!!!{Colors.RESET}") 
    img_match = Matcher() 
    try:
        rclpy.spin(img_match)
    except KeyboardInterrupt:
        img_match.get_logger().info(f"{Colors.RED}Keyboard Interrupt{Colors.RESET}")
        
    img_match.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    try:
        main()
    except:
        logging.error('Error in the main function')
        rclpy.shutdown()
        pass