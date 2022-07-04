import argparse
import pandas as pd
import rosbag
from bagpy import bagreader
import os
import numpy as np
import matplotlib.pyplot as pl
from cv_bridge import CvBridge
import cv2
from os import path
from scipy.ndimage.filters import gaussian_filter1d, median_filter
from scipy.signal import savgol_filter
import pickle
import json
from derived_object_msgs.msg import ObjectArray
from std_msgs.msg import Header
import yaml

def load_bag(args):
    log_path = "{}/{}.bag".format(args.save_path, args.bag_file)
    try:
        return bagreader(log_path)
    except:
        print("Could not find bag file")
        return False

def bag_to_csv(args):
    files, bag = [], []

    if not os.path.isfile("{}/{}/joy.csv".format(args.save_path, args.bag_file)):
        bag = load_bag(args)
        print(bag.topic_table)
        # Process amount of feedback and get episode reset flags
        csv = bag.message_by_topic('/joy')
    else:
        csv = "{}/{}/joy.csv".format(args.save_path, args.bag_file)

    files.append(csv)

    if not os.path.isfile("{}/{}/human_action.csv".format(args.save_path, args.bag_file)):
        if not bag:
            bag = load_bag(args)
            print(bag.topic_table)
        # Create human action csv file
        csv = bag.message_by_topic('/human_action')
    else:
        csv = "{}/{}/human_action.csv".format(args.save_path, args.bag_file)

    files.append(csv)

    if not os.path.isfile("{}/{}/network_action.csv".format(args.save_path, args.bag_file)):
        if not bag:
            bag = load_bag(args)
            print(bag.topic_table)
        # Create human action csv file
        csv = bag.message_by_topic('/network_action')
    else:
        csv = "{}/{}/network_action.csv".format(args.save_path, args.bag_file)

    files.append(csv)

    if not os.path.isfile("{}/{}/feedback.csv".format(args.save_path, args.bag_file)):
        if not bag:
            bag = load_bag(args)
            print(bag.topic_table)
        # Create human action csv file
        csv = bag.message_by_topic('/feedback')
    else:
        csv = "{}/{}/feedback.csv".format(args.save_path, args.bag_file)

    files.append(csv)

    if not os.path.isfile("{}/{}/lmpcc-initialpose.csv".format(args.save_path, args.bag_file)):
        if not bag:
            bag = load_bag(args)
            print(bag.topic_table)
        # Create human action csv file
        csv = bag.message_by_topic('/lmpcc/initialpose')
    else:
        csv = "{}/{}/lmpcc-initialpose.csv".format(args.save_path, args.bag_file)

    files.append(csv)

    if not os.path.isfile("{}/{}/h_counter.csv".format(args.save_path, args.bag_file)):
        if not bag:
            bag = load_bag(args)
            print(bag.topic_table)
        # Create human action csv file
        csv = bag.message_by_topic('/h_counter')
    else:
        csv = "{}/{}/h_counter.csv".format(args.save_path, args.bag_file)

    files.append(csv)

    if not os.path.isfile("{}/{}/done.csv".format(args.save_path, args.bag_file)):
        if not bag:
            bag = load_bag(args)
            print(bag.topic_table)
        # Create human action csv file
        csv = bag.message_by_topic('/done')
    else:
        csv = "{}/{}/done.csv".format(args.save_path, args.bag_file)

    files.append(csv)

    if not os.path.isfile("{}/{}/carla-ego_vehicle-odometry.csv".format(args.save_path, args.bag_file)):
        if not bag:
            bag = load_bag(args)
            print(bag.topic_table)
        # Create human action csv file
        csv = bag.message_by_topic('/carla/ego_vehicle/odometry')
    else:
        csv = "{}/{}/carla-ego_vehicle-odometry.csv".format(args.save_path, args.bag_file)

    files.append(csv)

    if not os.path.isfile("{}/{}/carla-objects.csv".format(args.save_path, args.bag_file)):
        if not bag:
            bag = load_bag(args)
            print(bag.topic_table)
        # Create human action csv file
        csv = bag.message_by_topic('/carla/objects')
    else:
        csv = "{}/{}/carla-objects.csv".format(args.save_path, args.bag_file)

    files.append(csv)

    return files

def process_episode_data(files):
    """ Requirements to consider episode information into the statistic results """

    if os.path.isfile("{}/{}/episode_info.pickle".format(args.save_path, args.bag_file)):

        with open("{}/{}/episode_info.pickle".format(args.save_path, args.bag_file),encoding='utf-8') as f:
            episode_info = pickle.load(f)

    else:

        feedback = pd.read_csv(files[3])
        reset = pd.read_csv(files[4])['Time']
        accumulated_feedback = pd.read_csv(files[5])
        done_signal = pd.read_csv(files[6])
        vehicle_state = pd.read_csv(files[7])
        human_actions = pd.read_csv(files[1])
        network_actions = pd.read_csv(files[2])
        carla_objects = pd.read_csv(files[8])

        min_episode_length = 10  # in meters equivalent to traveled distance

        time = np.round(network_actions['Time'].to_numpy(),1)
        t0 = np.round(done_signal['Time'].to_numpy()[0],1)
        t0_index = np.where(time == t0)[0][0]
        dones = done_signal['data'].to_numpy()
        k = 0

        trajectory = []
        episode_info = []
        traveled_distance = 0
        episodes_counter = 0
        feedback_counter = 0

        for t in range(t0_index,dones.shape[0]+t0_index,1):

            # Collect trajectory data
            position = vehicle_state.iloc[t,6:8].to_numpy()
            velocity = vehicle_state.iloc[t,14:16].to_numpy()
            feedback_counter += int(feedback.iloc[k].to_numpy()[1])
            feed = feedback.iloc[k].to_numpy()[1]
            human_action = human_actions.iloc[k].to_numpy()[1]
            network_action = network_actions.iloc[k].to_numpy()[1]
            objects = carla_objects.iloc[k][5][1:].split("classification_age")

            objects_dict = []
            for obj in objects:
                id = obj.find('header')
                if len(obj) > 10:
                    objects_dict.append(yaml.safe_load(obj[id:]))
            header = Header()
            header.seq = carla_objects.iloc[k][1]
            header.stamp.secs = carla_objects.iloc[k][2]
            header.stamp.nsecs = carla_objects.iloc[k][3]
            header.frame_id = carla_objects.iloc[k][4]

            dict = {'t' : time[t],
                'position': position,
                'velocity': velocity,
                'feedback_counter': feedback_counter,
                'feedback': feed,
                'human_action': human_action,
                'network_action': network_action,
                'objects': objects_dict,
                'header': header
                }

            trajectory.append(dict)

            if (len(trajectory)>1):
                traveled_distance += np.linalg.norm(trajectory[len(trajectory)-1]['position'] - trajectory[len(trajectory)-2]['position'])

            if(done_signal.iloc[k,1] and traveled_distance > min_episode_length):
                episode_info.append(trajectory)
                trajectory = []
                traveled_distance = 0
                episodes_counter +=1
            k += 1


        print("Total numer of episodes: {}", episodes_counter)

        with open("{}/{}/episode_info.pickle".format(args.save_path, args.bag_file), 'wb') as f:
            pickle.dump(episode_info, f,protocol=0)

    return episode_info

def draw_feedback_plot(args,episode_info):

    time = []
    accumulated_feedback = []
    avg_feedback_error = 0
    feedback_error= []
    counter = 0

    for ep_info in episode_info:
        if len(ep_info) > 100:
            for t in range(0,len(ep_info)):
                time.append(ep_info[t]['t']-100)
                accumulated_feedback.append(ep_info[t]['feedback_counter']/1000)
                if ep_info[t]['feedback']:
                    error = np.abs(ep_info[t]['human_action'] - ep_info[t]['network_action'])
                    avg_feedback_error = (avg_feedback_error*counter+error)/(counter+1)
                    counter += 1
                feedback_error.append(avg_feedback_error)

    time = np.array(time)
    accumulated_feedback = np.array(accumulated_feedback)
    feedback_error = np.array(feedback_error)

    fig, (ax , ax2) = pl.subplots(2)  # Create a figure containing a single axes.

    ax.plot(time, accumulated_feedback,linewidth=4)
    ax.grid(True)
    ax.set_xlabel('Time [s]')  # Add an x-label to the axes.
    ax.set_ylabel('Total # of demonstrations')  # Add a y-label to the axes.
    ax.set_xlim(0, time[-1])
    ax.set_title("Training Evolution")  # Add a title to the axes.
    #fig.suptitle("Training Evolution")  # Add a title to the axes.
    ax2.plot(time, feedback_error,linewidth=4)
    ax2.grid(True)
    ax2.set_xlabel('Time [s]')  # Add an x-label to the axes.
    ax2.set_ylabel('Feedback Error [m/s]')  # Add a y-label to the axes.
    ax2.legend()  # Add a legend.
    ax2.set_xlim(0, time[-1])

    fig.tight_layout()
    file = "{}/{}/trainig_evolution.png".format(args.save_path, args.bag_file)
    pl.savefig(file)

def bar_feedback_plot(args,episode_info):

    time = []
    accumulated_feedback = []
    avg_feedback_error = 0#np.zeros([50])
    feedback_error= []
    counter = 0
    episodes_index = []
    idx = 0

    for ep_info in episode_info:
        if len(ep_info) > 100:
            start_counting = False
            accumulated_feedback.append(0)
            episodes_index.append(idx+1)
            counter2 = 0
            for t in range(0,len(ep_info)):
                time.append(ep_info[t]['t']-100)

                if ep_info[t]['feedback']:
                    start_counting = True

                accumulated_feedback[idx] += ep_info[t]['feedback']
                if ep_info[t]['feedback']:
                    error = np.abs(ep_info[t]['human_action'] - ep_info[t]['network_action'])
                    avg_feedback_error = (avg_feedback_error*counter+error)/(counter+1)
                    #avg_feedback_error = np.roll(avg_feedback_error, 1)
                    #avg_feedback_error[0] = error
                    counter += 1

                if start_counting:
                    counter2 += 1

                feedback_error.append(avg_feedback_error)
            accumulated_feedback[idx] /= counter2

            idx +=1

    time = np.array(time)
    accumulated_feedback = np.array(accumulated_feedback)
    feedback_error = np.array(feedback_error)

    fig, (ax , ax2) = pl.subplots(2)  # Create a figure containing a single axes.
    pl.grid(True)
    fig.tight_layout()
    rects = ax.bar(episodes_index, accumulated_feedback,
                     align='center')
    ax.set_xlabel('Episode number')  # Add an x-label to the axes.
    ax.set_ylabel('Percentage of demonstrations')  # Add a y-label to the axes.
    #ax.set_title("Training Evolution")  # Add a title to the axes.
    fig.suptitle("Training Evolution")  # Add a title to the axes.
     # Or equivalently,  "plt.tight_layout()"
    ax2.plot(time, feedback_error,linewidth=4)
    ax2.set_xlabel('Time [s]')  # Add an x-label to the axes.
    ax2.set_ylabel('Feedback Error [m/s]')  # Add a y-label to the axes.
    ax2.legend()  # Add a legend.
    ax2.set_xlim(500, time[-1])
    file = "{}/{}/training_evolution.png".format(args.save_path, args.bag_file)
    pl.savefig(file)

def save_env_image(args):
    log_path = "{}/{}.bag".format(args.save_path, args.bag_file)

    # Get image topic
    bag = rosbag.Bag(log_path)
    for topic, msg, t in bag.read_messages(topics=['/lmpcc/view']):
        image = msg

    bridge = CvBridge()
    cv_image = bridge.imgmsg_to_cv2(image, desired_encoding='passthrough')

    save_path = "{}/{}.png".format(args.save_path, args.bag_file)
    cv2.imwrite(save_path, cv_image)

    print("saving image")

def save_images(args):
    log_path = "{}/{}.bag".format(args.save_path, args.bag_file)

    save_path = "{}/{}/images".format(args.save_path, args.bag_file)

    if not path.isdir(save_path):
        os.mkdir(save_path)

    # Get image topic
    bag = rosbag.Bag(log_path)
    bridge = CvBridge()
    """
    i = 0
    for topic, msg, t in bag.read_messages(topics=['/carla/ego_vehicle/camera/rgb/front/image_color']):
        save_path = "{}/{}/images/{}.png".format(args.save_path, args.bag_file,t.secs+np.round(t.nsecs*10**-9,1))
        image = msg
        cv_image = bridge.imgmsg_to_cv2(image, desired_encoding='passthrough')
        cv2.imwrite(save_path, cv_image)
        i +=1
    """
    i = 0
    save_path = "{}/{}/processed_view_images".format(args.save_path, args.bag_file)
    if not path.isdir(save_path):
        os.mkdir(save_path)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for topic, msg, t in bag.read_messages(topics=['/lmpcc/processed_view']):
        time = t.secs+np.round(t.nsecs*10**-9,1)
        save_path = "{}/{}/processed_view_images/{}.png".format(args.save_path, args.bag_file,time)
        image = msg
        cv_image = bridge.imgmsg_to_cv2(image, desired_encoding='passthrough')
        #new_image = cv2.putText(cv_image, "t ={}".format(time), (420, 450),   fontFace = cv2.FONT_HERSHEY_DUPLEX,
        #  fontScale = 1.0,
        #  color = (0, 0, 0),
        #  thickness = 3)
        new_image = cv_image
        cv2.imwrite(save_path, new_image)
        cv2.imshow("test", new_image)
        i +=1

    print("saving images done!")

def record_video(args):
    log_path = "{}/{}.bag".format(args.save_path, args.bag_file)

    save_path = "{}/{}/project.avi".format(args.save_path, args.bag_file)

    if path.isdir(save_path):
        os.mkdir(save_path)

    # Get image topic
    bag = rosbag.Bag(log_path)
    bridge = CvBridge()
    i = 0
    size = (640, 640)
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    out = cv2.VideoWriter(save_path, fourcc, 10, size)
    for topic, msg, t in bag.read_messages(topics=['/lmpcc/processed_view']):
        image = msg
        cv_image = bridge.imgmsg_to_cv2(image, desired_encoding='passthrough')
        resized = cv2.resize(cv_image, size, interpolation=cv2.INTER_AREA)
        out.write(resized[:,:,:3])
        i +=1
    out.release()
    print("video recording done!")

def draw_vel_ref_plot(args,episode_info):

    print("draw_vel_ref_plot")
    save_path = "{}/{}/vel_ref_images".format(args.save_path, args.bag_file)
    if not path.isdir(save_path):
        os.mkdir(save_path)

    for ep_info in episode_info:
        if len(ep_info) > 100:
            time_window = 75 # equivalent to 10s
            for t in range(time_window,len(ep_info)-time_window):
                info_window = ep_info[t-time_window:t+time_window]
                time, vel_ref = [],[]
                for info in info_window:
                    vel_ref.append(info['network_action'])
                    time.append(info['t'])

                ysmoothed = savgol_filter(vel_ref,21,1)
                pl.rcParams.update({'font.size': 32})
                fig, ax= pl.subplots()
                fig.set_size_inches(36, 8)# Create a figure containing a single axes.
                ax.plot(time, ysmoothed,linewidth=4)
                ax.set_xlabel('Time [s]')  # Add an x-label to the axes.
                ax.set_ylabel('Velocity Reference [m/s]')  # Add a y-label to the axes.
                ax.set_xlim(time[0], time[-1])
                ax.set_ylim(0, 8)
                ax.vlines(x=1500,ymin=0,ymax=8)
                file = "{}/{}.png".format(save_path,time[time_window])
                pl.grid(True)
                pl.savefig(file, bbox_inches='tight')
                pl.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, default="/home/amr/coachmpc/bags")
    parser.add_argument('--bag_file', type=str, default="mpc_coach_2021-11-23-16-48-19")

    args = parser.parse_args()

    files = bag_to_csv(args)

    # process episode data
    episode_info = process_episode_data(files)

    # Generate Accumulated Feedback plot
    #draw_feedback_plot(args, episode_info)

    res = ObjectArray(episode_info[0][0]['header'],episode_info[0][0]['objects'])

    # Velocity Reference Plot
    draw_vel_ref_plot(args,episode_info)

    # Bar plot with feedback per episode
    #bar_feedback_plot(args, episode_info)

    # Save images
    #save_images(args)

    # Record Video
    #record_video(args)
    """
    save_env_image(args)
    """

