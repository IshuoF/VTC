import json
import os 
import random
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import argparse

def draw_plot(ori_pos3d, frame_pos3d):
    ori_x_values = [coord[0] for coord in ori_pos3d.values()]
    ori_y_values = [coord[1] for coord in ori_pos3d.values()]
    ori_z_values = [coord[2] for coord in ori_pos3d.values()]

    x_values = [coord[0] for coord in frame_pos3d.values()]
    y_values = [coord[1] for coord in frame_pos3d.values()]
    z_values = [coord[2] for coord in frame_pos3d.values()]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    draw_volleyball_court(ax)
    ax.scatter(ori_x_values, ori_y_values, ori_z_values, marker='o', color='skyblue', label='original traj')
    ax.scatter(x_values, y_values, z_values, marker='o', color='tomato', label='augmented traj')

    # ax.set_title('3D Scatter Plot of frame_pos3d')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')

    ax.set_xlim3d(left=0, right=9)
    ax.set_ylim3d(bottom=-1, top=19)
    ax.set_zlim3d(bottom=0, top=5)
    ax.view_init(30, 200)
    plt.legend()
    plt.show()
    # plt.savefig('aug.png')
    plt.clf()

def draw_volleyball_court(ax):
    points = np.array([
        [0, 0, 0], [9, 0, 0], [0, 6, 0], [9, 6, 0], [0, 9, 0],
        [9, 9, 0], [0, 12, 0], [9, 12, 0], [0, 18, 0], [9, 18, 0]
    ])
    courtedge = [2, 0, 1, 3, 2, 4, 5, 3, 5, 7, 6, 4, 6, 8, 9, 7]
    curves = points[courtedge]

    netpoints = np.array([
        [0, 9, 0], [0, 9, 1.24], [0, 9, 2.24], [9, 9, 0], [9, 9, 1.24], [9, 9, 2.24]])
    netedge = [0, 1, 2, 5, 4, 1, 4, 3]
    netcurves = netpoints[netedge]

    court = points.T
    courtX, courtY, courtZ = court
    # plot 3D court reference points

    ax.scatter(courtX, courtY, courtZ, c='black', marker='o', s=1)
    ax.plot(curves[:, 0], curves[:, 1], c='k',
            linewidth=2, alpha=0.5)  # plot 3D court edges
    ax.plot(netcurves[:, 0], netcurves[:, 1], netcurves[:, 2],
            c='k', linewidth=2, alpha=0.5)  # plot 3D net edges

    # ground = Rectangle([0, 0], 9, 18, zorder=3)
    # ax.add_patch(ground)
    # art3d.pathpatch_2d_to_3d(ground, z=0, zdir='z')
    """
    # plot 2D court reference points
    ax2D.scatter(courtX, courtY, c='b', marker='o')
    ax2D.plot(curves[:, 0], curves[:, 1], c='k',
              linewidth=3, alpha=0.5)  # plot 2D court edges
    """
    
def load_trajectory_data(traj_path):
    if os.path.exists(traj_path):
        with open(traj_path, 'r') as file:
            traj_data = json.load(file)
            frame_data = traj_data.get('frame_data')
            layout_data = traj_data.get('layout_data')
            frame_pos3d = traj_data.get('frame_pos3d')
            frame_pos3d_color = traj_data.get('frame_pos3d_color')  
            collided_frame_id_list = traj_data.get('collided_frame_id_list')
            spike_estimations = traj_data.get('spike_estimations')
            label = traj_data.get('label') 
            length_of_frame_data = len(frame_data)
            # print(f"The length of frame_data is: {length_of_frame_data}")
            length_of_frame_pos3d = len(frame_pos3d)
            # print(f"The length of frame_pos3d is: {length_of_frame_pos3d}")

        return frame_pos3d, length_of_frame_data, length_of_frame_pos3d,frame_pos3d_color, \
               frame_data, layout_data, collided_frame_id_list, spike_estimations, label

        
def on_circle_augment(traj_path, output_dir, modified_ratio): # choose radius ??
    
    frame_pos3d, length_of_frame_data, length_of_frame_pos3d, frame_pos3d_color,\
    frame_data, layout_data, collided_frame_id_list, spike_estimations = load_trajectory_data(traj_path)
    
    if collided_frame_id_list is not None:
        ori_pos3d = frame_pos3d.copy()
        num_points_to_modify=int(modified_ratio*length_of_frame_pos3d)  # User can modify the ratio of points to modify
        frames_to_modify = random.sample(list(set(frame_pos3d.keys()) - set(collided_frame_id_list)), num_points_to_modify)
        
        for frame_id in frames_to_modify:
            pos3d = frame_pos3d[frame_id]
            
            # Calculate the radius as 10% of the distance between the current point and the next point
            next_frame_id = str(int(frame_id) + 1)
            if next_frame_id in frame_pos3d:
                radius = 0.1 * math.sqrt(sum((pos3d[i] - frame_pos3d[next_frame_id][i])**2 for i in range(3)))
                # print(f"radius: {radius}")
            else:
                radius = 0

            # Randomly select a degree for modifications
            degree = random.uniform(0, 360)

            # Calculate the new point 
            new_x = pos3d[0] + radius * math.cos(math.radians(degree))
            new_y = pos3d[1] + radius * math.sin(math.radians(degree))
            new_z = pos3d[2] 

            frame_pos3d[frame_id] = [new_x, new_y, new_z]
            
        length_of_frame_data = len(frame_data)
        # print(f"The length of frame_data is: {length_of_frame_data}")

        length_of_frame_pos3d = len(frame_pos3d)
        # print(f"The length of frame_pos3d is: {length_of_frame_pos3d}")

        new_data = {
            'frame_data': frame_data,
            'layout_data': layout_data,
            'frame_pos3d': frame_pos3d,
            'frame_pos3d_color': frame_pos3d_color,
            'collided_frame_id_list': collided_frame_id_list,
            'spike_estimations': spike_estimations,
        }

        new_json_data = json.dumps(new_data, indent=2)

        new_data_name = os.path.join(output_dir, traj_path.split('/')[-1].split('.')[0] + f'_aug_on_1.json')
        # print(new_data_name)
        with open(new_data_name, 'w') as file:
            file.write(new_json_data)

        draw_plot(ori_pos3d, frame_pos3d)
        
def in_circle_augment(traj_path, output_dir, modified_ratio):
    frame_pos3d, length_of_frame_data, length_of_frame_pos3d, frame_pos3d_color,\
    frame_data, layout_data, collided_frame_id_list, spike_estimations = load_trajectory_data(traj_path)

    if collided_frame_id_list is not None:
        ori_pos3d = frame_pos3d.copy()
        num_points_to_modify=int(modified_ratio*length_of_frame_pos3d)  # Randomly select num_points_to_modify points
        frames_to_modify = random.sample(list(set(frame_pos3d.keys()) - set(collided_frame_id_list)), num_points_to_modify)

        for frame_id in frames_to_modify:
            pos3d = frame_pos3d[frame_id]

            # Randomly select a distance in a circular region around the current point
            next_frame_id = str(int(frame_id) + 1)
            if next_frame_id in frame_pos3d:
                distance_limit = 0.1 * math.sqrt(sum((pos3d[i] - frame_pos3d[next_frame_id][i])**2 for i in range(3)))

                # Randomly select a distance smaller than the distance between the current point and the next
                distance = random.uniform(0, distance_limit)
                # print(f"distance: {distance}")

                # Randomly select a degree for modifications
                degree = random.uniform(0, 360)

                # Calculate the new point at the aforementioned distance in the direction of the selected degree
                new_x = pos3d[0] + distance * math.cos(math.radians(degree))
                new_y = pos3d[1] + distance * math.sin(math.radians(degree))
                new_z = pos3d[2]

                frame_pos3d[frame_id] = [new_x, new_y, new_z]

        length_of_frame_data = len(frame_data)
        # print(f"The length of frame_data is: {length_of_frame_data}")

        length_of_frame_pos3d = len(frame_pos3d)
        # print(f"The length of frame_pos3d is: {length_of_frame_pos3d}")

        new_data = {
            'frame_data': frame_data,
            'layout_data': layout_data,
            'frame_pos3d': frame_pos3d,
            'frame_pos3d_color': frame_pos3d_color,
            'collided_frame_id_list': collided_frame_id_list,
            'spike_estimations': spike_estimations,
        }

        new_json_data = json.dumps(new_data, indent=2)

        new_data_name = os.path.join(output_dir, traj_path.split('/')[-1].split('.')[0] + f'_aug_in_2.json')
        # print(new_data_name)
        with open(new_data_name, 'w') as file:
            file.write(new_json_data)

        draw_plot(ori_pos3d, frame_pos3d)
        
def point_stretching_augment(traj_path, output_dir,  modified_ratio, max_distance, method): 
    frame_pos3d, length_of_frame_data, length_of_frame_pos3d, frame_pos3d_color,\
    frame_data, layout_data, collided_frame_id_list, spike_estimations = load_trajectory_data(traj_path)

    if collided_frame_id_list is not None:
        ori_pos3d = frame_pos3d.copy()
        num_points_to_modify=int(modified_ratio*length_of_frame_pos3d)  # Randomly select num_points_to_modify points
        frames_to_modify = random.sample(list(set(frame_pos3d.keys()) - set(collided_frame_id_list)), num_points_to_modify)

        for frame_id in frames_to_modify:
            pos3d = frame_pos3d[frame_id]

            # Calculate maximum latitudes and longitudes in each direction based on the maximum allowed distance
            max_latitude = pos3d[0] + (max_distance / 2)
            min_latitude = pos3d[0] - (max_distance / 2)
            max_longitude = pos3d[1] + (max_distance / 2)
            min_longitude = pos3d[1] - (max_distance / 2)

            # Calculate the new point based on the user-given method
            if method == "min":
                new_x = min_latitude
                new_y = min_longitude
            elif method == "max":
                new_x = max_latitude
                new_y = max_longitude
            elif method == "random":
                if random.choice([True, False]):  # Randomly select either the maximum or the minimum point
                    new_x = max_latitude
                    new_y = max_longitude
                else:
                    new_x = min_latitude
                    new_y = min_longitude
            elif method == "between":
                new_x = random.uniform(min_latitude, max_latitude)
                new_y = random.uniform(min_longitude, max_longitude)
            else:
                raise ValueError("Invalid method")

            new_z = pos3d[2]

            frame_pos3d[frame_id] = [new_x, new_y, new_z]

        length_of_frame_data = len(frame_data)
        # print(f"The length of frame_data is: {length_of_frame_data}")

        length_of_frame_pos3d = len(frame_pos3d)
        # print(f"The length of frame_pos3d is: {length_of_frame_pos3d}")

        new_data = {
            'frame_data': frame_data,
            'layout_data': layout_data,
            'frame_pos3d': frame_pos3d,
            'frame_pos3d_color': frame_pos3d_color,
            'collided_frame_id_list': collided_frame_id_list,
            'spike_estimations': spike_estimations,
        }

        new_json_data = json.dumps(new_data, indent=2)

        new_data_name = os.path.join(output_dir, traj_path.split('/')[-1].split('.')[0] + f'_aug_ps_3.json')
        # print(new_data_name)
        with open(new_data_name, 'w') as file:
            file.write(new_json_data)
        
        draw_plot(ori_pos3d, frame_pos3d)
            
def point_dropping_augment(traj_path, output_dir,  modified_ratio):
    frame_pos3d, length_of_frame_data, length_of_frame_pos3d, frame_pos3d_color,\
    frame_data, layout_data, collided_frame_id_list, spike_estimations,label = load_trajectory_data(traj_path)
    
    if collided_frame_id_list is not None:
        ori_pos3d = frame_pos3d.copy()

        # Randomly drop points based on the given probability
        frames_to_keep = [frame_id for frame_id in frame_pos3d.keys() if frame_id not in collided_frame_id_list and random.uniform(0, 1) >  modified_ratio]
        frame_pos3d = {frame_id: frame_pos3d[frame_id] for frame_id in frames_to_keep}
        collided_frame_id_list = list(set(collided_frame_id_list) - set(frames_to_keep))

        length_of_frame_data = len(frame_data)
        # print(f"The length of frame_data is: {length_of_frame_data}")

        length_of_frame_pos3d = len(frame_pos3d)
        # print(f"The length of frame_pos3d is: {length_of_frame_pos3d}")

        new_data = {
            'frame_data': frame_data,
            'layout_data': layout_data,
            'frame_pos3d': frame_pos3d,
            'frame_pos3d_color': frame_pos3d_color,
            'collided_frame_id_list': collided_frame_id_list,
            'spike_estimations': spike_estimations,
            'label': label
        }

        new_json_data = json.dumps(new_data, indent=2)

        new_data_name = os.path.join(output_dir, traj_path.split('/')[-1].split('.')[0] + f'_aug_pd_4.json')
        # print(new_data_name)
        with open(new_data_name, 'w') as file:
            file.write(new_json_data)

        # draw_plot(ori_pos3d, frame_pos3d)

def mirror_augment(traj_path, output_dir):
    frame_pos3d, length_of_frame_data, length_of_frame_pos3d, frame_pos3d_color,\
    frame_data, layout_data, collided_frame_id_list, spike_estimations,label = load_trajectory_data(traj_path)

    if collided_frame_id_list is not None:
        ori_pos3d = frame_pos3d.copy()
        transformed_data = {key: [9 - x, 18 - y, z] for key, [x, y, z] in ori_pos3d.items()}

        new_data = {
            'frame_data': frame_data,
            'layout_data': layout_data,
            'frame_pos3d': transformed_data,
            'frame_pos3d_color': frame_pos3d_color,
            'collided_frame_id_list': collided_frame_id_list,
            'spike_estimations': spike_estimations,
            'label': label
        }

        new_json_data = json.dumps(new_data, indent=2)

        new_data_name = os.path.join(output_dir, traj_path.split('/')[-1].split('.')[0] + f'_aug_mi_5.json')
        # print(new_data_name)
        
        with open(new_data_name, 'w') as file:
            file.write(new_json_data)


        # draw_plot(ori_pos3d, transformed_data)

def shift_left_augment(traj_path, output_dir, shift_length):
    frame_pos3d, length_of_frame_data, length_of_frame_pos3d, frame_pos3d_color,\
    frame_data, layout_data, collided_frame_id_list, spike_estimations, label = load_trajectory_data(traj_path)

    if collided_frame_id_list is not None:
        ori_pos3d = frame_pos3d.copy()
        transformed_data = {key: [x - shift_length, y, z] for key, [x, y, z] in ori_pos3d.items()}

        new_data = {
            'frame_data': frame_data,
            'layout_data': layout_data,
            'frame_pos3d': transformed_data,
            'frame_pos3d_color': frame_pos3d_color,
            'collided_frame_id_list': collided_frame_id_list,
            'spike_estimations': spike_estimations,
            'label': label
        }

        new_json_data = json.dumps(new_data, indent=2)

        new_data_name = os.path.join(output_dir, traj_path.split('/')[-1].split('.')[0] + f'_aug_sl_{shift_length}.json')
        # print(new_data_name)
        with open(new_data_name, 'w') as file:
            file.write(new_json_data)


        # draw_plot(ori_pos3d, transformed_data)


def shift_right_augment(traj_path, output_dir, shift_length):
    frame_pos3d, length_of_frame_data, length_of_frame_pos3d, frame_pos3d_color,\
    frame_data, layout_data, collided_frame_id_list, spike_estimations,label = load_trajectory_data(traj_path)

    if collided_frame_id_list is not None:
        ori_pos3d = frame_pos3d.copy()
        transformed_data = {key: [x + shift_length, y, z] for key, [x, y, z] in ori_pos3d.items()}

        new_data = {
            'frame_data': frame_data,
            'layout_data': layout_data,
            'frame_pos3d': transformed_data,
            'frame_pos3d_color': frame_pos3d_color,
            'collided_frame_id_list': collided_frame_id_list,
            'spike_estimations': spike_estimations,
            'label': label
        }

        new_json_data = json.dumps(new_data, indent=2)

        new_data_name = os.path.join(output_dir, traj_path.split('/')[-1].split('.')[0] + f'_aug_sr_{shift_length}.json')
        # print(new_data_name)
        with open(new_data_name, 'w') as file:
            file.write(new_json_data)

        # draw_plot(ori_pos3d, transformed_data)


def vertical_movement_augment(traj_path, output_dir, shift_length):
    frame_pos3d, length_of_frame_data, length_of_frame_pos3d, frame_pos3d_color,\
    frame_data, layout_data, collided_frame_id_list, spike_estimations,label = load_trajectory_data(traj_path)

    if collided_frame_id_list is not None:
        ori_pos3d = frame_pos3d.copy()
        transformed_data = {key: [x , y, z + shift_length] for key, [x, y, z] in ori_pos3d.items()}

        new_data = {
            'frame_data': frame_data,
            'layout_data': layout_data,
            'frame_pos3d': transformed_data,
            'frame_pos3d_color': frame_pos3d_color,
            'collided_frame_id_list': collided_frame_id_list,
            'spike_estimations': spike_estimations,
            'label': label
        }

        new_json_data = json.dumps(new_data, indent=2)

        new_data_name = os.path.join(output_dir, traj_path.split('/')[-1].split('.')[0] + f'_aug_vm_{shift_length}.json')
        # print(new_data_name)
        with open(new_data_name, 'w') as file:
            file.write(new_json_data)

        # draw_plot(ori_pos3d, transformed_data)

            
def check_max_distance(value):
    float_value = float(value)
    min = 0.01
    max = 0.025

    if not (min<= float_value <= max):
        raise argparse.ArgumentTypeError(f"{value} must be a float between {min} and {max}")

    return float_value

def parse_arguments():
    parser = argparse.ArgumentParser(description="Trajectory Data Augmentation Script")

    # Required arguments
    parser.add_argument("--input", help="Path to the input trajectory JSON file", required=True)
    parser.add_argument("--output", help="Path to the output folder", required=True)

    # Augmentation mode
    parser.add_argument("--mode", help="Augmentation mode ", choices=["on", "in", "ps", "pd", "mi", "sl", "sr"], required=True)

    # Common argument for all modes
    parser.add_argument("--modified_ratio", type=float, help="Ratio of trajectories to be modified", required=True)

    args, paths = parser.parse_known_args()

    # Process paths separately, and append them to args
    args.paths = paths

    if args.mode == "on" or args.mode == "in":
        parser.add_argument("--radius", type=float, help="Radius for on-circle and in-circle augmentation", required=True)
        args = parser.parse_args()
        
    if args.mode == "ps":
        parser.add_argument("--max_distance", type=check_max_distance, help="Maximum distance for point stretching 0.1 ~ 0.025", required=True)
        parser.add_argument("--method", help="Method for selecting the new point", choices=["min", "max", "random", "between"], required=True)

        # Parse the arguments again with the updated parser
        args = parser.parse_args()

    return args

if __name__ == "__main__":
    # args = parse_arguments()
    # if not os.path.exists(args.output):
    #     os.makedirs(args.output)
    
    # if args.mode == "on":
    #     on_circle_augment(args.input, args.output, args.modified_ratio)
    # elif args.mode == "in":
    #     in_circle_augment(args.input, args.output, args.modified_ratio)
    # elif args.mode == "ps":
    #     point_stretching_augment(args.input, args.output, args.modified_ratio, args.max_distance, args.method)
    # elif args.mode == "pd":
    #     point_dropping_augment(args.input, args.output, args.modified_ratio)
    # elif args.mode == "mi":
    #     mirror_augment(args.input, args.output)
    # elif args.mode == "sl":
    #     shift_left_augment(args.input, args.output, 0.1)
    # elif args.mode == "sr":
    #     shift_right_augment(args.input, args.output, 0.1)

    # input = '/home/jason/Documents/Volleyball_TrajText/data/second_augment_dataset/train/trajectory/0205_man/HDR80_A_Live_20230205_163636_000.json'
    # vertical_movement_augment(input, './out', 0.1)
    
    inputfolder = '../data/test_by_label/3/aug'
    outputfolder = '../data/test_by_label/3/aug'
    
    for file in os.listdir(inputfolder):
        if file.endswith('.json'):
            input = os.path.join(inputfolder, file).replace('\\', '/')
            
            mirror_augment(input, outputfolder)
            point_dropping_augment(input, outputfolder, 0.1)
            shift_right_augment(input, outputfolder, 0.3)
            # shift_right_augment(input, outputfolder, 0.4)
            shift_right_augment(input, outputfolder, 0.5)
            # shift_left_augment(input, outputfolder, 0.3)
            shift_left_augment(input, outputfolder, 0.4)
            # vertical_movement_augment(input, outputfolder, 0.1)
            vertical_movement_augment(input, outputfolder, 0.2)
            # vertical_movement_augment(input, outputfolder, 0.3)
            # shift_left_augment(input, outputfolder, 0.6)
    
        
        