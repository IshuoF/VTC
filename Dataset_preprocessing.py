import json
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

def preprocessing(folder_path, max_len):
    data = []
    subfolder_names = os.listdir(folder_path)
    for subfolder_name in subfolder_names:
        subfolder_name = subfolder_name.replace("\\", "").replace("/", "")
        for i in range(2):
            
            if i == 0:
                json_files = os.path.join(folder_path, subfolder_name)
                print(json_files)
            else:
                json_files = os.path.join(folder_path, subfolder_name, "aug")
                print(json_files)
        
            for files in os.listdir(json_files):
                if files.endswith(".json"):
                    
                    file = os.path.join(json_files, files)
                    file = file.replace("\\", "/")
                    # print(file)
                    try:
                        with open(file, 'r') as json_file:
                            json_data = json.load(json_file)
                            
                        if json_data.get("label") is None:
                            print(f'{file} label is None')
                            continue
                        
                        if json_data.get("frame_pos3d") is None:
                            print(f'{file} frame_pos3d is None')
                            continue
                        
                        label = int(json_data["label"])
                        
                        frame_pos3d = json_data.get("frame_pos3d", {})
                        
                        
                        interpolated_traj,time = interpolate_data(frame_pos3d, max_len)
                        velocities = calculate_velocity(interpolated_traj, time, max_len)                     
                        timestamp = np.arange(0, max_len)/60
                        data.append((label, interpolated_traj, velocities, timestamp)) 
                          
                    except Exception as e:
                        print(e)
                        print(f'{file} json file open failed')
                        wait = input("PRESS ENTER TO CONTINUE.")
                        continue
    return data
    
def calculate_velocity(interpolated_traj,time, max_len):
    dt = time/max_len
    # print(dt)
    velocities = np.zeros(max_len)
    velocities[0] = 0

    for i in range(1, len(interpolated_traj)):
        displacement = interpolated_traj[i] - interpolated_traj[i-1]
        
        velocities[i] = np.sqrt(sum(displacement**2))/dt
    # print("calculate_velocity pass")
    return velocities
     
        
def interpolate_data(pos3d, max_len):
    min_frame_id = min(pos3d.keys(), key=(lambda k: int(k)))
    max_frame_id = max(pos3d.keys(), key=(lambda k: int(k)))
    
    # print(f"min_frame_id: {min_frame_id}, max_frame_id: {max_frame_id}, frame_len: {len(pos3d)}")
    
    
    if int(max_frame_id) > max_len:
        start_frame_id = int(max_frame_id) - max_len
        
        if str(start_frame_id) not in pos3d.keys():       
            
            start_frame_id = min(key for key in pos3d.keys() if int(key) >= int(start_frame_id))
            
        # print(f" max frame id > max len  start_frame_id: {start_frame_id}")
    
    if int(max_frame_id) < max_len:
        start_frame_id = min_frame_id
    
        # print(f" max frame id < max len  start_frame_id: {start_frame_id}")
    # wait = input("PRESS ENTER TO CONTINUE.")  
    # print("find start_frame_id pass")
    sorted_keys = sorted(pos3d.keys(), key=lambda x: int(x))
    original_traj = np.array([pos3d[key] for key in sorted_keys])
    # print(f' original len {len(original_traj)}')
    
    
    original_traj_extract = {frame_id: pos3d[str(frame_id)] for frame_id in sorted_keys if int(frame_id) >= int(start_frame_id)}
    original_traj_extract = np.array([original_traj_extract[key] for key in original_traj_extract.keys()])
    # print(f' extract key {original_traj_extract.keys()}')
    # print(f' extract value {original_traj_extract.values()}')
    # print(f' extract len {len(original_traj_extract)}')
    
    # print(original_traj_extract[0][0])
    idx = np.linspace(0, len(original_traj_extract)-1, max_len)
    interpolated_traj = np.zeros((len(idx), 3))
    
    for i in range(3):
        interpolated_traj[:, i] = np.interp(idx, np.arange(len(original_traj_extract)), original_traj_extract[:, i])
    # print("interpolated_traj pass")
    # print(interpolated_traj)
    # print(interpolated_traj.shape)
    times = (int(max_frame_id) - int(start_frame_id))/60
    # print ("interpolate_data pass")
    return interpolated_traj, times
    # x = interpolated_traj[:, 0]
    # y = interpolated_traj[:, 1]
    # z = interpolated_traj[:, 2]
    
    # ox = original_traj[:, 0]
    # oy = original_traj[:, 1]
    # oz = original_traj[:, 2]
    
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.set_xlim3d(left=0, right=9)
    # ax.set_ylim3d(bottom=-1, top=19)
    # ax.set_zlim3d(bottom=0, top=5)
    # draw_volleyball_court(ax)
    # ax.scatter(x, y, z, c='r')
    # ax.scatter(ox, oy, oz, c='b')
    # ax.view_init(elev=5, azim=195)

    # plt.show()
    


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

def save_data(data, file_path):
    print("Data Length :" + str(len(data)))
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
        
def main():
    types = ["train", "test", "valid"]
    max_length = [200,300,400,500,600,700,800]
    aug_or_not = [""]
    # types_num = 0
    # max_length_num = 1
    for aug in range(len(aug_or_not)):
        for types_num in range(len(types)):
            
            folder_path = f'./data/{types[types_num]}_by_label'
            
            for i in range(len(max_length)):
                output_file = f'./dataset/{types[types_num]}/{types[types_num]}_last{max_length[i]}_d.pkl'
                
                    
                print(f"Start preprocessing {types[types_num]} length {max_length[i]} data \n")
                result = preprocessing(folder_path, max_length[i])
                print(f"Preprocessing {types[types_num]} length {max_length[i]} data done \n")
                # print(len(result))
                save_data(result, output_file)
                print("Data saved")
    
    
    Isfine = False
    
    
    for aug in range(len(aug_or_not)):
        for types_num in range(len(types)):
            for i in range(len(max_length)):
            
                output_file = f'./dataset/{types[types_num]}/{types[types_num]}_last{max_length[i]}_d.pkl'
                
                
                print(f"Start checking {types[types_num]} {aug_or_not[aug]}  length {max_length[i]} data \n")
                with open(output_file, 'rb') as f:
                
                    data = pickle.load(f)
                for j in range(len(data)):
                    if len(data[j][1]) != max_length[i]:
                        print(j)
                        print(len(data[j][1]))
                        Isfine = True

                if not Isfine:
                    print("Data check is fine")

if __name__ == "__main__":
    main()