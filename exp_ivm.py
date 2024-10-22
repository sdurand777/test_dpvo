import cv2
import numpy as np
import glob
import os.path as osp
import os
import torch
from pathlib import Path
from multiprocessing import Process, Queue

from plyfile import PlyElement, PlyData

import open3d as o3d

from dpvo.utils import Timer
from dpvo.dpvo import DPVO
from dpvo.config import cfg
from dpvo.stream import image_stream, video_stream, image_ivm_300_stream
from dpvo.plot_utils import plot_trajectory, save_trajectory_tum_format

import time

DEBUG = True

# import torch
# import numpy as np
# import torch.nn.functional as F
#
# from dpvo import fastba
# from dpvo import altcorr
# from dpvo import lietorch
# from dpvo.lietorch import SE3
#
# from dpvo.net import VONet
# from dpvo.utils import *
# from dpvo import projective_ops as pops
#
# autocast = torch.cuda.amp.autocast
# Id = SE3.Identity(1, device="cuda")



SKIP = 0

def show_image(image, t=0):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(t)


@torch.no_grad()
def run(cfg, network, imagedir, calib, stride=1, skip=0, viz=False, timeit=False, save_reconstruction=False):

    slam = None
    queue = Queue(maxsize=8)

    if os.path.isdir(imagedir):
        #reader = Process(target=image_stream, args=(queue, imagedir, calib, stride, skip))
        reader = Process(target=image_ivm_300_stream, args=(queue, imagedir, stride, skip))
    else:
        reader = Process(target=video_stream, args=(queue, imagedir, calib, stride, skip))

    reader.start()

    # viewer = None
    # # launch viewer    
    # if viewer is None:
    #     viewer = ViewerX()

    imagedir_img = imagedir+"/left/"
    number_of_images = len([file for file in os.listdir(imagedir_img) if file.endswith('.JPG')])
        
    print("number_of_images : ", number_of_images)

    while 1:
        #try:
        (t, image, disp_sens, intrinsics) = queue.get(timeout=10)
        # except queue.empty:
        #     print("Empty queue")
        # except FileNotFoundError as e:
        #     print("FileNotFoundError")
        #if t < 0: break
        if t < 0: 
            break

        # if t >= 100:
        #     break

        if t >= number_of_images-10:
           break
        #while 1: 
            #     print("wait ...")
            #     time.sleep(1)

        #image = torch.from_numpy(image).permute(2,0,1).cuda()

        print("image.shape : ", image.shape)
        print("image \n : ", image)
        image = image.to(torch.uint8).cuda()

        #intrinsics = torch.from_numpy(intrinsics).cuda()
        intrinsics = intrinsics.cuda()

        if slam is None:
            slam = DPVO(cfg, network, ht=image.shape[1], wd=image.shape[2], viz=viz)

        image = image.cuda()

        image_clone = image.clone()

        #viewer(image_clone, slam.poses_)

        #time.sleep(10)

        intrinsics = intrinsics.cuda()

        #with Timer("SLAM", enabled=timeit):
        slam(t, image, disp_sens, intrinsics)

    points = slam.points_.cpu().numpy()[:slam.m]
    colors = slam.colors_.view(-1, 3).cpu().numpy()[:slam.m]
    points = np.array([(x,y,z,r,g,b) for (x,y,z),(r,g,b) in zip(points, colors)],
                      dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'),('red', 'u1'), ('green', 'u1'),('blue', 'u1')])
    el = PlyElement.describe(points, 'vertex',{'some_property': 'f8'},{'some_property': 'u4'})

    # Filtrer les points qui sont dans la distance maximale
    max_distance = 3.0

    # Calculer la distance euclidienne des points par rapport à l'origine
    x = points['x']
    y = points['y']
    z = points['z']
    
    distances = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    
    # Filtrer les points qui sont dans la distance maximale
    filtered_points = points[distances <= max_distance]

    el = PlyElement.describe(filtered_points, 'vertex', 
                             {'x': 'f4', 'y': 'f4', 'z': 'f4', 'red': 'u1', 'green': 'u1', 'blue': 'u1'})

    ply_data = PlyData([el], text=True)
    ply_data.write("output_test.ply")

# Lire le fichier PLY
    ply_data = PlyData.read("output_test.ply")

# Compter le nombre de points
    num_points = len(ply_data['vertex'].data)
    print(f"Nombre de points dans le fichier output PLY : {num_points}")

    if DEBUG: import pdb; pdb.set_trace()


    ply_data = PlyData([el], text=True)
    ply_data.write("pointcloud_test.ply")

# Lire le fichier PLY
    ply_data = PlyData.read("pointcloud_test.ply")

# Compter le nombre de points
    num_points = len(ply_data['vertex'].data)
    print(f"Nombre de points dans le fichier PLY : {num_points}")
    
# Charger le fichier PLY
    ply_file_path = "pointcloud_test.ply"
    pcd = o3d.io.read_point_cloud(ply_file_path)

# Visualiser le point cloud
    o3d.visualization.draw_geometries([pcd])

    print("ply saved ...")
    time.sleep(10)



    print("---- END TRACKING")

    # for _ in range(12):
    #     slam.update()
    #
    # print("---- END GLOBAL BA")

    # while True:
    #     print("wait ...")
    #     time.sleep(10)


    reader.join()

    if save_reconstruction:
        points = slam.points_.cpu().numpy()[:slam.m]
        colors = slam.colors_.view(-1, 3).cpu().numpy()[:slam.m]
        points = np.array([(x,y,z,r,g,b) for (x,y,z),(r,g,b) in zip(points, colors)],
                          dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'),('red', 'u1'), ('green', 'u1'),('blue', 'u1')])
        el = PlyElement.describe(points, 'vertex',{'some_property': 'f8'},{'some_property': 'u4'})
        return slam.terminate(), PlyData([el], text=True)

    return slam.terminate()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str, default='/home/smith/test_dpvo/dpvo.pth')
    parser.add_argument('--imagedir', type=str, default='/home/smith/test_pipe/' )
    parser.add_argument('--calib', type=str)
    parser.add_argument('--stride', type=int, default=2)
    parser.add_argument('--skip', type=int, default=0)
    parser.add_argument('--buffer', type=int, default=2048)
    parser.add_argument('--config', default="config/default.yaml")
    parser.add_argument('--timeit', action='store_true')
    parser.add_argument('--viz', action="store_true")
    parser.add_argument('--plot', action="store_true")
    parser.add_argument('--save_reconstruction', action="store_true")
    parser.add_argument('--save_trajectory', action="store_true")
    args = parser.parse_args()

    cfg.merge_from_file(args.config)
    cfg.BUFFER_SIZE = args.buffer

    print("Running with config...")
    print(cfg)

    pred_traj = run(cfg, args.network, args.imagedir, args.calib, args.stride, args.skip, args.viz, args.timeit, args.save_reconstruction)
    name = Path(args.imagedir).stem

    # if args.save_reconstruction:
    #     pred_traj, ply_data = pred_traj
    #     ply_data.write(f"{name}.ply")
    #     print(f"Saved {name}.ply")
    #
    # if args.save_trajectory:
    #     Path("saved_trajectories").mkdir(exist_ok=True)
    #     save_trajectory_tum_format(pred_traj, f"saved_trajectories/{name}.txt")
    #
    # if args.plot:
    #     Path("trajectory_plots").mkdir(exist_ok=True)
    #     plot_trajectory(pred_traj, title=f"DPVO Trajectory Prediction for {name}", filename=f"trajectory_plots/{name}.pdf")


        

