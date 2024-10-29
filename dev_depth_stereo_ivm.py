import cv2
import numpy as np
import glob
import os.path as osp
import os
import torch
from pathlib import Path
from multiprocessing import Process, Queue
from plyfile import PlyElement, PlyData

from dpvo.utils import Timer
from dpvo.dpvo import DPVO
from dpvo.config import cfg
from dpvo.stream import image_stream, video_stream, image_stream_stereo, image_stream_depth_stereo_ivm
from dpvo.plot_utils import plot_trajectory, save_trajectory_tum_format

import time

SKIP = 0

STEREO = True

def show_image(image, t=0):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(t)

@torch.no_grad()
def run(cfg, network, imagedir, calib, stride=0, skip=0, viz=False, timeit=False, save_reconstruction=False):

    slam = None
    queue = Queue(maxsize=8)
    disps = None

    # if os.path.isdir(imagedir):
    #     reader = Process(target=image_stream, args=(queue, imagedir, calib, stride, skip))
    # else:
    #     reader = Process(target=video_stream, args=(queue, imagedir, calib, stride, skip))

    reader = Process(target=image_stream_depth_stereo_ivm, args=(queue, imagedir, calib, stride, skip))
    reader.start()

    while 1:
        (t, images, intrinsics, disps) = queue.get()

        print("image t : ", t)

        if t < 0: break
    
        # if STEREO:
        #     if t > 700: break
        # else:
        if t > 430: break

        
        # mettre sur cuda 
        images = images.cuda()
        intrinsics = intrinsics.cuda()
        disps = disps.cuda()

        # image = torch.from_numpy(image).permute(2,0,1).cuda()
        # intrinsics = torch.from_numpy(intrinsics).cuda()

        if slam is None:
            slam = DPVO(cfg, network, ht=images.shape[2], wd=images.shape[3], viz=viz, stereo=STEREO)

        #with Timer("SLAM", enabled=timeit):
        slam(t, images, disps, intrinsics)

# Extract the translation vectors (first three elements)
    t1 = slam.poses[0, slam.n-1][:3]
    t2 = slam.poses[0, 0][:3]

    print("slam.n ", slam.n)
    print("t1 ", t1)
    print("t2 ", t2)

# Compute the Euclidean distance (norm)
    translation_norm = torch.norm(t1 - t2)
    print("TRANSLATION POSES")
    print(translation_norm)

    points = slam.points_.cpu().numpy()[:slam.m]
    colors = slam.colors_.view(-1, 3).cpu().numpy()[:slam.m]
    points = np.array([(x,y,z,r,g,b) for (x,y,z),(r,g,b) in zip(points, colors)],
                      dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'),('red', 'u1'), ('green', 'u1'),('blue', 'u1')])
    el = PlyElement.describe(points, 'vertex',{'some_property': 'f8'},{'some_property': 'u4'})

    # Filtrer les points qui sont dans la distance maximale
    max_distance = 5.0

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
    ply_data.write("output_test_stereo_depth.ply")

    import pdb; pdb.set_trace

    time.sleep(60)


    # for _ in range(12):
    #     print("GLOBAL IT")
    #     slam.update()




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
    parser.add_argument('--imagedir', type=str, default='/home/smith/test_pipe/')
    parser.add_argument('--calib', type=str)
    parser.add_argument('--stride', type=int, default=1)
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

    #import pdb; pdb.set_trace()

    print("Running with config...")
    print(cfg)

    pred_traj = run(cfg, args.network, args.imagedir, args.calib, args.stride, args.skip, args.viz, args.timeit, args.save_reconstruction)
    name = Path(args.imagedir).stem

    if args.save_reconstruction:
        pred_traj, ply_data = pred_traj
        ply_data.write(f"{name}.ply")
        print(f"Saved {name}.ply")

    if args.save_trajectory:
        Path("saved_trajectories").mkdir(exist_ok=True)
        save_trajectory_tum_format(pred_traj, f"saved_trajectories/{name}.txt")

    if args.plot:
        Path("trajectory_plots").mkdir(exist_ok=True)
        plot_trajectory(pred_traj, title=f"DPVO Trajectory Prediction for {name}", filename=f"trajectory_plots/{name}.pdf")


        

