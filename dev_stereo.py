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
from dpvo.stream import image_stream, video_stream, image_stream_stereo

from dpvo.plot_utils import plot_trajectory, save_trajectory_tum_format

import time

SKIP = 0

STEREO = True

def show_image(image, t=0):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(t)

@torch.no_grad()
def run(cfg, network, imagedir, calib, stride=1, skip=0, viz=False, timeit=False, save_reconstruction=False):

    slam = None
    queue = Queue(maxsize=8)
    disps = None

    # if os.path.isdir(imagedir):
    #     reader = Process(target=image_stream, args=(queue, imagedir, calib, stride, skip))
    # else:
    #     reader = Process(target=video_stream, args=(queue, imagedir, calib, stride, skip))

    reader = Process(target=image_stream_stereo, args=(queue, imagedir, calib, stride, skip))
    reader.start()


    imagedir_img = imagedir+"/image_left/"
    number_of_images = len([file for file in os.listdir(imagedir_img) if file.endswith('.png')])
        
    print("number_of_images : ", number_of_images)

    import pdb; pdb.set_trace()

    while 1:

        (t, images, intrinsics) = queue.get()

        print("image t : ", t)
        import pdb; pdb.set_trace()

        if t < 0: 
            import pdb; pdb.set_trace()
            break

        if t >= number_of_images-5:
            import pdb; pdb.set_trace()
            break

        # mettre sur cuda 
        images = images.cuda()
        intrinsics = intrinsics.cuda()

        # image = torch.from_numpy(image).permute(2,0,1).cuda()
        # intrinsics = torch.from_numpy(intrinsics).cuda()


        if slam is None:
            slam = DPVO(cfg, network, ht=images.shape[2], wd=images.shape[3], viz=viz, stereo=STEREO)

        #with Timer("SLAM", enabled=timeit):
        slam(t, images, disps, intrinsics)

    print("--- tracking termine ---")

    # for _ in range(12):
    #     slam.update()

    
    time.sleep(30)


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
    parser.add_argument('--imagedir', type=str)
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


        

