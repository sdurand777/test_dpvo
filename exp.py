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
from dpvo.stream import image_stream, video_stream
from dpvo.plot_utils import plot_trajectory, save_trajectory_tum_format


from viewerx import Viewer

import time

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


class ViewerX:
    def __init__(self):
# data for viewer
        self.intrinsics_ = torch.zeros(1, 4, dtype=torch.float32, device="cuda")
        self.image_ = torch.zeros(528, 960, 3, dtype=torch.uint8, device="cpu") 
        self.poses_ =  torch.zeros(2048, 7, dtype=torch.float, device="cuda")
        # initialize poses to identity matrix
        self.poses_[:,6] = 1.0
        self.points_ = torch.zeros(96*100, 3, dtype=torch.float, device="cuda")
        self.colors_ = torch.zeros(100, 96, 3, dtype=torch.uint8, device="cuda")
        # define Viewer to render SLAM
        self.viewer = Viewer(
            self.image_,
            self.poses_,
            self.points_,
            self.colors_,
            self.intrinsics_)
        #time.sleep(1)

    def __call__(self, image, poses):

        if self.viewer is not None:
            print("viewer update : ")
            #image = torch.rand(480,640,3)
            print("image.shape : ", image.shape)
            self.viewer.update_image(image)

            # update poses
            self.poses_.copy_(poses)

            print("poses.shape : ", poses.shape)
            print("poses[:10] : ", poses[:10])

    # @property
    # def poses(self):
    #     return self.poses_.view(1, 2048, 7)




@torch.no_grad()
def run(cfg, network, imagedir, calib, stride=1, skip=0, viz=False, timeit=False, save_reconstruction=False):

    slam = None
    queue = Queue(maxsize=8)

    if os.path.isdir(imagedir):
        reader = Process(target=image_stream, args=(queue, imagedir, calib, stride, skip))
    else:
        reader = Process(target=video_stream, args=(queue, imagedir, calib, stride, skip))

    reader.start()

    viewer = None
    # launch viewer    
    if viewer is None:
        viewer = ViewerX()

    while 1:
        (t, image, intrinsics) = queue.get()
        if t < 0: break

        image = torch.from_numpy(image).permute(2,0,1).cuda()
        intrinsics = torch.from_numpy(intrinsics).cuda()

        if slam is None:
            slam = DPVO(cfg, network, ht=image.shape[1], wd=image.shape[2], viz=viz)

        image = image.cuda()

        image_clone = image.clone()

        viewer(image_clone, slam.poses_)

        #time.sleep(10)

        intrinsics = intrinsics.cuda()

        #with Timer("SLAM", enabled=timeit):
        slam(t, image, intrinsics)

    for _ in range(12):
        slam.update()

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
    parser.add_argument('--network', type=str, default='dpvo.pth')
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


        

