import torch
import numpy as np
import torch.nn.functional as F

from . import fastba
from . import altcorr
from . import lietorch
from .lietorch import SE3

from .net import VONet
from .utils import *
from . import projective_ops as pops

DEBUG = False

# import fastba
# import altcorr
# import lietorch
# from lietorch import SE3
#
# from net import VONet
# from utils import *
# import projective_ops as pops




autocast = torch.cuda.amp.autocast
Id = SE3.Identity(1, device="cuda")



# class Patchifier_Wrapper:
#     def __init__(self, method):
#         self.method = method
#
#     def __call__(self, *args, **kwargs):
#         #print(f"Calling {self.method.__name__} with args: {args}, kwargs: {kwargs}")
#         result = self.method(*args, **kwargs)
#         #print(f"Result of {self.method.__name__}: {result}")
#         return result

# class Update_Wrapper:
#     def __init__(self, method):
#         self.method = method
#
#     def __call__(self, *args, **kwargs):
#         #print(f"Calling {self.method.__name__} with args: {args}, kwargs: {kwargs}")
#         result = self.method(*args, **kwargs)
#         #print(f"Result of {self.method.__name__}: {result}")
#         return result




class DPVO:
    def __init__(self, cfg, network, ht=480, wd=640, viz=False, stereo=False):
        self.cfg = cfg
        self.load_weights(network)
        self.is_initialized = False
        self.enable_timing = False
        
        self.n = 0      # number of frames
        self.m = 0      # number of patches
        self.M = self.cfg.PATCHES_PER_FRAME
        self.N = self.cfg.BUFFER_SIZE

        self.ht = ht    # image height
        self.wd = wd    # image width

        DIM = self.DIM
        RES = self.RES

        ### state attributes ###
        self.tlist = []
        self.counter = 0

        # dummy image for visualization
        self.image_ = torch.zeros(self.ht, self.wd, 3, dtype=torch.uint8, device="cpu")

        self.tstamps_ = torch.zeros(self.N, dtype=torch.long, device="cuda")
        self.poses_ = torch.zeros(self.N, 7, dtype=torch.float, device="cuda")
        self.patches_ = torch.zeros(self.N, self.M, 3, self.P, self.P, dtype=torch.float, device="cuda")
        self.intrinsics_ = torch.zeros(self.N, 4, dtype=torch.float, device="cuda")

        self.points_ = torch.zeros(self.N * self.M, 3, dtype=torch.float, device="cuda")
        self.colors_ = torch.zeros(self.N, self.M, 3, dtype=torch.uint8, device="cuda")

        self.index_ = torch.zeros(self.N, self.M, dtype=torch.long, device="cuda")
        self.index_map_ = torch.zeros(self.N, dtype=torch.long, device="cuda")

        ### network attributes ###
        self.mem = 32

        if self.cfg.MIXED_PRECISION:
            self.kwargs = kwargs = {"device": "cuda", "dtype": torch.half}
        else:
            self.kwargs = kwargs = {"device": "cuda", "dtype": torch.float}
        
        self.imap_ = torch.zeros(self.mem, self.M, DIM, **kwargs)
        self.gmap_ = torch.zeros(self.mem, self.M, 128, self.P, self.P, **kwargs)

        ht = ht // RES
        wd = wd // RES

        self.fmap1_ = torch.zeros(1, self.mem, 128, ht // 1, wd // 1, **kwargs)
        self.fmap2_ = torch.zeros(1, self.mem, 128, ht // 4, wd // 4, **kwargs)

        # for right frames
        self.fmap1_right = torch.zeros(1, self.mem, 128, ht // 1, wd // 1, **kwargs)
        self.fmap2_right = torch.zeros(1, self.mem, 128, ht // 4, wd // 4, **kwargs)


        # feature pyramid
        self.pyramid = (self.fmap1_, self.fmap2_)

        # pyramid right
        self.pyramid_right = (self.fmap1_right, self.fmap2_right)

        self.net = torch.zeros(1, 0, DIM, **kwargs)
        self.ii = torch.as_tensor([], dtype=torch.long, device="cuda")
        self.jj = torch.as_tensor([], dtype=torch.long, device="cuda")
        self.kk = torch.as_tensor([], dtype=torch.long, device="cuda")
        
        # initialize poses to identity matrix
        self.poses_[:,6] = 1.0

        # store relative poses for removed frames
        self.delta = {}

        self.stereo = stereo

        self.viewer = None
        if viz:
            self.start_viewer()

    def load_weights(self, network):
        # load network from checkpoint file
        if isinstance(network, str):
            from collections import OrderedDict
            state_dict = torch.load(network)
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if "update.lmbda" not in k:
                    new_state_dict[k.replace('module.', '')] = v

            self.network = VONet()
            self.network.load_state_dict(new_state_dict)
            
            # Vérifier les clés du state_dict
            print("Clés du state_dict chargées :")
            for k in new_state_dict.keys():
                print(k)

            if DEBUG: import pdb; pdb.set_trace()

            #self.network.patchify.forward = Patchifier_Wrapper(self.network.patchify.forward)
            #self.network.update.forward = Update_Wrapper(self.network.update.forward)

        else:
            self.network = network

        # steal network attributes
        self.DIM = self.network.DIM
        self.RES = self.network.RES
        self.P = self.network.P

        self.network.cuda()
        self.network.eval()

        # if self.cfg.MIXED_PRECISION:
        #     self.network.half()


    def start_viewer(self):
        #from dpviewer import Viewer
        from viewerdpvo import Viewer
        #from viewerx import Viewer

        intrinsics_ = torch.zeros(1, 4, dtype=torch.float32, device="cuda")

        self.viewer = Viewer(
            self.image_,
            self.poses_,
            self.points_,
            self.colors_,
            intrinsics_)

    @property
    def poses(self):
        return self.poses_.view(1, self.N, 7)

    @property
    def patches(self):
        return self.patches_.view(1, self.N*self.M, 3, 3, 3)

    @property
    def intrinsics(self):
        return self.intrinsics_.view(1, self.N, 4)

    @property
    def ix(self):
        return self.index_.view(-1)

    @property
    def imap(self):
        return self.imap_.view(1, self.mem * self.M, self.DIM)

    @property
    def gmap(self):
        return self.gmap_.view(1, self.mem * self.M, 128, 3, 3)

    def get_pose(self, t):
        if t in self.traj:
            return SE3(self.traj[t])

        t0, dP = self.delta[t]
        return dP * self.get_pose(t0)

    def terminate(self):
        """ interpolate missing poses """
        self.traj = {}
        for i in range(self.n):
            self.traj[self.tstamps_[i].item()] = self.poses_[i]

        poses = [self.get_pose(t) for t in range(self.counter)]
        poses = lietorch.stack(poses, dim=0)
        poses = poses.inv().data.cpu().numpy()
        tstamps = np.array(self.tlist, dtype=np.float)

        if self.viewer is not None:
            self.viewer.join()

        return poses, tstamps


    def terminate_dev(self):
        # if self.viewer is not None:
        #     self.viewer.join()
        # Nettoyer les tensors sur le GPU pour libérer la mémoire
        try:
            torch.cuda.empty_cache()
            del self.image_
            del self.tstamps_
            del self.poses_
            del self.patches_
            del self.intrinsics_
            del self.points_
            del self.colors_
            del self.index_
            del self.index_map_
            del self.imap_
            del self.gmap_
            del self.fmap1_
            del self.fmap2_
            del self.net
            del self.ii
            del self.jj
            del self.kk
        except Exception as e:
            print(f"Erreur lors du nettoyage des tensors GPU : {e}")

        print("slam terminate dev")

        return 0


    def corr(self, coords, indicies=None, stereo=False):
        """ local correlation volume """

        #import pdb; pdb.set_trace()

        if stereo:
            # on recupere les indices
            ii, jj, kk = indicies if indicies is not None else (self.ii, self.jj, self.kk)
            ii1 = ii % (self.mem)
            jj1 = jj % (self.mem)
            kk1 = kk % (self.M * self.mem)

            #import pdb; pdb.set_trace()

            corr1 = altcorr.corr_stereo(
                                        self.gmap, 
                                        self.pyramid[0], 
                                        self.pyramid_right[0], 
                                        coords / 1, 
                                        ii1, 
                                        jj1, 
                                        kk1, 
                                        3)
            corr1 = corr1[0]
            # level 2 divise par 4
            # corr2 shape [1, 96, 7 ,7, 3, 3]
            corr2 = altcorr.corr_stereo(
                                        self.gmap, 
                                        self.pyramid[1], 
                                        self.pyramid_right[1], 
                                        coords / 4, 
                                        ii1, 
                                        jj1, 
                                        kk1, 
                                        3)
            corr2 = corr2[0]

            #corr2 = altcorr.corr(self.gmap, self.pyramid[1], coords / 4, kk1, jj1, 3)
        else:
            # on recupere les indices
            ii, jj = indicies if indicies is not None else (self.kk, self.jj)
            # on utilise la memoire circulaire self.mem pour recuperer les bons indices sur les 32 stockes
            # indices des patches
            ii1 = ii % (self.M * self.mem)
            # indices de la frame sur laquelle reprojete les patches
            jj1 = jj % (self.mem)

#             # recuperer les data entrees pour rejouer correlation
#             if len(ii) == 6144:
#                 my_values = {
#                     'gmap': self.gmap.to('cpu'),
#                     'pyramid': self.pyramid[0].to('cpu'),
#                     'coords': coords.to('cpu'),
#                     'kk1': ii1.to('cpu'),
#                     'jj1': jj1.to('cpu'),
#                 }
#
#                 class Container(torch.nn.Module):
#                     def __init__(self, my_values):
#                         super().__init__()
#                         for key in my_values:
#                             setattr(self, key, my_values[key])
#
# # Save arbitrary values supported by TorchScript
# # https://pytorch.org/docs/master/jit.html#supported-type
#                 container = torch.jit.script(Container(my_values))
#                 container.save("container_corr_dpvo.pt")
#
#                 import pdb; pdb.set_trace()

            # on recupere directement les correlations dans la pyramide
            # level 1 pleine taille 132 240
            # corr1 shape [1, 96, 7 ,7, 3, 3]
            corr1 = altcorr.corr(self.gmap, self.pyramid[0], coords / 1, ii1, jj1, 3)
            # level 2 divise par 4
            # corr2 shape [1, 96, 7 ,7, 3, 3]
            corr2 = altcorr.corr(self.gmap, self.pyramid[1], coords / 4, ii1, jj1, 3)
        # on stack tout
        return torch.stack([corr1, corr2], -1).view(1, len(ii), -1)



    def reproject(self, indicies=None, stereo=False):
        """ reproject patch k from i -> j """

        # recuperation des indices si None
        (ii, jj, kk) = indicies if indicies is not None else (self.ii, self.jj, self.kk)
        # on a les coords de patches a reprojeter dans self.patches
        # import pdb; pdb.set_trace()
        # print("stereo ", stereo)
        coords = pops.transform(SE3(self.poses), self.patches, self.intrinsics, ii, jj, kk, stereo=stereo)
        # reshape coords
        return coords.permute(0, 1, 4, 2, 3).contiguous()


    # fonction pour ajouter les factors
    def append_factors(self, ii, jj):
        
        #import pdb; pdb.set_trace()

        # ii ici indices des patches avec les fonctions edges_for et edges_back
        # jj indices des frames

        # incides des frames cibles
        self.jj = torch.cat([self.jj, jj])
        # indices patches 
        self.kk = torch.cat([self.kk, ii])
        # avec self.ix recuperation des indices des frames sources a partir des patches
        self.ii = torch.cat([self.ii, self.ix[ii]])

        # update net latent state for the conv gru with the new edges
        net = torch.zeros(1, len(ii), self.DIM, **self.kwargs)

        self.net = torch.cat([self.net, net], dim=1)



    def remove_factors(self, m):
        self.ii = self.ii[~m]
        self.jj = self.jj[~m]
        self.kk = self.kk[~m]
        self.net = self.net[:,~m]



    def motion_probe(self, stereo=False):
        """ kinda hacky way to ensure enough motion for initialization """

        #import pdb; pdb.set_trace()

        # indice des patches
        if stereo:
            # on recuperer les indices des patches a -2
            kk = torch.arange(self.m-self.M*2, self.m-self.M, device="cuda")
            # incide noeud ou image actuel
            jj = self.n * torch.ones_like(kk)
            # indice noeuds des patches qui vont etre projete
            ii = self.ix[kk]
        else:
            kk = torch.arange(self.m-self.M, self.m, device="cuda")
            # incide noeud ou image actuel
            jj = self.n * torch.ones_like(kk)
            # indice noeuds des patches qui vont etre projete
            ii = self.ix[kk]

        # latent state for raft
        net = torch.zeros(1, len(ii), self.DIM, **self.kwargs)

        # get coords reprojection of patches kk of  ii to jj 
        coords = self.reproject(indicies=(ii, jj, kk))

        with autocast(enabled=self.cfg.MIXED_PRECISION):
            # compute correlation
            corr = self.corr(coords, indicies=(kk, jj))
            # get context features for the patches kk
            ctx = self.imap[:,kk % (self.M * self.mem)]
            # apply raft to the patches
            net, (delta, weight, _) = \
                self.network.update(net, ctx, corr, None, ii, jj, kk)

        return torch.quantile(delta.norm(dim=-1).float(), 0.5)


    # mesure amplitude du mouvement entre i et j
    def motionmag(self, i, j):
        k = (self.ii == i) & (self.jj == j)
        ii = self.ii[k]
        jj = self.jj[k]
        kk = self.kk[k]

        flow = pops.flow_mag(SE3(self.poses), self.patches, self.intrinsics, ii, jj, kk, beta=0.5)
        return flow.mean().item()


    def keyframe(self):

        #import pdb; pdb.set_trace()

        # KEYFRAME INDEX set to 4 on regarde les frames proches
        i = self.n - self.cfg.KEYFRAME_INDEX - 1 # KF - 5
        j = self.n - self.cfg.KEYFRAME_INDEX + 1 # KF - 3
        m = self.motionmag(i, j) + self.motionmag(j, i)
 

        # KEYFRAME THRESH set to 15 en pixel pour le deplacement on divise m par deux pour la moyenne 
        # si le deplacement est inferieur au seuil alors on elimine KF - 4 trop redondante
        if m / 2 < self.cfg.KEYFRAME_THRESH:
            # recuperation indice frame k -3 par rapport a la KF actuelle
            k = self.n - self.cfg.KEYFRAME_INDEX # KF - 4
            # pose k-1
            t0 = self.tstamps_[k-1].item() # KF - 5
            # pose k
            t1 = self.tstamps_[k].item() # KF - 4

            # deplacement relatif
            dP = SE3(self.poses_[k]) * SE3(self.poses_[k-1]).inv()

            # save delta pour avoir la trajectoir complete a la fin
            self.delta[t1] = (t0, dP)

            # on supprime la frame k du graph
            to_remove = (self.ii == k) | (self.jj == k)
            self.remove_factors(to_remove)

            self.kk[self.ii > k] -= self.M
            self.ii[self.ii > k] -= 1
            self.jj[self.jj > k] -= 1

            # gestion des indices
            for i in range(k, self.n-1):
                self.tstamps_[i] = self.tstamps_[i+1]
                self.colors_[i] = self.colors_[i+1]
                self.poses_[i] = self.poses_[i+1]
                self.patches_[i] = self.patches_[i+1]
                self.intrinsics_[i] = self.intrinsics_[i+1]

                self.imap_[i%self.mem] = self.imap_[(i+1) % self.mem]
                self.gmap_[i%self.mem] = self.gmap_[(i+1) % self.mem]
                self.fmap1_[0,i%self.mem] = self.fmap1_[0,(i+1)%self.mem]
                self.fmap2_[0,i%self.mem] = self.fmap2_[0,(i+1)%self.mem]

            self.n -= 1
            self.m-= self.M

        # on elimine les edges trop vieux au de la de REMOVAL_WINDOW set a 22
        to_remove = self.ix[self.kk] < self.n - self.cfg.REMOVAL_WINDOW
        self.remove_factors(to_remove)


    # main update fonction reprojection to index corr to raft to compute targets then BA to update pose and disp values based on targets
    def update(self):

        #import pdb; pdb.set_trace()

        # reprojection RAFT tp update flow weights so target for BA
        # ce with timer just pour encapsuler le temps pour cette partie du code
        with Timer("other", enabled=self.enable_timing):
            # calculer toutes les reprojection pour le graph
            # coords shape [1, 6144, 2, 3, 3]
            #import pdb; pdb.set_trace()
            #coords = self.reproject(stereo=self.stereo)
            coords = self.reproject()

            with autocast(enabled=True):
                # gestion du cas stereo
                #corr = self.corr(coords, stereo=self.stereo)
                corr = self.corr(coords)

                # uniquement context de gauche pour rappel
                ctx = self.imap[:,self.kk % (self.M * self.mem)]
                self.net, (delta, weight, _) = \
                    self.network.update(self.net, ctx, corr, None, self.ii, self.jj, self.kk)

            lmbda = torch.as_tensor([1e-4], device="cuda")
            weight = weight.float()
            # on recupere le centre de coords et on applique le terme correctif delta
            target = coords[...,self.P//2,self.P//2] + delta.float()

        # BA to update poses and depth values
        with Timer("BA", enabled=self.enable_timing):
            # t0 minimale OPTIMIZATION WINDOW set to 10
            t0 = self.n - self.cfg.OPTIMIZATION_WINDOW if self.is_initialized else 1
            # t0 min 1 on optimise pas la frame 0 sa pose reste Identite definitivement
            t0 = max(t0, 1)

            try:
                #import pdb; pdb.set_trace()

                # recuperation des data pour faire tourner directement en cuda
#                 my_values = {
#                     'poses': self.poses.to('cpu'),
#                     'patches': self.patches.to('cpu'),
#                     'intrinsics': self.intrinsics[0].to('cpu'),
#                     'target': target.to('cpu'),
#                     'weight': weight.to('cpu'),
#                     'lmbda': lmbda.to('cpu'),
#                     't0': t0,
#                     'n': self.n,
#                     'ii': self.ii.to('cpu'),
#                     'jj': self.jj.to('cpu'),
#                     'kk': self.kk.to('cpu'),
#                 }
#
#                 class Container(torch.nn.Module):
#                     def __init__(self, my_values):
#                         super().__init__()
#                         for key in my_values:
#                             setattr(self, key, my_values[key])
#
# # Save arbitrary values supported by TorchScript
# # https://pytorch.org/docs/master/jit.html#supported-type
#                 container = torch.jit.script(Container(my_values))
#                 container.save("container.pt")

                #import pdb; pdb.set_trace()

                # fastba.BA(self.poses, self.patches, self.intrinsics, 
                #     target, weight, lmbda, self.ii, self.jj, self.kk, t0, self.n, 2, self.stereo)
                fastba.BA(self.poses, self.patches, self.intrinsics, 
                    target, weight, lmbda, self.ii, self.jj, self.kk, t0, self.n, 2)

                #import pdb; pdb.set_trace()

            except:
                print("Warning BA failed...")
            
            # extraire points 3D 
            # points shape [1,768,3,3,4]
            points = pops.point_cloud(SE3(self.poses), self.patches[:, :self.m], self.intrinsics, self.ix[:self.m])
            # on recupere uniquement le centre du patch en divisant par d pour recuperer le point 3D
            # points [768,3]
            points = (points[...,1,1,:3] / points[...,1,1,3:]).reshape(-1, 3)
            self.points_[:len(points)] = points[:]
            #import pdb; pdb.set_trace()

                
    def __edges_all(self):
        return flatmeshgrid(
            torch.arange(0, self.m, device="cuda"),
            torch.arange(0, self.n, device="cuda"), indexing='ij')

    # edges forward
    def __edges_forw(self):

        #import pdb; pdb.set_trace()

        # set to 13
        r=self.cfg.PATCH_LIFETIME
        # self.M donc en nombre de patch
        t0 = self.M * max((self.n - r), 0)
        t1 = self.M * max((self.n - 1), 0)
        return flatmeshgrid(
            torch.arange(t0, t1, device="cuda"),
            torch.arange(self.n-1, self.n, device="cuda"), indexing='ij')

    def __edges_back(self):

        #import pdb; pdb.set_trace()

        r=self.cfg.PATCH_LIFETIME
        t0 = self.M * max((self.n - 1), 0)
        t1 = self.M * max((self.n - 0), 0)
        return flatmeshgrid(torch.arange(t0, t1, device="cuda"),
            torch.arange(max(self.n-r, 0), self.n, device="cuda"), indexing='ij')




    def __call__(self, tstamp, image, disp=None, intrinsics=None):

        """ track new frame """
        if DEBUG: import pdb; pdb.set_trace()

        #import pdb; pdb.set_trace()

        # self.N buffer et self.n+1 current frame id
        if (self.n+1) >= self.N:
            raise Exception(f'The buffer size is too small. You can increase it using "--buffer {self.N*2}"')

        # on recupere que la gauche pour tester


        if self.viewer is not None:
            if self.stereo:
                self.viewer.update_image(image[0])
            else:
                self.viewer.update_image(image)


        # post traitement image
        if self.stereo:
            # normalisation image avant patchifier
            image = 2 * (image[None,None] / 255.0) - 0.5
        else:
            # normalisation image avant patchifier
            image = 2 * (image[None] / 255.0) - 0.5

        # recuperer disp pour patchify
        with autocast(enabled=self.cfg.MIXED_PRECISION):

            if self.stereo:
                # recuperer info image gauche
                #import pdb; pdb.set_trace()
                fmap, gmap, imap, patches, _, clr, fmap_right = \
                        self.network.patchify(image,
                                            disp,
                                            patches_per_image=self.cfg.PATCHES_PER_FRAME, 
                                            gradient_bias=self.cfg.GRADIENT_BIAS, 
                                            return_color=True,
                                            stereo=self.stereo)

            else:
                # info image system monoculaire
                fmap, gmap, imap, patches, _, clr = \
                    self.network.patchify(  image,
                                            disp,
                                            patches_per_image=self.cfg.PATCHES_PER_FRAME, 
                                            gradient_bias=self.cfg.GRADIENT_BIAS, 
                                            return_color=True)




        """ patchifier done """
        if DEBUG: import pdb; pdb.set_trace()


        ### update state attributes ###
        self.tlist.append(tstamp)
        self.tstamps_[self.n] = self.counter
        self.intrinsics_[self.n] = intrinsics / self.RES

        # color info for visualization
        clr = (clr[0,:,[2,1,0]] + 0.5) * (255.0 / 2)
        self.colors_[self.n] = clr.to(torch.uint8)

        # index nombre de frames
        self.index_[self.n + 1] = self.n + 1
        # index nombre de patches
        self.index_map_[self.n + 1] = self.m + self.M

        if self.n > 1:
            if self.cfg.MOTION_MODEL == 'DAMPED_LINEAR':
                P1 = SE3(self.poses_[self.n-1])
                P2 = SE3(self.poses_[self.n-2])
                
                # calcul dans la lie algebra du deplacement relatif
                xi = self.cfg.MOTION_DAMPING * (P1 * P2.inv()).log()
                # retractation sur SE3  sur la frame n - 1
                tvec_qvec = (SE3.exp(xi) * P1).data
                # initialise pose actuelle avec tvec_qvec
                self.poses_[self.n] = tvec_qvec
            else:
                tvec_qvec = self.poses[self.n-1]
                self.poses_[self.n] = tvec_qvec

        # Utiliser la stereo pour initialiser la depth mieux et des le debut

        # TODO better depth initialization
        patches[:,:,2] = torch.rand_like(patches[:,:,2,0,0,None,None])
        if self.is_initialized:
            s = torch.median(self.patches_[self.n-3:self.n,:,2])
            patches[:,:,2] = s

        self.patches_[self.n] = patches

        ### update network attributes ###
        # on utilise self.mem pour reecrir par dessus les donnees au dela de self.mem
        self.imap_[self.n % self.mem] = imap.squeeze()
        self.gmap_[self.n % self.mem] = gmap.squeeze()

        # pyramid gauche
        self.fmap1_[:, self.n % self.mem] = F.avg_pool2d(fmap[0], 1, 1)
        self.fmap2_[:, self.n % self.mem] = F.avg_pool2d(fmap[0], 4, 4)

        # pyramid droite
        if self.stereo:
            self.fmap1_right[:, self.n % self.mem] = F.avg_pool2d(fmap_right[0], 1, 1)
            self.fmap2_right[:, self.n % self.mem] = F.avg_pool2d(fmap_right[0], 4, 4)

        self.counter += 1        
        if self.n > 0 and not self.is_initialized:
            if self.motion_probe() < 2.0:
                # on garde les poses pour reconstituer toute la trajectoire
                self.delta[self.counter - 1] = (self.counter - 2, Id[0])
                return

        self.n += 1
        self.m += self.M

        #import pdb; pdb.set_trace()

        # relative pose
        # edges patches passes vers frame actuel
        self.append_factors(*self.__edges_forw())

        # edges patches actuels vers frames passes et actuels
        self.append_factors(*self.__edges_back())


        # initialisation
        if self.n == 8 and not self.is_initialized:
            self.is_initialized = True            

            #import pdb; pdb.set_trace()
            # 12 iterations update
            for itr in range(12):
                self.update()
        
        # classic update
        elif self.is_initialized:
            self.update()
            self.keyframe()

                





