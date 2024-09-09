


from pycallgraph2 import PyCallGraph
from pycallgraph2.output import GraphvizOutput
from pycallgraph2 import Config
from pycallgraph2 import GlobbingFilter

import sys
import os

import cv2
import numpy as np
import glob
import os.path as osp
import os
import torch
from pathlib import Path
from multiprocessing import Process, Queue
from queue import Empty  # Import correct d'Empty
from plyfile import PlyElement, PlyData

from dpvo.utils import Timer
from dpvo.dpvo import DPVO
from dpvo.config import cfg
from dpvo.stream import image_stream, video_stream
from dpvo.plot_utils import plot_trajectory, save_trajectory_tum_format

SKIP = 0
DEBUG = False


def show_image(image, t=0):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(t)


@torch.no_grad()
def run(cfg, network, imagedir, calib, stride=1, skip=0, viz=False, timeit=False, save_reconstruction=False):
    print("Début de la fonction run")
    slam = None
    queue = Queue(maxsize=8)

    STOP_SIGNAL = -1
    TIMEOUT = 1  # Temps maximum d'attente en secondes pour queue.get()
    JOIN_TIMEOUT = 1  # Temps d'attente maximum pour joindre le processus reader

    # if os.path.isdir(imagedir):
    #     reader = Process(target=image_stream, args=(queue, imagedir, calib, stride, skip))
    # else:
        
    reader = Process(target=video_stream, args=(queue, imagedir, calib, stride, skip))

    reader.start()

    disp = None
    
    try:
        while True:
            try:
                (t, image, intrinsics) = queue.get(timeout=TIMEOUT)  # Utilisation du timeout
            except Empty:
                print("Aucun message dans la queue, arrêt du processus.")
                break

            if t == STOP_SIGNAL or t == 20:
                break

            print("Frame : ", t)

            """ stop dans la loop """
#             if DEBUG: import pdb; pdb.set_trace()

            image = torch.from_numpy(image).permute(2, 0, 1).cuda()
            intrinsics = torch.from_numpy(intrinsics).cuda()

            if slam is None:
                slam = DPVO(cfg, network, ht=image.shape[1], wd=image.shape[2], viz=viz)

            image = image.cuda()
            intrinsics = intrinsics.cuda()

            slam(t, image, disp, intrinsics)

    except Exception as e:
        print(f"Erreur pendant le traitement : {e}")

    finally:
        print("Début de l'arrêt")

        if reader.is_alive():
            # Envoyer le signal d'arrêt au processus reader
            queue.put((STOP_SIGNAL, None, None))
            print("Signal d'arrêt envoyé au processus reader.")
            
            # Attendre que le processus se termine
            reader.join(timeout=JOIN_TIMEOUT)
            if reader.is_alive():
                print("Le processus reader n'a pas pu se terminer à temps.")
                reader.terminate()
                reader.join()
                print("Processus reader terminé force.")

        if slam:
            try:
                slam.terminate_dev()
            except Exception as e:
                print(f"Erreur lors de la terminaison de slam : {e}")

        print("Processus principal terminé.")
        #os._exit(0)  # Utilisé pour forcer la fermeture

        
    # try:
    #     while True:
    #         try:
    #             (t, image, intrinsics) = queue.get(timeout=TIMEOUT)
    #         except Empty:
    #             print("Aucun message dans la queue, arrêt du processus principal.")
    #             break
    #
    #         if t == STOP_SIGNAL or t == 2:
    #             print("Signal d'arrêt reçu, arrêt du processus principal.")
    #             break
    #
    #         print(f"Traitement de l'image à t={t}")
    #
    #         image = torch.from_numpy(image).permute(2, 0, 1).cuda()
    #         intrinsics = torch.from_numpy(intrinsics).cuda()
    #
    #         if slam is None:
    #             slam = DPVO(cfg, network, ht=image.shape[1], wd=image.shape[2], viz=viz)
    #
    #         slam(t, image, intrinsics)
    #
    # finally:
    #     if slam:
    #         slam.terminate_dev()
    #
    #     # Envoyer le signal d'arrêt au processus reader
    #     if reader.is_alive():
    #         try:
    #             queue.put((STOP_SIGNAL, None, None), timeout=TIMEOUT)
    #         except Full:
    #             print("La queue est pleine, attente pour l'envoi du signal d'arrêt.")
    #             while queue.full():
    #                 pass  # Attendre que la queue ne soit plus pleine
    #             queue.put((STOP_SIGNAL, None, None))
    #
    #         reader.join()
    #     print("Processus principal terminé.")


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


    try:
        config = Config()
        
        #config.trace_filter = GlobbingFilter(exclude=['pycallgraph2.*'])
        config.trace_filter = GlobbingFilter(
                #include=['dpvo.net.*'],  # Inclure explicitement le module torch
                exclude=['numpy.*', 
                         'pdb.*',
                         'pycallgraph2.*' ,
                         '_*', 
                         'shutil', 
                         'os', 
                         're', 
                         'sys', 
                         'module_from_spec.*',
                         'module_from_spec',
                         'SourceFileLoader.*',
                         'FileFinder.*',
                         'find_spec', 
                         '<listcomp>',
                         '<genexpr>',
                         'spec_from_file_location',
                         'cache_from_source',
                         'cb',
                         '<lambda>',
                         'VFModule.*',
                         'ModuleSpec.*',
                         'dpvo.lietorch.*',
                         'dpvo.utils.*',
                         'dpvo.blocks.*',
                         'dpvo.altcorr.*',
                         'dpvo.projective_ops.*',
                         'dpvo.extractor.*'])

        graphviz = GraphvizOutput()
        #graphviz.output_file = 'tmp.pdf'
        graphviz.output_file = 'callgraph.pdf'
        graphviz.output_type = 'pdf'  # Spécifier le format de sortie en PDF

        # graphviz.output_file = 'callgraph.png'
        # graphviz.output_type = 'png'  # Spécifier le format de sortie en PDF

        # Générer le graphe d'appel
        with PyCallGraph(output=graphviz, config=config):
            run(cfg, args.network, args.imagedir, args.calib, args.stride, args.skip, args.viz, args.timeit, args.save_reconstruction)
    except KeyboardInterrupt:
        print("Programme interrompu par l'utilisateur.")
    except Exception as e:
        print(f"Erreur inattendue : {e}")
    finally:
        print("fin du programme")
        #sys.exit(0)        
        os._exit(0)  # Utilisé pour forcer la fermeture

   

