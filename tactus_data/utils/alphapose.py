"""Script for single-gpu/multi-gpu demo."""
import argparse
import os
import platform
import sys
import time
from typing import List
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
import natsort

from trackers.tracker_api import Tracker
from trackers.tracker_cfg import cfg as tcfg
from trackers import track
from alphapose.models import builder
from alphapose.utils.config import update_config
from alphapose.utils.detector import DetectionLoader
from alphapose.utils.vis import getTime
from alphapose.utils.writer_smpl import DataWriterSMPL

"""----------------------------- Demo options -----------------------------"""
parser = argparse.ArgumentParser(description='AlphaPose Demo')
parser.add_argument('--cfg', type=str, required=True,
                    help='experiment configure file name')
parser.add_argument('--checkpoint', type=str, required=True,
                    help='checkpoint file name')
parser.add_argument('--sp', default=False, action='store_true',
                    help='Use single process for pytorch')
parser.add_argument('--detector', dest='detector',
                    help='detector name', default="yolo")
parser.add_argument('--detfile', dest='detfile',
                    help='detection result file', default="")
parser.add_argument('--indir', dest='inputpath',
                    help='image-directory', default="")
parser.add_argument('--list', dest='inputlist',
                    help='image-list', default="")
parser.add_argument('--image', dest='inputimg',
                    help='image-name', default="")
parser.add_argument('--outdir', dest='outputpath',
                    help='output-directory', default="examples/res/")
parser.add_argument('--save_img', default=False, action='store_true',
                    help='save result as image')
parser.add_argument('--vis', default=False, action='store_true',
                    help='visualize image')
parser.add_argument('--showbox', default=False, action='store_true',
                    help='visualize human bbox')
parser.add_argument('--show_skeleton', default=False, action='store_true',
                    help='visualize 3d human skeleton')
parser.add_argument('--profile', default=False, action='store_true',
                    help='add speed profiling at screen output')
parser.add_argument('--format', type=str,
                    help='save in the format of cmu or coco or openpose, option: coco/cmu/open')
parser.add_argument('--min_box_area', type=int, default=0,
                    help='min box area to filter out')
parser.add_argument('--detbatch', type=int, default=5,
                    help='detection batch size PER GPU')
parser.add_argument('--posebatch', type=int, default=64,
                    help='pose estimation maximum batch size PER GPU')
parser.add_argument('--eval', dest='eval', default=False, action='store_true',
                    help='save the result json as coco format, using image index(int) instead of image name(str)')
parser.add_argument('--gpus', type=str, dest='gpus', default="0",
                    help='choose which cuda device to use by index and input comma to use multi gpus, e.g. 0,1,2,3. (input -1 for cpu only)')
parser.add_argument('--qsize', type=int, dest='qsize', default=1024,
                    help='the length of result buffer, where reducing it will lower requirement of cpu memory')
parser.add_argument('--flip', default=False, action='store_true',
                    help='enable flip testing')
parser.add_argument('--debug', default=False, action='store_true',
                    help='print detail information')
"""----------------------------- Video options -----------------------------"""
parser.add_argument('--video', dest='video',
                    help='video-name', default="")
parser.add_argument('--webcam', dest='webcam', type=int,
                    help='webcam number', default=-1)
parser.add_argument('--save_video', dest='save_video',
                    help='whether to save rendered video', default=False, action='store_true')
parser.add_argument('--vis_fast', dest='vis_fast',
                    help='use fast rendering', action='store_true', default=False)
"""----------------------------- Tracking options -----------------------------"""
parser.add_argument('--pose_flow', dest='pose_flow',
                    help='track humans in video with PoseFlow', action='store_true', default=False)
parser.add_argument('--pose_track', dest='pose_track',
                    help='track humans in video with reid', action='store_true', default=False)


if platform.system() != 'Windows':
    torch.multiprocessing.set_start_method('forkserver', force=True)
    torch.multiprocessing.set_sharing_strategy('file_system')


def get_detector(args: dict):
    detector: str = args.detector

    if detector == 'yolo':
        from detector.yolo_api import YOLODetector
        from detector.yolo_cfg import cfg
        return YOLODetector(cfg, args)
    if 'yolox' in detector:
        from detector.yolox_api import YOLOXDetector
        from detector.yolox_cfg import cfg
        return YOLOXDetector(cfg, args)
    if detector == 'tracker':
        from detector.tracker_api import Tracker
        from detector.tracker_cfg import cfg
        return Tracker(cfg, args)
    if detector.startswith('efficientdet_d'):
        from detector.effdet_api import EffDetDetector
        from detector.effdet_cfg import cfg
        return EffDetDetector(cfg, args)

    raise NotImplementedError


def load_pose_model(cfg, args, pose_track: bool, weights_path: Path, device_ids: List[int] = [0], device: torch.device):
    pose_model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)

    print(f'Loading pose model from {weights_path}...')

    pose_model.load_state_dict(torch.load(weights_path, map_location=device))

    if pose_track:
        pose_tracker = Tracker(tcfg, args)

    if len(device_ids) > 1:
        pose_model = torch.nn.DataParallel(pose_model, device_ids=device_ids).to(device)
    else:
        pose_model.to(device)
    pose_model.eval()

    return pose_model, pose_tracker


def alphapose(input_path: Path, output_path: Path, weights_path: Path, pose_track: bool, device_ids: List[int] = [0], detection_batch_size: int = 5):

    args = {}
    args.gpus = device_ids
    args.device = device
    args.tracking = True
    args.inputpath = input_path
    args.sp = platform.system() != 'Windows'

    cfg = update_config(args.cfg)

    if torch.cuda.device_count() == 0:
        device_ids = [-1]

    if device_ids[0] >= 0:
        device = torch.device("cuda:" + str(device_ids[0]))
    else:
        device = torch.device("cuda:cpu")

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    det_loader = DetectionLoader(img_names, get_detector(args), cfg, args, batchSize=detection_batch_size, mode='image', queueSize=args.qsize)
    det_loader.start()

    # Load pose model
    pose_model, pose_tracker = load_pose_model(cfg, args, pose_track, weights_path, device_ids, device)

    # Init data writer
    queueSize = args.qsize

    writer = DataWriterSMPL(cfg, args, save_video=False, queueSize=queueSize).start()

    data_len = det_loader.length
    im_names_desc = tqdm(range(data_len), dynamic_ncols=True)

    try:
        for i in im_names_desc:
            start_time = getTime()
            with torch.no_grad():
                (inps, orig_img, im_name, boxes, scores, ids, cropped_boxes) = det_loader.read()
                if orig_img is None:
                    break
                if boxes is None or boxes.nelement() == 0:
                    writer.save(None, None, None, None, None, orig_img, im_name)
                    continue
                
                # Pose Estimation
                inps = inps.to(args.device)

                img_center = torch.Tensor((orig_img.shape[1], orig_img.shape[0])).float().to(args.device) / 2
                img_center = img_center.unsqueeze(0).repeat(inps.shape[0], 1)

                pose_output = pose_model(
                    inps, flip_test=args.flip,
                    bboxes=cropped_boxes.to(args.device),
                    img_center=img_center
                )
                
                # Not support for now
                if args.pose_track:
                    old_ids = torch.arange(boxes.shape[0]).long()
                    _, _, ids, new_ids, _ = track(
                        pose_tracker, args, orig_img, inps,
                        boxes, old_ids, cropped_boxes,
                        im_name, scores)
                    new_ids = new_ids.long()
                else:
                    new_ids = torch.arange(boxes.shape[0]).long()
                    ids = new_ids + 1

                boxes = boxes[new_ids]
                cropped_boxes = cropped_boxes[new_ids]
                scores = scores[new_ids]

                smpl_output = {
                    'pred_uvd_jts': pose_output.pred_uvd_jts.cpu()[new_ids],
                    'maxvals': pose_output.maxvals.cpu()[new_ids],
                    'transl': pose_output.transl.cpu()[new_ids],
                    'pred_vertices': pose_output.pred_vertices.cpu()[new_ids],
                    'pred_xyz_jts_24': pose_output.pred_xyz_jts_24_struct.cpu()[new_ids] * 2,   # convert to meters
                    'smpl_faces': torch.from_numpy(pose_model.smpl.faces.astype(np.int32))
                }

                writer.save(boxes, scores, ids, smpl_output,
                            cropped_boxes, orig_img, im_name)

        while(writer.running()):
            time.sleep(1)
            print('===========================> Rendering remaining ' + str(writer.count()) + ' images in the queue...', end='\r')
        writer.stop()
        det_loader.stop()
    except KeyboardInterrupt:
        # Thread won't be killed when press Ctrl+C
        if platform.system() == 'Windows':
            det_loader.terminate()
            while(writer.running()):
                time.sleep(1)
                print('===========================> Rendering remaining ' + str(writer.count()) + ' images in the queue...', end='\r')
            writer.stop()
        else:
            # subprocesses are killed, manually clear queues

            det_loader.terminate()
            writer.terminate()
            writer.clear_queues()
            det_loader.clear_queues()
