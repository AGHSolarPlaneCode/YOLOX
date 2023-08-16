#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
# import random
# import warnings

import tensorflow.lite as tflite
# import torch
# import torch.backends.cudnn as cudnn
from loguru import logger
# from torch.nn.parallel import DistributedDataParallel as DDP

from yolox.core import launch
from yolox.exp import get_exp
from YOLOX.yolox.utils import (
    configure_module,
    configure_nccl,
    fuse_model,
    get_local_rank,
    get_model_info,
    setup_logger,
)


def make_parser():
    parser = argparse.ArgumentParser("YOLOX Eval")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument(
        "-n", "--name", type=str, default=None, help="model name"
    )

    # distributed
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--dist-url",
        default=None,
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "-b", "--batch-size", type=int, default=64, help="batch size"
    )
    parser.add_argument(
        "-d", "--devices", default=None, type=int, help="device for training"
    )
    parser.add_argument(
        "--num_machines", default=1, type=int, help="num of node for training"
    )
    parser.add_argument(
        "--machine_rank",
        default=0,
        type=int,
        help="node rank for multi-node training",
    )
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="please input your experiment description file",
    )
    parser.add_argument(
        "-c", "--ckpt", default=None, type=str, help="ckpt for eval"
    )
    parser.add_argument(
        "--dataset-path", default=None, type=str, help="evaluation dataset path"
    )
    parser.add_argument(
        "--ann-path", default=None, type=str, help="evaluation annotations path"
    )
    parser.add_argument("--conf", default=None, type=float, help="test conf")
    parser.add_argument(
        "--nms", default=None, type=float, help="test nms threshold"
    )
    parser.add_argument(
        "--tsize", default=None, type=int, help="test img size"
    )
    parser.add_argument("--seed", default=None, type=int, help="eval seed")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--legacy",
        dest="legacy",
        default=False,
        action="store_true",
        help="To be compatible with older versions",
    )
    parser.add_argument(
        "--test",
        dest="test",
        default=False,
        action="store_true",
        help="Evaluating on test-dev set.",
    )
    parser.add_argument(
        "--speed",
        dest="speed",
        default=False,
        action="store_true",
        help="speed test only.",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


@logger.catch
def main(
    # exp,
    args, num_gpu
):
    is_distributed = num_gpu > 1

    rank = get_local_rank()
    
    output_dir = "YOLOX_outputs"

    file_name = os.path.join(output_dir, "test")

    if rank == 0:
        os.makedirs(file_name, exist_ok=True)

    setup_logger(
        file_name, distributed_rank=rank, filename="val_log.txt", mode="a"
    )
    logger.info("Args: {}".format(args))

    if args.conf is not None:
        test_conf = args.conf
    if args.nms is not None:
        nmsthre = args.nms
    if args.tsize is not None:
        test_size = (args.tsize, args.tsize)
    data_dir = args.dataset_path if args.dataset_path is not None else "datasets/coco"
    test_ann = args.ann_path if args.ann_path is not None else "coco_test.json"

    from yolox.data import COCODataset, ValTransform
    dataset = COCODataset(
            data_dir=data_dir,
            json_file=test_ann,
            name="images",
            img_size=test_size,
            preproc=ValTransform(legacy=args.legacy),
        )
    from YOLOX.yolox.evaluators.coco_evaluator_tf import COCOEvaluator
    evaluator = COCOEvaluator(
        dataset,
        test_size,
        test_conf,
        nmsthre,
        1,
    )
    evaluator.per_class_AP = True
    evaluator.per_class_AR = True

    if not args.speed:
        if args.ckpt is None:
            ckpt_file = "../models/tf_lite/fp16.tflite"
        else:
            ckpt_file = args.ckpt
        interpreter = tflite.Interpreter(
            model_path=ckpt_file, num_threads=2 #?
        )
        interpreter.allocate_tensors()
        logger.info("loaded checkpoint done.")

    trt_file = None
    decoder = None

    # start evaluate
    *_, summary = evaluator.evaluate(
        interpreter, is_distributed, args.fp16, trt_file, decoder, test_size
    )
    logger.info("\n" + summary)


if __name__ == "__main__":
    configure_module()
    args = make_parser().parse_args()

    num_gpu = (
        0 if args.devices is None else args.devices
    )
    # assert num_gpu <= torch.cuda.device_count()

    dist_url = "auto" if args.dist_url is None else args.dist_url
    launch(
        main,
        num_gpu,
        args.num_machines,
        args.machine_rank,
        backend=args.dist_backend,
        dist_url=dist_url,
        args=(
            # exp, 
            args, num_gpu
        ),
    )
