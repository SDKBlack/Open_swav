# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import math
import os
import shutil
import time
from logging import getLogger

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
try:
    import apex
    from apex.parallel.LARC import LARC
except ImportError:
    apex = None
    LARC = None
# from src.larc import LARC
from src.boundary_loss import BoundaryLoss

from src.utils import (
    bool_flag,
    initialize_exp,
    restart_from_checkpoint,
    fix_random_seeds,
    AverageMeter,
    init_distributed_mode,
)
from src.multicropdataset import MultiCropDataset
import src.resnet50 as resnet_models
from src.s3r_dataset import S3RDataset
from src.eval_openset import evaluate_openset
from src.wtnet import WTNet

logger = getLogger()

parser = argparse.ArgumentParser(description="Implementation of SwAV")

#########################
#### data parameters ####
#########################
parser.add_argument("--data_path", type=str, default="/path/to/imagenet",
                    help="path to dataset repository")
parser.add_argument("--nmb_crops", type=int, default=[2], nargs="+",
                    help="list of number of crops (example: [2, 6])")
parser.add_argument("--size_crops", type=int, default=[224], nargs="+",
                    help="crops resolutions (example: [224, 96])")
parser.add_argument("--min_scale_crops", type=float, default=[0.14], nargs="+",
                    help="argument in RandomResizedCrop (example: [0.14, 0.05])")
parser.add_argument("--max_scale_crops", type=float, default=[1], nargs="+",
                    help="argument in RandomResizedCrop (example: [1., 0.14])")

#########################
## swav specific params #
#########################
parser.add_argument("--crops_for_assign", type=int, nargs="+", default=[0, 1],
                    help="list of crops id used for computing assignments")
parser.add_argument("--temperature", default=0.1, type=float,
                    help="temperature parameter in training loss")
parser.add_argument("--epsilon", default=0.05, type=float,
                    help="regularization parameter for Sinkhorn-Knopp algorithm")
parser.add_argument("--sinkhorn_iterations", default=3, type=int,
                    help="number of iterations in Sinkhorn-Knopp algorithm")
parser.add_argument("--feat_dim", default=128, type=int,
                    help="feature dimension")
parser.add_argument("--nmb_prototypes", default=36, type=int,
                    help="number of prototypes")
parser.add_argument("--n_active_prototypes", default=0, type=int,
                    help="(optional) number of active prototypes to use for SwAV assignments; if 0, fallback to num_classes or heuristic")
parser.add_argument("--queue_length", type=int, default=8096,
                    help="length of the queue (0 for no queue)")
parser.add_argument("--epoch_queue_starts", type=int, default=15,
                    help="from this epoch, we start using a queue")

#########################
#### optim parameters ###
#########################
parser.add_argument("--epochs", default=100, type=int,
                    help="number of total epochs to run")
parser.add_argument("--batch_size", default=64, type=int,
                    help="batch size per gpu, i.e. how many unique instances per gpu")
parser.add_argument("--base_lr", default=4.8, type=float, help="base learning rate")
parser.add_argument("--final_lr", type=float, default=0, help="final learning rate")
parser.add_argument("--freeze_prototypes_niters", default=313, type=int,
                    help="freeze the prototypes during this many iterations from the start")
parser.add_argument("--wd", default=1e-6, type=float, help="weight decay")
parser.add_argument("--warmup_epochs", default=10, type=int, help="number of warmup epochs")
parser.add_argument("--start_warmup", default=0, type=float,
                    help="initial warmup learning rate")

#########################
#### dist parameters ###
#########################
parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up distributed
                    training; see https://pytorch.org/docs/stable/distributed.html""")
parser.add_argument("--world_size", default=-1, type=int, help="""
                    number of processes: it is set automatically and
                    should not be passed as argument""")
parser.add_argument("--rank", default=0, type=int, help="""rank of this process:
                    it is set automatically and should not be passed as argument""")
parser.add_argument("--local_rank", default=0, type=int,
                    help="this argument is not used and should be ignored")

#########################
#### other parameters ###
#########################
parser.add_argument("--arch", default="resnet50", type=str, help="convnet architecture")
parser.add_argument("--hidden_mlp", default=2048, type=int,
                    help="hidden layer dimension in projection head")
parser.add_argument("--workers", default=10, type=int,
                    help="number of data loading workers")
parser.add_argument("--checkpoint_freq", type=int, default=25,
                    help="Save the model periodically")
parser.add_argument("--use_fp16", type=bool_flag, default=True,
                    help="whether to train with mixed precision or not")
parser.add_argument("--sync_bn", type=str, default="pytorch", help="synchronize bn")
parser.add_argument("--syncbn_process_group_size", type=int, default=8, help=""" see
                    https://github.com/NVIDIA/apex/blob/master/apex/parallel/__init__.py#L58-L67""")
parser.add_argument("--dump_path", type=str, default=".",
                    help="experiment dump path for checkpoints and log")
parser.add_argument("--seed", type=int, default=31, help="seed")
parser.add_argument("--split_path", type=str, default=None, help="path to split file")
parser.add_argument("--test_split_path", type=str, default=None, help="path to test split file")
parser.add_argument("--unknown_split_path", type=str, default=None, help="path to unknown split file")
parser.add_argument("--num_classes", type=int, default=0, help="number of classes for CE loss")
parser.add_argument("--swav_weight", type=float, default=1.0, help="weight for SwAV loss")

#########################
#### Boundary Loss params
#########################
parser.add_argument("--use_boundary_loss", type=bool_flag, default=False, help="whether to use boundary loss")
parser.add_argument("--boundary_pos_thresh", type=float, default=0.5, help="threshold for positive class distance")
parser.add_argument("--boundary_neg_thresh", type=float, default=1.0, help="threshold for negative class distance")
parser.add_argument("--boundary_proto_thresh", type=float, default=1.0, help="threshold for inter-prototype distance")
parser.add_argument("--boundary_loss_weight", type=float, default=0.1, help="weight for boundary loss")
parser.add_argument("--boundary_warmup_epochs", type=int, default=0, help="number of epochs to wait before enabling boundary loss")
parser.add_argument("--boundary_pos_start", type=float, default=None, help="starting positive threshold for boundary loss (will anneal to boundary_pos_thresh)")
parser.add_argument("--boundary_pos_anneal_epochs", type=int, default=0, help="number of epochs over which to linearly anneal boundary_pos from start to target (0 disables annealing)")
parser.add_argument("--use_specaugment", type=bool_flag, default=False, help="Use SpecAugment (time+freq masking) instead of RandomErasing for spectrogram data")
parser.add_argument("--use_freq_pos_enc", type=bool_flag, default=False, help="Use learnable frequency positional encoding")
parser.add_argument("--spec_freq_masks", type=int, default=2, help="number of frequency masks to apply")
parser.add_argument("--spec_time_masks", type=int, default=2, help="number of time masks to apply")
parser.add_argument("--spec_max_freq", type=int, default=30, help="maximum width (in bins) for a frequency mask")
parser.add_argument("--spec_max_time", type=int, default=40, help="maximum width (in frames) for a time mask")
parser.add_argument("--tb_log_interval", type=int, default=50, help="TensorBoard logging interval in iterations (only on rank 0)")
parser.add_argument("--use_mixup", type=bool_flag, default=False, help="Use mixup for virtual unknowns")
parser.add_argument("--mixup_loss_weight", type=float, default=1, help="Weight for mixup loss")

#########################
#### WTNet specific params
#########################
parser.add_argument("--use_shared_stem", type=bool_flag, default=False,
                    help="Whether to use a shared stem for WTNet branches")
parser.add_argument("--shared_stem_blocks", type=int, default=2,
                    help="Number of blocks in the shared stem (1-6)")
parser.add_argument("--use_sk_fusion", type=bool_flag, default=False,
                    help="Whether to use Selective Kernel Fusion for WTNet branches")
parser.add_argument("--pooling_type", type=str, default="gem", choices=["gem", "mpn", "avg"],
                    help="Pooling type: gem, mpn (MPN-COV), or avg")
parser.add_argument("--use_aux_heads", type=bool_flag, default=False,
                    help="Whether to use auxiliary classification heads for each branch")
parser.add_argument("--aux_loss_weight", type=float, default=1.0,
                    help="Weight for auxiliary classification loss")


def main():
    global args
    args = parser.parse_args()
    init_distributed_mode(args)
    fix_random_seeds(args.seed)
    logger, training_stats = initialize_exp(args, "epoch", "loss")

    # If no explicit start is provided, start from the configured target so nothing changes
    # unless the user explicitly sets a different starting threshold.
    if getattr(args, "boundary_pos_start", None) is None:
        args.boundary_pos_start = args.boundary_pos_thresh

    # build data
    if args.split_path:
        train_dataset = S3RDataset(
            args.data_path,
            args.split_path,
            args.size_crops,
            args.nmb_crops,
            args.min_scale_crops,
            args.max_scale_crops,
            is_train=True,
            # SpecAugment params
            use_specaugment=args.use_specaugment,
            spec_freq_masks=args.spec_freq_masks,
            spec_time_masks=args.spec_time_masks,
            spec_max_freq=args.spec_max_freq,
            spec_max_time=args.spec_max_time,
        )
        if args.num_classes == 0:
            args.num_classes = train_dataset.num_classes
            logger.info(f"Automatically detected number of classes: {args.num_classes}")
    else:
        train_dataset = MultiCropDataset(
            args.data_path,
            args.size_crops,
            args.nmb_crops,
            args.min_scale_crops,
            args.max_scale_crops,
        )
    sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info("Building data done with {} images loaded.".format(len(train_dataset)))

    # build model
    if args.arch == 'wtnet':
        # Determine max input size for frequency pos encoding
        max_input_size = max(args.size_crops) if args.size_crops else 224
        
        model = WTNet(
            normalize=True,
            hidden_mlp=args.hidden_mlp,
            output_dim=args.feat_dim,
            nmb_prototypes=args.nmb_prototypes,
            num_classes=args.num_classes,
            use_shared_stem=args.use_shared_stem,
            shared_stem_blocks=args.shared_stem_blocks,
            use_sk_fusion=args.use_sk_fusion,
            pooling_type=args.pooling_type,
            use_aux_heads=args.use_aux_heads,
            use_freq_pos_enc=args.use_freq_pos_enc,
            input_size=[max_input_size, max_input_size],
        )
    else:
        model = resnet_models.__dict__[args.arch](
            normalize=True,
            hidden_mlp=args.hidden_mlp,
            output_dim=args.feat_dim,
            nmb_prototypes=args.nmb_prototypes,
            num_classes=args.num_classes,
        )
    # synchronize batch norm layers
    if args.sync_bn == "pytorch":
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    elif args.sync_bn == "apex":
        # with apex syncbn we sync bn per group because it speeds up computation
        # compared to global syncbn
        # process_group = apex.parallel.create_syncbn_process_group(args.syncbn_process_group_size)
        # model = apex.parallel.convert_syncbn_model(model, process_group=process_group)
        logger.warning("Apex SyncBN is not supported without apex installed. Using PyTorch SyncBN instead.")
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # copy model to GPU
    model = model.cuda()
    if args.rank == 0:
        logger.info(model)
    logger.info("Building model done.")

    # build boundary loss
    boundary_criterion = None
    if args.use_boundary_loss:
        # Determine backbone feature dimension
        if "resnet18" in args.arch:
            backbone_dim = 512
        elif "wtnet" in args.arch:
            backbone_dim = 128
        elif "resnet50" in args.arch:
            if "w2" in args.arch:
                backbone_dim = 2048 * 2
            elif "w4" in args.arch:
                backbone_dim = 2048 * 4
            elif "w5" in args.arch:
                backbone_dim = 2048 * 5
            else:
                backbone_dim = 2048
        else:
            # Default fallback or error
            backbone_dim = 2048
            
        boundary_criterion = BoundaryLoss(
            num_classes=args.num_classes,
            feat_dim=backbone_dim,
            pos_thresh=args.boundary_pos_thresh,
            neg_thresh=args.boundary_neg_thresh,
            proto_thresh=args.boundary_proto_thresh
        ).cuda()

    # build optimizer
    params = list(model.parameters())
    if boundary_criterion is not None:
        params += list(boundary_criterion.parameters())

    optimizer = torch.optim.SGD(
        params,
        lr=args.base_lr,
        momentum=0.9,
        weight_decay=args.wd,
    )
    if LARC is not None:
        optimizer = LARC(optimizer=optimizer, trust_coefficient=0.001, clip=False)
    else:
        logger.warning("NVIDIA Apex LARC not found. Training without LARC.")
    warmup_lr_schedule = np.linspace(args.start_warmup, args.base_lr, len(train_loader) * args.warmup_epochs)
    iters = np.arange(len(train_loader) * (args.epochs - args.warmup_epochs))
    cosine_lr_schedule = np.array([args.final_lr + 0.5 * (args.base_lr - args.final_lr) * (1 + \
                         math.cos(math.pi * t / (len(train_loader) * (args.epochs - args.warmup_epochs)))) for t in iters])
    lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
    logger.info("Building optimizer done.")

    # init mixed precision
    scaler = None
    if args.use_fp16:
        # model, optimizer = apex.amp.initialize(model, optimizer, opt_level="O1")
        scaler = torch.cuda.amp.GradScaler()
        logger.info("Initializing mixed precision done.")

    # wrap model
    model = nn.parallel.DistributedDataParallel(
        model,
        device_ids=[args.gpu_to_work_on],
        find_unused_parameters=True
    )

    # optionally resume from a checkpoint
    to_restore = {"epoch": 0}
    restart_from_checkpoint(
        os.path.join(args.dump_path, "checkpoint.pth.tar"),
        run_variables=to_restore,
        state_dict=model,
        optimizer=optimizer,
        scaler=scaler,
    )
    start_epoch = to_restore["epoch"]

    # build the queue
    queue = None
    queue_path = os.path.join(args.dump_path, "queue" + str(args.rank) + ".pth")
    if os.path.isfile(queue_path):
        queue = torch.load(queue_path)["queue"]
    # the queue needs to be divisible by the batch size
    args.queue_length -= args.queue_length % (args.batch_size * args.world_size)

    cudnn.benchmark = True

    for epoch in range(start_epoch, args.epochs):

        # train the network for one epoch
        logger.info("============ Starting epoch %i ... ============" % epoch)

        # set sampler
        train_loader.sampler.set_epoch(epoch)

        # optionally starts a queue
        if args.queue_length > 0 and epoch >= args.epoch_queue_starts and queue is None:
            queue = torch.zeros(
                len(args.crops_for_assign),
                args.queue_length // args.world_size,
                args.feat_dim,
            ).cuda()

        # train the network
        scores, queue = train(train_loader, model, optimizer, epoch, lr_schedule, queue, scaler, boundary_criterion)
        training_stats.update(scores)

        # save checkpoints
        if args.rank == 0:
            save_dict = {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            if args.use_fp16:
                save_dict["scaler"] = scaler.state_dict()
            torch.save(
                save_dict,
                os.path.join(args.dump_path, "checkpoint.pth.tar"),
            )
            if epoch % args.checkpoint_freq == 0 or epoch == args.epochs - 1:
                shutil.copyfile(
                    os.path.join(args.dump_path, "checkpoint.pth.tar"),
                    os.path.join(args.dump_checkpoints, "ckp-" + str(epoch) + ".pth"),
                )
        if queue is not None:
            torch.save({"queue": queue}, queue_path)

    # build test data
    test_loader = None
    if args.test_split_path:
        test_dataset = S3RDataset(
            args.data_path,
            args.test_split_path,
            args.size_crops,
            args.nmb_crops,
            args.min_scale_crops,
            args.max_scale_crops,
            is_train=False
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
        )
        
        logger.info("============ Starting validation ... ============")
        validate(test_loader, model)

    # build unknown data
    unknown_loader = None
    if args.unknown_split_path:
        unknown_dataset = S3RDataset(
            args.data_path,
            args.unknown_split_path,
            args.size_crops,
            args.nmb_crops,
            args.min_scale_crops,
            args.max_scale_crops,
            is_train=False
        )
        unknown_loader = torch.utils.data.DataLoader(
            unknown_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
        )

    if args.test_split_path and args.unknown_split_path:
        # Create a train loader for evaluation (no augmentations)
        train_dataset_eval = S3RDataset(
            args.data_path,
            args.split_path,
            args.size_crops,
            args.nmb_crops,
            args.min_scale_crops,
            args.max_scale_crops,
            is_train=False
        )
        train_loader_eval = torch.utils.data.DataLoader(
            train_dataset_eval,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
        )
        evaluate_openset(model, train_loader_eval, test_loader, unknown_loader, args)

    if dist.is_initialized():
        dist.destroy_process_group()


def train(train_loader, model, optimizer, epoch, lr_schedule, queue, scaler, boundary_criterion=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    swav_losses = AverageMeter()
    boundary_losses = AverageMeter()
    mixup_losses = AverageMeter()

    ce_losses = AverageMeter()
    aux_losses = AverageMeter()

    model.train()
    use_the_queue = False

    end = time.time()
    for it, batch in enumerate(train_loader):
        if isinstance(batch, list) and len(batch) == 2 and isinstance(batch[1], torch.Tensor):
             inputs, labels = batch
             labels = labels.cuda(non_blocking=True)
        else:
             inputs = batch
             labels = None
        # measure data loading time
        data_time.update(time.time() - end)

        # update learning rate
        iteration = epoch * len(train_loader) + it
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_schedule[iteration]

        # normalize the prototypes
        with torch.no_grad():
            w = model.module.prototypes.weight.data.clone()
            w = nn.functional.normalize(w, dim=1, p=2)
            model.module.prototypes.weight.copy_(w)

        # ============ multi-res forward passes ... ============
        aux_logits = None
        if args.use_fp16:
            with torch.amp.autocast('cuda'):
                ret = model(inputs, labels)
                embedding = ret[0]
                output = ret[1]
                logits = None
                aux_logits = None
                
                if len(ret) == 5:
                    logits = ret[2]
                    aux_logits = ret[3]
                elif len(ret) == 4:
                    if isinstance(ret[2], dict):
                        aux_logits = ret[2]
                    else:
                        logits = ret[2]
        else:
            ret = model(inputs, labels)
            embedding = ret[0]
            output = ret[1]
            logits = None
            aux_logits = None
            
            if len(ret) == 5:
                logits = ret[2]
                aux_logits = ret[3]
            elif len(ret) == 4:
                if isinstance(ret[2], dict):
                    aux_logits = ret[2]
                else:
                    logits = ret[2]
            
        embedding_detached = embedding.detach()
        bs = inputs[0].size(0)

        # ============ swav loss ... ============
        loss = 0
        boundary_loss = 0
        aux_loss = 0
        
        # === 新增：定义活跃原型的数量 ===
        # 优先使用用户手动指定的 args.n_active_prototypes（若 >0）
        if hasattr(args, 'n_active_prototypes') and args.n_active_prototypes > 0:
            n_active = args.n_active_prototypes
        else:
            # 否则回退到 num_classes（如果设置了），再回退到一个默认值（18）
            n_active = args.num_classes if args.num_classes > 0 else 18
        # 确保不超过总原型数
        n_active = min(n_active, args.nmb_prototypes)
        
        for i, crop_id in enumerate(args.crops_for_assign):
            with torch.no_grad():
                out = output[bs * crop_id: bs * (crop_id + 1)].detach()

                # === 修改点 1: 仅截取“活跃原型”的 logits 传给 Sinkhorn ===
                # out 的形状是 [Batch_Size, nmb_prototypes]
                # 我们只取前 n_active 列 [Batch_Size, n_active]
                out_active = out[:, :n_active]

                # time to use the queue
                if queue is not None:
                    if use_the_queue or not torch.all(queue[i, -1, :] == 0):
                        use_the_queue = True
                        
                        # 队列中的特征也只跟活跃原型做计算
                        queue_proto_logits = torch.mm(
                            queue[i],
                            model.module.prototypes.weight[:n_active].t() # 只用活跃原型的权重
                        )
                        out_active = torch.cat((queue_proto_logits, out_active))
                    
                    # fill the queue
                    queue[i, bs:] = queue[i, :-bs].clone()
                    queue[i, :bs] = embedding_detached[crop_id * bs: (crop_id + 1) * bs]

                # get assignments (q 现在的维度是 [Batch_Size, n_active])
                # Sinkhorn 现在只会把样本分配给这 n_active 个原型，完美避开了幽灵原型
                q = distributed_sinkhorn(out_active)[-bs:]

            # cluster assignment prediction
            subloss = 0
            for v in np.delete(np.arange(np.sum(args.nmb_crops)), crop_id):
                # === 修改点 2: 计算 Loss 时也只看活跃原型 ===
                # 获取 prediction logits
                logits_all = output[bs * v: bs * (v + 1)] / args.temperature
                
                # 截取活跃部分
                logits_active = logits_all[:, :n_active]
                
                # 标准 SwAV Loss (只在活跃原型上计算)
                subloss -= torch.mean(torch.sum(q * F.log_softmax(logits_active, dim=1), dim=1))
                
                # === 修改点 3: 幽灵原型的“互斥损失” (Ghost Push-Away Loss) ===
                # 我们希望已知类样本与幽灵原型的相似度越低越好
                if n_active < args.nmb_prototypes:
                    # 取出幽灵原型的 logits
                    logits_ghost = logits_all[:, n_active:]
                    # 目标：最小化这些 logits 的值 (或者最大化它们的负值)
                    # 简单做法：L2 正则化或者 LogSumExp 最小化
                    # 这里用一个简单的 hinge loss 风格：强迫相似度小于某个负阈值，或者直接 minimize mean
                    # 因为 SwAV 的 logits 是点积 / temp，值越大越相似。我们希望值越小越好。
                    
                    # 策略：Minimize Mean(Logits_Ghost)
                    # 赋予一个权重 (ghost_loss_weight)，例如 0.1
                    ghost_penalty = torch.mean(logits_ghost)
                    subloss += 0.1 * ghost_penalty

            loss += subloss / (np.sum(args.nmb_crops) - 1)

        loss /= len(args.crops_for_assign)
        # Track raw (unweighted) SwAV loss
        try:
            swav_losses.update(float(loss.item()), bs)
        except Exception:
            swav_losses.update(float(loss), bs)

        # ============ CE loss ... ============
        ce_loss = 0
        if logits is not None and labels is not None:
            ce_loss = nn.CrossEntropyLoss()(logits[:bs], labels)
            ce_losses.update(ce_loss.item(), bs)
        else:
            ce_losses.update(0.0, bs)

        # ============ Aux loss ... ============
        if aux_logits is not None and labels is not None:
            for k, v in aux_logits.items():
                aux_loss += nn.CrossEntropyLoss()(v[:bs], labels)
            aux_losses.update(aux_loss.item(), bs)
        else:
            aux_losses.update(0.0, bs)
            
        # ============ Boundary Loss ... ============
        boundary_loss = 0
        boundary_loss_raw = 0
        mixup_loss_raw = 0
        # Compute current positive threshold (possibly annealed). If annealing is
        # disabled (boundary_pos_anneal_epochs == 0), this will simply be
        # args.boundary_pos_thresh. We update the criterion's pos_thresh so the
        # loss uses the current value.
        if boundary_criterion is not None:
            if args.boundary_pos_anneal_epochs > 0:
                t = min(epoch, args.boundary_pos_anneal_epochs)
                frac = float(t) / float(args.boundary_pos_anneal_epochs)
                current_pos = args.boundary_pos_start + frac * (args.boundary_pos_thresh - args.boundary_pos_start)
            else:
                current_pos = args.boundary_pos_thresh
            # write current value into the criterion (this is an attribute used in forward)
            try:
                boundary_criterion.pos_thresh = float(current_pos)
            except Exception:
                # if criterion doesn't expose the attribute for some reason, ignore
                pass

        # Only compute boundary loss after the warmup/wait period to avoid
        # destabilizing early training. Controlled by --boundary_warmup_epochs.
        if boundary_criterion is not None and labels is not None and epoch >= args.boundary_warmup_epochs:
            # Get backbone features
            # model is DDP wrapped, so use model.module
            backbone_feats = getattr(model.module, '_last_backbone', None)
            
            if backbone_feats is not None:
                # backbone_feats is [B * sum(nmb_crops), D]
                # labels is [B]
                # We need to expand labels to match backbone_feats
                n_crops = backbone_feats.size(0) // bs
                labels_expanded = labels.repeat(n_crops)
                
                # Raw (unweighted) boundary loss
                boundary_loss_raw = boundary_criterion(backbone_feats, labels_expanded)
                # Weight meters by number of backbone features (often B * n_crops)
                n_feats = int(backbone_feats.size(0))
                boundary_losses.update(float(boundary_loss_raw.item()), n_feats)

                # Start from raw boundary loss; mixup term (if enabled) is added separately
                boundary_loss = boundary_loss_raw

                # ============ Mixup for Virtual Unknowns ... ============
                if args.use_mixup:
                    # 1. Random Mixup
                    lam = np.random.beta(1.0, 1.0)
                    batch_size_feats = backbone_feats.size(0)
                    index = torch.randperm(batch_size_feats).cuda()
                    virtual_unknowns = lam * backbone_feats + (1 - lam) * backbone_feats[index]

                    # 2. & 3. Calculate distance to prototypes and force > neg_thresh
                    mixup_loss_raw = boundary_criterion.forward_virtual(virtual_unknowns)
                    mixup_losses.update(float(mixup_loss_raw.item()), n_feats)
                    boundary_loss += args.mixup_loss_weight * mixup_loss_raw
                else:
                    mixup_losses.update(0.0, n_feats)
            else:
                # boundary enabled but backbone feats missing
                boundary_losses.update(0.0, bs)
                mixup_losses.update(0.0, bs)
        else:
            # boundary disabled / no labels / warmup not reached
            boundary_losses.update(0.0, bs)
            mixup_losses.update(0.0, bs)
        
        total_loss = args.swav_weight * loss + ce_loss + args.boundary_loss_weight * boundary_loss + args.aux_loss_weight * aux_loss

        # ============ backward and optim step ... ============
        optimizer.zero_grad()
        if args.use_fp16:
            scaler.scale(total_loss).backward()
            # with apex.amp.scale_loss(total_loss, optimizer) as scaled_loss:
            #     scaled_loss.backward()
        else:
            total_loss.backward()
        # cancel gradients for the prototypes
        if iteration < args.freeze_prototypes_niters:
            for name, p in model.named_parameters():
                if "prototypes" in name:
                    p.grad = None
        
        if args.use_fp16:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        # ============ misc ... ============
        losses.update(total_loss.item(), inputs[0].size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        if args.rank == 0 and it % args.tb_log_interval == 0:
            lr_val = optimizer.optimizer.param_groups[0]["lr"] if hasattr(optimizer, "optimizer") else optimizer.param_groups[0]["lr"]
            logger.info(
                "Epoch: [{0}][{1}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                    "Total Loss {tot.val:.4f} ({tot.avg:.4f})\t"
                    "SWAV Loss {swav.val:.4f} ({swav.avg:.4f})\t"
                    "CE Loss {ce_loss.val:.4f} ({ce_loss.avg:.4f})\t"
                    "Aux Loss {aux_loss.val:.4f} ({aux_loss.avg:.4f})\t"
                    "B Loss {b_loss.val:.4f} ({b_loss.avg:.4f})\t"
                    "Mixup Loss {m_loss.val:.4f} ({m_loss.avg:.4f})\t"
                "Lr: {lr:.4f}".format(
                    epoch,
                    it,
                    batch_time=batch_time,
                    data_time=data_time,
                    tot=losses,
                    swav=swav_losses,
                    ce_loss=ce_losses,
                        aux_loss=aux_losses,
                        b_loss=boundary_losses,
                        m_loss=mixup_losses,
                    lr=lr_val,
                )
            )

            # TensorBoard logging (per-iteration)
            try:
                tb = getattr(logger, 'tb_writer', None)
                if tb is not None:
                    # global step: iteration
                    global_step = iteration
                    # total loss
                    tb.add_scalar('train/total_loss_iter', float(total_loss.item()), global_step)
                    # swav loss (per-crop loss variable `loss`)
                    try:
                        tb.add_scalar('train/swav_loss_iter', float(loss.item()), global_step)
                    except Exception:
                        tb.add_scalar('train/swav_loss_iter', float(loss), global_step)
                    # ce loss
                    try:
                        tb.add_scalar('train/ce_loss_iter', float(ce_loss.item()), global_step)
                    except Exception:
                        tb.add_scalar('train/ce_loss_iter', float(ce_loss), global_step)
                    # boundary loss
                    try:
                        tb.add_scalar('train/boundary_loss_iter', float(boundary_loss.item()), global_step)
                    except Exception:
                        tb.add_scalar('train/boundary_loss_iter', float(boundary_loss), global_step)
                    # boundary raw (unweighted)
                    try:
                        tb.add_scalar('train/boundary_loss_raw_iter', float(boundary_loss_raw.item()), global_step)
                    except Exception:
                        tb.add_scalar('train/boundary_loss_raw_iter', float(boundary_loss_raw), global_step)
                    # mixup raw (unweighted)
                    try:
                        tb.add_scalar('train/mixup_loss_raw_iter', float(mixup_loss_raw.item()), global_step)
                    except Exception:
                        tb.add_scalar('train/mixup_loss_raw_iter', float(mixup_loss_raw), global_step)
                    # aux loss
                    try:
                        tb.add_scalar('train/aux_loss_iter', float(aux_loss.item()), global_step)
                    except Exception:
                        try:
                            tb.add_scalar('train/aux_loss_iter', float(aux_loss), global_step)
                        except Exception:
                            pass
                    # learning rate
                    try:
                        current_lr = None
                        try:
                            current_lr = optimizer.param_groups[0]["lr"]
                        except Exception:
                            try:
                                current_lr = optimizer.optimizer.param_groups[0]["lr"]
                            except Exception:
                                current_lr = lr_schedule[iteration]
                        tb.add_scalar('train/lr', float(current_lr), global_step)
                    except Exception:
                        pass
            except Exception:
                # avoid breaking training if tensorboard write fails
                pass
    # Epoch-level TensorBoard logging (averages)
    try:
        tb = getattr(logger, 'tb_writer', None)
        if tb is not None and args.rank == 0:
            tb.add_scalar('train/total_loss_epoch', float(losses.avg), epoch)
            tb.add_scalar('train/swav_loss_epoch', float(swav_losses.avg), epoch)
            tb.add_scalar('train/ce_loss_epoch', float(ce_losses.avg), epoch)
            tb.add_scalar('train/aux_loss_epoch', float(aux_losses.avg), epoch)
            tb.add_scalar('train/boundary_loss_raw_epoch', float(boundary_losses.avg), epoch)
            tb.add_scalar('train/mixup_loss_raw_epoch', float(mixup_losses.avg), epoch)
            # record lr at epoch end (first param group)
            try:
                try:
                    current_lr = optimizer.param_groups[0]["lr"]
                except Exception:
                    current_lr = optimizer.optimizer.param_groups[0]["lr"]
                tb.add_scalar('train/lr_epoch', float(current_lr), epoch)
            except Exception:
                pass
    except Exception:
        pass

    return (epoch, losses.avg), queue


@torch.no_grad()
def distributed_sinkhorn(out):
    Q = torch.exp(out / args.epsilon).t() # Q is K-by-B for consistency with notations from our paper
    B = Q.shape[1] * args.world_size # number of samples to assign
    K = Q.shape[0] # how many prototypes

    # make the matrix sums to 1
    sum_Q = torch.sum(Q)
    dist.all_reduce(sum_Q)
    Q /= sum_Q

    for it in range(args.sinkhorn_iterations):
        # normalize each row: total weight per prototype must be 1/K
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        dist.all_reduce(sum_of_rows)
        Q /= sum_of_rows
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B

    Q *= B # the colomns must sum to 1 so that Q is an assignment
    return Q.t()


def validate(val_loader, model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            
            ret = model(inputs)
            if len(ret) == 3:
                _, _, logits = ret
            else:
                logits = None
            
            if logits is not None:
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
    if total > 0:
        acc = 100 * correct / total
        logger.info(f"Validation Accuracy: {acc:.2f}%")
        return acc
    return 0


if __name__ == "__main__":
    main()


# torchrun --nproc_per_node=1 main_swav.py --arch wtnet --data_path /root/autodl-tmp/S3R --split_path /root/autodl-tmp/S3R/experiment_groups/1-known_for_train --test_split_path /root/autodl-tmp/S3R/experiment_groups/1-known_for_test --unknown_split_path /root/autodl-tmp/S3R/experiment_groups/1-unknown --swav_weight 0.5 --epochs 200 --batch_size 128 --base_lr 0.1 --final_lr 0.001 --size_crops 224 --nmb_crops 6 --min_scale_crops 0.8 --max_scale_crops 1.0 --dump_path ./test_active_pro --use_fp16 False --use_boundary_loss true --boundary_pos_start 1.0 --boundary_pos_thresh 0.2 --boundary_pos_anneal_epochs 50 --boundary_neg_thresh 1.3 --boundary_proto_thresh 1.3 --boundary_loss_weight 1.0 --nmb_prototypes 72 --use_aux_heads True --aux_loss_weight 0.1 --n_active_prototypes 54