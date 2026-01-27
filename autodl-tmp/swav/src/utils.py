# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
from logging import getLogger
import pickle
import os
import sys

import numpy as np
import torch

from .logger import create_logger, PD_Stats

import torch.distributed as dist

# Optional TensorBoard SummaryWriter
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

FALSY_STRINGS = {"off", "false", "0"}
TRUTHY_STRINGS = {"on", "true", "1"}


logger = getLogger()


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")


def init_distributed_mode(args):
    """
    Initialize the following variables:
        - world_size
        - rank
    """

    args.is_slurm_job = "SLURM_JOB_ID" in os.environ

    if args.is_slurm_job:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.world_size = int(os.environ["SLURM_NNODES"]) * int(
            os.environ["SLURM_TASKS_PER_NODE"][0]
        )
    else:
        # multi-GPU job (local or multi-node) - jobs started with torch.distributed.launch
        # read environment variables
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            args.rank = int(os.environ["RANK"])
            args.world_size = int(os.environ["WORLD_SIZE"])
            args.gpu_to_work_on = args.rank % torch.cuda.device_count()
        else:
            print("Not using distributed mode")
            args.rank = 0
            args.world_size = 1
            args.gpu_to_work_on = 0

            # Do NOT initialize a process group in single-process mode.
            # This makes the script runnable on Windows with plain:
            #   python main_swav.py ...
            # where NCCL isn't available.
            args.distributed = False
            if torch.cuda.is_available():
                torch.cuda.set_device(args.gpu_to_work_on)
            return

    args.distributed = True

    # prepare distributed: NCCL on Linux when available, GLOO on Windows.
    backend = "nccl"
    if sys.platform.startswith("win") or os.name == "nt":
        backend = "gloo"
    else:
        try:
            if not dist.is_nccl_available():
                backend = "gloo"
        except Exception:
            backend = "gloo"

    dist.init_process_group(
        backend=backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )

    # set cuda device
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_to_work_on)
    return


def initialize_exp(params, *args, dump_params=True):
    """
    Initialize the experience:
    - dump parameters
    - create checkpoint repo
    - create a logger
    - create a panda object to keep track of the training statistics
    """

    # dump parameters
    if dump_params:
        if not os.path.exists(params.dump_path):
            os.makedirs(params.dump_path, exist_ok=True)
        pickle.dump(params, open(os.path.join(params.dump_path, "params.pkl"), "wb"))

    # create repo to store checkpoints
    params.dump_checkpoints = os.path.join(params.dump_path, "checkpoints")
    if not params.rank and not os.path.isdir(params.dump_checkpoints):
        os.makedirs(params.dump_checkpoints, exist_ok=True)

    # create a panda object to log loss and acc
    training_stats = PD_Stats(
        os.path.join(params.dump_path, "stats" + str(params.rank) + ".pkl"), args
    )

    # create a logger
    logger = create_logger(
        os.path.join(params.dump_path, "train.log"), rank=params.rank
    )
    logger.info("============ Initialized logger ============")
    logger.info(
        "\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(params)).items()))
    )
    logger.info("The experiment will be stored in %s\n" % params.dump_path)
    logger.info("")
    # Create a TensorBoard SummaryWriter for this experiment (only on rank 0)
    try:
        tb_writer = None
        # experiment name: last path component of dump_path
        exp_name = os.path.basename(os.path.abspath(params.dump_path))
        tb_base = '/root/tf-logs'
        tb_dir = os.path.join(tb_base, exp_name)
        if params.rank == 0 and SummaryWriter is not None:
            os.makedirs(tb_dir, exist_ok=True)
            tb_writer = SummaryWriter(log_dir=tb_dir)
            logger.info(f"TensorBoard writer created at: {tb_dir}")
        else:
            logger.info("TensorBoard writer not created on this rank or SummaryWriter unavailable")
        # attach to logger for easy access: logger.tb_writer may be None on non-master ranks
        logger.tb_writer = tb_writer
    except Exception:
        logger.tb_writer = None

    return logger, training_stats


def restart_from_checkpoint(ckp_paths, run_variables=None, **kwargs):
    """
    Re-start from checkpoint
    """

    def _should_use_prefix(keys, prefix, sample_size=64, threshold=0.8):
        """Heuristic: do most keys start with a prefix?"""
        try:
            keys = list(keys)
        except Exception:
            keys = list(keys)
        if not keys:
            return False
        sample = keys[: min(sample_size, len(keys))]
        if not sample:
            return False
        hits = sum(1 for k in sample if isinstance(k, str) and k.startswith(prefix))
        return (hits / float(len(sample))) >= threshold

    def _adapt_state_dict_keys_for_target(target, state_dict):
        """Adapt checkpoint state_dict keys to match target.state_dict() keys.

        Common case: checkpoints saved from DDP/DataParallel have 'module.' prefix.
        When loading into a non-wrapped model (or vice-versa), keys won't match and
        you'll see huge missing_keys/unexpected_keys.
        """
        if not isinstance(state_dict, dict) or target is None:
            return state_dict
        if not hasattr(target, "state_dict"):
            return state_dict

        try:
            target_keys = list(target.state_dict().keys())
        except Exception:
            return state_dict

        ckpt_keys = list(state_dict.keys())
        if not ckpt_keys or not target_keys:
            return state_dict

        ckpt_has_module = _should_use_prefix(ckpt_keys, "module.")
        target_has_module = _should_use_prefix(target_keys, "module.")

        if ckpt_has_module and not target_has_module:
            # Strip 'module.'
            adapted = {}
            for k, v in state_dict.items():
                if isinstance(k, str) and k.startswith("module."):
                    adapted[k[len("module.") :]] = v
                else:
                    adapted[k] = v
            logger.info("Adapting checkpoint keys: stripping 'module.' prefix for loading.")
            return adapted

        if (not ckpt_has_module) and target_has_module:
            # Add 'module.'
            adapted = {}
            for k, v in state_dict.items():
                if isinstance(k, str) and not k.startswith("module."):
                    adapted["module." + k] = v
                else:
                    adapted[k] = v
            logger.info("Adapting checkpoint keys: adding 'module.' prefix for loading.")
            return adapted

        return state_dict
    # look for a checkpoint in exp repository
    if isinstance(ckp_paths, list):
        for ckp_path in ckp_paths:
            if os.path.isfile(ckp_path):
                break
    else:
        ckp_path = ckp_paths

    if not os.path.isfile(ckp_path):
        return

    logger.info("Found checkpoint at {}".format(ckp_path))

    # open checkpoint file
    # Safe map_location even when torch.distributed isn't initialized.
    map_location = "cpu"
    if torch.cuda.is_available():
        try:
            if dist.is_available() and dist.is_initialized() and torch.cuda.device_count() > 0:
                map_location = "cuda:" + str(dist.get_rank() % torch.cuda.device_count())
            else:
                map_location = "cuda:0"
        except Exception:
            map_location = "cuda:0"

    checkpoint = torch.load(
        ckp_path,
        map_location=map_location,
        weights_only=False,
    )

    # key is what to look for in the checkpoint file
    # value is the object to load
    # example: {'state_dict': model}
    for key, value in kwargs.items():
        if key in checkpoint and value is not None:
            try:
                # Preferred: try loading with strict=False when supported
                try:
                    payload = checkpoint[key]
                    if key == "state_dict":
                        payload = _adapt_state_dict_keys_for_target(value, payload)
                    msg = value.load_state_dict(payload, strict=False)
                    # Some load_state_dict return a message
                    if msg is not None:
                        print(msg)
                except TypeError:
                    # Older or wrapped optimizers (like our LARC) might not accept strict kwarg
                    try:
                        payload = checkpoint[key]
                        if key == "state_dict":
                            payload = _adapt_state_dict_keys_for_target(value, payload)
                        msg = value.load_state_dict(payload)
                        if msg is not None:
                            print(msg)
                    except ValueError as ve:
                        # Happens when optimizer param groups differ between runs (common when
                        # adding/removing parameters). Warn and skip restoring optimizer state.
                        logger.warning(
                            "Could not load state for '{}' due to ValueError: {}. Skipping this key.".format(key, ve)
                        )
            except Exception as e:
                # Catch any other exception to avoid crashing the restart.
                logger.warning(
                    "Unexpected error while loading '{}' from checkpoint '{}': {}. Skipping.".format(key, ckp_path, e)
                )
            else:
                logger.info("=> loaded {} from checkpoint '{}'".format(key, ckp_path))
        else:
            logger.warning(
                "=> failed to load {} from checkpoint '{}'".format(key, ckp_path)
            )

    # re load variable important for the run
    if run_variables is not None:
        for var_name in run_variables:
            if var_name in checkpoint:
                run_variables[var_name] = checkpoint[var_name]


def fix_random_seeds(seed=31):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


class AverageMeter(object):
    """computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
