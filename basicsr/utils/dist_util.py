# jittor_dist_utils.py
# Modified from your PyTorch version to Jittor (MPI-based)

import os
import functools
import subprocess
import multiprocessing as mp

import jittor as jt
# Jittor 的分布式接口以 MPI 为核心（自动检测 mpirun/srun 环境）
# 常用标志：
#   jt.in_mpi     -> 是否处于分布式环境
#   jt.rank       -> 当前进程 rank（单卡为 0）
#   jt.world_size -> 进程总数（单卡为 1）
# 常用装饰器 / 工具：
#   @jt.single_process_scope() 让包裹代码在单进程上执行
#   Var.mpi_all_reduce("sum"/"mean") 等操作用于全局统计


def init_dist(launcher, backend="nccl", **kwargs):
    """
    Initialize distributed env for Jittor.
    """
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method("spawn")

    if launcher == "pytorch":
        _init_dist_env_from_env(**kwargs)
    elif launcher == "slurm":
        _init_dist_slurm(**kwargs)
    else:
        raise ValueError(f"Invalid launcher type: {launcher}")

    try:
        jt.flags.use_cuda = 1
    except Exception:
        pass


def _init_dist_env_from_env(**kwargs):
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")

    port = kwargs.get("port", None)
    if port is not None:
        os.environ["MASTER_PORT"] = str(port)
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")


def _init_dist_slurm(port=None):
    proc_id = int(os.environ["SLURM_PROCID"])
    ntasks = int(os.environ["SLURM_NTASKS"])
    node_list = os.environ["SLURM_NODELIST"]

    addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1").strip()

    # MASTER_PORT
    if port is not None:
        os.environ["MASTER_PORT"] = str(port)
    else:
        os.environ.setdefault("MASTER_PORT", "29500")

    os.environ["MASTER_ADDR"] = addr
    os.environ["WORLD_SIZE"] = str(ntasks)
    os.environ["RANK"] = str(proc_id)

    os.environ.setdefault("LOCAL_RANK", str(proc_id))


def get_dist_info():
    try:
        if getattr(jt, "in_mpi", False):
            return int(jt.rank), int(jt.world_size)
    except Exception:
        pass
    return 0, 1


def master_only(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        rank, _ = get_dist_info()
        if rank == 0:
            return func(*args, **kwargs)

    return wrapper
