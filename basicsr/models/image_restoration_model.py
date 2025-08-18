import importlib
import numpy as np
import jittor as jt
from jittor import nn

from copy import deepcopy
from os import path as osp
from collections import OrderedDict

import os
import random
from functools import partial

from basicsr.models.archs import define_network
from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger, imwrite, tensor2img

loss_module = importlib.import_module("basicsr.models.losses")
metric_module = importlib.import_module("basicsr.metrics")


class Beta:
    """
    A minimal, torch.distributions-like Beta for Jittor.

    Notes
    -----
    - Uses the Gamma-ratio trick: Beta(a,b) = X / (X + Y),
    with X ~ Gamma(a, 1), Y ~ Gamma(b, 1).
    - We implement sampling via NumPy and wrap back into jt.Var.
    (Good enough for mixup; reparameterized gradients are not provided.)
    - rsample() here is the same as sample(); it does NOT provide
    implicit reparameterization gradients like PyTorch.
    """

    def __init__(self, alpha: jt.Var, beta: jt.Var, dtype="float32", device="cuda"):
        self.alpha_np = alpha.numpy().astype(np.float32)
        self.beta_np = beta.numpy().astype(np.float32)

        # Jittor “tensors” for convenience / printing
        self.alpha = jt.array(self.alpha_np).astype(dtype)
        self.beta = jt.array(self.beta_np).astype(dtype)

        self.dtype = dtype
        self.device = device  # kept for API similarity; Jittor manages device globally

    @property
    def batch_shape(self):
        # Torch-style: shape that alpha/beta broadcast to
        return np.broadcast(self.alpha_np, self.beta_np).shape

    @property
    def event_shape(self):
        # Scalar distribution
        return ()

    def _np_beta_sample(self, out_shape):
        # Broadcast alpha/beta to target shape, then sample elementwise via gamma
        target_shape = out_shape if out_shape is not None else self.batch_shape or ()
        # np.random.gamma supports array-shaped shape parameters
        x = np.random.gamma(shape=self.alpha_np, scale=1.0, size=target_shape)
        y = np.random.gamma(shape=self.beta_np, scale=1.0, size=target_shape)
        lam = x / (x + y + 1e-12)
        return lam.astype(np.float32)

    def sample(self, shape=None):
        """Sample without gradients; returns jt.Var (stop_grad)."""
        lam = self._np_beta_sample(shape)
        var = jt.array(lam).astype(self.dtype)
        return var.stop_grad()  # make it explicit: no grad

    def rsample(self, shape=None):
        """
        Torch-compatible name; identical to sample() here.
        (No implicit reparameterization gradients provided.)
        """
        return self.sample(shape)

    def to(self, dtype=None, device=None):
        """Lightweight 'to' for API familiarity."""
        if dtype is not None:
            self.dtype = dtype
            self.alpha = self.alpha.astype(dtype)
            self.beta = self.beta.astype(dtype)
        if device is not None:
            self.device = device  # Jittor uses global flags; kept for symmetry
        return self

    def __repr__(self):
        return f"Beta(alpha={self.alpha.numpy()}, beta={self.beta.numpy()}, dtype={self.dtype}, device={self.device})"


class BetaDict:
    def __init__(self):
        pass


class Mixing_Augment:
    def __init__(self, mixup_beta, use_identity, device):
        self.dist = Beta(jt.array([mixup_beta]), jt.array([mixup_beta]))
        self.device = device
        self.use_identity = use_identity
        self.augments = [self.mixup]

    def mixup(self, target, input_):
        lam = self.dist.rsample((1, 1)).item()
        r_index = jt.randperm(target.size(0)).to(self.device)

        target = lam * target + (1 - lam) * target[r_index, :]
        input_ = lam * input_ + (1 - lam) * input_[r_index, :]

        return target, input_

    def __call__(self, target, input_):
        if self.use_identity:
            augment = random.randint(0, len(self.augments))
            if augment < len(self.augments):
                target, input_ = self.augments[augment](target, input_)
        else:
            augment = random.randint(0, len(self.augments) - 1)
            target, input_ = self.augments[augment](target, input_)
        return target, input_


class ImageCleanModel(BaseModel):
    """Base Deblur model for single image deblur."""

    def __init__(self, opt):
        super(ImageCleanModel, self).__init__(opt)

        # define network

        self.mixing_flag = self.opt["train"]["mixing_augs"].get("mixup", False)
        if self.mixing_flag:
            mixup_beta = self.opt["train"]["mixing_augs"].get("mixup_beta", 1.2)
            use_identity = self.opt["train"]["mixing_augs"].get("use_identity", False)
            self.mixing_augmentation = Mixing_Augment(
                mixup_beta, use_identity, self.device
            )

        self.net_g: nn.Module = define_network(deepcopy(opt["network_g"]))
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt["path"].get("pretrain_network_g", None)
        if load_path is not None:
            self.load_network(
                self.net_g,
                load_path,
                self.opt["path"].get("strict_load_g", True),
                param_key=self.opt["path"].get("param_key", "params"),
            )

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt["train"]

        self.ema_decay = train_opt.get("ema_decay", 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f"Use Exponential Moving Average with decay: {self.ema_decay}")
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = define_network(self.opt["network_g"]).to(self.device)
            # load pretrained model
            load_path = self.opt["path"].get("pretrain_network_g", None)
            if load_path is not None:
                self.load_network(
                    self.net_g_ema,
                    load_path,
                    self.opt["path"].get("strict_load_g", True),
                    "params_ema",
                )
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get("pixel_opt"):
            pixel_type = train_opt["pixel_opt"].pop("type")
            cri_pix_cls = getattr(loss_module, pixel_type)
            self.cri_pix = cri_pix_cls(**train_opt["pixel_opt"])
        else:
            raise ValueError("pixel loss are None.")

        if train_opt.get("fft_loss_opt"):
            fft_type = train_opt["fft_loss_opt"].pop("type")
            cri_fft_cls = getattr(loss_module, fft_type)
            self.cri_fft = cri_fft_cls(**train_opt["fft_loss_opt"])
        else:
            self.cri_fft = None

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt["train"]
        optim_params = []

        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f"Params {k} will not be optimized.")

        optim_type = train_opt["optim_g"].pop("type")
        if optim_type == "Adam":
            self.optimizer_g = jt.optim.Adam(optim_params, **train_opt["optim_g"])
        elif optim_type == "AdamW":
            self.optimizer_g = jt.optim.AdamW(optim_params, **train_opt["optim_g"])
        else:
            raise NotImplementedError(f"optimizer {optim_type} is not supperted yet.")
        self.optimizers.append(self.optimizer_g)

    def feed_train_data(self, data):
        self.lq = data["lq"].to(self.device)
        if "gt" in data:
            self.gt = data["gt"].to(self.device)

        if self.mixing_flag:
            self.gt, self.lq = self.mixing_augmentation(self.gt, self.lq)

    def feed_data(self, data):
        self.lq = data["lq"].to(self.device)
        if "gt" in data:
            self.gt = data["gt"].to(self.device)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        preds = self.net_g(self.lq)

        if not isinstance(preds, list):
            preds = [preds]

        self.output = preds[-1]

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = 0.0
            for pred in preds:
                l_pix += self.cri_pix(pred, self.gt)

            # print('l pix ... ', l_pix)
            l_total += l_pix
            loss_dict["l_pix"] = l_pix

        # fft loss
        if self.cri_fft:
            l_fft = self.cri_fft(preds[-1], self.gt)
            l_total += l_fft
            loss_dict["l_fft"] = l_fft

        # Ensure l_total is a scalar Var before backward
        if not isinstance(l_total, jt.Var):
            l_total = jt.float32(l_total)
        l_total = l_total.mean()

        self.optimizer_g.pre_step(l_total)

        if self.opt["train"]["use_grad_clip"] and current_iter > 1:
            self.optimizer_g.clip_grad_norm(0.01)
        self.optimizer_g.step(l_total)

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def pad_test(self, window_size):
        scale = self.opt.get("scale", 1)
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = self.lq.size()
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        img = jt.nn.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), "reflect")
        self.nonpad_test(img)
        _, _, h, w = self.output.size()
        self.output = self.output[
            :, :, 0 : h - mod_pad_h * scale, 0 : w - mod_pad_w * scale
        ]

    def nonpad_test(self, img=None):
        if img is None:
            img = self.lq
        if hasattr(self, "net_g_ema"):
            self.net_g_ema.eval()
            with jt.no_grad():
                pred = self.net_g_ema(img)
            if isinstance(pred, list):
                pred = pred[-1]
            self.output = pred
        else:
            self.net_g.eval()
            with jt.no_grad():
                pred = self.net_g(img)
            if isinstance(pred, list):
                pred = pred[-1]
            self.output = pred
            self.net_g.train()

    def dist_validation(
        self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image
    ):
        if os.environ["LOCAL_RANK"] == "0":
            return self.nondist_validation(
                dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image
            )
        else:
            return 0.0

    def nondist_validation(
        self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image
    ):
        dataset_name = dataloader.dataset.opt["name"]
        with_metrics = self.opt["val"].get("metrics") is not None
        if with_metrics:
            self.metric_results = {
                metric: 0 for metric in self.opt["val"]["metrics"].keys()
            }
        # pbar = tqdm(total=len(dataloader), unit='image')

        window_size = self.opt["val"].get("window_size", 0)

        if window_size:
            test = partial(self.pad_test, window_size)
        else:
            test = self.nonpad_test

        cnt = 0

        for _, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data["lq_path"][0]))[0]

            self.feed_data(val_data)
            test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals["result"]], rgb2bgr=rgb2bgr)
            if "gt" in visuals:
                gt_img = tensor2img([visuals["gt"]], rgb2bgr=rgb2bgr)
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output

            if save_img:
                if self.opt["is_train"]:
                    save_img_path = osp.join(
                        self.opt["path"]["visualization"],
                        img_name,
                        f"{img_name}_{current_iter}.png",
                    )

                    save_gt_img_path = osp.join(
                        self.opt["path"]["visualization"],
                        img_name,
                        f"{img_name}_{current_iter}_gt.png",
                    )
                else:
                    save_img_path = osp.join(
                        self.opt["path"]["visualization"],
                        dataset_name,
                        f"{img_name}.png",
                    )
                    save_gt_img_path = osp.join(
                        self.opt["path"]["visualization"],
                        dataset_name,
                        f"{img_name}_gt.png",
                    )

                imwrite(sr_img, save_img_path)
                imwrite(gt_img, save_gt_img_path)

            if with_metrics:
                # calculate metrics
                opt_metric = deepcopy(self.opt["val"]["metrics"])
                if use_image:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop("type")
                        self.metric_results[name] += getattr(metric_module, metric_type)(
                            sr_img, gt_img, **opt_
                        )
                else:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop("type")
                        self.metric_results[name] += getattr(metric_module, metric_type)(
                            visuals["result"], visuals["gt"], **opt_
                        )

            cnt += 1

        current_metric = 0.0
        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= cnt
                current_metric = self.metric_results[metric]

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
        return current_metric

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f"Validation {dataset_name},\t"
        for metric, value in self.metric_results.items():
            log_str += f"\t # {metric}: {value:.4f}"
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f"metrics/{metric}", value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict["lq"] = self.lq.detach().cpu()
        out_dict["result"] = self.output.detach().cpu()
        if hasattr(self, "gt"):
            out_dict["gt"] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if self.ema_decay > 0:
            self.save_network(
                [self.net_g, self.net_g_ema],
                "net_g",
                current_iter,
                param_key=["params", "params_ema"],
            )
        else:
            self.save_network(self.net_g, "net_g", current_iter)
        self.save_training_state(epoch, current_iter)
