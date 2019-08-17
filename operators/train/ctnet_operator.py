import os
import random
from models.rrnet import RRNet
from modules.loss.focalloss import FocalLossHM
import numpy as np
from modules.loss.regl1loss import RegL1Loss
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torch.nn.parallel import DistributedDataParallel

import os
from datasets import make_train_dataloader
from utils.vis.logger import Logger
from datasets.transforms.functional import denormalize
from utils.vis.annotations import visualize
from ext.nms.nms_wrapper import nms


class CTNetTrainOperator(object):
    def __init__(self, cfg):
        self.cfg = cfg

        random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed(cfg.seed)

        model = RRNet(cfg).cuda(cfg.Distributed.gpu_id)
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        self.optimizer = optim.Adam(model.parameters(), lr=cfg.Train.lr)
        self.lr_sch = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=cfg.Train.lr_milestones, gamma=0.1)
        self.training_loader = make_train_dataloader(cfg)

        self.model = DistributedDataParallel(model, find_unused_parameters=True, device_ids=[self.cfg.Distributed.gpu_id])

        self.hm_focal_loss = FocalLossHM()
        self.l1_loss = RegL1Loss()

        self.main_proc_flag = cfg.Distributed.gpu_id == 0

    def criterion(self, outs, targets):
        s1_hms, s1_whs, s1_offsets, _ = outs
        gt_hms, gt_whs, gt_inds, gt_offsets, gt_reg_masks, gt_annos = targets
        hm_loss = 0
        wh_loss = 0
        off_loss = 0

        # I. Stage 1
        s1_hm = s1_hms
        s1_wh = s1_whs
        s1_offset = s1_offsets
        s1_hm = torch.clamp(torch.sigmoid(s1_hm), min=1e-4, max=1-1e-4)
        # Heatmap Loss
        hm_loss += self.hm_focal_loss(s1_hm, gt_hms) / self.cfg.Model.num_stacks
        # WH Loss
        wh_loss += self.l1_loss(s1_wh, gt_reg_masks, gt_inds, gt_whs) / self.cfg.Model.num_stacks
        # OffSet Loss
        off_loss += self.l1_loss(s1_offset, gt_reg_masks, gt_inds, gt_offsets) / self.cfg.Model.num_stacks

        return hm_loss, wh_loss, off_loss

    def training_process(self):
        if self.main_proc_flag:
            logger = Logger(self.cfg)

        self.model.train()

        total_loss = 0
        total_hm_loss = 0
        total_wh_loss = 0
        total_off_loss = 0

        for step in range(self.cfg.Train.iter_num):
            self.lr_sch.step()
            self.optimizer.zero_grad()
            
            try:
                imgs, annos, gt_hms, gt_whs, gt_inds, gt_offsets, gt_reg_masks, names = self.training_loader.get_batch()
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print("WARNING: ran out of memory with exception at step {}.".format(step))
                continue

            outs = self.model(imgs)
            targets = gt_hms, gt_whs, gt_inds, gt_offsets, gt_reg_masks, annos
            hm_loss, wh_loss, offset_loss = self.criterion(outs, targets)

            loss = hm_loss + (0.1 * wh_loss) + offset_loss
            loss.backward()
            self.optimizer.step()

            total_loss += float(loss)
            total_hm_loss += float(hm_loss)
            total_wh_loss += float(wh_loss)
            total_off_loss += float(offset_loss)

            if self.main_proc_flag:
                if step % self.cfg.Train.print_interval == self.cfg.Train.print_interval - 1:
                    # Loss
                    for param_group in self.optimizer.param_groups:
                        lr = param_group['lr']
                    log_data = {'scalar': {
                        'train/total_loss': total_loss / self.cfg.Train.print_interval,
                        'train/hm_loss': total_hm_loss / self.cfg.Train.print_interval,
                        'train/wh_loss': total_wh_loss / self.cfg.Train.print_interval,
                        'train/off_loss': total_off_loss / self.cfg.Train.print_interval,
                        'train/lr': lr
                    }}

                    # Generate bboxs
                    pred_bbox = self.generate_bbox(outs, batch_idx=0)
                    pred_bbox = pred_bbox[pred_bbox[:, 4] > 0.1]
                    pred_bbox = self._ext_nms(pred_bbox)
                    # Visualization
                    img = (denormalize(imgs[0].cpu(), mean=self.cfg.mean, std=self.cfg.std).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    #
                    pred_on_img = visualize(img.copy(), pred_bbox, xywh=True, with_score=True)
                    gt_img = visualize(img.copy(), annos[0, :, :6], xywh=True)

                    pred_on_img = torch.from_numpy(pred_on_img).permute(2, 0, 1).unsqueeze(0).float() / 255.
                    gt_on_img = torch.from_numpy(gt_img).permute(2, 0, 1).unsqueeze(0).float() / 255.
                    log_data['imgs'] = {'Train': [pred_on_img, gt_on_img]}
                    logger.log(log_data, step)

                    total_loss = 0
                    total_hm_loss = 0
                    total_wh_loss = 0
                    total_off_loss = 0

                if step % self.cfg.Train.checkpoint_interval == self.cfg.Train.checkpoint_interval - 1 or \
                        step == self.cfg.Train.iter_num - 1:
                    self.save_ckp(self.model.module, step, logger.log_dir)

    def generate_bbox(self, outs, batch_idx=0):
        s1_hms, s1_whs, s1_offsets, bxyxy = outs
        xyxy = bxyxy[batch_idx]
        xyxy[:, 0:4] *= self.cfg.scale_factor
        xyxy[:, 2:4] -= xyxy[:, 0:2]
        xyxy[:, 5] += 1
        return xyxy

    @staticmethod
    def _ext_nms(pred_bbox, per_cls=True):
        if pred_bbox.size(0) == 0:
            return pred_bbox
        keep_bboxs = []
        if per_cls:
            cls_unique = pred_bbox[:, 5].unique()
            for cls in cls_unique:
                cls_idx = pred_bbox[:, 5] == cls
                bbox_for_nms = pred_bbox[cls_idx].detach().cpu().numpy()
                bbox_for_nms[:, 2] = bbox_for_nms[:, 0] + bbox_for_nms[:, 2]
                bbox_for_nms[:, 3] = bbox_for_nms[:, 1] + bbox_for_nms[:, 3]
                keep_bbox = nms(bbox_for_nms, thresh=0.3)
                keep_bboxs.append(keep_bbox)
            keep_bboxs = np.concatenate(keep_bboxs, axis=0)
        else:
            bbox_for_nms = pred_bbox.detach().cpu().numpy()
            bbox_for_nms[:, 2] = bbox_for_nms[:, 0] + bbox_for_nms[:, 2]
            bbox_for_nms[:, 3] = bbox_for_nms[:, 1] + bbox_for_nms[:, 3]
            keep_bboxs = nms(bbox_for_nms, thresh=0.3)
        keep_bboxs[:, 2:4] -= keep_bboxs[:, 0:2]
        return torch.from_numpy(keep_bboxs)

    @staticmethod
    def save_ckp(models, step, path):
        """
        Save checkpoint of the model.
        :param models: nn.Module
        :param step: step of the checkpoint.
        :param path: save path.
        """
        torch.save(models.state_dict(), os.path.join(path, 'ckp-{}.pth'.format(step)))
