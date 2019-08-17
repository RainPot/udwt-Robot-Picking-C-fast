import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
from utils.model_tools import get_backbone
from detectors.detector import CenterNetDetector
from detectors.detector import FasterRCNNDetector
from ext.nms.nms_wrapper import soft_nms


class RRNet(nn.Module):
    def __init__(self, cfg):
        super(RRNet, self).__init__()
        self.num_stacks = cfg.Model.num_stacks
        self.num_classes = cfg.num_classes
        self.nms_type = cfg.Model.nms_type_for_stage1
        self.nms_per_class = cfg.Model.nms_per_class_for_stage1

        self.backbone = get_backbone(cfg.Model.backbone, num_stacks=self.num_stacks)
        self.hm = CenterNetDetector(planes=self.num_classes, hm=True)
        self.wh = CenterNetDetector(planes=2)
        self.offset_reg = CenterNetDetector(planes=2)
        self.use_rr = cfg.Model.use_rr
        if cfg.Model.use_rr:
            self.head_detector = FasterRCNNDetector()

    def forward(self, x, k=1500):
        # I. Forward Backbone
        pre_feat = self.backbone(x)
        # II. Forward Stage 1 to generate heatmap, wh and offset.
        hms, whs, offsets = self.forward_stage1(pre_feat)
        # III. Generate the true xywh for Stage 1.
        bboxs = self.transform_bbox(self._ctnet_nms(hms), whs, offsets, k)  # (bs, k, 6)
        if not self.use_rr:
            return hms, whs, offsets, bboxs
        # IV. Stage 2.
        bxyxys = []
        scores = []
        clses = []
        for b_idx in range(bboxs.size(0)):
            # Do nms
            bbox = bboxs[b_idx]
            xyxy = bbox[:, :4]
            scores.append(bbox[:, 4])
            clses.append(bbox[:, 5])
            batch_idx = torch.ones((xyxy.size(0), 1), device=xyxy.device) * b_idx
            bxyxy = torch.cat((batch_idx, xyxy), dim=1)
            bxyxys.append(bxyxy)
        bxyxys = torch.cat(bxyxys, dim=0)
        scores = torch.cat(scores, dim=0)
        clses = torch.cat(clses, dim=0)
        #  Generate the ROIAlign features.
        roi_feat = torchvision.ops.roi_align(torch.relu(pre_feat), bxyxys, (3, 3))
        # Forward Stage 2 to predict and wh offset.
        stage2_reg = self.forward_stage2(roi_feat)
        return hms, whs, offsets, stage2_reg, bxyxys, scores, clses

    @staticmethod
    def _ctnet_nms(heat, kernel=3):
        heat = torch.sigmoid(heat)
        pad = (kernel - 1) // 2

        hmax = F.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
        keep = (hmax == heat).float()
        return heat * keep

    @staticmethod
    def _gather_feat(feat, ind, mask=None):
        dim = feat.size(2)
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        feat = feat.gather(1, ind)
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
        return feat

    def _topk(self, scores, k=1500):
        batch, cat, height, width = scores.size()

        topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), k)

        topk_inds = topk_inds % (height * width)
        topk_ys = (topk_inds / width).int().float()
        topk_xs = (topk_inds % width).int().float()

        topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), k)
        topk_clses = (topk_ind / k).int()
        topk_inds = self._gather_feat(
            topk_inds.view(batch, -1, 1), topk_ind).view(batch, k)
        topk_ys = self._gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, k)
        topk_xs = self._gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, k)

        return topk_score, topk_inds, topk_clses, topk_ys, topk_xs

    def _transpose_and_gather_feat(self, feat, ind):
        feat = feat.permute(0, 2, 3, 1).contiguous()
        feat = feat.view(feat.size(0), -1, feat.size(3))
        feat = self._gather_feat(feat, ind)
        return feat

    def transform_bbox(self, hm, wh, offset, k=250):
        batchsize, cls_num, h, w = hm.size()

        scores, inds, clses, ys, xs = self._topk(hm, k)

        offset = self._transpose_and_gather_feat(offset, inds)
        offset = offset.view(batchsize, k, 2)
        xs = xs.view(batchsize, k, 1) + offset[:, :, 0:1]
        ys = ys.view(batchsize, k, 1) + offset[:, :, 1:2]
        wh = self._transpose_and_gather_feat(wh, inds).clamp(min=0)

        wh = wh.view(batchsize, k, 2)
        clses = clses.view(batchsize, k, 1).float()
        scores = scores.view(batchsize, k, 1)

        pred_x = (xs - wh[..., 0:1] / 2)
        pred_y = (ys - wh[..., 1:2] / 2)
        pred_w = wh[..., 0:1]
        pred_h = wh[..., 1:2]
        pred = torch.cat([pred_x, pred_y, pred_w + pred_x, pred_h + pred_y, scores, clses], dim=2)
        return pred

    def forward_stage1(self, feats):
        hm = self.hm(feats)
        wh = self.wh(feats)
        offset = self.offset_reg(feats)
        return hm, wh, offset

    def forward_stage2(self, feats,):
        stage2_reg = self.head_detector(feats)
        return stage2_reg
