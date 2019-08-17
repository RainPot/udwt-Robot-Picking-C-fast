import os
from models.rrnet import RRNet
import numpy as np
import torch
import time
from ext.nms.nms_wrapper import nms


class CTNetTestOperator(object):
    def __init__(self, cfg):
        self.cfg = cfg

        self.model = RRNet(cfg).cuda()
        state_dict = torch.load(self.cfg.model_path, map_location='cpu')
        self.model.load_state_dict(state_dict)
        self.model.eval()

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
    def save_result(file_path, pred_bbox):
        pred_bbox = torch.clamp(pred_bbox, min=0.)
        # Check Here
        with open(file_path, 'w') as f:
            for i in range(pred_bbox.size()[0]):
                bbox = pred_bbox[i]
                line = '%f,%f,%f,%f,%.4f,%d,-1,-1\n' % (
                    float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]),
                    float(bbox[4]), int(bbox[5])
                )
                f.write(line)

    def evaluation_process(self, dataloader):
        with torch.no_grad():
            torch.cuda.synchronize()
            st = time.time()
            for img, name, ratio in dataloader:
                img = img.cuda()
                out = self.model(img)
                raw_pred_bbox = self.generate_bbox(out)
                pred_bbox = raw_pred_bbox[raw_pred_bbox[:, 4] > 0.01]
                if pred_bbox.size(0) == 0:
                    pred_bbox = raw_pred_bbox[0:1, :]
                _, idx = torch.sort(pred_bbox[:, 4], descending=True)
                pred_bbox = pred_bbox[idx]
                pred_bbox[:, 0] *= float(ratio[1])
                pred_bbox[:, 1] *= float(ratio[0])
                pred_bbox[:, 2] *= float(ratio[1])
                pred_bbox[:, 3] *= float(ratio[0])
                pred_bbox = pred_bbox.cpu()
                pred_bbox = self._ext_nms(pred_bbox)

                # Check Here.
                _, idx = torch.sort(pred_bbox[:, 4], descending=True)
                pred_bbox = pred_bbox[idx]
                file_path = os.path.join(self.cfg.result_dir, name[0] + '.txt')
                self.save_result(file_path, pred_bbox)

                del out
                del pred_bbox
            torch.cuda.synchronize()
            fps = 1. / ((time.time() - st) / len(dataloader))
            print('=> Evaluation Done! FPS: {}'.format(fps))
