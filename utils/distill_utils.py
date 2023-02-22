import torch
import numpy as np


def bbox_overlaps_batch(anchors, gt_boxes, img_size):
    # anchors [N, 4]
    # gt_boxes [b, K, 6]
    batch_size = gt_boxes.size(0)
    if anchors.dim() == 2:
        N = anchors.size(0)
        K = gt_boxes.size(1)

        # [batch, N, 4]
        anchors = anchors.view(1, N, 4).expand(batch_size, N, 4).contiguous()
        # [batch, K, 4]
        gt_boxes = gt_boxes[:, :, 2:].contiguous()
        # [batch, K]
        gt_boxes_x = gt_boxes[:, :, 2] - gt_boxes[:, :, 0] + 1
        # [batch, K]
        gt_boxes_y = gt_boxes[:, :, 3] - gt_boxes[:, :, 1] + 1
        # 目标框的面积 [batch, 1, K]
        gt_boxes_area = (gt_boxes_x * gt_boxes_y).view(batch_size, 1, K)

        # [batch, N]
        anchors_boxes_x = (anchors[:, :, 2] - anchors[:, :, 0] + 1)
        # [batch, N]
        anchors_boxes_y = (anchors[:, :, 3] - anchors[:, :, 1] + 1)
        # [batch, N, 1]
        anchors_area = (anchors_boxes_x * anchors_boxes_y).view(batch_size, N, 1)

        gt_area_zero = (gt_boxes_x == 1) & (gt_boxes_y == 1)
        anchors_area_zero = (anchors_boxes_x == 1) & (anchors_boxes_y == 1)

        boxes = anchors.view(batch_size, N, 1, 4).expand(batch_size, N, K, 4)
        query_boxes = gt_boxes.view(batch_size, 1, K, 4).expand(batch_size, N, K, 4)

        iw = (torch.min(boxes[:, :, :, 2], query_boxes[:, :, :, 2]) -
              torch.max(boxes[:, :, :, 0], query_boxes[:, :, :, 0] + 1))
        iw[iw < 0] = 0

        ih = (torch.min(boxes[:, :, :, 3], query_boxes[:, :, :, 3]) -
              torch.max(boxes[:, :, :, 1], query_boxes[:, :, :, 1]) + 1)
        ih[ih < 0] = 0
        ua = anchors_area + gt_boxes_area - (iw * ih)
        overlaps = iw * ih / ua

        overlaps.masked_fill_(gt_area_zero.view(batch_size, 1, K).expand(batch_size, N, K), 0)
        overlaps.masked_fill_(anchors_area_zero.view(batch_size, N, 1).expand(batch_size, N, K), -1)
    elif anchors.dim() == 3:
        N = anchors.size(1)
        K = gt_boxes.size(1)

        if anchors.size(2) == 4:
            anchors = anchors[:, :, :4].contiguous()
        else:
            anchors = anchors[:, :, 1:5].contiguous()

        gt_boxes = gt_boxes[:, :, :4].contiguous()

        gt_boxes_x = (gt_boxes[:, :, 2] - gt_boxes[:, :, 0] + 1)
        gt_boxes_y = (gt_boxes[:, :, 3] - gt_boxes[:, :, 1] + 1)
        gt_boxes_area = (gt_boxes_x * gt_boxes_y).view(batch_size, 1, K)

        anchors_boxes_x = (anchors[:, :, 2] - anchors[:, :, 0] + 1)
        anchors_boxes_y = (anchors[:, :, 3] - anchors[:, :, 1] + 1)

        anchors_area = (anchors_boxes_x * anchors_boxes_y).view(batch_size, N, 1)

        gt_area_zero = (gt_boxes_x == 1) & (gt_boxes_y == 1)
        anchors_area_zero = (anchors_boxes_x == 1) & (anchors_boxes_y == 1)

        boxes = anchors.view(batch_size, N, 1, 4).expand(batch_size, N, K, 4)
        query_boxes = gt_boxes.view(
            batch_size, 1, K, 4).expand(batch_size, N, K, 4)

        iw = (torch.min(boxes[:, :, :, 2], query_boxes[:, :, :, 2]) -
              torch.max(boxes[:, :, :, 0], query_boxes[:, :, :, 0]) + 1)
        iw[iw < 0] = 0

        ih = (torch.min(boxes[:, :, :, 3], query_boxes[:, :, :, 3]) -
              torch.max(boxes[:, :, :, 1], query_boxes[:, :, :, 1]) + 1)
        ih[ih < 0] = 0
        ua = anchors_area + gt_boxes_area - (iw * ih)

        overlaps = iw * ih / ua

        # mask the overlap here.
        overlaps.masked_fill_(gt_area_zero.view(
            batch_size, 1, K).expand(batch_size, N, K), 0)
        overlaps.masked_fill_(anchors_area_zero.view(
            batch_size, N, 1).expand(batch_size, N, K), -1)
    else:
        raise ValueError("anchors input dim is not correct")
    overlap_shape = overlaps.shape
    return overlaps


def generate_anchors(base_size, anchors):
    base_anchor = np.array([0, 0, base_size - 1, base_size - 1])
    x_ctr, y_ctr = _whctrs(base_anchor)
    aim_anchor = []
    for anchor in anchors:
        x1 = x_ctr - 0.5 * anchor[0] * base_size
        y1 = y_ctr - 0.5 * anchor[1] * base_size
        x2 = x_ctr + 0.5 * anchor[0] * base_size
        y2 = y_ctr + 0.5 * anchor[1] * base_size
        aim_anchor.append([x1, y1, x2, y2])
    return np.array(aim_anchor)

# 计算特征图中的一个像素点在原图中的中心点坐标


def _whctrs(anchor):
    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return x_ctr, y_ctr


def make_gt_boxes(gt_boxes, max_num_box, batch, img_size):
    new_gt_boxes = []
    for i in range(batch):
        # 获取第i个batch的所有真实框
        boxes = gt_boxes[gt_boxes[:, 0] == i]
        # 真实框的个数
        num_boxes = boxes.size(0)
        if num_boxes < max_num_box:
            gt_boxes_padding = torch.zeros([max_num_box, gt_boxes.size(1)], dtype=torch.float)
            gt_boxes_padding[:num_boxes, :] = boxes
        else:
            gt_boxes_padding = boxes[:max_num_box]
        new_gt_boxes.append(gt_boxes_padding.unsqueeze(0))
    new_gt_boxes = torch.cat(new_gt_boxes)
    # transfer [x, y, w, h] to [x1, y1, x2, y2]
    new_gt_boxes_aim = torch.zeros(size=new_gt_boxes.size())
    new_gt_boxes_aim[:, :, 2] = (new_gt_boxes[:, :, 2] - 0.5 * new_gt_boxes[:, :, 4]) * img_size[1]
    new_gt_boxes_aim[:, :, 3] = (new_gt_boxes[:, :, 3] - 0.5 * new_gt_boxes[:, :, 5]) * img_size[0]
    new_gt_boxes_aim[:, :, 4] = (new_gt_boxes[:, :, 2] + 0.5 * new_gt_boxes[:, :, 4]) * img_size[1]
    new_gt_boxes_aim[:, :, 5] = (new_gt_boxes[:, :, 3] + 0.5 * new_gt_boxes[:, :, 5]) * img_size[0]
    return new_gt_boxes_aim


def getMask(batch_size, gt_boxes, img_size, feat, anchors, max_num_box, device):
    # [b, K, 4]
    gt_boxes = make_gt_boxes(gt_boxes, max_num_box, batch_size, img_size)
    # 原图相对于当前特征图的步长
    feat_stride = img_size[0] / feat.size(2)
    anchors = torch.from_numpy(generate_anchors(feat_stride, anchors))
    feat = feat.cpu()
    height, width = feat.size(2), feat.size(3)
    feat_height, feat_width = feat.size(2), feat.size(3)
    shift_x = np.arange(0, feat_width) * feat_stride
    shift_y = np.arange(0, feat_height) * feat_stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = torch.from_numpy(np.vstack((shift_x.ravel(), shift_y.ravel(),
                                         shift_x.ravel(), shift_y.ravel())).transpose())
    shifts = shifts.contiguous().type_as(feat).float()

    # num of anchors [3]
    A = anchors.size(0)
    K = shifts.size(0)

    anchors = anchors.type_as(gt_boxes)
    # all_anchors [K, A, 4]
    all_anchors = anchors.view(1, A, 4) + shifts.view(K, 1, 4)
    all_anchors = all_anchors.view(K * A, 4)
    # compute iou [all_anchors, gt_boxes]
    IOU_map = bbox_overlaps_batch(all_anchors, gt_boxes, img_size).view(batch_size, height, width, A, gt_boxes.shape[1])

    mask_batch = []
    for i in range(batch_size):
        max_iou, _ = torch.max(IOU_map[i].view(height * width * A, gt_boxes.shape[1]), dim=0)
        mask_per_im = torch.zeros([height, width], dtype=torch.int64).to(device)
        for k in range(gt_boxes.shape[1]):
            if torch.sum(gt_boxes[i][k]) == 0:
                break
            max_iou_per_gt = max_iou[k] * 0.5
            mask_per_gt = torch.sum(IOU_map[i][:, :, :, k] > max_iou_per_gt, dim=2)
            mask_per_im += mask_per_gt.to(device)
        mask_batch.append(mask_per_im)
    return mask_batch


def compute_mask_loss(mask_batch, student_feature, teacher_feature, imitation_loss_weight):
    mask_list = []
    for mask in mask_batch:
        mask = (mask > 0).float().unsqueeze(0)
        mask_list.append(mask)
    # [batch, height, widt
    mask_batch = torch.stack(mask_list, dim=0)
    norms = mask_batch.sum() * 2
    mask_batch_s = mask_batch.unsqueeze(4)
    no = student_feature.size(-1)
    bs, na, height, width, _ = mask_batch_s.shape
    mask_batch_no = mask_batch_s.expand((bs, na, height, width, no))
    sup_loss = (torch.pow(teacher_feature - student_feature, 2) * mask_batch_no).sum() / norms
    sup_loss = sup_loss * imitation_loss_weight
    return sup_loss

if __name__ == "__main__":
    anchors = torch.tensor()
    gt_boxes = torch.rand()
