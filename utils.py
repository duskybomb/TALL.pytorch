import pickle
import json
import operator


def read_load_pickle(pickle_file):
    with open(pickle_file, 'rb') as file:
        return pickle.load(file, encoding='latin1')


def calculate_IoU(i0, i1):
    union = (min(i0[0], i1[0]), max(i0[1], i1[1]))
    inter = (max(i0[0], i1[0]), min(i0[1], i1[1]))
    iou = 1.0 * (inter[1] - inter[0]) / (union[1] - union[0])
    return iou


def calculate_nIoL(base, sliding_clip):
    inter = (max(base[0], sliding_clip[0]), min(base[1], sliding_clip[1]))
    inter_l = inter[1] - inter[0]
    length = sliding_clip[1] - sliding_clip[0]
    nIoL = 1.0 * (length - inter_l) / length
    return nIoL


def read_json(filename):
    with open(filename, 'r') as read_file:
        return json.load(read_file)


def nms_temporal(x1, x2, s, overlap):
    pick = []
    assert len(x1) == len(s)
    assert len(x2) == len(s)
    if len(x1) == 0:
        return pick

    union = list(map(operator.sub, x2, x1))  # union = x2-x1

    I = [i[0] for i in sorted(enumerate(s), key=lambda x: x[1])]  # sort and get index

    while len(I) > 0:
        i = I[-1]
        pick.append(i)

        xx1 = [max(x1[i], x1[j]) for j in I[:-1]]
        xx2 = [min(x2[i], x2[j]) for j in I[:-1]]
        inter = [max(0.0, k2 - k1) for k1, k2 in zip(xx1, xx2)]
        o = [inter[u] / (union[i] + union[I[u]] - inter[u]) for u in range(len(I) - 1)]
        I_new = []
        for j in range(len(o)):
            if o[j] <= overlap:
                I_new.append(I[j])
        I = I_new
    return pick


def compute_IoU_recall_top_n_forreg(top_n, iou_thresh, sentence_image_mat, sentence_image_reg_mat, sclips, iclips):
    correct_num = 0.0
    for k in range(sentence_image_mat.shape[0]):
        gt = sclips[k]
        gt_start = float(gt.split("_")[1])
        gt_end = float(gt.split("_")[2])

        sim_v = [v for v in sentence_image_mat[k]]
        starts = [s for s in sentence_image_reg_mat[k, :, 0]]
        ends = [e for e in sentence_image_reg_mat[k, :, 1]]
        picks = nms_temporal(starts, ends, sim_v, iou_thresh - 0.05)

        if top_n < len(picks): picks = picks[0:top_n]
        for index in picks:
            pred_start = sentence_image_reg_mat[k, index, 0]
            pred_end = sentence_image_reg_mat[k, index, 1]
            iou = calculate_IoU((gt_start, gt_end), (pred_start, pred_end))
            if iou >= iou_thresh:
                correct_num += 1
                break
    return correct_num
