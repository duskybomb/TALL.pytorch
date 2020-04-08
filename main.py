import argparse
import random
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from tqdm import tqdm
from dataset.dataset import TrainingDataSet, TestDataset
from model.model import TALL
from model.loss import compute_loss_reg
from utils import compute_IoU_recall_top_n_forreg

best_R1_IOU5 = 0
best_R5_IOU5 = 0
best_R1_IOU5_epoch = 0
best_R5_IOU5_epoch = 0


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def train(epoch, model, trainloader, optimizer):
    model.train()
    train_loss = 0

    for batch_idx, (images, sentences, offsets) in tqdm(enumerate(trainloader), total=len(trainloader),
                                                        desc='batch_idx', leave=False):
        images, sentences, offsets = images.cuda(), sentences.cuda(), offsets.cuda()

        outputs = model(images, sentences)

        loss, loss_align, loss_reg = compute_loss_reg(outputs, offsets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    print('Epoch: %d | Step: %d | Loss: %.3f | loss_align: %.3f | loss_reg: %.3f' % (
        epoch, batch_idx, train_loss / (batch_idx + 1), loss_align, loss_reg))


def test(epoch, model, test_dataset, test_result_output, path):
    global best_R1_IOU5
    global best_R5_IOU5
    global best_R1_IOU5_epoch
    global best_R5_IOU5_epoch

    model.eval()

    IoU_thresh = [0.1, 0.3, 0.5, 0.7]
    all_correct_num_10 = [0.0] * 5
    all_correct_num_5 = [0.0] * 5
    all_correct_num_1 = [0.0] * 5
    all_retrievd = 0.0
    all_number = len(test_dataset.movie_names)
    idx = 0
    for movie_name in test_dataset.movie_names:
        idx += 1
        print("%d/%d" % (idx, all_number))

        movie_clip_featmaps, movie_clip_sentences = test_dataset.load_movie_sliding_clip(movie_name, 16)
        print("sentences: " + str(len(movie_clip_sentences)))
        print("clips: " + str(len(movie_clip_featmaps)))  # candidate clips)

        sentence_image_mat = np.zeros([len(movie_clip_sentences), len(movie_clip_featmaps)])
        sentence_image_reg_mat = np.zeros([len(movie_clip_sentences), len(movie_clip_featmaps), 2])

        for k in range(len(movie_clip_sentences)):
            sent_vec = movie_clip_sentences[k][1]
            sent_vec = np.reshape(sent_vec, [1, sent_vec.shape[0]])  # 1,4800
            sent_vec = torch.from_numpy(sent_vec).cuda()

            for t in range(len(movie_clip_featmaps)):
                featmap = movie_clip_featmaps[t][1]
                visual_clip_name = movie_clip_featmaps[t][0]

                start = float(visual_clip_name.split("_")[1])
                end = float(visual_clip_name.split("_")[2].split("_")[0])

                featmap = np.reshape(featmap, [1, featmap.shape[0]])
                featmap = torch.from_numpy(featmap).cuda()

                outputs = model(featmap, sent_vec)

                outputs = outputs.squeeze(1).squeeze(1)

                sentence_image_mat[k, t] = outputs[0]

                # sentence_image_mat[k, t] = expit(outputs[0]) * conf_score
                reg_end = end + outputs[2]
                reg_start = start + outputs[1]

                sentence_image_reg_mat[k, t, 0] = reg_start
                sentence_image_reg_mat[k, t, 1] = reg_end

        iclips = [b[0] for b in movie_clip_featmaps]
        sclips = [b[0] for b in movie_clip_sentences]

        # calculate Recall@m, IoU=n
        for k in range(len(IoU_thresh)):
            IoU = IoU_thresh[k]
            correct_num_10 = compute_IoU_recall_top_n_forreg(10, IoU, sentence_image_mat, sentence_image_reg_mat,
                                                             sclips, iclips)
            correct_num_5 = compute_IoU_recall_top_n_forreg(5, IoU, sentence_image_mat, sentence_image_reg_mat, sclips,
                                                            iclips)
            correct_num_1 = compute_IoU_recall_top_n_forreg(1, IoU, sentence_image_mat, sentence_image_reg_mat, sclips,
                                                            iclips)
            print(movie_name + " IoU=" + str(IoU) + ", R@10: " + str(correct_num_10 / len(sclips)) + "; IoU=" + str(
                IoU) + ", R@5: " + str(correct_num_5 / len(sclips)) + "; IoU=" + str(IoU) + ", R@1: " + str(
                correct_num_1 / len(sclips)))

            all_correct_num_10[k] += correct_num_10
            all_correct_num_5[k] += correct_num_5
            all_correct_num_1[k] += correct_num_1
        all_retrievd += len(sclips)

    for k in range(len(IoU_thresh)):
        print(" IoU=" + str(IoU_thresh[k]) + ", R@10: " + str(all_correct_num_10[k] / all_retrievd) + "; IoU=" + str(
            IoU_thresh[k]) + ", R@5: " + str(all_correct_num_5[k] / all_retrievd) + "; IoU=" + str(
            IoU_thresh[k]) + ", R@1: " + str(all_correct_num_1[k] / all_retrievd))

        test_result_output.write("Epoch " + str(epoch) + ": IoU=" + str(IoU_thresh[k]) + ", R@10: " + str(
            all_correct_num_10[k] / all_retrievd) + "; IoU=" + str(IoU_thresh[k]) + ", R@5: " + str(
            all_correct_num_5[k] / all_retrievd) + "; IoU=" + str(IoU_thresh[k]) + ", R@1: " + str(
            all_correct_num_1[k] / all_retrievd) + "\n")

    R1_IOU5 = all_correct_num_1[2] / all_retrievd
    R5_IOU5 = all_correct_num_5[2] / all_retrievd

    if R1_IOU5 > best_R1_IOU5:
        print("best_R1_IOU5: %0.3f" % R1_IOU5)
        state = {
            'model': model.state_dict(),
            'best_R1_IOU5': best_R1_IOU5,
        }
        if not os.path.isdir(path):
            os.mkdir(path)
        torch.save(state, os.path.join(path, 'best_R1_IOU5_model.t7'))
        best_R1_IOU5 = R1_IOU5
        best_R1_IOU5_epoch = epoch

    if R5_IOU5 > best_R5_IOU5:
        print("best_R5_IOU5: %0.3f" % R5_IOU5)
        state = {
            'model': model.state_dict(),
            'best_R5_IOU5': best_R5_IOU5,
        }
        if not os.path.isdir(path):
            os.mkdir(path)
        torch.save(state, os.path.join(path, 'best_R5_IOU5_model.t7'))
        best_R5_IOU5 = R5_IOU5
        best_R5_IOU5_epoch = epoch


def main(config):
    train_dataset = TrainingDataSet(sliding_clip_path='exp_data/Interval64_128_256_512_overlap0.8_c3d_fc6/',
                                    clip_sentence_vector_path='exp_data/TACoS/train_clip-sentvec.pkl')

    test_dataset = TestDataset(sliding_clip_path='exp_data/Interval128_256_overlap0.8_c3d_fc6/',
                               clip_sentence_vector_path='exp_data/TACoS/test_clip-sentvec.pkl')
    # build model architecture, then print to console

    trainloader = DataLoader(dataset=train_dataset,
                             batch_size=config.batch_size,
                             shuffle=True,
                             num_workers=4)

    model = TALL().cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    setup_seed(0)

    start_epoch = 0
    total_epoch = 20

    path = 'results'

    if not os.path.isdir(path):
        os.mkdir(path)

    test_result_output = open(os.path.join(path, "test_results.txt"), "w")

    for epoch in range(start_epoch, total_epoch):
        train(epoch, model, trainloader, optimizer)

    test(0, model, test_dataset, test_result_output, path)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='TALL: Pytorch')
    args.add_argument('-b', '--batch_size', default=8, type=int,
                      help='batch size (default: 8)')
    args.add_argument('-l', '--lr', default=0.001, type=float,
                      help='learning rate (default: 0.001)')
    config = args.parse_args()
    main(config)
