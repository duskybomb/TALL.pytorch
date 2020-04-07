import numpy as np
import os

from utils import read_load_pickle, calculate_IoU, calculate_nIoL
from torch.utils.data import Dataset


class TrainingDataSet(Dataset):
    def __init__(self, sliding_clip_path, clip_sentence_vector_path, context_num=1, context_window=128,
                 visual_features_dim=4096, channels=3, sentence_vector_dim=4800):
        self.sliding_clip_path = sliding_clip_path

        self.context_num = context_num
        self.context_window = context_window

        cs_dict = read_load_pickle(clip_sentence_vector_path)
        # movie_length_info = read_load_pickle(movie_length_path)

        self.clip_sentence_pairs = []
        for cs in cs_dict:
            clip_name, sentence_vectors = cs
            for sentence_vector in sentence_vectors:
                self.clip_sentence_pairs.append((clip_name, sentence_vector))

        movie_names_set = set()
        self.movie_clip_names = {}

        for i, csv in enumerate(self.clip_sentence_pairs):
            clip_name = csv[0]
            movie_name = clip_name.split('_')[0]
            if movie_name not in movie_names_set:
                movie_names_set.add(movie_name)
                self.movie_clip_names[movie_name] = []
            self.movie_clip_names[movie_name].append(i)

        self.movie_names = list(movie_names_set)
        self.visual_feature_dim = visual_features_dim * channels
        self.sentence_vector_dim = sentence_vector_dim

        sliding_clips_temp = os.listdir(self.sliding_clip_path)
        self.clip_sentence_pairs_iou = []
        for clip_name in sliding_clips_temp:
            if clip_name[-3] == 'npy':
                movie_name = clip_name.split('_')[0]
                for clip_sentence in self.clip_sentence_pairs:
                    original_clip_name = clip_sentence[0]
                    original_movie_name = original_clip_name.split('_')[0]
                    if original_movie_name == movie_name:
                        start = int(clip_name.split('_')[1])
                        end = int(clip_name.split('_')[2])
                        o_start = int(original_clip_name.split('_')[1])
                        o_end = int(original_clip_name.split('_')[1].split('.')[0])
                        iou = calculate_IoU((start, end), (o_start, o_end))
                        if iou > 0.5:
                            nIoL = calculate_nIoL((o_start, o_end), (start, end))
                            if nIoL < 0.15:
                                # movie_length = movie_length_info[movie_name.split(".")[0]]
                                start_offset = o_start - start
                                end_offset = o_end - end
                                self.clip_sentence_pairs_iou.append(
                                    (clip_sentence[0], clip_sentence[1], clip_name, start_offset, end_offset))

        self.num_samples_iou = len(self.clip_sentence_pairs_iou)

    def __len__(self):
        return self.num_samples_iou

    def get_context_window(self, clip_name):
        movie_name = clip_name.split('_')[0]
        start = int(clip_name.split('_')[1])
        end = int(clip_name.split('_')[2].split('.')[0])

        clip_length = self.context_window
        left_context_features = np.zeros([self.context_num, self.visual_feature_dim], dtype=np.float32)
        right_context_features = np.zeros([self.context_num, self.visual_feature_dim], dtype=np.float32)

        last_left_feature = np.load(os.path.join(self.sliding_clip_path, clip_name))
        last_right_feature = np.load(os.path.join(self.sliding_clip_path, clip_name))

        for i in range(self.context_num):
            left_context_start = start - clip_length * (i + 1)
            left_context_end = start - clip_length * i
            right_context_start = end + clip_length * i
            right_context_end = end + clip_length * (i + 1)
            left_context_name = f'{movie_name}_{str(left_context_start)}_{str(left_context_end)}.npy'
            right_context_name = f'{movie_name}_{str(right_context_start)}_{str(right_context_end)}.npy'

            if os.path.exists(self.sliding_clip_path + left_context_name):
                left_context_feature = np.load(os.path.join(self.sliding_clip_path, left_context_name))
                last_left_feature = left_context_feature
            else:
                left_context_feature = last_left_feature
            if os.path.exists(self.sliding_clip_path + right_context_name):
                right_context_feature = np.load(os.path.join(self.sliding_clip_path, right_context_name))
                last_right_feature = right_context_feature
            else:
                right_context_feature = last_right_feature

            left_context_features[i] = left_context_feature
            right_context_features[i] = right_context_feature

        return np.mean(left_context_features, axis=0), np.mean(right_context_features, axis=0)

    def __getitem__(self, item):
        # image = np.zeros([self.visual_feature_dim])

        original_clip_name, sentence_vector, clip_name, p_offset, l_offset = self.clip_sentence_pairs_iou[item]
        features_path = os.path.join(self.sliding_clip_path, clip_name)
        feature_map = np.load(features_path)
        left_context_feat, right_context_feat = self.get_context_window(clip_name)
        image = np.hstack((left_context_feat, feature_map, right_context_feat))
        sentence = sentence_vector[:self.sentence_vector_dim]
        offset = [p_offset, l_offset]

        return image, sentence, offset


class TestDataset(Dataset):
    def __init__(self, sliding_clip_path, clip_sentence_vector_path, context_window=128, visual_features_dim=4096,
                 channels=3, sentence_vector_dim=4800, context_num=1):
        self.sliding_clip_path = sliding_clip_path

        self.context_window = context_window
        self.context_num = context_num
        cs_dict = read_load_pickle(clip_sentence_vector_path)

        self.clip_sentence_pairs = []
        for cs in cs_dict:
            clip_name, sentence_vectors = cs
            for sentence_vector in sentence_vectors:
                self.clip_sentence_pairs.append((clip_name, sentence_vector))

        movie_names_set = set()
        self.movie_clip_names = {}

        for i, csv in enumerate(self.clip_sentence_pairs):
            clip_name = csv[0]
            movie_name = clip_name.split('_')[0]
            if movie_name not in movie_names_set:
                movie_names_set.add(movie_name)
                self.movie_clip_names[movie_name] = []
            self.movie_clip_names[movie_name].append(i)

        self.movie_names = list(movie_names_set)
        self.visual_feature_dim = visual_features_dim * channels
        self.sentence_vector_dim = sentence_vector_dim

        self.clip_num_per_movie_max = 0

        for movie_name in self.movie_clip_names:
            if len(self.movie_clip_names[movie_name]) > self.clip_num_per_movie_max:
                self.clip_num_per_movie_max = len(self.movie_clip_names[movie_name])

        sliding_clip_temp = os.listdir(self.sliding_clip_path)
        self.sliding_clip_names = []
        for clip_name in sliding_clip_temp:
            if clip_name[-3] == "npy":
                movie_name = clip_name.split('_')[0]
                if movie_name in self.movie_clip_names:
                    self.sliding_clip_names.append(clip_name[:-4])

        self.num_samples = len(self.clip_sentence_pairs)

    def __len__(self):
        return self.num_samples

    def get_context_window(self, clip_name):
        movie_name = clip_name.split('_')[0]
        start = int(clip_name.split('_')[1])
        end = int(clip_name.split('_')[2].split('.')[0])

        clip_length = self.context_window
        left_context_features = np.zeros([self.context_num, self.visual_feature_dim], dtype=np.float32)
        right_context_features = np.zeros([self.context_num, self.visual_feature_dim], dtype=np.float32)

        last_left_feature = np.load(os.path.join(self.sliding_clip_path, clip_name))
        last_right_feature = np.load(os.path.join(self.sliding_clip_path, clip_name))

        for i in range(self.context_num):
            left_context_start = start - clip_length * (i + 1)
            left_context_end = start - clip_length * i
            right_context_start = end + clip_length * i
            right_context_end = end + clip_length * (i + 1)
            left_context_name = f'{movie_name}_{str(left_context_start)}_{str(left_context_end)}.npy'
            right_context_name = f'{movie_name}_{str(right_context_start)}_{str(right_context_end)}.npy'

            if os.path.exists(self.sliding_clip_path + left_context_name):
                left_context_feature = np.load(os.path.join(self.sliding_clip_path, left_context_name))
                last_left_feature = left_context_feature
            else:
                left_context_feature = last_left_feature
            if os.path.exists(self.sliding_clip_path + right_context_name):
                right_context_feature = np.load(os.path.join(self.sliding_clip_path, right_context_name))
                last_right_feature = right_context_feature
            else:
                right_context_feature = last_right_feature

            left_context_features[i] = left_context_feature
            right_context_features[i] = right_context_feature

        return np.mean(left_context_features, axis=0), np.mean(right_context_features, axis=0)

    def load_movie_sliding_clip(self, movie_name, sample_num):
        movie_clip_sentences = []
        movie_clip_feature_map = []
        clip_set = set()

        for i in range(len(self.clip_sentence_pairs)):
            if movie_name in self.clip_sentence_pairs[i][0]:
                movie_clip_sentences.append(
                    (self.clip_sentence_pairs[i][0], self.clip_sentence_pairs[i][1][:self.sentence_vector_dim]))
        for i in range(len(self.sliding_clip_names)):
            if movie_name in self.sliding_clip_names[i]:
                visual_feature_path = os.path.join(self.sliding_clip_path, self.sliding_clip_names[i], ".npy")
                left_context_feature, right_context_feature = self.get_context_window(
                    self.sliding_clip_names[i] + ".npy")
                feature_data = np.load(visual_feature_path)
                comb_feat = np.hstack((left_context_feature, feature_data, right_context_feature))
                movie_clip_feature_map.append((self.sliding_clip_names[i], comb_feat))

        return movie_clip_feature_map, movie_clip_sentences
