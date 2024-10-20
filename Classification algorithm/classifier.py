import csv
import os
import numpy as np
from scipy.spatial.distance import euclidean, chebyshev, mahalanobis, cityblock
# 数据加载
class PoseSample(object):

    def __init__(self, name, landmarks, class_name, embedding):
        self.name = name
        self.landmarks = landmarks
        self.class_name = class_name
        self.embedding = embedding
# 加载CSV文件中每一行数据
def load_pose_samples(pose_samples_folder, file_extension, file_separator, n_landmarks, n_dimensions, pose_embedder):
    pose_samples = []
    pose_labels = []
    for file_name in os.listdir(pose_samples_folder):
        if file_name.endswith(file_extension):
            class_name = file_name[:-(len(file_extension) )]
            with open(os.path.join(pose_samples_folder, file_name)) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=file_separator)
                for row in csv_reader:
                    assert len(row) == n_landmarks * n_dimensions + 1, 'Wrong number of values: {}'.format(len(row))
                    landmarks = np.array(row[1:], np.float32).reshape([n_landmarks, n_dimensions])
                    pose_sample = PoseSample(
                        name=row[0],
                        landmarks=landmarks,
                        class_name=class_name,
                        embedding=pose_embedder(landmarks),
                        )
                    pose_samples.append(pose_sample)
                    pose_labels.append(class_name)

    return pose_samples, pose_labels


# 分类算法
class PoseClassifier:
    def __init__(self, pose_samples, n_landmarks, n_dimensions, pose_embedder,
                 top_n_by_max_distance, top_n_by_mean_distance,axes_weights=(1., 1., 0.2)):
        self._pose_samples = pose_samples
        self._n_landmarks = n_landmarks
        self._n_dimensions = n_dimensions
        self._pose_embedder = pose_embedder
        self._axes_weights = axes_weights
        self._top_n_by_max_distance = top_n_by_max_distance
        self._top_n_by_mean_distance = top_n_by_mean_distance
# 基础
    def classify_pose(self, pose_landmarks):
        assert pose_landmarks.shape == (self._n_landmarks, self._n_dimensions), 'Unexpected shape: {}'.format(pose_landmarks.shape)

        pose_embedding = self._pose_embedder(pose_landmarks)
        flipped_pose_embedding = self._pose_embedder(pose_landmarks * np.array([-1, 1, 1]))

        max_dist_heap = []

        for sample_idx, sample in enumerate(self._pose_samples):
            max_dist = min(
                np.max(np.abs(sample.embedding - pose_embedding) * self._axes_weights),
                np.max(np.abs(sample.embedding - flipped_pose_embedding) * self._axes_weights)
            )
            max_dist_heap.append([max_dist, sample_idx])

        max_dist_heap = sorted(max_dist_heap, key=lambda x: x[0])
        max_dist_heap = max_dist_heap[:self._top_n_by_max_distance]

        mean_dist_heap = []

        for _, sample_idx in max_dist_heap:
            sample = self._pose_samples[sample_idx]
            mean_dist = min(
                np.mean(np.abs(sample.embedding - pose_embedding) * self._axes_weights),
                np.mean(np.abs(sample.embedding - flipped_pose_embedding) * self._axes_weights)
            )
            mean_dist_heap.append([mean_dist, sample_idx])

        mean_dist_heap = sorted(mean_dist_heap, key=lambda x: x[0])
        mean_dist_heap = mean_dist_heap[:self._top_n_by_mean_distance]
        #print(mean_dist_heap)

        class_names = [self._pose_samples[sample_idx].class_name for _, sample_idx in mean_dist_heap]
        class_name_counts = {class_name: class_names.count(class_name) for class_name in set(class_names)}

        max_class_name = max(class_name_counts, key=class_name_counts.get)

        return max_class_name

    # 除去自己本身
    def classify_pose1(self, pose_landmarks,current_image_index):
        assert pose_landmarks.shape == (self._n_landmarks, self._n_dimensions), 'Unexpected shape: {}'.format(pose_landmarks.shape)

        pose_embedding = self._pose_embedder(pose_landmarks) # 待分类的数据
        flipped_pose_embedding = self._pose_embedder(pose_landmarks * np.array([-1, 1, 1]))

        max_dist_heap = []
        for sample_idx, sample in enumerate(self._pose_samples):
            if sample.name == current_image_index:
                continue  # 跳过当前图像本身
            max_dist = min(
                np.max(np.abs(sample.embedding - pose_embedding) * self._axes_weights),
                np.max(np.abs(sample.embedding - flipped_pose_embedding) * self._axes_weights)
            )
            max_dist_heap.append([max_dist, sample_idx])

        max_dist_heap = sorted(max_dist_heap, key=lambda x: x[0])
        max_dist_heap = max_dist_heap[:self._top_n_by_max_distance]

        mean_dist_heap = []

        for _, sample_idx in max_dist_heap:
            sample = self._pose_samples[sample_idx]
            mean_dist = min(
                np.mean(np.abs(sample.embedding - pose_embedding) * self._axes_weights),
                np.mean(np.abs(sample.embedding - flipped_pose_embedding) * self._axes_weights)
            )
            mean_dist_heap.append([mean_dist, sample_idx])

        mean_dist_heap = sorted(mean_dist_heap, key=lambda x: x[0])
        mean_dist_heap = mean_dist_heap[:self._top_n_by_mean_distance]
        #print(mean_dist_heap)

        class_names = [self._pose_samples[sample_idx].class_name for _, sample_idx in mean_dist_heap]
        class_name_counts = {class_name: class_names.count(class_name) for class_name in set(class_names)}

        max_class_name = max(class_name_counts, key=class_name_counts.get)

        return max_class_name

#带有降维
    def classify_pose_p(self, pose_landmarks):
        assert pose_landmarks.shape == (self._n_landmarks, self._n_dimensions), 'Unexpected shape: {}'.format(pose_landmarks.shape)

        pose_embedding = self._pose_embedder(pose_landmarks)
        flipped_pose_embedding = self._pose_embedder(pose_landmarks * np.array([-1, 1, 1]))

        max_dist_heap = []
        # self._pose_samples是处理好的数据,包括图片名称name,坐标landmark，分类名class_name，特征向量embedding
        for sample_idx, sample in enumerate(self._pose_samples):  # enumerate（）初始化的自我。
            action_name=sample.class_name.split('_')[0]
            # print(action_name)
            if action_name == 'upper':
                max_dist = min(
                    np.max(np.abs(sample.embedding[:12] - pose_embedding[:12])  # np.abs（）返回的是绝对值，
                           * self._axes_weights),
                    np.max(np.abs(sample.embedding[:12] - flipped_pose_embedding[:12])
                           * self._axes_weights), )
                # 如果采用的是欧式距离
            elif action_name == 'lower':
                max_dist = min(
                    np.max(np.abs(sample.embedding[10:] - pose_embedding[10:])  # np.abs（）返回的是绝对值，
                           # sample.embedding -，pose_embedding是来自于同一个数据集中的不同样本
                           * self._axes_weights),
                    np.max(np.abs(sample.embedding[10:] - flipped_pose_embedding[10:])
                           * self._axes_weights), )
            else:
                max_dist = min(
                    np.max(np.abs(sample.embedding - pose_embedding)  # np.abs（）返回的是绝对值，
                           # sample.embedding -，pose_embedding是来自于同一个数据集中的不同样本
                           * self._axes_weights),
                    np.max(np.abs(sample.embedding - flipped_pose_embedding)
                           * self._axes_weights), )
            max_dist_heap.append([max_dist, sample_idx])
        max_dist_heap = sorted(max_dist_heap, key=lambda x: x[0])
        max_dist_heap = max_dist_heap[:self._top_n_by_max_distance]

        # Filter by mean distance.
        # 在去除异常值后，我们可以通过平均距离找到最近的姿势。
        # After removing outliers we can find the nearest pose by mean distance.
        mean_dist_heap = []
        for _, sample_idx in max_dist_heap:
            sample = self._pose_samples[sample_idx]  # 去除异常值之后的，再找到对应之前的那个
            if action_name == 'upper':
                mean_dist = min(
                    np.mean(
                           # np.abs(sample.embedding[12:] - pose_embedding[12:]) * self._axes_weights * 0.3 +
                            np.mean(np.abs(sample.embedding[:12] - pose_embedding[:12]) * self._axes_weights)),
                    np.mean(
                           # np.abs(sample.embedding[12:] - flipped_pose_embedding[12:] * self._axes_weights) * 0.3 +
                            np.mean(np.abs(sample.embedding[:12] - flipped_pose_embedding[:12]) * self._axes_weights)),
                )

            elif action_name == 'lower':
                mean_dist = min(
                    np.mean(
                            #np.abs(sample.embedding[:10] - pose_embedding[:10]) * self._axes_weights * 0.3 +
                            np.mean(np.abs(sample.embedding[10:] - pose_embedding[10:]) * self._axes_weights)),
                    np.mean(
                            #np.abs(sample.embedding[:10] - flipped_pose_embedding[:10] * self._axes_weights) * 0.3 +
                            np.mean(np.abs(sample.embedding[10:] - flipped_pose_embedding[10:]) * self._axes_weights)),
                )
            else:
                mean_dist = min(  # 使用的是曼哈顿距离，后续可以做消融实验（包括K值的选择，距离度量函数的选择）
                    np.mean(np.abs(sample.embedding - pose_embedding)
                            * self._axes_weights),
                    np.mean(np.abs(sample.embedding - flipped_pose_embedding)
                            * self._axes_weights),
                )

            mean_dist_heap.append([mean_dist, sample_idx])

        mean_dist_heap = sorted(mean_dist_heap, key=lambda x: x[0])
        mean_dist_heap = mean_dist_heap[:self._top_n_by_mean_distance]

        class_names = [self._pose_samples[sample_idx].class_name for _, sample_idx in mean_dist_heap]
        class_name_counts = {class_name: class_names.count(class_name) for class_name in set(class_names)}

        max_class_name = max(class_name_counts, key=class_name_counts.get)

        return max_class_name

# 带有权重
    def classify_pose_w(self, pose_landmarks):
        assert pose_landmarks.shape == (self._n_landmarks, self._n_dimensions), 'Unexpected shape: {}'.format(pose_landmarks.shape)

        pose_embedding = self._pose_embedder(pose_landmarks)
        flipped_pose_embedding = self._pose_embedder(pose_landmarks * np.array([-1, 1, 1]))

        max_dist_heap = []

        for sample_idx, sample in enumerate(self._pose_samples):
            # 计算加权距离
            max_dist = min(
                np.max(np.abs(sample.embedding - pose_embedding) * self._axes_weights),
                np.max(np.abs(sample.embedding - flipped_pose_embedding) * self._axes_weights)
            )
            max_dist_heap.append([max_dist, sample_idx])

        max_dist_heap = sorted(max_dist_heap, key=lambda x: x[0])
        max_dist_heap = max_dist_heap[:self._top_n_by_max_distance]

        mean_dist_heap = []

        for _, sample_idx in max_dist_heap:
            sample = self._pose_samples[sample_idx]
            # 计算加权距离
            mean_dist = min(
                np.mean(np.abs(sample.embedding - pose_embedding) * self._axes_weights),
                np.mean(np.abs(sample.embedding - flipped_pose_embedding) * self._axes_weights)
            )
            mean_dist_heap.append([mean_dist, sample_idx])

        mean_dist_heap = sorted(mean_dist_heap, key=lambda x: x[0])
        mean_dist_heap = mean_dist_heap[:self._top_n_by_mean_distance]  # 此部分就是我们需要进行排序的地方
        #print(mean_dist_heap)

        mean_dist_weight_list = []  # 保存 mean_dist_heap 中每行的 distance、class_name 和归一化之后的 weight

        class_name_weights = {class_name: 0 for class_name in
                              set([self._pose_samples[sample_idx].class_name for _, sample_idx in
                                   mean_dist_heap])}  # 获取每个类别的权重
        total_weight = 0
        for (mean_dist, sample_idx) in mean_dist_heap:
            sample = self._pose_samples[sample_idx]
            class_name = sample.class_name   # 获取当前样本的分类结果
            if mean_dist == 0:
                weight = 0.75
            else:
                weight = 1 / mean_dist   # 使用 mean_dist 作为权重
            total_weight += weight
            mean_dist_weight_list.append([mean_dist, class_name, weight])

            # 归一化权重
        mean_dist_weight_normalized = [[dist, name, weight / total_weight] for dist, name, weight in
                                       mean_dist_weight_list]
        #print(mean_dist_weight_normalized)

        # 计算每个类别的权重之和
        class_name_sum_weights = {
            class_name: sum(entry[2] for entry in mean_dist_weight_normalized if entry[1] == class_name)
            for class_name in set(entry[1] for entry in mean_dist_weight_normalized)}

        # 找到具有最大权重之和的类别
        max_class_name = max(class_name_sum_weights, key=class_name_sum_weights.get)

        return max_class_name

#带有权重，除去本身
    def classify_pose_w1(self, pose_landmarks,current_image_index):
        assert pose_landmarks.shape == (self._n_landmarks, self._n_dimensions), 'Unexpected shape: {}'.format(pose_landmarks.shape)

        pose_embedding = self._pose_embedder(pose_landmarks)
        flipped_pose_embedding = self._pose_embedder(pose_landmarks * np.array([-1, 1, 1]))

        max_dist_heap = []

        for sample_idx, sample in enumerate(self._pose_samples):
            if sample.name == current_image_index:
                continue  # 跳过当前图像本身
            # 计算加权距离
            max_dist = min(
                np.max(np.abs(sample.embedding - pose_embedding) * self._axes_weights),
                np.max(np.abs(sample.embedding - flipped_pose_embedding) * self._axes_weights)
            )
            max_dist_heap.append([max_dist, sample_idx])

        max_dist_heap = sorted(max_dist_heap, key=lambda x: x[0])
        max_dist_heap = max_dist_heap[:self._top_n_by_max_distance]

        mean_dist_heap = []

        for _, sample_idx in max_dist_heap:
            sample = self._pose_samples[sample_idx]
            # 计算加权距离
            mean_dist = min(
                np.mean(np.abs(sample.embedding - pose_embedding) * self._axes_weights),
                np.mean(np.abs(sample.embedding - flipped_pose_embedding) * self._axes_weights)
            )
            mean_dist_heap.append([mean_dist, sample_idx])

        mean_dist_heap = sorted(mean_dist_heap, key=lambda x: x[0])
        mean_dist_heap = mean_dist_heap[:self._top_n_by_mean_distance]  # 此部分就是我们需要进行排序的地方
        #print(mean_dist_heap)

        mean_dist_weight_list = []  # 保存 mean_dist_heap 中每行的 distance、class_name 和归一化之后的 weight

        class_name_weights = {class_name: 0 for class_name in
                              set([self._pose_samples[sample_idx].class_name for _, sample_idx in
                                   mean_dist_heap])}  # 获取每个类别的权重
        total_weight = 0
        for (mean_dist, sample_idx) in mean_dist_heap:
            sample = self._pose_samples[sample_idx]
            class_name = sample.class_name
            if mean_dist == 0:
                weight = 0.8
            else:
                weight = 1 / mean_dist   # 使用 mean_dist 作为权重
            total_weight += weight
            mean_dist_weight_list.append([mean_dist, class_name, weight])

            # 归一化权重
        mean_dist_weight_normalized = [[dist, name, weight / total_weight] for dist, name, weight in
                                       mean_dist_weight_list]
        #print(mean_dist_weight_normalized)

        # 计算每个类别的权重之和
        class_name_sum_weights = {
            class_name: sum(entry[2] for entry in mean_dist_weight_normalized if entry[1] == class_name)
            for class_name in set(entry[1] for entry in mean_dist_weight_normalized)}

        # 找到具有最大权重之和的类别
        max_class_name = max(class_name_sum_weights, key=class_name_sum_weights.get)

        return max_class_name


#不同距离度量
    def classify_pose_D(self, pose_landmarks, distance_function_name):
        assert pose_landmarks.shape == (self._n_landmarks, self._n_dimensions), 'Unexpected shape: {}'.format(pose_landmarks.shape)

        pose_embedding = self._pose_embedder(pose_landmarks)
        flipped_pose_embedding = self._pose_embedder(pose_landmarks * np.array([-1, 1, 1]))

        if distance_function_name == "manhattan_distance":
            distance_function = lambda x, y: np.sum(np.abs(x - y))
        elif distance_function_name == "chebyshev_distance":
            distance_function = lambda x, y: np.max(np.abs(x - y))
        elif distance_function_name == "euclidean_distance":
            distance_function = lambda x, y: np.linalg.norm(x - y)
        elif distance_function_name == "minkowski_distance":
            distance_function = lambda x, y: np.power(np.sum(np.power(np.abs(x - y), 3)), 1 / 3)
        else:
            raise ValueError(f"Unsupported distance function: {distance_function_name}")

        max_dist_heap = []

        for sample_idx, sample in enumerate(self._pose_samples):
            max_dist = min(
                distance_function(sample.embedding, pose_embedding),
                distance_function(sample.embedding, flipped_pose_embedding)
            )
            max_dist_heap.append([max_dist, sample_idx])

        max_dist_heap = sorted(max_dist_heap, key=lambda x: x[0])
        max_dist_heap = max_dist_heap[:self._top_n_by_max_distance]

        mean_dist_heap = []

        for _, sample_idx in max_dist_heap:
            sample = self._pose_samples[sample_idx]
            mean_dist = min(
                distance_function(sample.embedding, pose_embedding),
                distance_function(sample.embedding, flipped_pose_embedding)
            )
            mean_dist_heap.append([mean_dist, sample_idx])

        mean_dist_heap = sorted(mean_dist_heap, key=lambda x: x[0])
        mean_dist_heap = mean_dist_heap[:self._top_n_by_mean_distance]

        class_names = [self._pose_samples[sample_idx].class_name for _, sample_idx in mean_dist_heap]
        class_name_counts = {class_name: class_names.count(class_name) for class_name in set(class_names)}

        max_class_name = max(class_name_counts, key=class_name_counts.get)

        return max_class_name

# 带有权重的距离度量
    def classify_pose_DW(self, pose_landmarks, distance_function_name):
        assert pose_landmarks.shape == (self._n_landmarks, self._n_dimensions), 'Unexpected shape: {}'.format(pose_landmarks.shape)

        pose_embedding = self._pose_embedder(pose_landmarks)
        flipped_pose_embedding = self._pose_embedder(pose_landmarks * np.array([-1, 1, 1]))

        if distance_function_name == "manhattan_distance":
            distance_function = lambda x, y: np.sum(np.abs(x - y))
        elif distance_function_name == "chebyshev_distance":
            distance_function = lambda x, y: np.max(np.abs(x - y))
        elif distance_function_name == "euclidean_distance":
            distance_function = lambda x, y: np.linalg.norm(x - y)
        elif distance_function_name == "minkowski_distance":
            distance_function = lambda x, y: np.power(np.sum(np.power(np.abs(x - y), 3)), 1 / 3)
        else:
            raise ValueError(f"Unsupported distance function: {distance_function_name}")

        max_dist_heap = []
        # self._pose_samples是处理好的数据,包括图片名称name,坐标landmark，分类名class_name，特征向量embedding
        for sample_idx, sample in enumerate(self._pose_samples):  # enumerate（）初始化的自我。
            action_name = sample.class_name.split('_')[0]
            # print(action_name)

            max_dist = min(
                    distance_function(sample.embedding , pose_embedding),  # np.abs（）返回的是绝对值，
                    distance_function(sample.embedding ,flipped_pose_embedding) )
            max_dist_heap.append([max_dist, sample_idx])
        max_dist_heap = sorted(max_dist_heap, key=lambda x: x[0])
        max_dist_heap = max_dist_heap[:self._top_n_by_max_distance]

        # Filter by mean distance.
        # 在去除异常值后，我们可以通过平均距离找到最近的姿势。
        # After removing outliers we can find the nearest pose by mean distance.
        mean_dist_heap = []
        for _, sample_idx in max_dist_heap:
            sample = self._pose_samples[sample_idx]  # 去除异常值之后的，再找到对应之前的那个
            mean_dist = min(
                    distance_function(sample.embedding, pose_embedding),
                    distance_function(sample.embedding, flipped_pose_embedding))

            mean_dist_heap.append([mean_dist, sample_idx])

        # Move these lines outside the for loop
        mean_dist_heap = sorted(mean_dist_heap, key=lambda x: x[0])
        mean_dist_heap = mean_dist_heap[:self._top_n_by_mean_distance]

        mean_dist_weight_list = []  # 保存 mean_dist_heap 中每行的 distance、class_name 和归一化之后的 weight

        class_name_weights = {class_name: 0 for class_name in
                              set([self._pose_samples[sample_idx].class_name for _, sample_idx in
                                   mean_dist_heap])}  # 获取每个类别的权重
        total_weight = 0
        for (mean_dist, sample_idx) in mean_dist_heap:
            sample = self._pose_samples[sample_idx]
            class_name = sample.class_name
            if mean_dist == 0:
                weight = 0.75
            else:
                weight = 1 / mean_dist  # 使用 mean_dist 作为权重
            total_weight += weight
            mean_dist_weight_list.append([mean_dist, class_name, weight])

            # 归一化权重
        mean_dist_weight_normalized = [[dist, name, weight / total_weight] for dist, name, weight in
                                       mean_dist_weight_list]
        # print(mean_dist_weight_normalized)

        # 计算每个类别的权重之和
        class_name_sum_weights = {
            class_name: sum(entry[2] for entry in mean_dist_weight_normalized if entry[1] == class_name)
            for class_name in set(entry[1] for entry in mean_dist_weight_normalized)}

        # 找到具有最大权重之和的类别
        max_class_name = max(class_name_sum_weights, key=class_name_sum_weights.get)

        return max_class_name