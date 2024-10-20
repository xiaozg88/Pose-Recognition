'''
Descripttion: 
version: 
Author: LiQiang
Date: 2022-02-27 22:04:56
LastEditTime: 2022-02-27 22:06:11
'''
# 人体姿态分类
import csv
import numpy as np
import os


# 样本分类
class PoseSample(object):

    def __init__(self, name, landmarks, class_name, embedding):
        self.name = name
        self.landmarks = landmarks
        self.class_name = class_name

        self.embedding = embedding

# 选出异常值
class PoseSampleOutlier(object):

    def __init__(self, sample, detected_class, all_classes):
        self.sample = sample
        self.detected_class = detected_class      # 检测类
        self.all_classes = all_classes


class PoseClassifier(object):
    """ClassE:\研究生\blazeposeifies pose landmarks.分类"""

    def __init__(self,
                 pose_samples_folder,   #样本所在文件夹
                 pose_embedder,         #向量化的坐标
                 file_extension='csv',
                 file_separator=',',   #分隔符
                 n_landmarks=33,       #关键点坐标点个数
                 n_dimensions=3,       #坐标的维度
                 top_n_by_max_distance=30,
                 top_n_by_mean_distance=11,
                 axes_weights=(1., 1., 0.2)): #各个维度的权重
        self._pose_embedder = pose_embedder
        self._n_landmarks = n_landmarks
        self._n_dimensions = n_dimensions
        self._top_n_by_max_distance = top_n_by_max_distance
        self._top_n_by_mean_distance = top_n_by_mean_distance
        self._axes_weights = axes_weights
        self.action_name = pose_samples_folder.split("\\")[-1].split("_")[0]
        self._pose_samples = self._load_pose_samples(pose_samples_folder,
                                                     file_extension,
                                                     file_separator,
                                                     n_landmarks,
                                                     n_dimensions,
                                                     pose_embedder)
    # 数据加载函数

    def _load_pose_samples(self,
                           pose_samples_folder,
                           file_extension,  # 文件类型
                           file_separator, #文件分隔符
                           n_landmarks,
                           n_dimensions,
                           pose_embedder):
        """Loads pose samples from a given folder.
        从给定的文件夹中加载样本
        Required folder structure:  需要的文件结构
          neutral_standing.csv
          pushups_down.csv
          pushups_up.csv
          squats_down.csv
          ...

        Required CSV structure: 要求的CSV文件样式
          sample_00001,x1,y1,z1,x2,y2,z2,....
          sample_00002,x1,y1,z1,x2,y2,z2,....
          ...
        """
        # Each file in the folder represents one pose class.一个文件代表一个类别
        file_names = [name for name in os.listdir(
            pose_samples_folder) if name.endswith(file_extension)]
        # action_name=pose_samples_folder.split("\\")[-1].split("_")[0]
        # print(pose_samples_folder)
        # print(action_name)
        # os.listdir（）列出指定目录下的所有文件和子目录
        # Python endswith() 方法用于判断字符串是否以指定后缀结尾
        pose_samples = []     # 用于存放样本
        for file_name in file_names:  # 从文件名中选出类名
            # Use file name as pose class name.
            class_name = file_name[:-(len(file_extension) + 1)]  # 是为了获取类名，比如：up,down

            # Parse CSV.解析，将每个坐标点以正确的形式存储到列表中
            # os.path.join()函数：连接两个或更多的路径名组件、
            with open(os.path.join(pose_samples_folder, file_name)) as csv_file: # 打开对应的文件
                csv_reader = csv.reader(csv_file, delimiter=file_separator)  # 按照分隔符读取文件
                for row in csv_reader:  # 按行读取CSV文件
                    assert len(row) == n_landmarks * n_dimensions + \
                        1, 'Wrong number of values: {}'.format(len(row))
                    # 检查每一行的数据个数是否正确：33个关键点
                    landmarks = np.array(row[1:], np.float32).reshape(
                        [n_landmarks, n_dimensions])         # 获取关键点坐标，进行矩阵存储坐标
                    pose_samples.append(PoseSample(          # 将数据进行存储
                        name=row[0],
                        landmarks=landmarks,
                        class_name=class_name,
                        embedding=pose_embedder(landmarks),  # 获取的就是特征向量

                    )) # pose_samples是一个列表，包括图片名，关键点坐标，分类名，以及特征向量坐标

        return pose_samples

    def find_pose_sample_outliers(self):
        """Classifies each sample against the entire database.根据整个数据库对每个样本进行分类"""
        # Find outliers in target poses
        outliers = [] # 存放异常值
        for sample in self._pose_samples:  # self._pose_samples中保存的技术一个列表
            # 处理好的数据,包括图片名称name,坐标landmark，分类名class_name，特征向量embedding
            # Find nearest poses for the target one.
            pose_landmarks = sample.landmarks.copy()
            pose_classification = self.__call__(pose_landmarks)  # 得到分类结果，结果为down': 8,up': 2,
            # print("pose_classification:", pose_classification)
            # print("type(pose_classification):", type(pose_classification))
            # 找到一个pose_classificatio字典中找出出现次数最多的类名，并将这些类名放入列表class_names 中
            class_names = [class_name for class_name, count in pose_classification.items(
            ) if count == max(pose_classification.values())]
            # class_names 将包含在 pose_classification 字典中具有最大计数/频率的类名列表。
            # 如果有多个类具有相同的最大计数，则所有这些类将包含在 class_names 列表中。

            # Sample is an outlier if nearest poses have different class or more than
            # one pose class is detected as nearest.
            if sample.class_name not in class_names or len(class_names) != 1:
                # 如果与标准的分类不同或者说，分类名不唯一（down:5,up:5），说明分类不唯一
                outliers.append(PoseSampleOutlier(
                    sample, class_names, pose_classification))

        return outliers

    def __call__(self, pose_landmarks):  # 33个关键点坐标
        """Classifies given pose. 分类主函数

        Classification is done in two stages:
          * First we pick top-N samples by MAX distance. It allows to remove samples
            that are almost the same as given pose, but has few joints bent in the
            other direction.
          * Then we pick top-N samples by MEAN distance. After outliers are removed
            on a previous step, we can pick samples that are closes on average.
        *首先，我们根据最大距离选取top-N个样本。它允许移除样品
        它们与给定的姿势几乎相同，但几乎没有向其他方向弯曲的关节。
        然后根据平均距离选取前n个样本。在上一步移除离群值之后，我们可以选择接近平均值的样本。
        Args:
          pose_landmarks: NumPy array with 3D landmarks of shape (N, 3).

        Returns: 使用数据库中最近的姿态样本计数。
          Dictionary with count of nearest pose samples from the database. Sample:
            {
              'pushups_down': 8,
              'pushups_up': 2,
            }
        """
        # Check that provided and target poses have the same shape.
        assert pose_landmarks.shape == (
            self._n_landmarks, self._n_dimensions), 'Unexpected shape: {}'.format(pose_landmarks.shape)

        #降维技术

        # if action_name=='upper':

        # Get given pose embedding.镜面反转
        pose_embedding = self._pose_embedder(pose_landmarks)   # 获取特征向量
        flipped_pose_embedding = self._pose_embedder(
            pose_landmarks * np.array([-1, 1, 1]))   # 镜像翻转

        # 去除异常值——几乎与给定的姿势相同，但有一个关节弯曲到另一个方向，实际上代表了一个不同的姿势类别。
        max_dist_heap = []
        # self._pose_samples是处理好的数据,包括图片名称name,坐标landmark，分类名class_name，特征向量embedding
        for sample_idx, sample in enumerate(self._pose_samples):  # enumerate（）初始化的自我。
            if self.action_name == 'upper':
                max_dist = min(
                np.max(np.abs(sample.embedding[:12] - pose_embedding[:12])   # np.abs（）返回的是绝对值，
                       * self._axes_weights),
                np.max(np.abs(sample.embedding[:12] - flipped_pose_embedding[:12])
                       * self._axes_weights),)
                # 如果采用的是欧式距离
            elif self.action_name== 'lower':
                max_dist = min(
                    np.max(np.abs(sample.embedding[10:] - pose_embedding[10:])  # np.abs（）返回的是绝对值，
                           # sample.embedding -，pose_embedding是来自于同一个数据集中的不同样本
                           * self._axes_weights),
                    np.max(np.abs(sample.embedding[10:] - flipped_pose_embedding[10:])
                           * self._axes_weights),)
            else:
                max_dist = min(
                    np.max(np.abs(sample.embedding- pose_embedding)  # np.abs（）返回的是绝对值，
                           # sample.embedding -，pose_embedding是来自于同一个数据集中的不同样本
                           * self._axes_weights),
                    np.max(np.abs(sample.embedding- flipped_pose_embedding)
                           * self._axes_weights),)
            max_dist_heap.append([max_dist, sample_idx])
        max_dist_heap = sorted(max_dist_heap, key=lambda x: x[0])  # 进行升序排序
        max_dist_heap = max_dist_heap[:self._top_n_by_max_distance]  # 取前30个值


        # Filter by mean distance.
        # 在去除异常值后，我们可以通过平均距离找到最近的姿势。
        # After removing outliers we can find the nearest pose by mean distance.
        mean_dist_heap = []
        for _, sample_idx in max_dist_heap:
            sample = self._pose_samples[sample_idx]    # 去除异常值之后的，再找到对应之前的那个
            if self.action_name=='upper':
                mean_dist = min(
                    np.mean(np.abs(sample.embedding[12:] - pose_embedding[12:])* self._axes_weights* 0.3 +
                        np.mean(np.abs(sample.embedding[:12] - pose_embedding[:12])* self._axes_weights))        ,
                    np.mean(np.abs(sample.embedding[12:] - flipped_pose_embedding[12:]* self._axes_weights)* 0.3 +
                        np.mean(np.abs(sample.embedding[:12] - flipped_pose_embedding[:12])* self._axes_weights)),
                )

            elif self.action_name=='lower':
                mean_dist = min(
                    np.mean(np.abs(sample.embedding[:10] - pose_embedding[:10])* self._axes_weights * 0.3 +
                            np.mean(np.abs(sample.embedding[10:] - pose_embedding[10:])* self._axes_weights)),
                    np.mean(np.abs(sample.embedding[:10] - flipped_pose_embedding[:10]* self._axes_weights) * 0.3 +
                            np.mean( np.abs(sample.embedding[10:] - flipped_pose_embedding[10:])* self._axes_weights)),
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
        mean_dist_heap = mean_dist_heap[:self._top_n_by_mean_distance]   # 取前十个
        print('============')
        print(mean_dist_heap)

        # Collect results into map: (class_name -> n_samples)
        class_names = [self._pose_samples[sample_idx].class_name for _,
                       sample_idx in mean_dist_heap]
        result = {class_name: class_names.count(
            class_name) for class_name in set(class_names)}

        return result


# 姿态分类结果平滑
class EMADictSmoothing(object):
    """Smoothes pose classification."""

    def __init__(self, window_size=10, alpha=0.2):        # 设置显示窗口的大小
        self._window_size = window_size  # 平滑的时间窗口大小
        self._alpha = alpha  # 指数移动平均的平滑因子

        self._data_in_window = []  # 存储在时间窗口内的姿势分类数据

    def __call__(self, data):
        """Smoothes given pose classification.
        平滑给定的姿态分类
        平滑是通过对给定时间窗内观察到的每个姿态类计算指数移动平均来完成的。错过的姿势类被替换为0。
        Smoothing is done by computing Exponential Moving Average for every pose
        class observed in the given time window. Missed pose classes are replaced
        with 0.

        Args:
          data: Dictionary with pose classification. Sample:
              {
                'pushups_down': 8,
                'pushups_up': 2,
              }

        Result:
          Dictionary in the same format but with smoothed and float instead of
          integer values. Sample:
            {
              'pushups_down': 8.3,
              'pushups_up': 1.7,
            }
        """
        # Add new data to the beginning of the window for simpler code.
        # 将新数据添加到窗口的开头，以获得更简单的代码。
        self._data_in_window.insert(0, data)  # 新的姿势分类数据添加到 _data_in_window 列表的开头
        self._data_in_window = self._data_in_window[:self._window_size]  #保证只取窗口大小的数据，保证每次只取10个数据

        # Get all keys.
        keys = set(  # 获取键名（即姿势类名）set()将列表转换成集合
            [key for data in self._data_in_window for key, _ in data.items()])

        # Get smoothed values.
        smoothed_data = dict()
        for key in keys:
            factor = 1.0
            top_sum = 0.0  # 累加的分子 top_sum
            bottom_sum = 0.0  # 累加的分母 bottom_sum。
            for data in self._data_in_window:
                value = data[key] if key in data else 0.0

                top_sum += factor * value
                bottom_sum += factor

                # Update factor.
                factor *= (1.0 - self._alpha)

            smoothed_data[key] = top_sum / bottom_sum
            print(smoothed_data)
            print(type(smoothed_data))

        return smoothed_data
