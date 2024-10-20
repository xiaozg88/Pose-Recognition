'''
Descripttion: 
version: 
Author: LiQiang
Date: 2022-02-27 22:04:05
LastEditTime: 2022-02-27 22:35:02
'''

# 人体姿态编码，对坐标进行标准化，归一化。距离向量化
import numpy as np


class FullBodyPoseEmbedder(object):
    """Converts 3D pose landmarks into 3D embedding."""

    def __init__(self, torso_size_multiplier=2.5):
        # Multiplier to apply to the torso to get minimal body size.
        self._torso_size_multiplier = torso_size_multiplier

        # Names of the landmarks as they appear in the prediction.33个关键点名称
        self._landmark_names = [
            'nose',
            'left_eye_inner', 'left_eye', 'left_eye_outer',
            'right_eye_inner', 'right_eye', 'right_eye_outer',
            'left_ear', 'right_ear',
            'mouth_left', 'mouth_right',
            'left_shoulder', 'right_shoulder',
            'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist',
            'left_pinky_1', 'right_pinky_1',
            'left_index_1', 'right_index_1',
            'left_thumb_2', 'right_thumb_2',
            'left_hip', 'right_hip',
            'left_knee', 'right_knee',
            'left_ankle', 'right_ankle',
            'left_heel', 'right_heel',
            'left_foot_index', 'right_foot_index',
        ]

    def __call__(self, landmarks):  # 坐标的保存形式
        """Normalizes pose landmarks and converts to embedding

        Args:
          landmarks - NumPy array with 3D landmarks of shape (N, 3).

        Result:
          Numpy array with pose embedding of shape (M, 3) where `M` is the number of
          pairwise distances defined in `_get_pose_distance_embedding`.
        """
        # 判断是不是有33个关键点
        assert landmarks.shape[0] == len(
            self._landmark_names), 'Unexpected number of landmarks: {}'.format(landmarks.shape[0])

        # Get pose landmarks.
        landmarks = np.copy(landmarks)

        # Normalize landmarks. 返回标准化，归一化的坐标结果
        landmarks = self._normalize_pose_landmarks(landmarks)

        # Get embedding. 向量化的结果
        embedding = self._get_pose_distance_embedding(landmarks)
        # print("embedding-------------")
        # print(embedding)

        return embedding

    def _normalize_pose_landmarks(self, landmarks):
        """Normalizes landmarks translation and scale.标准化，归一化"""
        landmarks = np.copy(landmarks)

        # Normalize translation. 标准化
        pose_center = self._get_pose_center(landmarks)
        landmarks -= pose_center

        # Normalize scale.归一化
        pose_size = self._get_pose_size(landmarks, self._torso_size_multiplier)
        landmarks /= pose_size
        # Multiplication by 100 is not required, but makes it eaasier to debug.
        landmarks *= 100

        return landmarks

    def _get_pose_center(self, landmarks):
        """Calculates pose center as point between hips.将髋关节中点为圆心"""
        left_hip = landmarks[self._landmark_names.index('left_hip')]
        right_hip = landmarks[self._landmark_names.index('right_hip')]
        center = (left_hip + right_hip) * 0.5
        return center
    # embedding就是用一个低维的向量表示一个物体，
    # embedding向量的性质是能使距离相近的向量对应的物体有相近的含义
    # Embedding能够用低维向量对物体进行编码还能保留其含义的特点
    def _get_pose_size(self, landmarks, torso_size_multiplier):
        """Calculates pose size.

        It is the maximum of two values:
          * Torso size multiplied by `torso_size_multiplier`
          * Maximum distance from pose center to any pose landmark
        """
        # This approach uses only 2D landmarks to compute pose size.
        # 这种方法只使用2D地标来计算姿势大小。
        landmarks = landmarks[:, :2]

        # Hips center.
        left_hip = landmarks[self._landmark_names.index('left_hip')]
        right_hip = landmarks[self._landmark_names.index('right_hip')]
        hips = (left_hip + right_hip) * 0.5

        # Shoulders center.
        left_shoulder = landmarks[self._landmark_names.index('left_shoulder')]
        right_shoulder = landmarks[self._landmark_names.index(
            'right_shoulder')]
        shoulders = (left_shoulder + right_shoulder) * 0.5

        # Torso size as the minimum body size.躯干尺寸
        torso_size = np.linalg.norm(shoulders - hips)
        # np.linalg.norm()返回的是绝对值
        # Max dist to pose center.求各个关键点到中心的最大距离
        pose_center = self._get_pose_center(landmarks)
        max_dist = np.max(np.linalg.norm(landmarks - pose_center, axis=1))

        return max(torso_size * torso_size_multiplier, max_dist)
        # 求外切圆用于将人体与y轴平行
    # 初始化动作嵌入器
    def _get_pose_distance_embedding(self, landmarks):
        """Converts pose landmarks into 3D embedding.

        We use several pairwise 3D distances to form pose embedding. All distances
        include X and Y components with sign. We differnt types of pairs to cover
        different pose classes. Feel free to remove some or add new.
         使用几个成对的3D距离来形成姿态向量化。
         方便KNN算法来计算两个向量之间的距离，以找到最接近目标的姿态样本
        Args:
          landmarks - NumPy array with 3D landmarks of shape (N, 3).

        Result:
          Numpy array with pose embedding of shape (M, 3) where `M` is the number of
          pairwise distances.其中M=23，保存了23个特征向量
        """
        embedding = np.array([
            # 上肢.

            # self._get_distance(
            #     self._get_average_by_names(landmarks, 'left_hip', 'right_hip'),
            #     self._get_average_by_names(landmarks, 'left_shoulder', 'right_shoulder')),

            self._get_distance_by_names(
                landmarks, 'left_shoulder', 'left_elbow'), # 肩部到肘部
            self._get_distance_by_names(
                landmarks, 'right_shoulder', 'right_elbow'),

            self._get_distance_by_names(landmarks, 'left_elbow', 'left_wrist'), # 肘部到手腕
            self._get_distance_by_names(
                landmarks, 'right_elbow', 'right_wrist'),

            self._get_distance_by_names(
                landmarks, 'left_shoulder', 'left_wrist'),  # 肩膀到手腕
            self._get_distance_by_names(
                landmarks, 'right_shoulder', 'right_wrist'),

            self._get_distance_by_names(landmarks, 'left_hip', 'left_wrist'),  # 髋关节到手腕
            self._get_distance_by_names(landmarks, 'right_hip', 'right_wrist'),

            self._get_distance_by_names(
                landmarks, 'left_elbow', 'right_elbow'),  # 左手肘到右手肘

            self._get_distance_by_names(
                landmarks, 'left_wrist', 'right_wrist'),  # 左手腕到右手腕


            # Body bent direction.身体弯曲方向

            self._get_distance(
                self._get_average_by_names(landmarks, 'left_wrist', 'left_ankle'),
                landmarks[self._landmark_names.index('left_hip')]),
            self._get_distance(
                self._get_average_by_names(landmarks, 'right_wrist', 'right_ankle'),
                landmarks[self._landmark_names.index('right_hip')]),

            # 下肢
            self._get_distance_by_names(landmarks, 'left_hip', 'left_knee'), # 髋关节到膝盖
            self._get_distance_by_names(landmarks, 'right_hip', 'right_knee'),

            self._get_distance_by_names(landmarks, 'left_knee', 'left_ankle'), # 膝盖到脚踝
            self._get_distance_by_names(
                landmarks, 'right_knee', 'right_ankle'),

            self._get_distance_by_names(landmarks, 'left_hip', 'left_ankle'), # 髋关节到脚踝
            self._get_distance_by_names(landmarks, 'right_hip', 'right_ankle'),

            self._get_distance_by_names(
                landmarks, 'left_shoulder', 'left_ankle'), # 肩膀到脚踝
            self._get_distance_by_names(
                landmarks, 'right_shoulder', 'right_ankle'),

            self._get_distance_by_names(landmarks, 'left_knee', 'right_knee'), # 左膝盖到右膝盖

            self._get_distance_by_names(
                landmarks, 'left_ankle', 'right_ankle'),]) # 左脚踝到右脚踝


        return embedding
    # 获取关键点对之间的距离（为了防止身体弯曲方向不同，而造成的姿势识别错误）
    def _get_average_by_names(self, landmarks, name_from, name_to):
        lmk_from = landmarks[self._landmark_names.index(name_from)]
        lmk_to = landmarks[self._landmark_names.index(name_to)]
        return (lmk_from + lmk_to) * 0.5

    def _get_distance_by_names(self, landmarks, name_from, name_to):
        lmk_from = landmarks[self._landmark_names.index(name_from)]
        lmk_to = landmarks[self._landmark_names.index(name_to)]
        return self._get_distance(lmk_from, lmk_to)

    def _get_distance(self, lmk_from, lmk_to):
        return lmk_to - lmk_from
