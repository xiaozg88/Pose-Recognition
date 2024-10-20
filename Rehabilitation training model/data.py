
"""
squat_dataset/
  up/
    image_001.jpg
    image_002.jpg
    ...
  down/
    image_001.jpg
    image_002.jpg
    ...
  ...
"""

# 提取训练集关节点坐标




from PIL import ImageFont
from PIL import ImageDraw
import cv2
from matplotlib import pyplot as plt
import numpy as np
import os
from PIL import Image
from utils import show_image
import sys
import tqdm
import csv
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import pose as mp_pose
from pose_encoding import *
class BootstrapHelper(object):
    """Helps to bootstrap images and filter pose samples for classification.
    帮助引导图像和过滤器的姿势样本分类。"""

    def __init__(self,
                 images_in_folder,    # 输入图片的文件夹地址.
                 images_out_folder,    # 输出图片的保存地址
                 csvs_out_folder,    # 保存关键点CSV文件地址
                 csvs_embedding_out_folder):     # 输出关键点对于特征向量保存CSV文件地址
        self._images_in_folder = images_in_folder
        self._images_out_folder = images_out_folder
        self._csvs_out_folder = csvs_out_folder
        self._csvs_out_embedding_folder=csvs_embedding_out_folder    # 新建一个文件夹用于保存 embedding坐标

        # Get list of pose classes and print image statistics.
        # 获取pose类列表和打印图像统计信息
        self._pose_class_names = sorted(    # 获取分类的名字，在本系统中就包括up和down两种
            [n for n in os.listdir(self._images_in_folder) if not n.startswith('.')])   # 如果文件夹images_in_folder下的不是以“.”开头，就将设置为分类名

    def bootstrap(self, per_pose_class_limit=None):
        """Bootstraps启动 images in a given folder.

        Required image in folder (same use for image out folder):
          pushups_up/
            image_001.jpg
            image_002.jpg
            ...
          pushups_down/
            image_001.jpg
            image_002.jpg
            ...
          ...
        结果形式：
        Produced CSVs out folder: 产生一个csv文件，形式如下
          pushups_up.csv
          pushups_down.csv

        Produced CSV structure with pose 3D landmarks:
          sample_00001,x1,y1,z1,x2,y2,z2,....
          sample_00002,x1,y1,z1,x2,y2,z2,....
        """
        # Create output folder for CVSs.创建一个文件夹用于保存CSV文件
        if not os.path.exists(self._csvs_out_folder):
            os.makedirs(self._csvs_out_folder)
        if not os.path.exists(self._csvs_out_embedding_folder):
            os.makedirs(self._csvs_out_embedding_folder)
        # 新建一个文件夹用于存放生成的CSV文件

        pose_embedder = FullBodyPoseEmbedder()

        # Create a dictionary to store embeddings.
        class_embeddings_dict = {}

        for pose_class_name in self._pose_class_names:   # 遍历每一个类别
            print('Bootstrapping ', pose_class_name, file=sys.stderr)

            # Paths for the pose class.保存到对应的文件夹
            # os.path.join()函数用于路径拼接文件路径，获取对应分类的文件夹或者CSV名称
            images_in_folder = os.path.join(
                self._images_in_folder, pose_class_name) # 路径为images_in_folder/pose_class_name
            images_out_folder = os.path.join(
                self._images_out_folder, pose_class_name)
            csv_out_path = os.path.join(
                self._csvs_out_folder, pose_class_name + '.csv')
            csv_embedding_out_path = os.path.join(
                self._csvs_out_embedding_folder, pose_class_name + '.csv')  # 获取文件夹的路径
            if not os.path.exists(images_out_folder):
                os.makedirs(images_out_folder)
                # 用于存放处理后的图片

            with open(csv_out_path, 'w') as csv_out_file:     # 打开对应的CSV文件将关键点坐标写入
                # 在下面语句中加入“,lineterminator='\n'”，数据写入的时候不产生空行
                # 目的是向CSV写向量文件时，一行行写，不产生空行
                csv_out_writer = csv.writer(
                    csv_out_file, delimiter=',', quoting=csv.QUOTE_MINIMAL,lineterminator='\n')
                # writer对象只引用那些包含特殊字符
                # Create a separate CSV writer for embeddings.
                # with open(csv_embedding_out_path, 'w') as csv_embedding_out_file:
                #     csv_embedding_out_writer = csv.writer(
                #         csv_embedding_out_file, delimiter=',', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')

                # 获取分类名字
                image_names = sorted([n for n in os.listdir(   # 将照片名进行排序
                    images_in_folder) if not n.startswith('.')])
                if per_pose_class_limit is not None:    # 是否对照片个数有要求
                    image_names = image_names[:per_pose_class_limit]

                # Bootstrap every image.
                for image_name in tqdm.tqdm(image_names):  # 进度条
                    # Load image.
                    input_frame = cv2.imread(  # 读入一张图片
                        os.path.join(images_in_folder, image_name))
                    input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
                    # 颜色空间转换，cv2.COLOR_BGR2RGB 将BGR格式转换成RGB格式

                    # Initialize fresh pose tracker and run it.初始化新的姿态跟踪器并运行它。
                    with mp_pose.Pose() as pose_tracker:
                        result = pose_tracker.process(image=input_frame)
                        # 处理图片并且返回检测到的人的姿态坐标
                        pose_landmarks = result.pose_landmarks
                    # landmark {
                    #   x: 0.5501323938369751
                    #   y: 0.6993732452392578
                    #   z: -1.0857504606246948
                    #   visibility: 0.9993054866790771
                    # } 这样的形式存储的

                    # Save image with pose prediction (if pose was detected).保存具有姿态预测的图像
                    output_frame = input_frame.copy()  # 输入图片
                    # 将图片与识别的姿态坐标相互结合
                    if pose_landmarks is not None:   # 图片识别到关键点
                        mp_drawing.draw_landmarks(
                            image=output_frame,
                            landmark_list=pose_landmarks,
                            connections=mp_pose.POSE_CONNECTIONS) #将关键点连接起来
                    output_frame = cv2.cvtColor(
                        output_frame, cv2.COLOR_RGB2BGR)     # 格式转换
                    cv2.imwrite(os.path.join(
                        images_out_folder, image_name), output_frame)   # 将处理后的图片保存到文件夹
                    # 保存图像，第一个参数是要保存为的文件名，第二个参数是要保存的图像

                    # Save landmarks if pose was detected.保存坐标，修改格式
                    if pose_landmarks is not None:
                        # Get landmarks.
                        frame_height, frame_width = output_frame.shape[0], output_frame.shape[1]
                        # output_frame是指处理后的图像（原图与关键点坐标的结合）
                        pose_landmarks = np.array(   #转化成array数组格式
                            [[lmk.x * frame_width, lmk.y * frame_height, lmk.z * frame_width]
                             for lmk in pose_landmarks.landmark],dtype=np.float32)
                       # pose_landmarks.landmark的格式如下：
                        # x: 0.5411455631256104
                        # y: 0.5091587901115417
                        # z: -1.2511787414550781
                        # visibility: 0.9998051524162292,


                        assert pose_landmarks.shape == (   # 判断格式是否正确
                            33, 3), 'Unexpected landmarks shape: {}'.format(pose_landmarks.shape)
                        csv_out_writer.writerow(   # 保存格式
                            [image_name] + pose_landmarks.flatten().astype(np.str).tolist())
                        # flatten()将其转换成一维，astype(np.str)将数组的副本，转换为str类型
                        # tolist()返回一个a.ndim级别的列表

                        # Compute embedding using FullBodyPoseEmbedder.
                        embedding = pose_embedder(pose_landmarks)  # 计算嵌入坐标
                        # Append embedding to the dictionary by class.
                        if pose_class_name not in class_embeddings_dict:
                            class_embeddings_dict[pose_class_name] = []
                        class_embeddings_dict[pose_class_name].append([image_name] + embedding.tolist())

                    # Write embeddings to separate CSV files for each class.
                    for pose_class_name, embeddings_list in class_embeddings_dict.items():
                        csv_embedding_out_path = os.path.join(
                            self._csvs_out_embedding_folder, f'{pose_class_name}.csv')

                        with open(csv_embedding_out_path, 'w') as csv_embedding_out_file:
                            csv_embedding_out_writer = csv.writer(
                                csv_embedding_out_file, delimiter=',', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')

                            # # Write the header to the CSV file.
                            # header = ['image_name'] + [f'embedding_{i}' for i in range(len(embeddings_list[0]) - 1)]
                            # csv_embedding_out_writer.writerow(header)

                            # Write each embedding to the CSV file.
                            for embedding_row in embeddings_list:
                                csv_embedding_out_writer.writerow(embedding_row)


                    # Draw XZ projection and concatenate with the image.绘制XZ投影并与图像连接。
                    projection_xz = self._draw_xz_projection(
                        output_frame=output_frame, pose_landmarks=pose_landmarks)
                    output_frame = np.concatenate(
                        (output_frame, projection_xz), axis=1)


                    # 沿现有轴连接数组序列。
             # Write embeddings to the csv_embedding_out_path file.
            # csv_out_embedding_path = os.path.join(
            #     self._csvs_out_embedding_folder, pose_class_name + '.csv')



    def _draw_xz_projection(self, output_frame, pose_landmarks, r=0.5, color='red'):
        # 投影的意义：Z坐标设置为常数（例如0），将三维坐标投影到了XZ平面上
        frame_height, frame_width = output_frame.shape[0], output_frame.shape[1]  # 获取原始图像的高度和宽度
        img = Image.new('RGB', (frame_width, frame_height), color='white')
        #  创建一个新的 RGB 图像，大小与原始图像相同，背景颜色为白色

        if pose_landmarks is None:
            return np.asarray(img)
         # 如果没有姿势关键点坐标信息，直接返回一个与原始图像大小相同的空图像

        # Scale radius according to the image width.根据图像宽度缩放半径。
        r *= frame_width * 0.01  # 根据图像宽度调整圆圈半径，乘以0.01用于比例缩放

        draw = ImageDraw.Draw(img)     # 一个简单的二维PIL图像绘图界面。
        for idx_1, idx_2 in mp_pose.POSE_CONNECTIONS: # 遍历姿势关键点之间的连接，一共有35对
            # Flip Z and move hips center to the center of the image.翻转Z和移动臀部中心到图像的中心。
            x1, y1, z1 = pose_landmarks[idx_1] * \
                [1, 1, -1] + [0, 0, frame_height * 0.5]  #  Z 坐标翻转并将 Y 坐标调整为图像中心
            x2, y2, z2 = pose_landmarks[idx_2] * \
                [1, 1, -1] + [0, 0, frame_height * 0.5]
            # 绘制关节点并连线
            draw.ellipse([x1 - r, z1 - r, x1 + r, z1 + r], fill=color)  # 以此为半径画圆
            draw.ellipse([x2 - r, z2 - r, x2 + r, z2 + r], fill=color)
            draw.line([x1, z1, x2, z2], width=int(r), fill=color)    # 画一条线，或一串相连的线段。

        return np.asarray(img)  # 将绘制好的图像转换为 NumPy 数组并返回

    def align_images_and_csvs(self, print_removed_items=False):
        """Makes sure that image folders and CSVs have the same sample.
        确保图像文件夹和csv具有相同的样本
        Leaves only intersetion of samples in both image folders and CSVs.
        保留经过处理并且在两个地方都有有效数据的样本。为了保证数据的一致性
        """
        for pose_class_name in self._pose_class_names:
            # Paths for the pose class.输出图像与CSV文件中图片的分类路径
            images_out_folder = os.path.join(  # 输出图片保存路劲
                self._images_out_folder, pose_class_name)
            csv_out_path = os.path.join(   # 输出CSV文件的保存路径
                self._csvs_out_folder, pose_class_name + '.csv')
            csv_embedding_out_path=os.path.join(
                self._csvs_out_embedding_folder,pose_class_name + '.csv'
            )

            # Read CSV into memory.读入内存
            csv_rows = []
            with open(csv_out_path) as csv_out_file:
                csv_out_reader = csv.reader(csv_out_file, delimiter=',')
                for row in csv_out_reader:
                    csv_rows.append(row)   # 将每一行保存在rows列表中



            # Image names left in CSV.图片名在CSV文件的最左侧一列
            image_names_in_csv = []

            # Get the image names that are present in the CSV.
            image_names_in_csv = [row[0] for row in csv_rows]

            # Re-write the CSV removing lines without corresponding images.重写CSV，删除没有对应图像的行。
            # 因为有些图片未能进行识别，对于的CSV文件中的行要删除
            with open(csv_out_path, 'w') as csv_out_file:
                csv_out_writer = csv.writer(
                    csv_out_file, delimiter=',', quoting=csv.QUOTE_MINIMAL,lineterminator='\n')
                for row in csv_rows:
                    image_name = row[0]
                    image_path = os.path.join(images_out_folder, image_name)
                    # os.path.exists()就是判断括号里的文件是否存在的意思，括号内的可以是文件路径。
                    if os.path.exists(image_path):   # 判断在CSV中保存的图像坐标是否在图像文件夹中存在
                        image_names_in_csv.append(image_name)
                        csv_out_writer.writerow(row)
                    elif print_removed_items:
                        print('Removed image from CSV: ', image_path)


            # Remove images without corresponding line in CSV.删除CSV中没有对应行的图像
            for image_name in os.listdir(images_out_folder):
                if image_name not in image_names_in_csv:
                    image_path = os.path.join(images_out_folder, image_name)
                    os.remove(image_path)
                    if print_removed_items:
                        print('Removed image from folder: ', image_path)
                    # Re-write the embedding CSV removing lines without corresponding images.

    def remove_embedding_outliers(self):
        for pose_class_name in self._pose_class_names:
            # Paths for the pose class.输出图像与CSV文件中图片的分类路径
            csv_out_path = os.path.join(   # 输出CSV文件的保存路径
                self._csvs_out_folder, pose_class_name + '.csv')
            csv_embedding_out_path=os.path.join(
                self._csvs_out_embedding_folder,pose_class_name + '.csv')

            # Read CSV into memory.读入内存
            csv_rows = []
            with open(csv_out_path) as csv_out_file:
                csv_out_reader = csv.reader(csv_out_file, delimiter=',')
                for row in csv_out_reader:
                    csv_rows.append(row)  # 将每一行保存在rows列表中
            # Get the image names that are present in the CSV.
            image_names_in_csv = [row[0] for row in csv_rows]
            new_embedding_rows = []
            with open(csv_embedding_out_path, 'r') as csv_embedding_file:
                csv_embedding_reader = csv.reader(csv_embedding_file, delimiter=',')
                for row in csv_embedding_reader:
                    image_name = row[0]
                    if image_name in image_names_in_csv:
                        new_embedding_rows.append(row)


            # Write the filtered embedding rows back to the embedding CSV file.
            with open(csv_embedding_out_path, 'w', newline='') as csv_embedding_file:
                csv_embedding_writer = csv.writer(
                    csv_embedding_file, delimiter=',', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
                csv_embedding_writer.writerows(new_embedding_rows)

    def analyze_outliers(self, outliers):
        """Classifies each sample agains all other to find outliers.
        对每个样本再次进行分类，以找到离群值。
        如果样本与原来的分类不同-它要么被删除，要么应该增加更多类似的样本
        If sample is classified differrrently than the original class - it sould
        either be deleted or more similar samples should be aadded.
        """
        for outlier in outliers:  #寻找离群值
            image_path = os.path.join(
                self._images_out_folder, outlier.sample.class_name, outlier.sample.name)
                # 找到离群值的类名和照片名字
            print('Outlier')
            print('  sample path =    ', image_path)
            print('  sample class =   ', outlier.sample.class_name)
            print('  detected class = ', outlier.detected_class)  # 检测类
            print('  all classes =    ', outlier.all_classes)

            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            show_image(img, figsize=(20, 20))  # 将离群值所在的图片显示出来

    def remove_outliers(self, outliers):  # 移除离群值
        """Removes outliers from the image folders."""
        for outlier in outliers:
            image_path = os.path.join(
                self._images_out_folder, outlier.sample.class_name, outlier.sample.name)
            os.remove(image_path)

    def print_images_in_statistics(self):
        """Prints statistics from the input image folder.从输入的图像文件夹中打印统计信息"""
        self._print_images_statistics(
            self._images_in_folder, self._pose_class_names)

    def print_images_out_statistics(self):
        """Prints statistics from the output image folder."""
        self._print_images_statistics(
            self._images_out_folder, self._pose_class_names)

    # 打印统计信息，每个姿势类的图像数量:
    def _print_images_statistics(self, images_folder, pose_class_names):
        print('Number of images per pose class:')
        for pose_class_name in pose_class_names:
            n_images = len([
                n for n in os.listdir(os.path.join(images_folder, pose_class_name))
                if not n.startswith('.')])
            print('  {}: {}'.format(pose_class_name, n_images))
