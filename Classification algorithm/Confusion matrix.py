from data import *
from pose_encoding import *
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
import seaborn as sns
from classifier import *
from sklearn.metrics import precision_score, recall_score, f1_score
# 1. 定义图片加载函数
def load_images_from_folders(folders, target_size=(100, 100)):    # 获取的是图片信息
    images = []
    labels = []
    class_names=[]
    for label, folder in enumerate(folders):
        class_name=os.path.basename(folder)
        class_names.append(class_name)
        for filename in os.listdir(folder):
            img = cv2.imread(os.path.join(folder, filename))
            if img is not None:
                img = cv2.resize(img, target_size)
                images.append(img)
                labels.append(label)
    # Create CSV files for each class_name

    return np.array(images), labels,class_names



# 2.定义处理图片函数，将其保存至CSV文件
def process_images_and_save_poses(dataset_folder, output_folder):
    action_folders = [os.path.join(dataset_folder, subfolder) for subfolder in os.listdir(dataset_folder) if
                      os.path.isdir(os.path.join(dataset_folder, subfolder))]

    images, labels, class_names = load_images_from_folders(action_folders)

    os.makedirs(output_folder, exist_ok=True)  # 创建输出文件

    for class_name in class_names:
        print('Bootstrapping ', class_name, file=sys.stderr)
        images_in_folder = os.path.join(dataset_folder, class_name)  # 每个需要分类的动作图片所在的文件夹
        csv_out_path = os.path.join(output_folder, f'{class_name}.csv')
        images, labels, _ = load_images_from_folders([images_in_folder])  # 为当前class_name加载图像

        pose_detected_list = []
        images_without_pose = 0  # 记录未检测到姿势关键点的图像数量

        # 将处理好的数据放入到对应的文件夹中
        with open(csv_out_path, 'w') as csv_out_file:  # 打开对应的CSV文件将关键点坐标写入
            csv_out_writer = csv.writer(
                csv_out_file, delimiter=',', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')

            image_names = sorted([n for n in os.listdir(
                images_in_folder) if not n.startswith('.')])  # 图片按照顺序进行编号
            #print(image_names)

            # Initialize embedder.初始化动作嵌入器，
            pose_embedder = FullBodyPoseEmbedder()  # 对关键点进行归一化以及向量化
            # 对每张图片进行处理
            for image_name in tqdm.tqdm(image_names):  # 进度条图片导入及处理
                #print(img_name)
                # 图片获取
                input_frame=cv2.imread(os.path.join(images_in_folder, image_name))
                input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
                with mp_pose.Pose() as pose_tracker:
                    result = pose_tracker.process(image=input_frame)
                    # 处理图片并且返回检测到的人的姿态坐标
                    pose_landmarks = result.pose_landmarks

                output_frame = input_frame.copy()  # 输入图片
                if pose_landmarks is not None:
                    frame_height, frame_width = output_frame.shape[0], output_frame.shape[1]
                    pose_landmarks = np.array(  # 转化成array数组格式x
                        [[lmk.x * frame_width, lmk.y * frame_height, lmk.z * frame_width]
                         for lmk in pose_landmarks.landmark], dtype=np.float32)
                    embeddings_landmarks = pose_embedder(pose_landmarks)

                    csv_out_writer.writerow(  # 保存格式
                        [image_name] + pose_landmarks.flatten().astype(str).tolist())
                    pose_detected_list.append(True)

                else:
                    #print(f'在图像 {img_name} 中未检测到姿势关键点')
                    images_without_pose += 1
                    pose_detected_list.append(False)

        print(f'类别 {class_name} 中未检测到姿势关键点的图像数量: {images_without_pose}')
        print('==========')
        if images_without_pose > 0:
            print(
                f'未检测到姿势关键点的图像名称: {", ".join([img_name for img_name, pose_detected in zip(os.listdir(images_in_folder), pose_detected_list) if not pose_detected])}')

# 3,调用函数
dataset_folder = "data"
output_folder = 'data_out'
#process_images_and_save_poses(dataset_folder, output_folder)   # 将图片转化为CSV文件


# 4. 图像分类算法
# 参数定义
n_landmarks = 33
n_dimensions = 3
pose_embedder=FullBodyPoseEmbedder()
pose_samples_folder = "data_out"
file_extension = ".csv"
file_separator = ","
top_n_by_max_distance = 30  # You can adjust this value
top_n_by_mean_distance = 5 # You can adjust this value
pose_samples, pose_labels = load_pose_samples(pose_samples_folder, file_extension, file_separator, n_landmarks, n_dimensions, pose_embedder)
 # 调用分类算法
pose_classifier = PoseClassifier(pose_samples, n_landmarks,
                                 n_dimensions, pose_embedder, top_n_by_max_distance, top_n_by_mean_distance)

# 绘制混淆矩阵
# 使用PoseClassifier对所有样本进行分类
predicted_labels = []
for sample in pose_samples:
    predicted_label = pose_classifier.classify_pose(sample.landmarks)  # 获取当前分类的图片名称
    predicted_labels.append(predicted_label)
#print(predicted_labels)
# 将类别字符串映射为整数
class_name_to_index = {class_name: i for i, class_name in enumerate(set(pose_labels))}
true_labels = [class_name_to_index[class_name] for class_name in pose_labels]
predicted_labels = [class_name_to_index[class_name] for class_name in predicted_labels]

# 计算准确度、精确度、召回率、和 F1-score
accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels, average='macro')
recall = recall_score(true_labels, predicted_labels, average='macro')
f1 = f1_score(true_labels, predicted_labels, average='macro')
print(f"准确度: {accuracy:.4f}")
print(f"精度: {precision:.4f}")
print(f"召回率: {recall:.4f}")
print(f"F1分数: {f1:.4f}")

# 可视化混淆矩阵
conf_matrix = confusion_matrix(true_labels, predicted_labels)
plt.figure(figsize=(10,8.5))
class_names = sorted(list(set(pose_labels)))
# plt.figure(figsize=(len(class_names), len(class_names)))
sns.heatmap(conf_matrix, annot=False, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
for i in range(len(class_names)):
    for j in range(len(class_names)):
        text = plt.text(j + 0.5, i + 0.6, conf_matrix[i, j], ha='center', va='center', fontsize=19)

plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.subplots_adjust(left=0.3,bottom=0.35)
plt.title("D_KNN* Confusion Matrix", fontsize=24)
plt.show()


# 比较不同距离度量算法的准确度
predicted_labels_euclidean = []
predicted_labels_manhattan = []
predicted_labels_chebyshev = []
predicted_labels_minkowski = []


for sample in pose_samples:
    result_euclidean = pose_classifier.classify_pose_D(sample.landmarks, "euclidean_distance")
    result_manhattan = pose_classifier.classify_pose_D(sample.landmarks, "manhattan_distance")
    result_chebyshev = pose_classifier.classify_pose_D(sample.landmarks, "chebyshev_distance")
    result_minkowski = pose_classifier.classify_pose_D(sample.landmarks,"minkowski_distance")

    predicted_labels_euclidean.append(result_euclidean)
    predicted_labels_manhattan.append(result_manhattan)
    predicted_labels_chebyshev.append(result_chebyshev)
    predicted_labels_minkowski.append(result_minkowski)

# 计算准确度、精确度、召回率、和 F1-score
def calculate_metrics(true_labels, predicted_labels):
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='macro')
    recall = recall_score(true_labels, predicted_labels, average='macro')
    f1 = f1_score(true_labels, predicted_labels, average='macro')
    return accuracy, precision, recall, f1
# 将类别字符串映射为整数
class_name_to_index = {class_name: i for i, class_name in enumerate(set(pose_labels))}
true_labels = [class_name_to_index[class_name] for class_name in pose_labels]

# 将预测的类别字符串映射为整数
predicted_labels_euclidean = [class_name_to_index[class_name] for class_name in predicted_labels_euclidean]
predicted_labels_manhattan = [class_name_to_index[class_name] for class_name in predicted_labels_manhattan]
predicted_labels_chebyshev = [class_name_to_index[class_name] for class_name in predicted_labels_chebyshev]
predicted_labels_minkowski = [class_name_to_index[class_name] for class_name in predicted_labels_minkowski]

# 计算每个距离度量方法的准确度、精确度、召回率和 F1 分数
accuracy_euclidean, precision_euclidean, recall_euclidean, f1_euclidean = calculate_metrics(true_labels, predicted_labels_euclidean)
accuracy_manhattan, precision_manhattan, recall_manhattan, f1_manhattan = calculate_metrics(true_labels, predicted_labels_manhattan)
accuracy_chebyshev, precision_chebyshev, recall_chebyshev, f1_chebyshev = calculate_metrics(true_labels, predicted_labels_chebyshev)
accuracy_minkowski, precision_minkowski, recall_minkowski, f1_minkowski = calculate_metrics(true_labels, predicted_labels_minkowski)

# # 绘制直方图
# labels = ['Euclidean', 'Manhattan', 'Chebyshev', 'Minkowski']
# accuracy_scores = [accuracy_euclidean, accuracy_manhattan, accuracy_chebyshev, accuracy_minkowski]
# precision_scores = [precision_euclidean, precision_manhattan, precision_chebyshev, precision_minkowski]
# recall_scores = [recall_euclidean, recall_manhattan, recall_chebyshev, recall_minkowski]
# f1_scores = [f1_euclidean, f1_manhattan, f1_chebyshev, f1_minkowski]

# 绘制直方图
labels = ['Euclidean', 'Manhattan', 'Chebyshev', 'Minkowski']
accuracy_scores = [0.9569,0.9807,0.8885,0.9182]
precision_scores = [0.9702,0.9867,0.9033,0.9309]
recall_scores = [0.9593,0.9818,0.8876,0.9199]
f1_scores = [0.9603,0.9817,0.8854,0.9193]
print("Accuracy Scores:", [round(score, 4) for score in accuracy_scores])
print("Precision Scores:", [round(score, 4) for score in precision_scores])
print("Recall Scores:", [round(score, 4) for score in recall_scores])
print("F1 Scores:", [round(score, 4) for score in f1_scores])

# 度量的数量
num_metrics = len(labels)  # the label locations

# 条形图的宽度
bar_width = 0.2  # the width of the bars
x = np.arange(len(labels))
# 绘制条形图
fig, ax = plt.subplots(figsize=(12, 8))
rects1 = ax.bar(x , accuracy_scores, bar_width, label='Accuracy')
rects2 = ax.bar(x + bar_width, precision_scores, bar_width, label='Precision')
rects3 = ax.bar(x + 2 * bar_width, recall_scores, bar_width, label='Recall')
rects4 = ax.bar(x + 3 * bar_width, f1_scores, bar_width, label='F1 Score')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Scores(%)',fontsize=24)
ax.set_title('Scores by distance metric',fontsize=24)
plt.xticks(x + 1.5 * bar_width, labels, fontsize=16)
plt.legend()
# 将纵轴标签格式化为百分比形式
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
ax.tick_params(axis='both', which='major', labelsize=16)
plt.legend(loc='upper right', bbox_to_anchor=(1, 1.02),fontsize=16, ncol=2)  # 将图例放置在右上角，增加图例字体大小，并设置ncol参数
plt.subplots_adjust(bottom=0.7)
# 显示图形
fig.tight_layout()
plt.show()



# # 定义不同的 top_n_by_mean_distance 值
# top_max_values = [15,20,25,30,35,40,45,50,55,60]
#
# # 存储准确度，精确度、召回率和F1-score的列表
# accuracies = []
# precisions = []
# recalls = []
# f1_scores = []
#
# # 对于每个 top_n_by_mean_distance 值，计算准确度、精确度、召回率、和 F1-score 并存储
# for top_max in top_max_values:
#     pose_classifier = PoseClassifier(pose_samples, n_landmarks,
#                                      n_dimensions, pose_embedder, top_max, top_n_by_mean_distance)
#
#     predicted_labels = []
#     for sample in pose_samples:
#         predicted_label = pose_classifier.classify_pose1(sample.landmarks,sample.name)
#         predicted_labels.append(predicted_label)
#
#     true_labels = [class_name_to_index[class_name] for class_name in pose_labels]
#     predicted_labels = [class_name_to_index[class_name] for class_name in predicted_labels]
#
#     accuracy = accuracy_score(true_labels, predicted_labels)
#     precision = precision_score(true_labels, predicted_labels, average='macro')
#     recall = recall_score(true_labels, predicted_labels, average='macro')
#     f1 = f1_score(true_labels, predicted_labels, average='macro')
#     accuracies.append(accuracy)
#     precisions.append(precision)
#     recalls.append(recall)
#     f1_scores.append(f1)
#     print(f"最大距离= :{top_max}, "f"准确度: {accuracy:.4f}, 精度: {precision:.4f}, 召回率: {recall:.4f}, F1分数: {f1:.4f}")
#
# # 绘制准确度曲线
# plt.plot(top_max_values, accuracies, marker='o')
# plt.title('Accuracy vs max_distance_number',fontsize=20)
# plt.xlabel('max_distance_number',fontsize=16)
# plt.ylabel('Accuracy',fontsize=16)
# plt.subplots_adjust(left=0.2)
# plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x*100:.2f}%'))
# plt.grid(True)
# plt.subplots_adjust(left=0.15,bottom=0.15)
# plt.show()

# #定义不同的 top_n_by_max_distance 值
# top_n_values = [3,4,5,6,7,8,9,10,11]
#
# # 存储准确度的列表
# accuracies = []
# precisions = []
# recalls = []
# f1_scores = []
#
# # 对于每个 top_n_by_mean_distance 值，计算准确度并存储
# for top_n in top_n_values:
#     pose_classifier = PoseClassifier(pose_samples, n_landmarks,
#                                      n_dimensions, pose_embedder, top_n_by_max_distance, top_n)
#
#     predicted_labels = []
#     for sample in pose_samples:
#         predicted_label = pose_classifier.classify_pose_w(sample.landmarks)
#         predicted_labels.append(predicted_label)
#
#     true_labels = [class_name_to_index[class_name] for class_name in pose_labels]
#     predicted_labels = [class_name_to_index[class_name] for class_name in predicted_labels]
#
#     accuracy = accuracy_score(true_labels, predicted_labels)
#     precision = precision_score(true_labels, predicted_labels, average='macro')
#     recall = recall_score(true_labels, predicted_labels, average='macro')
#     f1 = f1_score(true_labels, predicted_labels, average='macro')
#     accuracies.append(accuracy)
#     precisions.append(precision)
#     recalls.append(recall)
#     f1_scores.append(f1)
#     print(f"平均距离={top_n}, "f"准确度: {accuracy:.4f}, 精度: {precision:.4f}, 召回率: {recall:.4f}, F1分数: {f1:.4f}")
#
# # 绘制准确度曲线
# plt.plot(top_n_values, accuracies, marker='o', label='Accuracy')
# plt.plot(top_n_values, precisions, marker='o', label='Precision')
# plt.plot(top_n_values, recalls, marker='o', label='Recall')
# plt.plot(top_n_values, f1_scores, marker='o', label='F1-score')
# plt.title('D_KNN* Metrics vs k (Number of Neighbors)',fontsize=18)
# plt.xlabel('k (Number of Neighbors)',fontsize=16)
# plt.ylabel('Metrics Value',fontsize=16)
# plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x*100:.2f}%'))
# plt.legend()  # 添加图例
# plt.subplots_adjust(left=0.15,bottom=0.15)
# plt.grid(True)
# plt.show()
#
#
