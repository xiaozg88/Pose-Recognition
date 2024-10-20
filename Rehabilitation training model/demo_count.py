
from mediapipe.python.solutions import drawing_utils as mp_drawing
import tqdm
import os
from mediapipe.python.solutions import pose as mp_pose
import cv2
from action_count import *
from data import *
from pose_classifier import *
from pose_encoding import *
from utils import *
from image_similarity_calculator import ImageSimilarityCalculator
import time
import pickle
import copy  # 导入 copy 模块

# 指定要训练的动作名称
folder_name = input("Enter name of folder containing CSV files: ")
class_name = 'up'  # 动作识别
out_video_path = 'squat-output.mp4'
# Folder with pose class CSVs. That should be the same folder you using while
# building classifier to output CSVs.
# 带有pose类csv的文件夹。这应该与您在构建分类器以输出csv时使用的文件夹相同。
pose_samples_folder = os.path.join('squat_csvs_out',folder_name)

# 读取视频
video_cap = cv2.VideoCapture(0)
# 参数0表示默认为笔记本的内置第一个摄像头，
# 如果需要读取已有的视频则参数改为视频所在路径路径q
def pocess_frame(img):
    return img
while video_cap.isOpened():
    # ret,frame=video_cap.read()
    # Get some video parameters to generate output video with classificaiton.
    # 得到一些视频参数，生成带有分类的输出视频
    video_n_frames =100
    # video_n_frames = video_cap.get(cv2.CAP_PROP_FRAME_COUNT)     # 输入视频的帧数,实时是无法获取总帧数的，需要给出一个
    video_fps = video_cap.get(cv2.CAP_PROP_FPS)  # 输入视频的帧率
    video_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 输入视频的宽
    video_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 输入视频的高

    # Initialize tracker.初始化关键点跟踪器,识别出人体关键点
    pose_tracker = mp_pose.Pose(static_image_mode=False)       # 识别出关键点

    # Initialize embedder.初始化动作嵌入器，
    pose_embedder = FullBodyPoseEmbedder()     # 对关键点进行归一化以及向量化


    # Initialize classifier. 初始化分类器，将每一帧进行实时分类
    # 注意一定要和做数据预处理的时候用的相同的参数
    # Ceck that you are using the same parameters as during bootstrapping.
    pose_classifier = PoseClassifier(               # 初始化分类器，将每一帧进行实时分类
        pose_samples_folder=pose_samples_folder,    # 样本所在的文件夹个/
        pose_embedder=pose_embedder,   # 分类过程需要对关键点进行向量特征处理
        top_n_by_max_distance=30,
        top_n_by_mean_distance=11)

    # # Uncomment to validate target poses used by classifier and find outliers.
    # outliers = pose_classifier.find_pose_sample_outliers()
    # print('Number of pose sample outliers (consider removing them): ', len(outliers))

    # Initialize EMA smoothing.初始化平滑处理
    pose_classification_filter = EMADictSmoothing(   # 进行滤波处理
        window_size=10,
        alpha=0.2)

    # 指定动作的两个阈值，只有超过这个阈值才会计数
    repetition_counter = RepetitionCounter(     # 计数处理
        class_name=class_name,
        enter_threshold=6,
        exit_threshold=4)

    # Initialize renderer.进行可视化
    pose_classification_visualizer = PoseClassificationVisualizer(    # 可视化界面
        class_name=class_name,
        plot_x_max=video_n_frames,
        # Graphic looks nicer if it's the same as `top_n_by_mean_distance`.
        plot_y_max=10)

    # 初始化评分
    similarity_calculator = ImageSimilarityCalculator(
        pkl_path='top_pose_landmarks_with_confidence.pkl',
        folder_name=folder_name)

    # 初始化一个字典，用于跟踪每个repetitions_count的最高置信度帧
    top_pose_landmarks_with_confidence = {}


    # Open output video.打开输出视频
    # cv2.VideoWriter（filename,fource,fps,frame_size）
    # 第一个参数是文件路径，2是表示压缩帧的codec，3是被创建视频流的帧率，4是视频流的大小
    out_video = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(
        *'mp4v'), video_fps, (video_width, video_height))
    frame_idx = 0
    output_frame = None
    # 进度条，对每一帧进行处理
    with tqdm.tqdm(total=video_n_frames, position=0, leave=True) as pbar:
        while True:
            # Get next frame of the video.
            success, input_frame = video_cap.read()
            # print(success, input_frame)
            if not success:    # 判断是不是捕捉到视频帧
                break

            # Run pose tracker.在当前帧上面执行关键点的推理，此时不需要进行归一化处理，因为我们需要在每一个视频帧上画出关键点并连线

            input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)  # input_frame是从摄像头实时获取的
            result = pose_tracker.process(image=input_frame)    # 将从摄像头获取的每一帧，送入mediapipe的pose模型中去
            pose_landmarks = result.pose_landmarks         # 获取33个关键点的坐标

            # Draw pose prediction.在当前帧画上关键点
            output_frame = input_frame.copy()  # 复制当前帧
            # 如果存在关键点就画出关键点，并将关键点进行连接
            if pose_landmarks is not None:
                mp_drawing.draw_landmarks(
                    image=output_frame,
                    landmark_list=pose_landmarks,
                    connections=mp_pose.POSE_CONNECTIONS)

            if pose_landmarks is not None:   # 处理关键点坐标
                # Get landmarks.获取当前帧关键点
                frame_height, frame_width = output_frame.shape[0], output_frame.shape[1]
                pose_landmarks = np.array([[lmk.x * frame_width, lmk.y * frame_height, lmk.z * frame_width]
                                           for lmk in pose_landmarks.landmark], dtype=np.float32)
                assert pose_landmarks.shape == (
                    33, 3), 'Unexpected landmarks shape: {}'.format(pose_landmarks.shape)

                # Classify the pose on the current frame.对当前帧进行分类
                pose_classification = pose_classifier(pose_landmarks)

                # Smooth classification using EMA.对当前的预测结果进行平滑处理了
                pose_classification_filtered = pose_classification_filter(
                    pose_classification)

                # Count repetitions.对当前结果进行计数
                repetitions_count = repetition_counter(
                    pose_classification_filtered)

                # 保存特征向量embedding
                embedding=pose_embedder(pose_landmarks)

                if repetitions_count >= 1:
                    # 为当前帧创建一个字典条目。
                    pose_confidence = pose_classification_filtered.get(class_name, 0)  # 前帧的分类类别获取其对应的置信度值
                    frame_data = {
                        'frame_idx': frame_idx,
                        'landmarks': pose_landmarks,
                        'confidence': pose_confidence,
                        'embedding': embedding,
                        'frame_image': input_frame,
                        'repetitions_count': repetitions_count
                    }
                    # 更新当前repetitions_count的字典
                    if repetitions_count not in top_pose_landmarks_with_confidence or pose_confidence > \
                            top_pose_landmarks_with_confidence[repetitions_count]['confidence']:
                        top_pose_landmarks_with_confidence[repetitions_count] = frame_data




                # # 更新前一个 repetitions_count
            # 如果没有检测到人体关键点
            else:
                # No pose => no classification on current frame.
                pose_classification = None

                # Still add empty classification to the filter to maintaing correct
                # smoothing for future frames.
                # 仍然把空的分类结果送入到平滑器里面，为了下一帧可以正常的运行
                pose_classification_filtered = pose_classification_filter(dict())
                pose_classification_filtered = None

                # Don't update the counter presuming that person is 'frozen'. Just
                # take the latest repetitions count.
                repetitions_count = repetition_counter.n_repeats




            # 在当前帧上画分类的图和计数
            output_frame = pose_classification_visualizer(   # 可视化的部分
                frame=output_frame,
                pose_classification=pose_classification,
                pose_classification_filtered=pose_classification_filtered,
                repetitions_count=repetitions_count)

            # Save the output frame.
            out_video.write(cv2.cvtColor(
                np.array(output_frame), cv2.COLOR_RGB2BGR))
            # cv2.show('12',cv2.cvtColor(np.array(output_frame)))

            # Show intermediate frames of the video to track progress.
            #if frame_idx % 50 == 0:
                #show_image(output_frame)

            frame_idx += 1
            # print(frame_idx)
            pbar.update()

    # Close output video.


    # Release MediaPipe resources.
    #pose_tracker.close()

    # Show the last frame of the video.q
            if output_frame is not None:
                img = cv2.cvtColor(np.array(output_frame), cv2.COLOR_RGBA2BGRA)  # PIL转cv2

                cv2.imshow('my_window', img)
                # 检查图表是否为空
            # if chart_image is not None:
            #     # 显示折线图的numpy数组
            #     cv2.imshow('Real-time Line Chart', chart_image)
                if cv2.waitKey(1) in [ord('q'), 27]:  # 按键盘上的q或esc退出（在英文输入法下）
                    out_video.release()
                    pose_tracker.close()
                    video_cap.release()
                    cv2.destroyAllWindows()
                    break
# video_cap.release()
# cv2.destroyAllWindows()
# After the video processing is complete, save the list containing top N landmarks and confidences.

with open('top_pose_landmarks_with_confidence.pkl', 'wb') as f:    # 对置信度最高的保存
    pickle.dump(top_pose_landmarks_with_confidence, f)

# Create an instance of the ImageSimilarityCalculator class


# Calculate similarities by providing the folder_name
similarities = similarity_calculator.calculate_similarity()

# Print the calculated similarities
print("All Overall Average Similarities:")
print(similarities)

# Create a directory to save the frames if it doesn't exist
output_dir = 'top_frames'
os.makedirs(output_dir, exist_ok=True)

# Iterate through the frames in top_pose_landmarks_with_confidence
print(type(top_pose_landmarks_with_confidence))
for repetition_count,frame_data in top_pose_landmarks_with_confidence.items():
    confidence = frame_data['confidence']
    frame_idx = frame_data['frame_idx']
    frame = frame_data['frame_image']
    landmark=frame_data ['landmarks']

    # # Define a filename for the frame
    filename = f"top_frame_{frame_idx}_conf_{confidence:.2f}.png"
    with mp_pose.Pose() as pose_tracker:
        result = pose_tracker.process(image=frame)
        # 处理图片并且返回检测到的人的姿态坐标
        pose_landmarks = result.pose_landmarks

    mp_drawing.draw_landmarks(
        image=frame,
        landmark_list=pose_landmarks,
        connections=mp_pose.POSE_CONNECTIONS)  # 将关键点连接起来

    # Save the frame as an image
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    frame_path = os.path.join(output_dir, filename)
    cv2.imwrite(frame_path, frame)

# Print a message indicating where the frames are saved
print(f"Top {len(top_pose_landmarks_with_confidence)} frames saved in '{output_dir}' folder.")
