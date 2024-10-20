# import pickle
#
# # Load the data from the pickle file
# with open('top_pose_landmarks_with_confidence.pkl', 'rb') as f:
#     top_pose_landmarks_data = pickle.load(f)
#
# # Print the contents of the loaded data
# for frame_data in top_pose_landmarks_data:
#     print(f"Frame Index: {frame_data['frame_idx']}")
#     print(f"Confidence: {frame_data['confidence']}")
#     # print(f"Landmarks: {frame_data['landmarks']}")
#     print(f"Embedding: {frame_data['embedding']}")
#     print("-----------")


import pickle
import csv
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 加载top_pose_landmarks_with_confidence.pkl中的数据
with open('top_pose_landmarks_with_confidence.pkl', 'rb') as f:
    top_frame_data_list = pickle.load(f)

# 提取top_pose_landmarks_with_confidence.pkl中的embedding
top_embeddings = [frame_data['embedding'] for frame_data in top_frame_data_list]

# 将3维的embedding转换为2维
top_embeddings_2d = [embedding.flatten() for embedding in top_embeddings]

# 加载csv中的embedding坐标数据
csv_embeddings = []
with open('squat_embedding_csvs_out/uplift/up.csv', 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    for row in csv_reader:
        image_name = row[0]
        embedding_strs = row[1:]  # 从第二列开始是embedding坐标字符串列表
        embedding = [float(val.strip("[] ")) for val in embedding_strs]  # 将字符串转换为float
        csv_embeddings.append(embedding)

# 将embedding数据转换为numpy数组
top_embeddings_2d = np.array(top_embeddings_2d)
csv_embeddings = np.array(csv_embeddings)

# 计算每个top embedding与csv中所有图片的余弦相似度
cosine_similarities = cosine_similarity(top_embeddings_2d, csv_embeddings)

# 计算每个top embedding的平均余弦相似度
average_cosine_similarities = cosine_similarities.mean(axis=1)

# 计算所有top embeddings的平均平均余弦相似度并转化为百分制
average_top_cosine_similarity = average_cosine_similarities.mean()
cosine_similarity_percentage = average_top_cosine_similarity * 100

print("Average Cosine Similarity Percentage for Top Embeddings:", cosine_similarity_percentage)




import pickle
import csv
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load embeddings from pickle file
with open('top_pose_landmarks_with_confidence.pkl', 'rb') as pkl_file:
    embeddings_data = pickle.load(pkl_file)

top_embeddings = [data['embedding'] for data in embeddings_data]

# Load embeddings from CSV file
csv_embeddings = []
csv_path = 'squat_embedding_csvs_out/uplift/up.csv'
with open(csv_path, 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    for row in csv_reader:
        embedding_strs = row[1:]  # Skip the image name in the first column
        embedding = [float(val.strip(" []")) for val in embedding_strs]
        csv_embeddings.append(embedding)

# Calculate cosine similarities only for corresponding positions
cosine_similarities = [cosine_similarity([top_embeddings[i]], [csv_embeddings[i]])[0][0]
                       for i in range(len(top_embeddings))]

# Calculate average cosine similarity
average_similarity = np.mean(cosine_similarities)

print("Average Cosine Similarity:", average_similarity)
