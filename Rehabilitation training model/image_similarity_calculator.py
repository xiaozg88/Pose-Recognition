import csv
import os
import ast
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity




class ImageSimilarityCalculator(object):
    def __init__(self, pkl_path, folder_name):
        self.pkl_path = pkl_path
        self.folder_name = folder_name

    def load_frame_embeddings(self):
        # Load the data from top_pose_landmarks_with_confidence.pkl
        with open(self.pkl_path, 'rb') as pkl_file:
            data = pickle.load(pkl_file)

        print("data的类型",type(data))
        # print(data)
        # Initialize an empty dictionary to store the embeddings
        frame_embeddings_dict = {}

        # Process each frame data
        for frame_idx,frame_data in data.items():
            frame_idx = frame_data['frame_idx']
            embedding = frame_data['embedding']

            # Convert the embedding to a 2D NumPy array
            embedding_2d = np.array(embedding)

            # Store the 2D array in the dictionary with frame_idx as key
            frame_embeddings_dict[frame_idx] = embedding_2d
        return frame_embeddings_dict

    def image_frame_embeddings(self):
        image_csv_file=os.path.join('squat_embedding_csvs_out', self.folder_name )
        csv_out_embedding_path = os.path.join(image_csv_file,'up.csv')
        # Initialize an empty dictionary to store the embeddings
        image_embeddings_dict = {}

        # Read embeddings from CSV and populate the dictionary
        with open(csv_out_embedding_path, 'r') as csv_out_embedding_file:
            csv_reader = csv.reader(csv_out_embedding_file)
            for row in csv_reader:
                image_name = row[0]
                embedding_strs = row[1:]  # Extract embedding strings

                # Parse each embedding and convert to NumPy array
                embeddings = [np.array(ast.literal_eval(embedding)) for embedding in embedding_strs]

                # Convert the list of arrays into a 2D NumPy array
                embeddings_2d = np.array(embeddings)

                # Store the 2D array in the dictionary
                image_embeddings_dict[image_name] = embeddings_2d
            return image_embeddings_dict

    def calculate_similarity(self):
        # Load frame embeddings
        frame_embeddings_dict = self.load_frame_embeddings()

        # Construct the CSV path
        image_embeddings_dict = self.image_frame_embeddings()

        # 初始化一个列表来存储每次计算得到的平均相似度
        all_average_similarities = []

        for frame_idx, frame_embedding in frame_embeddings_dict.items():
            # 初始化一个列表来存储每个frame_embedding与所有image_embeddings的相似度
            similarities = []

            for image_embedding in image_embeddings_dict.values():
                num_rows = frame_embedding.shape[0]
                similarity_sum = 0.0

                for i in range(num_rows):
                    vec1 = frame_embedding[i, :]
                    vec2 = image_embedding[i, :]
                    similarity = cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1))
                    similarity_sum += similarity[0][0]

                # 计算平均相似度并保存到similarities列表
                average_similarity = similarity_sum / num_rows
                similarities.append(average_similarity)

            # 计算所有相似度的平均值
            overall_average_similarity = sum(similarities) / len(similarities)
            all_average_similarities.append(overall_average_similarity * 100)
        return all_average_similarities








