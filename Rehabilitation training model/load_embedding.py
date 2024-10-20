import csv
import numpy as np
import ast
import os


# Initialize an empty dictionary to store the embeddings
embeddings_dict = {}
folder_name = input("Enter name of folder containing CSV files: ")
image_csv_file = os.path.join('squat_embedding_csvs_out', folder_name)
csv_out_embedding_path = os.path.join(image_csv_file, 'up.csv')
print(csv_out_embedding_path)
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
        embeddings_dict[image_name] = embeddings_2d
    print(embeddings_dict)
