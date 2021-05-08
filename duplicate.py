# import numpy as np
# import pandas as pd
# from imagededup.methods import PHash
# import matplotlib.pyplot as plt
# import os
# import pickle

# path = '/resized/'

# def convert_dict_to_df(duplicates_dict):
#     duplicates_list = []
#     scores = []

#     for key, dup_items in duplicates_dict.items():
#         image1 = key.split('_')[0]

#         for item in dup_items:
#             image2 = item[0].split('_')[0]
#             if image1!=image2 and ((image1, image2) not in duplicates_list) and ((image2, image1) not in duplicates_list):
#                     duplicates_list.append((image1, image2))
#                     scores.append(item[1])

#     duplicates_df = pd.DataFrame(duplicates_list, columns=['image1', 'image2'])
#     duplicates_df['score'] = scores
#     return duplicates_df

# def get_images(df, col):
#     mt = [path+(id+'_red.png') for id in df[col].values]
#     er = [path+(id+'_yellow.png') for id in df[col].values]
#     nu = [path+(id+'_blue.png') for id in df[col].values]
#     pr = [path+(id+'_green.png') for id in df[col].values]
#     titles = [title for title in df[col].values]
#     return mt, er, nu, pr, titles

# def read_images(mt, er, nu, pr):
#     mt = plt.imread(mt)
#     er = plt.imread(er)
#     nu = plt.imread(nu)
#     pr = plt.imread(pr)
#     img = np.dstack((mt, pr, nu))
#     return img


# def plot_duplicates(df, nrows=10):
    
#     mt, er, nu, pr, titles = get_images(df, 'image1')
#     mt2, er2, nu2, pr2, titles2 = get_images(df, 'image2')
    
#     fig, ax = plt.subplots(nrows, 2, figsize=(12, nrows*3.5))
#     for i in range(nrows):
#             img = read_images(mt[i], er[i], nu[i], pr[i])
#             img2 = read_images(mt2[i], er2[i], nu2[i], pr2[i])

#             ax[i, 0].set_title('ID: ' + titles[i])
#             ax[i, 0].imshow(img)
#             ax[i, 0].axis('off')
#             ax[i, 1].set_title('ID: ' + titles2[i])
#             ax[i, 1].imshow(img2)
#             ax[i, 1].axis('off')
            
#     plt.tight_layout()                
#     plt.show()

# phash = PHash()
# encodings = phash.encode_images(image_dir=path)
# duplicates = phash.find_duplicates(encoding_map = encodings, scores = True, max_distance_threshold = 8)

import os
from tqdm import tqdm
import numpy as np

import pandas as pd
import cv2

train = pd.read_csv('train.csv')
duplicate = pd.read_csv("duplicates.csv")
label1 = []
label2 = []
for i in range(len(duplicate)):
    lab1 = duplicate.iloc[i]["image1"]
    lab2 = duplicate.iloc[i]["image2"]
    l1 = str(train.loc[train["ID"] == lab1]["Label"].values)
    l2 = str(train.loc[train["ID"] == lab2]["Label"].values)
    label1.append(l1[2:-2])
    label2.append(l2[2:-2])
label1 = np.array(label1)
label2 = np.array(label2)
duplicate["label1"] = label1
duplicate["label2"] = label2
duplicate.to_csv("duplicate&label.csv")



