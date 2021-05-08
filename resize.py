import os
from tqdm import tqdm

import pandas as pd
import cv2

df = pd.read_csv('train.csv')
X = df['ID']
X = [os.path.join('images/train/', i) for i in X]
image_size = 768

for x in tqdm(X):
    save_name = x.split('/')[-1]
    if os.path.exists(x+"_red.png") :
        red = cv2.imread(x+'_red.png')[..., 0]
        red = cv2.resize(red, (image_size, image_size))
        cv2.imwrite(f'{save_name}_red.png', red)

        green = cv2.imread(x+'_green.png')[..., 0]
        green = cv2.resize(green, (image_size, image_size))
        cv2.imwrite(f'{save_name}_green.png', green)

        blue = cv2.imread(x+'_blue.png')[..., 0]
        blue = cv2.resize(blue, (image_size, image_size))
        cv2.imwrite(f'{save_name}_blue.png', blue)

        yellow = cv2.imread(x+'_yellow.png')[..., 0]
        yellow = cv2.resize(yellow, (image_size, image_size))
        cv2.imwrite(f'{save_name}_yellow.png', yellow)