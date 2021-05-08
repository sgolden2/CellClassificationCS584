import pandas as pd
from shutil import copyfile
import os
from tqdm import tqdm

train = pd.read_csv("train.csv")
sort = train.sort_values(by=["Label"])
sort.to_csv("sorted.csv",index=False)
# print(sort)

X = [os.path.join('images/train/', i) for i in sort["ID"]]
for x in tqdm(X):
    save_name = x.split('/')[-1]
    if os.path.exists(x+"_green.png") :
        dest = sort.loc[sort["ID"]==save_name]["Label"].values
        dest=str(dest)[2:-2]
        dest = dest.replace("|","&")
        if not os.path.exists("label\\"+dest):
            os.makedirs("label\\"+dest)
        # print("\\"+dest+"\\"+save_name+"_green.png")
        copyfile(x+'_green.png',"label\\"+dest+"\\"+save_name+"_green.png")
        # cv2.imwrite(f'{save_name}_green.png', green)
        
