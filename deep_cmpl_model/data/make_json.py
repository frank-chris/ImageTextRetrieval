import json
import csv
import random
import os
from math import floor

random.seed(10)

def get_len(csv_path,header=True):
    # Returns no of samples in the dataset
    with open(csv_path,"r") as csv_file:
        csv_reader=csv.reader(csv_file,delimiter=',')
        line_count=0
        l=[]
        for row in csv_reader:
            l.append(row[2])
            line_count+=1
        return line_count-1


def generate_split(num_samples,train_perc,val_perc):
    ids=list(range(num_samples))
    # print(len(ids))
    random.shuffle(ids)
    train_size = floor(num_samples*train_perc)
    val_size = floor(num_samples*val_perc)
    train_ids = ids[:train_size]
    val_ids = ids[train_size:train_size+val_size]
    test_ids=ids[train_size+val_size:]

    return train_ids,val_ids,test_ids

def make_file(num_samples=None):
    if num_samples==None:
        num_samples=get_len(csv_path)
    print(num_samples)
    # num_samples = 12305
    # train -> 10000, test-> 2305
    train_ids,val_ids,test_ids = generate_split(num_samples,0.8,0.1)

    data_list=[]
    id = 1
    with open(csv_path,"r") as csv_file:
        csv_reader=csv.reader(csv_file,delimiter=',')
        line_no=0
        for row in csv_reader:
            line_no+=1
            if line_no==1:
                continue
            # if(line_no>num_samples):
            #   break
            description, image_id = row[1],int(row[2])
            sample_dict={}
            if image_id in train_ids:
                split="train"
            elif image_id in val_ids:
                split="val"
            elif image_id in test_ids:
                split="test"
            else:
                # print("**",image_id)
                # raise Exception("Sample not alloted")
                continue
            
            sample_dict["split"] = split
            sample_dict["captions"] = [description]
            sample_dict["file_path"] = os.path.join(img_path,str(line_no-2)+".jpg")
            sample_dict["processed_tokens"]=[[]]
            sample_dict["id"]=image_id
            id+=1

            data_list.append(sample_dict)
        
    sorted(data_list,key=lambda x:x["id"])

    with open(out_path,"w") as f:
        json.dump(data_list,f)


if __name__=="__main__":
    parent_folder = "/content/Image_Text_Retrieval/deep_cmpl_model"
    csv_path = parent_folder + "/data/images.csv"
    img_path = "dataset"
    out_path = parent_folder + "/data/reid_raw.json"
    make_file()



# {"split": "train", 
# "captions": 
# ["A pedestrian with dark hair is wearing red and white shoes, a black hooded sweatshirt, and black pants.", 
# "The person has short black hair and is wearing black pants, a long sleeve black top, and red sneakers."],
#  "file_path": "CUHK01/0363004.png",
#   "processed_tokens": [["a", "pedestrian", "with", "dark", "hair", "is", "wearing", "red", "and", "white", "shoes", "a", "black", "hooded", "sweatshirt", "and", "black", "pants"],
#    ["the", "person", "has", "short", "black", "hair", "and", "is", "wearing", "black", "pants", "a", "long", "sleeve", "black", "top", "and", "red", "sneakers"]], 
#   "id": 1}