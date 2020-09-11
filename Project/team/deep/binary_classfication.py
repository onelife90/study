import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from natsort import natsorted
import glob
import numpy as np

#1. show binary origin image 
# define location of dataset
main_folder = 'D:/project/data/flow_from'
directory = os.listdir(main_folder)

# plot for each folder
for each in directory:
    fig = plt.figure() 
    cur_folder = main_folder + "/" + each
    print(f"cur_folder:{cur_folder}")
    # read 3 images for each folder 
    for i, file in enumerate(os.listdir(cur_folder)[0:3]):
        fullpath = main_folder + "/" + each + "/" + file
        print(fullpath)
        img = mpimg.imread(fullpath)
        fig.add_subplot(2, 3, i+1)
        plt.imshow(img)
    #plt.show()

#2. image down size
# sorted fight images
fight = []    
resized_fight = []  

for filename in natsorted(glob.glob('D:/project/data/flow_from/fighting/*.jpg')):
    fight_open = Image.open(filename) 
    fight.append(fight_open)    

# append resized images to list
for fight_img in fight:       
    fight_img = fight_img.resize((224, 224))
    resized_fight.append(fight_img)  

# save resized images to new folder
for i, new in enumerate(resized_fight):
    new.save(f"{'D:/project/data/binary/fight'},{i},{'.jpg'}") 

# sorted normal images
normal = []
resized_normal = [] 

for filename in natsorted(glob.glob('D:/project/data/flow_from/normal/*.jpg')):
    normal_open = Image.open(filename) 
    normal.append(normal_open)    

# append resized images to list
for normal_img in normal:       
    normal_img = normal_img.resize((224, 224))
    resized_normal.append(normal_img)  

# save resized images to new folder
for i, new in enumerate(resized_normal):
    new.save(f"{'D:/project/data/binary/normal'},{i},{'.jpg'}") 

#3. image dataset
groups_folder_path = "D:/project/data/binary"
categories = ["fight", "normal"]  

x = []
y = []

for index, categorie in enumerate(categories):
    label = [0 for i in range(2)]
    # label[index] = 1

    # image file path
    image_dir = groups_folder_path + categorie + '/'
    files = glob.glob(image_dir +  "*.jpg")

    for img, filename in enumerate(files):
        img = Image.open(filename)
        img = img.convert("RGB")
        data = np.asarray(img)
        x.append(data)
        y.append(label)

x = np.array(x)
y = np.array(y)

print(f"x.shape:{x.shape}")
print(f"y.shape:{y.shape}")

# memory error