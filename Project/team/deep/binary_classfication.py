import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

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
    plt.show()

#2. 