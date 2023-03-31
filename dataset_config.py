import PIL
import os
import os.path
from PIL import Image
import shutil

folder1 = "NORMAL"
folder2 = "PNEUMONIA"
output_folder = "DATASET"

for file in os.listdir(folder1):
    folder1_img = folder1+"/"+file
    img = Image.open(folder1_img)
    img = img.resize((200,200))
    img.save(folder1_img)
    source = os.path.join(folder1, file)
    end = os.path.join(output_folder, file)
    shutil.copy2(source, end)


for file in os.listdir(folder2):
    folder2_img = folder2+"/"+file
    img = Image.open(folder2_img)
    img = img.resize((200,200))
    img.save(folder2_img)
    source = os.path.join(folder2, file)
    end = os.path.join(output_folder, file)
    shutil.copy2(source, end)



