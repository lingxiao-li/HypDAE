from transformers import AutoImageProcessor, AutoModel, AutoProcessor, CLIPModel
from PIL import Image
import torch.nn as nn
import torch
from os import listdir

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
model = AutoModel.from_pretrained('facebook/dinov2-base').to(device)
# processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
# model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)


image_list_1 = listdir('images_1')
# 最好两组对比的图片是同名的，分别在不同的文件夹下, image_1, image_2
sim_list = []
print(image_list_1)
for img in image_list_1:
    image1 = Image.open(f'images_1/{img}')
    with torch.no_grad():
        inputs1 = processor(images=image1, return_tensors="pt").to(device)
        outputs1 = model(**inputs1)
        image_features1 = outputs1.last_hidden_state
        image_features1 = image_features1.mean(dim=1)

    image2 = Image.open(f'images_2/{img}')
    with torch.no_grad():
        inputs2 = processor(images=image2, return_tensors="pt").to(device)
        outputs2 = model(**inputs2)
        image_features2 = outputs2.last_hidden_state
        image_features2 = image_features2.mean(dim=1)

    cos = nn.CosineSimilarity(dim=0)
    sim = cos(image_features1[0],image_features2[0]).item()
    sim = (sim+1)/2
    print('Similarity:', sim)
    sim_list.append(sim)


# 平均sim
import numpy as np
mean_sim = np.mean(sim_list)

print(mean_sim)