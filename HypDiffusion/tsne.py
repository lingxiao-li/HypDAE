from email.mime import image
from argparse import Namespace
from transformers import CLIPModel, CLIPProcessor
import os
import clip
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm   
import random
import numpy as np
import sys
import cv2
sys.path.insert(0, '/data2/mhf/DXL/Lingxiao/Codes')
from DeltaHyperEditing.models.delta_hyp_clip import hae_clip


print("Loading Delta CLIP model...")
### 模型构建部分#########################
model_directory = "/data2/mhf/DXL/Lingxiao/Cache/huggingface/hub/models--openai--clip-vit-large-patch14"
model = CLIPModel.from_pretrained(model_directory)
processor = CLIPProcessor.from_pretrained(
    '/data2/mhf/DXL/Lingxiao/Cache/huggingface/hub/models--openai--clip-vit-large-patch14')
sys.path.insert(0, '/data2/mhf/DXL/Lingxiao/Codes')

model_path = '/data2/mhf/DXL/Lingxiao/Codes/DeltaHyperEditing/exp_out/hyper_ffhq_512_5_30_v2/checkpoints/iteration_62500.pt'

ckpt = torch.load(model_path, map_location='cpu')
opts = ckpt['opts']
del ckpt
opts['checkpoint_path'] = model_path
opts['load_mapper'] = True
# instantialize model with checkpoints and args
opts = Namespace(**opts)
net = hae_clip(opts)
print('Model successfully loaded!')
device = "cuda:3" if torch.cuda.is_available() else "cpu"
net.eval()
net.to(device)
########################################
print(f"Running on: {device}")
### 这里随机选了1000个样本，并且把样本序号保存在indecs1.txt文件了
indices = random.sample(range(30000), 1000)

def write_indices_to_file(indices, filename):
    with open(filename, 'w') as f:
        for index in indices:
            f.write(f"{index}\n")

write_indices_to_file(indices, "indices1.txt")
print("Indices have been written to indices1.txt")
########################################

### 设定图像和文本文件路径
image_folder = "/data2/mhf/DXL/Lingxiao/datasets/multi_modal_celeba/dataset/image/images"
text_folder = "/data2/mhf/DXL/Lingxiao/datasets/multi_modal_celeba/dataset/text/celeba-caption"
########################################

# load image
def load_img(path, size=[256, 256]):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    w, h = image.shape[:2]
    # print(f"loaded input image of size ({w}, {h}) from {path}")
    # resize to integer multiple of 32
    # w, h = map(lambda x: x - x % 32, (w, h))
    w, h = size
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LANCZOS4)
    image = np.array(image).astype(np.uint8)

    image = (image / 127.5 - 1.0).astype(np.float32)
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return image

# 提取图像特征
def extract_image_features(image_folder, indices):
    image_features = []
    image_labels = []
    for index in tqdm(indices, desc="Processing images"):
        image_file = f"{index}.jpg"  # 假设图片文件名格式为数字.jpg
        if image_file.endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(image_folder, image_file)
            image = Image.open(image_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                feature = model.get_image_features(**inputs).cpu().numpy()
            image_features.append(feature.flatten())
            image_labels.append(image_file)  
    return image_features, image_labels

# 提取文本特征
def extract_text_features(text_folder, indices):
    text_features = []
    text_labels = []
    for index in tqdm(indices, desc="Processing images"):
        text_file = f"{index}.txt"  # 假设文本文件名格式为数字.txt
        text_path = os.path.join(text_folder, text_file)
        with open(text_path, 'r', encoding='utf-8') as f:
            text = f.readline().strip()  # 读取第一行文本
        # 使用 processor 处理文本输入并获取特征
        inputs = processor(text=[text], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            feature = model.get_text_features(**inputs)
        text_features.append(feature.cpu().numpy().flatten())
        text_labels.append(text_file)  
    return text_features, text_labels


def extract_delta_features(image_folder, text_folder, indices):
    image_features = []
    image_labels = []
    text_features = []
    text_labels = []
    ref_image_path = "/data2/mhf/DXL/Lingxiao/datasets/multi_modal_celeba/dataset/image/images/0.jpg"
    # load ref image
    assert os.path.isfile(ref_image_path)
    ref_image = load_img(ref_image_path, [224, 224]).to(device)
    ref_text = 'The person has high cheekbones, and pointy nose. She is wearing lipstick.'
    
    for index in tqdm(indices, desc="Processing images"):
        # load image
        image_file = f"{index}.jpg"  # 假设图片文件名格式为数字.jpg
        if image_file.endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(image_folder, image_file)
            image = load_img(image_path, [224, 224]).to(device)
            # get delta
            with torch.no_grad():
                logits_1, feature_dist_1, feature_1, ocodes_1, feature_euc_1 = net.forward(
                    ref_image, batch_size=1, return_latents=True)
                logits_2, feature_dist_2, feature_2, ocodes_2, feature_euc_2 = net.forward(
                    image, batch_size=1, return_latents=True)
                clip_feature_1 = net.get_image_features(ref_image)
                clip_feature_2 = net.get_image_features(image)
                delta_c = clip_feature_2 - clip_feature_1
                delta_c_concate = torch.cat([clip_feature_1, delta_c], dim=1)
                delta_s = feature_2 - feature_1
                fake_delta_s_img = net.get_fake_delta_s(
                    delta_c_concate.unsqueeze(1), feature_1)
                image_feature = feature_1 + fake_delta_s_img
                
            image_features.append(image_feature.cpu().numpy().flatten())
            image_labels.append(image_file)
        # load text    
        text_file = f"{index}.txt"  # 假设文本文件名格式为数字.txt
        text_path = os.path.join(text_folder, text_file)
        with open(text_path, 'r', encoding='utf-8') as f:
            text = f.readline().strip()  # 读取第一行文本
        # 使用 processor 处理文本输入并获取特征
        with torch.no_grad():
            fake_delta_s_txt = net.get_fake_delta_s_given_data(
                ref_image, ref_text, text, feature_type='hyperbolic', device=device)
            text_feature = feature_1 + fake_delta_s_txt
        text_features.append(text_feature.cpu().numpy().flatten())
        text_labels.append(text_file)
    
    return image_features, image_labels, text_features, text_labels
'''
print("Extracting image features...")
image_features, image_labels = extract_image_features(image_folder,indices)
print("Extracting text features...")
text_features, text_labels = extract_text_features(text_folder,indices)
'''
print("Extracting delta features...")

batch_size = 256
image_features, image_labels, text_features, text_labels = extract_delta_features(image_folder, text_folder, indices)

# 合并图像和文本特征
features = np.array(image_features + text_features)
# labels = image_labels + text_labels

# 使用 t-SNE 降维
print("Performing t-SNE...")
tsne = TSNE(n_components=2, random_state=42)
features_2d = tsne.fit_transform(features)

# 可视化
print("Visualizing...")
plt.figure(figsize=(12, 8))
# for i, label in enumerate(labels):
#     x, y = features_2d[i]
#     plt.scatter(x, y)
#     # plt.text(x + 0.1, y + 0.1, label, fontsize=8)
image_features_2d = features_2d[:len(image_features)]
text_features_2d = features_2d[len(image_features):]
plt.scatter(image_features_2d[:, 0], image_features_2d[:, 1], color='blue', label='CLIP Image Features')
plt.scatter(text_features_2d[:, 0], text_features_2d[:, 1], color='red', label='CLIP Text Features')
plt.title("t-SNE Visualization of Image and Text Features in Hyperbolic Space")
plt.legend()
plt.savefig("tsne.png")
