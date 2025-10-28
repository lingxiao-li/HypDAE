from json import load
import geoopt.manifolds.stereographic.math as gmath
from notebook_utils.pmath import *
from ldm.models.diffusion.dpm_solver_org import DPMSolverSampler
from ldm.models.diffusion.ddim_org import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.util import instantiate_from_config
import time
import cv2
from pytorch_lightning import seed_everything
from contextlib import nullcontext
from torch import autocast
from einops import rearrange, repeat
from itertools import islice
from PIL import Image
from omegaconf import OmegaConf
from tqdm import tqdm, trange
from torchvision.utils import make_grid
import numpy as np
import glob
import random
import re
import torch
import PIL
import argparse
import os
import sys
import shutil
sys.path.insert(0, '/data2/mhf/DXL/Lingxiao/Codes/HypDiffusion')

# we utilize geoopt package for hyperbolic calculation


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


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


def get_unconditional_embedding(model, scale, n_samples, device, prompts):
    # return the learned unconditioning
    if scale != 1.0:
        _, _, _, _, uc = model.get_learned_conditioning(
            n_samples * [torch.zeros((1, 3, 224, 224)).to(device)])
    else:
        _, _, _, _, uc = model.get_learned_conditioning(prompts)

    return uc


def feature_fusion(model, prompt_1, prompt_2, alpha):
    # fuse the feature with different attribute levels in hyperbolic space
    '''
    inputs:
    prompt_1: the first image
    prompt_2: the second image
    alpha: the fusion ratio between the two prompts, value: [0, 6.2126]
    
    outputs:
    fused_hyp_code: the fused latent code in hyperbolic space
    '''
    _, _, _, hyp_code_1, _ = model.get_learned_conditioning(prompt_1)
    _, _, _, hyp_code_2, _ = model.get_learned_conditioning(prompt_2)
    rescaled_hyp_code_1 = rescale(alpha, hyp_code_1)
    rescaled_hyp_code_2 = rescale(alpha, hyp_code_2)
    delta_hyp_code_1 = mobius_add(hyp_code_1, -rescaled_hyp_code_1)
    delta_hyp_code_2 = mobius_add(hyp_code_2, -rescaled_hyp_code_2)
    fused_hyp_code = mobius_add(rescaled_hyp_code_1, delta_hyp_code_2)
    return fused_hyp_code


def get_hyp_codes(model, prompts):
    # return latent codes in the hyperbolic space for the given prompts
    _, _, feature, feature_dist, _ = model.get_learned_conditioning(prompts)
    return feature, feature_dist


def get_hyp_codes_given_feature(model, feature):
    # return latent codes in the hyperbolic space for the given latent codes in CLIP space
    _, _, feature, feature_dist, _ = model.get_learned_conditioning(
        feature, input_feature=False, input_code=True)
    return feature, feature_dist


def get_condition_given_feature(model, feature):
    # return latent codes in the CLIP space for the given latent codes in hyperbolic space
    _, _, _, _, feature_euc = model.get_learned_conditioning(
        feature, input_feature=False, input_code=True)
    return feature_euc


def get_condition_given_hyp_codes(model, hyp_codes):
    # return latent codes in the CLIP space for the given latent codes in hyperbolic space
    _, _, _, _, feature_euc = model.get_learned_conditioning(
        hyp_codes, input_feature=True)
    return feature_euc


# rescale function
def rescale(target_radius, x):
    r_change = target_radius / \
        dist0(gmath.mobius_scalar_mul(
            r=torch.tensor(1), x=x, k=torch.tensor(-1.0)))
    return gmath.mobius_scalar_mul(r=r_change, x=x, k=torch.tensor(-1.0))


# function for generating images with fixed radius (also contains raw geodesic images of 'shorten' images, and stretched images to boundary)
def geo_interpolate_fix_r(x, y, interval, target_radius, save_codes=False):
    feature_geo = []
    feature_geo_normalized = []
    dist_to_start = []
    feature_geo_current_target_boundaries = []
    target_radius_ratio = torch.tensor(target_radius/6.2126)
    geodesic_start_short = gmath.mobius_scalar_mul(
        r=target_radius_ratio, x=x, k=torch.tensor(-1.0))
    geodesic_end_short = gmath.mobius_scalar_mul(
        r=target_radius_ratio, x=y, k=torch.tensor(-1.0))
    index = 0
    for i in interval:
        # this is raw image on geodesic, instead of fixed radius
        feature_geo_current = gmath.geodesic(t=torch.tensor(
            i), x=geodesic_start_short, y=geodesic_end_short, k=torch.tensor(-1.0))

        # here we fix the radius and don't revert them now
        r_change = target_radius / \
            dist0(gmath.mobius_scalar_mul(r=torch.tensor(1),
                  x=feature_geo_current, k=torch.tensor(-1.0)))
        feature_geo.append(feature_geo_current)
        feature_geo_current_target_radius = gmath.mobius_scalar_mul(
            r=r_change, x=feature_geo_current, k=torch.tensor(-1.0))
        feature_geo_normalized.append(feature_geo_current_target_radius)
        dist = gmath.dist(
            geodesic_start_short, feature_geo_current_target_radius, k=torch.tensor(-1.0))
        dist_to_start.append(dist)

        # here is to revert the feature to boundary
        r_change_to_boundary = 6.2126 / \
            dist0(gmath.mobius_scalar_mul(r=torch.tensor(1),
                  x=feature_geo_current, k=torch.tensor(-1.0)))
        feature_geo_current_target_boundary = gmath.mobius_scalar_mul(
            r=r_change_to_boundary, x=feature_geo_current, k=torch.tensor(-1.0))
        feature_geo_current_target_boundaries.append(
            feature_geo_current_target_boundary)

    return feature_geo, feature_geo_normalized, feature_geo_current_target_boundaries, dist_to_start

# function for generating images with fixed radius with optional latent codes list output


def geo_interpolate_fix_r_with_codes(x, y, interval, target_radius):
    # please use this with batch_size = 1
    feature_geo = []
    feature_geo_normalized = []
    dist_to_start = []
    target_radius_ratio = torch.tensor(target_radius/6.2126)
    geodesic_start_short = gmath.mobius_scalar_mul(
        r=target_radius_ratio, x=x, k=torch.tensor(-1.0))
    geodesic_end_short = gmath.mobius_scalar_mul(
        r=target_radius_ratio, x=y, k=torch.tensor(-1.0))
    for i in interval:
        # this is raw image on geodesic, instead of fixed radius
        feature_geo_current = gmath.geodesic(t=torch.tensor(
            i), x=geodesic_start_short, y=geodesic_end_short, k=torch.tensor(-1.0))

        # here we fix the radius and don't revert them now
        r_change = target_radius / \
            dist0(gmath.mobius_scalar_mul(r=torch.tensor(1),
                  x=feature_geo_current, k=torch.tensor(-1.0)))
        feature_geo.append(feature_geo_current)
        feature_geo_current_target_radius = gmath.mobius_scalar_mul(
            r=r_change, x=feature_geo_current, k=torch.tensor(-1.0))
        feature_geo_normalized.append(feature_geo_current_target_radius)
        dist = gmath.dist(
            geodesic_start_short, feature_geo_current_target_radius, k=torch.tensor(-1.0))
        dist_to_start.append(dist)
        # print(feature_geo_current_target_radius.norm())

        # here is to revert the feature to boundary
        r_change_to_boundary = 6.2126 / \
            dist0(gmath.mobius_scalar_mul(r=torch.tensor(1),
                  x=feature_geo_current, k=torch.tensor(-1.0)))
        feature_geo_current_target_boundary = gmath.mobius_scalar_mul(
            r=r_change_to_boundary, x=feature_geo_current, k=torch.tensor(-1.0))
        # print(feature_geo_current_target_boundary.norm())

    return dist_to_start, feature_geo_current, feature_geo_current_target_radius, feature_geo_current_target_boundary


def geo_perturbation(x, distances, perturb_codes, target_radius=6.2126, num_samples=10, save_codes=False):
    """
    该函数在双曲空间围绕给定的点 x 生成随机扰动，并确保扰动点的半径与 x 相同。
    函数会为每个给定的双曲距离生成多个样本。

    参数:
        x (tensor): Poincaré圆盘中的起始点。
        distances (list): 目标的双曲距离列表，例如 [0.01, 0.1, 0.2]。
        target_radius (float): 要将扰动点缩放到的目标半径。
        num_samples (int): 每个距离生成的样本数量。
        save_codes (bool, 可选): 是否保存其他数据，默认为 False。
    
    返回:
        tuple: 包含以下列表的元组：
            - feature_geo: 未归一化的扰动点。
            - feature_geo_normalized: 归一化到目标半径的扰动点。
            - feature_geo_current_target_boundaries: 扰动点缩放到边界。
            - dist_to_start: 扰动点与起始点 x 的双曲距离。
            - perturbation_distances: 每个扰动点与起始点的双曲距离。
    """

    feature_geo = []
    feature_geo_normalized = []
    dist_to_start = []
    feature_geo_current_target_boundaries = []
    perturbation_distances = []  # 保存每个 feature_geo_current 和 geodesic_start_short 之间的距离

    # 1. 计算目标半径的比例，并缩放输入向量 x
    target_radius_ratio = torch.tensor(target_radius / 6.2126)

    # 缩放 x 到目标半径，得到 geodesic_start_short
    geodesic_start_short = gmath.mobius_scalar_mul(
        r=target_radius_ratio, x=x, k=torch.tensor(-1.0))
    print('start_radius', dist0(gmath.mobius_scalar_mul(
        r=torch.tensor(1), x=geodesic_start_short, k=torch.tensor(-1.0))))
    # 外层循环遍历每个距离
    for distance in distances:
        # 内层循环生成每个距离下的多个样本
        for perturb_code in perturb_codes:
            # 2. 生成随机方向
            # random_direction = torch.randn_like(geodesic_start_short)/100
            random_direction = perturb_code

            # print(dist0(gmath.mobius_scalar_mul(r=torch.tensor(1), x=random_direction, k=torch.tensor(-1.0))))
            # 3. 使用 dist0 计算随机方向的双曲距离，并调整使其与 geodesic_start_short 一样长
            r_change_direction_to = target_radius / \
                dist0(gmath.mobius_scalar_mul(r=torch.tensor(1),
                      x=random_direction, k=torch.tensor(-1.0)))
            geodesic_end_short = gmath.mobius_scalar_mul(
                r=r_change_direction_to, x=random_direction, k=torch.tensor(-1.0))
            # print(dist0(gmath.mobius_scalar_mul(r=torch.tensor(1), x=geodesic_end_short, k=torch.tensor(-1.0))))
            # 4. 计算 geodesic_start_short 和 geodesic_end_short 之间的双曲距离
            distance_x_y = gmath.dist(
                geodesic_start_short, geodesic_end_short, k=torch.tensor(-1.0))
            # print('current_distance',distance_x_y)
            # 5. 计算插值比例 t，确保 geodesic_start_short 和 feature_geo_current 之间的双曲距离等于指定的 distance
            if distance_x_y > 0:
                t = distance / distance_x_y  # 插值比例，确保按照双曲距离采样
            else:
                t = torch.tensor(1.0)  # 当距离极小时，设为1

            # 6. 沿着 geodesic_start_short 和 geodesic_end_short 插值生成 feature_geo_current
            feature_geo_current = gmath.geodesic(
                t=t, x=geodesic_start_short, y=geodesic_end_short, k=torch.tensor(-1.0))

            # 7. 保存未归一化的 feature_geo_current
            feature_geo.append(feature_geo_current)

            # print('current_radius', dist0(gmath.mobius_scalar_mul(r=torch.tensor(1), x=feature_geo_current, k=torch.tensor(-1.0))))
            # 8. 修正到目标半径，确保 feature_geo_current 的半径与 target_radius 一致
            r_change = target_radius / \
                dist0(gmath.mobius_scalar_mul(r=torch.tensor(1),
                      x=feature_geo_current, k=torch.tensor(-1.0)))
            # print('change ratio for interpolation sample',r_change)
            feature_geo_current_target_radius = gmath.mobius_scalar_mul(
                r=r_change, x=feature_geo_current, k=torch.tensor(-1.0))
            feature_geo_normalized.append(feature_geo_current_target_radius)

            # 9. 计算扰动点到起始点的双曲距离，并保存
            dist = gmath.dist(
                geodesic_start_short, feature_geo_current_target_radius, k=torch.tensor(-1.0))
            dist_to_start.append(dist)

            # 10. 保存每个 feature_geo_current 和 geodesic_start_short 的双曲距离
            perturbation_distances.append(gmath.dist(
                geodesic_start_short, feature_geo_current, k=torch.tensor(-1.0)))

            # 11. 将扰动点调整到边界
            r_change_to_boundary = 6.2126 / \
                dist0(gmath.mobius_scalar_mul(r=torch.tensor(1),
                      x=feature_geo_current, k=torch.tensor(-1.0)))
            feature_geo_current_target_boundary = gmath.mobius_scalar_mul(
                r=r_change_to_boundary, x=feature_geo_current, k=torch.tensor(-1.0))
            feature_geo_current_target_boundaries.append(
                feature_geo_current_target_boundary)

    return feature_geo, feature_geo_normalized, feature_geo_current_target_boundaries, dist_to_start, perturbation_distances


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                try:
                    load_img(os.path.join(root, fname))
                    path = os.path.join(root, fname)
                    images.append(path)
                except:
                    continue
                
    return images


def find_classes(directory):
    """Finds the class folders in a dataset.
    See :class:`DatasetFolder` for details.
    """
    classes = sorted(entry.name for entry in os.scandir(
        directory) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(
            f"Couldn't find any class folder in {directory}.")

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx


def copy_dataset(path, new_path):
    # copy all images in this dictionary to new_path
    images = make_dataset(path)
    print(len(images))
    for image in tqdm(images):
        img = cv2.imread(image)
        # print(image)
        image_name = image.split('/')[-2] + '_' + image.split('/')[-1]
        if not os.path.exists(new_path):
            os.makedirs(new_path)
        if img is not None:
            cv2.imwrite(new_path + '/' + image_name, img)


def copy_dataset_new(path, new_path):
    # copy all images in this dictionary to new_path
    classes, _ = find_classes(path)
    for class_name in tqdm(classes):
        class_path = os.path.join(path, class_name)
        new_class_path = os.path.join(new_path, class_name)
        images = make_dataset(class_path)
        for image in images:
            img = cv2.imread(image)
            if not os.path.exists(new_class_path):
                os.makedirs(new_class_path)
            if img is not None:
                cv2.imwrite(new_class_path + '/' + image.split('/')[-1], img)


def sample_dataset(path, new_path, size):
    images = make_dataset(path)
    sample_images = random.sample(images, k=size)
    # print(sample_images)

    for image in tqdm(sample_images):
        img = cv2.imread(image)
        # print(img)
        dirs = new_path
        if not os.path.exists(dirs):
            os.makedirs(dirs)
        if img is not None:
            cv2.imwrite(dirs + '/' + image.split('/')[-1], img)


def select_subset(path, new_path, num_classes, num_samples_per_class=None):
    classes, class_to_idx = find_classes(path)
    test_classes = random.sample(classes, k=num_classes)
    train_classes = set(classes)-set(test_classes)
    print(len(test_classes))
    print(len(train_classes))
    for i in range(num_classes):
        test_class = classes[i]
        source_dir = path + '/' + str(test_class)
        destination_dir = new_path + '/' + str(test_class)
        if num_samples_per_class is not None:
            sample_dataset(source_dir, destination_dir, num_samples_per_class)
        else:
            shutil.copytree(source_dir, destination_dir)
        print('finish {}/{}'.format(i, num_classes))
    '''
    for test_class in tqdm(test_classes):
        source_dir = path + 'valid/' + str(test_class)
        destination_dir = new_path + 'test/' + str(test_class)
        shutil.copytree(source_dir, destination_dir)
        '''
    print("Finish creating new dataset!")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image_folder",
        type=str,
        nargs="?",
        help="path to the input image folder"
    )

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/img2img-samples"
    )

    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )

    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save indiviual samples. For speed measurements.",
    )

    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    

    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor, most often 8 or 16",
    )
    parser.add_argument(
        "--n_classes",
        type=int,
        default=1,
        help="how many classes of the dataset to test",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=4,
        help="how many samples to produce for each given prompt. A.k.a batch size",
    )
    parser.add_argument(
        "--n_perturb_samples",
        type=int,
        default=5,
        help="how many samples to produce for each given image.",
    )
    parser.add_argument(
        "--n_selected",
        type=int,
        default=30,
        help="how many images to select for each class.",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--hyp_radius",
        type=float,
        default=5.6,
        help="radius in hyperbolic space for perturbation",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )

    parser.add_argument(
        "--strength",
        type=float,
        default=0.95,
        help="strength for noising/unnoising. 1.0 corresponds to full destruction of information in init image",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    

    opt = parser.parse_args()
    
    
    config_path = opt.config_path
    ckpt = opt.ckpt
    seed = opt.seed
    precision = opt.precision

    # seed_everything(seed)

    config = OmegaConf.load(f"{config_path}")
    model = load_model_from_config(config, f"{ckpt}")

    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    print("Successfully loaded model!")

    sampler = DDIMSampler(model)

    image_folder = opt.image_folder
    outdir = opt.outdir
    # files = glob.glob(opt.sample_folder+'/*.jpg')
    skip_save = opt.skip_save
    ddim_steps = opt.ddim_steps
    ddim_eta = opt.ddim_eta
    n_iter = opt.n_iter
    C = opt.C
    f = opt.f
    n_classes = opt.n_classes
    n_samples = opt.n_samples
    n_perturb_samples = opt.n_perturb_samples
    n_selected = opt.n_selected
    n_rows = opt.n_rows
    scale = opt.scale
    strength = opt.strength
    hyp_radius = opt.hyp_radius
    precision = opt.precision
    sampler.make_schedule(ddim_num_steps=ddim_steps,
                          ddim_eta=ddim_eta, verbose=False)


    assert 0. <= strength <= 1., 'can only work with strength in [0.0, 1.0]'
    precision_scope = autocast if precision == "autocast" else nullcontext
    batch_size = n_samples
    n_rows = n_rows if n_rows > 0 else batch_size
    classes, _ = find_classes(image_folder)
    count_class = 0
    samples = make_dataset(image_folder)
    for class_name in classes:
        class_path = os.path.join(image_folder, class_name)
        images = make_dataset(class_path)
        if n_selected < len(images):
            images = random.sample(images, n_selected)
        count = 0
        for image_path in images:
            assert os.path.isfile(image_path)
            init_image = load_img(image_path, [512, 512]).to(device)
            ref_image = load_img(image_path, [224, 224]).to(device)
            init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
            ref_image = repeat(ref_image, '1 ... -> b ...', b=batch_size)
            # sampling image
            sampled_imgs = random.sample(samples, n_perturb_samples)
            perturbed_codes = []
            logits, ocodes, feature, feature_dist, feature_euc = model.get_learned_conditioning(
                ref_image)
            perturbed_codes.append(feature_euc)
            # load sampled images
            if n_perturb_samples > 0:
                sampled_images = []
                for sampled_img in sampled_imgs:
                    assert os.path.isfile(sampled_img)
                    sampled_image = load_img(sampled_img, [224, 224]).to(device)
                    sampled_images.append(sampled_image)

                _, hyp_code = get_hyp_codes(model, ref_image[0].unsqueeze(0))
                _, perturb_codes = get_hyp_codes(model, sampled_images)
                distances = [hyp_radius]
                num_samples = len(sampled_images)
                feature_geo, feature_geo_normalized, feature_geo_current_target_boundaries, dist_to_start, perturbation_distances = geo_perturbation(
                    hyp_code, distances, target_radius=6.2126, perturb_codes=perturb_codes, num_samples=num_samples)
                for i in feature_geo_current_target_boundaries:
                    feature = repeat(i, '1 ... -> b ...', b=batch_size)
                    feature_euc = get_condition_given_hyp_codes(model, feature)
                    perturbed_codes.append(feature_euc)
                    
            init_latent = model.get_first_stage_encoding(
                model.encode_first_stage(init_image))  # move to latent space
            t_enc = int(strength * ddim_steps)

            uc = get_unconditional_embedding(
                model, scale, n_samples, device, ref_image)

            with torch.no_grad():
                with precision_scope("cuda"):
                    with model.ema_scope():
                        tic = time.time()
                        all_samples = list()
                        shape = [C, 64, 64]
                        # decode it
                        i = 0
                        # for c in rescaled_codes:
                        # for c in interpolated_codes:
                        # for c in reconstruct_codes:
                        for c in perturbed_codes:
                            # for c in compare_codes:
                            # for c in fused_codes:
                            # for c in code:
                            # encode (scaled latent)
                            if strength < 1.0:
                                z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc]*batch_size).to(device))
                            # print(f"z_enc shape: {z_enc.shape}")
                            
                            else:
                                z_enc = torch.randn(
                                    [n_samples, 4, 64, 64], device=device)
                            if strength == 1.0:
                                samples_ddim, _ = sampler.sample(S=ddim_steps,
                                                                conditioning=c,
                                                                batch_size=n_samples,
                                                                shape=shape,
                                                                verbose=False,
                                                                unconditional_guidance_scale=scale,
                                                                unconditional_conditioning=uc,
                                                                eta=ddim_eta,
                                                                x_T=z_enc)
                            else:
                                samples_ddim = sampler.decode(z_enc, c, t_enc, unconditional_guidance_scale=scale,
                                                            unconditional_conditioning=uc)

                            x_samples = model.decode_first_stage(samples_ddim)
                            x_samples = torch.clamp(
                                (x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                            if not skip_save:
                                for x_sample in x_samples:
                                    x_sample = 255. * \
                                        rearrange(x_sample.cpu().numpy(),
                                                'c h w -> h w c')
                                    output_dir = os.path.join(
                                        outdir, image_path.split('/')[-2])
                                    if not os.path.exists(output_dir):
                                        os.makedirs(output_dir)
                                    Image.fromarray(x_sample.astype(np.uint8)).save(
                                        os.path.join(output_dir, str(i) + '_' + image_path.split('/')[-1]))
                                    i += 1

                toc = time.time()
                count += 1
                print('finish {}/{} image'.format(count, len(images)))
                
        count_class += 1
        print('finish {}/{} class'.format(count_class, n_classes))
        if count_class == n_classes:
            break
    print('Saved images to {}'.format(outdir))
    print(f"Your samples are ready and waiting for you \n"
        f" \nEnjoy.")

if __name__ == "__main__":
    main()
