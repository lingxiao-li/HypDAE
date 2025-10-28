CUDA_VISIBLE_DEVICES=0  python -m pytorch_fid /data2/mhf/DXL/Lingxiao/Codes/HypDiffusion/outputs/flowers_eva_random_genrated_cfg_1.7_strength_0.95_r_5.5_10_samples_12_each_fid /data2/mhf/DXL/Lingxiao/datasets/flowers_eva_random/test_fid \
                                --device cuda:0 \
                                --model-type InceptionV3