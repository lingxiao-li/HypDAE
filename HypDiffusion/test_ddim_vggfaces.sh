CUDA_VISIBLE_DEVICES=2 python scripts/test_ddim.py  --ckpt /data2/mhf/DXL/Lingxiao/Codes/Paint-by-Example-test/models/Paint-by-Example/vgg_faces/2024-10-02T02-12-51_v1/checkpoints/epoch=000020.ckpt      \
                                --config ./configs/stable-diffusion/v2_vggfaces.yaml \
                                --init-img ./inputs/same_domain_test/vggfaces/2.png       \
                                --ref-img ./inputs/same_domain_test/vggfaces/2.png        \
                                --ddim_steps 50                    \
                                --strength 0.9                     \
                                --scale 7.5                       \
                                --outdir ./outputs/vggfaces                \
                                --skip_save                  \
                                --seed 3408                          