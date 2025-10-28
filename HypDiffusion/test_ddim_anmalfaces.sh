CUDA_VISIBLE_DEVICES=2 python scripts/test_ddim.py  --ckpt /data2/mhf/DXL/Lingxiao/Codes/Paint-by-Example-test/models/Paint-by-Example/animal_faces/2024-09-11T11-30-12_v1/checkpoints/epoch=000035.ckpt      \
                                --config ./configs/stable-diffusion/v2_animalfaces.yaml \
                                --init-img ./inputs/same_domain_test/animalfaces/test3.jpg       \
                                --ref-img ./inputs/same_domain_test/animalfaces/test3.jpg        \
                                --ddim_steps 50                    \
                                --strength 1.0                     \
                                --scale 7.5                       \
                                --outdir ./outputs/animalfaces                \
                                --skip_save                  \
                                --seed 3408                          