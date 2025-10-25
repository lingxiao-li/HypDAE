python -u main.py \
--logdir models/Paint-by-Example/ffhq \
--resume /data2/mhf/DXL/Lingxiao/Codes/Paint-by-Example-test/models/Paint-by-Example/vgg_faces/2024-10-02T02-12-51_v1/checkpoints/epoch=000020.ckpt \
--pretrained_model /data2/mhf/DXL/Lingxiao/Cache/model_weight/anydoor/v2-1_512-ema-pruned.ckpt \
--base configs/v1.yaml \
--scale_lr False