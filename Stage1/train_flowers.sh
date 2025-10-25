python -u main.py \
--logdir models/Paint-by-Example/flowers \
--resume /data2/mhf/DXL/Lingxiao/Codes/Paint-by-Example-test/models/Paint-by-Example/flowers/2024-10-12T05-45-07_v1/checkpoints/epoch=000083.ckpt \
--pretrained_model /data2/mhf/DXL/Lingxiao/Cache/model_weight/anydoor/v2-1_512-ema-pruned.ckpt \
--base configs/v1.yaml \
--scale_lr False