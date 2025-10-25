python -u main.py \
--logdir models/Paint-by-Example \
--resume /data2/mhf/DXL/Lingxiao/Codes/Paint-by-Example-test/models/Paint-by-Example/2024-09-11T11-30-12_v1/checkpoints/epoch=000025.ckpt \
--pretrained_model /data2/mhf/DXL/Lingxiao/Cache/model_weight/anydoor/v2-1_512-ema-pruned.ckpt \
--base configs/v1.yaml \
--scale_lr False