python demo/image_seq_demo.py \
    "/pub0/data/ivision/DJI_Data/Week-46-Sat-Aug-09-2025/Week-46-Sat-Aug-09-2025-LiDAR/" \
    --masa_config configs/masa-gdino/masa_gdino_swinb_inference.py \
    --masa_checkpoint checkpoints/gdino_masa.pth \
    --texts "vehicle . truck . excavator . crane . person . car . bulldozer . construction equipment" \
    --out results/day2_full.mp4 \
    --save_dir results/day2_full_frames \
    --save_json results/day2_full_tracks.json \
    --resize 1920 --score-thr 0.3 --fp16