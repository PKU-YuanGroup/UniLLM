#!/bin/bash

# 输入目录
input_dir="/storage/zhubin/Janus-MoE/videos_clip_v6_20241203"
# 输出目录
output_dir="/storage/zhubin/Janus-MoE/videos_clip_v6_20241203_split"

# 创建输出目录
mkdir -p "$output_dir"

# 遍历输入目录中的所有视频文件
for video in "$input_dir"/*/*; do
  # 检查是否是文件
  if [ -f "$video" ]; then
    # 获取视频时长（秒）
    duration=$(ffmpeg -i "$video" 2>&1 | grep "Duration" | awk '{print $2}' | tr -d , | awk -F: '{print ($1*3600)+($2*60)+$3}')
    
    # 计算切割点（中间点）
    split_point=$(echo "$duration / 2" | bc -l)
    
    # 生成随机文件名（0 到 100 万）
    random_name1=$(shuf -i 0-1000000 -n 1)
    random_name2=$(shuf -i 0-1000000 -n 1)
    
    # 切割视频
    ffmpeg -i "$video" -ss 0 -to "$split_point" -c copy "$output_dir/${random_name1}.mp4"
    ffmpeg -i "$video" -ss "$split_point" -to "$duration" -c copy "$output_dir/${random_name2}.mp4"
    
    echo "已切割: $video -> ${random_name1}.mp4 和 ${random_name2}.mp4"
  fi
done

echo "所有视频切割完成！"