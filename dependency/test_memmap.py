import numpy as np
from decord import VideoReader, cpu
import os
import time

# ---------- 参数配置 ----------
video_path = "source_video/camel10.mp4"  # 替换成你的视频路径
memmap_file = "source_video/video_frames.dat"  # 输出文件路径
batch_size = 100

# ---------- 视频读取 ----------
vid = VideoReader(video_path, ctx=cpu(0))
num_frames = len(vid)

# 获取视频尺寸
sample_frame = vid[0].asnumpy()
H, W, C = sample_frame.shape
print(f"Video shape: ({num_frames}, {H}, {W}, {C})")

# ---------- 创建 memmap ----------
if os.path.exists(memmap_file):
    os.remove(memmap_file)  # 保证干净地重新生成

frames_mmap = np.memmap(memmap_file, dtype='float32', mode='w+', shape=(num_frames, H, W, C))

# ---------- 批量读取写入 ----------
for i in range(0, num_frames, batch_size):
    start = time.time()
    end = min(i + batch_size, num_frames)
    batch_idx = list(range(i, end))

    batch = vid.get_batch(batch_idx).asnumpy().astype("float32") / 255.0
    frames_mmap[i:end] = batch

    print(f"Written frames {i} to {end-1} | Time: {time.time() - start:.3f}s")

frames_mmap.flush()  # 确保写入磁盘
print(f"\n✅ Finished. All frames written to '{memmap_file}'")

# ---------- 验证读取 ----------
frames_read = np.memmap(memmap_file, dtype='float32', mode='r', shape=(num_frames, H, W, C))
print(f"Check frame 0 shape: {frames_read[0].shape}, dtype: {frames_read[0].dtype}")
