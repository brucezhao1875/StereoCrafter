import os
import cv2
import numpy as np
from fire import Fire

import torch
from decord import VideoReader, cpu

from transformers import CLIPVisionModelWithProjection
from diffusers import (
    AutoencoderKLTemporalDecoder,
)
from diffusers import UNetSpatioTemporalConditionModel

from pipelines.stereo_video_inpainting import StableVideoDiffusionInpaintingPipeline, tensor2vid
def blend_h(a: torch.Tensor, b: torch.Tensor, overlap_size: int) -> torch.Tensor:
    weight_b = (torch.arange(overlap_size).view(1, 1, 1, -1) / overlap_size).to(
        b.device
    )
    b[:, :, :, :overlap_size] = (1 - weight_b) * a[
        :, :, :, -overlap_size:
    ] + weight_b * b[:, :, :, :overlap_size]
    return b


def blend_v(a: torch.Tensor, b: torch.Tensor, overlap_size: int) -> torch.Tensor:
    weight_b = (torch.arange(overlap_size).view(1, 1, -1, 1) / overlap_size).to(
        b.device
    )
    b[:, :, :overlap_size, :] = (1 - weight_b) * a[
        :, :, -overlap_size:, :
    ] + weight_b * b[:, :, :overlap_size, :]
    return b


'''
这段是大函数，在main方法中被调用。
1、分成tiles；2、逐个tile，调用process_func进行处理；3、tile之间进行过渡性的像素融合;4、最后返回pixels
2、输入的cond_frames,mask_frames：从main方法传入的frames和mask，是一批,大概是frames_warpped[当前起始帧,+23]这么多

3、process_func返回的结果中取出frames[0]，这是什么原因？
这个process_func是 StableVideoDiffusionInpaintingPipeline，在main方法中先构建pipeline，再作为参数调用spatial_tiled_process
查看StableVideoDiffusionInpaintingPipeline的输出格式，可以获知frames[0]是什么：
返回的是，a `tuple` is returned where the first element is a list of list with the generated frames.
那么frames[0]是：a list of list with the generated frames.
# tile_num 显存不够时，可能可以加大tile_num,从而使得4090也能训练4k高清
'''
def spatial_tiled_process(cond_frames, mask_frames, process_func, tile_num, spatial_n_compress=8, **kargs,):
    height = cond_frames.shape[2]       #高度
    width = cond_frames.shape[3]        #宽度

    '''
    这段的逻辑： 每个瓦片之间的重叠处是128个pixel；然后计算出整体需要的tile_size
    注意，tile是上下、左右都按照tile数进行划分的。就是说，如果tile_num=2，那就分成2*2=4片瓦；如果tile_num=4，那就分成4*4=16片瓦
    '''
    tile_overlap = (128, 128)
    tile_size = (    int((height + tile_overlap[0] *  (tile_num - 1)) / tile_num), int((width  + tile_overlap[1] * (tile_num - 1)) / tile_num)  )
    tile_stride = (   (tile_size[0] - tile_overlap[0]),     (tile_size[1] - tile_overlap[1])      )
    
    cols = []
    for i in range(0, tile_num):
        rows = []
        for j in range(0, tile_num):

            cond_tile = cond_frames[:,:,i * tile_stride[0] : i * tile_stride[0] + tile_size[0],j * tile_stride[1] : j * tile_stride[1] + tile_size[1],]
            mask_tile = mask_frames[:,:,i * tile_stride[0] : i * tile_stride[0] + tile_size[0],j * tile_stride[1] : j * tile_stride[1] + tile_size[1],]

            tile = process_func(frames=cond_tile,frames_mask=mask_tile,height=cond_tile.shape[2],width=cond_tile.shape[3],
                                num_frames=len(cond_tile),output_type="latent", **kargs,).frames[0]
            rows.append(tile)
        cols.append(rows)

    # latent_stride 和 latent_overlap 为什么要//spatial_n_compress :
    #
    latent_stride = (
        tile_stride[0] // spatial_n_compress,
        tile_stride[1] // spatial_n_compress,
    )
    latent_overlap = (
        tile_overlap[0] // spatial_n_compress,
        tile_overlap[1] // spatial_n_compress,
    )

    results_cols = []
    for i, rows in enumerate(cols):
        results_rows = []
        for j, tile in enumerate(rows):
            if i > 0:
                tile = blend_v(cols[i - 1][j], tile, latent_overlap[0])
            if j > 0:
                tile = blend_h(rows[j - 1], tile, latent_overlap[1])
            results_rows.append(tile)
        results_cols.append(results_rows)

    pixels = []
    for i, rows in enumerate(results_cols):
        for j, tile in enumerate(rows):
            if i < len(results_cols) - 1:
                # tile 是一个四维的 torch.Tensor，其形状为 (batch_size, channels, height, width)
                tile = tile[:, :, : latent_stride[0], :]
            if j < len(rows) - 1:
                tile = tile[:, :, :, : latent_stride[1]]
            rows[j] = tile
        pixels.append(torch.cat(rows, dim=3)) # 把各个tile先按照dim=3，witdh方向拼接起来
    x = torch.cat(pixels, dim=2) #再按照dim=2，height方向拼接起来
    return x #分块的拼接都完成了，返回x. 它的shape仍然是：(b,c,h,w)

def main(pre_trained_path,    unet_path,    input_video_path,    save_dir,    frames_chunk=23,    overlap=3,    tile_num=1):
    # svd包括了：image_encoder,text_encoder,unet（核心扩散模型）,vae(负责图像压缩和重建)    # 这里只加载了image_encoder    # 要理解svd的全景，需要阅读svd相关资料
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(pre_trained_path,subfolder="image_encoder",variant="fp16",torch_dtype=torch.float16)
    # 这是svd里的另一个组件，用于将视频潜变量解码成完整的视频帧
    vae = AutoencoderKLTemporalDecoder.from_pretrained(pre_trained_path, subfolder="vae", variant="fp16", torch_dtype=torch.float16)
    ## 输入：带噪声的潜变量+文本条件；输出：去噪后的潜变量，送入vae还原为视频帧# 这里使用StereoCrafter替代了svd-xt，因为：StereoCrafter是在svd-xt基础上用左右眼视差的资料训练微调过。 #
    unet = UNetSpatioTemporalConditionModel.from_pretrained(unet_path,subfolder="unet_diffusers",low_cpu_mem_usage=True,torch_dtype=torch.float16)
    image_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    #构建一个pipeline，输入需要修复的视频，输出修复后的视频
    pipeline = StableVideoDiffusionInpaintingPipeline.from_pretrained(pre_trained_path,image_encoder=image_encoder,vae=vae,unet=unet,torch_dtype=torch.float16,)
    pipeline = pipeline.to("cuda")
    os.makedirs(save_dir, exist_ok=True)
    video_name = input_video_path.split("/")[-1].replace(".mp4", "").replace("_splatting_results", "") + "_inpainting_results"

    video_reader = VideoReader(input_video_path, ctx=cpu(0))
    fps = video_reader.get_avg_fps()
    num_frames = len(video_reader)   #总的帧数
    height, width = video_reader[0].shape[0] // 2, video_reader[0].shape[1] // 2    #触发读入第一帧，获取尺寸
    height = height // 128 * 128
    width = width // 128 * 128

    # Create VideoWriter once at the start
    frames_sbs_path = os.path.join(save_dir, f"{video_name}_sbs.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(frames_sbs_path, fourcc, fps, (width * 2, height))

    generated = None

    '''
    for循环，遍历所有frames, frames_chunk=23, overlap=3 , 这样每批处理的23-3=20帧， 这个意思是：有迭加的overlap，又每批处理23个。
    input_frames_i 取自 frames_warpped[当前起始帧,+23];  mask_frames_i 也一样，取自 frames_mask
    然后调用spatial_tiled_process 生成这一批数据，放到 generated中；    
    '''
    for i in range(0, num_frames, frames_chunk - overlap):
        if i + overlap >= num_frames:
            break

        if generated is not None and i + frames_chunk > num_frames:
            cur_i = max(num_frames + overlap - frames_chunk, 0)
            cur_overlap = i - cur_i + overlap
        else:
            cur_i = i
            cur_overlap = overlap

        indices = list(range(cur_i, min(cur_i + frames_chunk, num_frames)))
        frames = video_reader.get_batch(indices)
        frames = torch.tensor(frames.asnumpy()).permute(0, 3, 1, 2).float() # [t,h,w,c] -> [t,c,h,w]

        h, w = frames.shape[2] // 2, frames.shape[3] // 2               #这是从中间结果2*2中获取真正的h，w
        frames_left = frames[:, :, :h, :w]
        frames_mask = frames[:, :, h:, :w]
        frames_warpped = frames[:, :, h:, w:]
        frames = torch.cat([frames_warpped, frames_left, frames_mask], dim=0) #按照batch维，concat起来
        frames = frames[:, :, 0:height, 0:width] / 255.0  #归一化到[0,1]

        frames_warpped, frames_left, frames_mask = torch.chunk(frames, chunks=3, dim=0) # 又分成3块
        frames_mask = frames_mask.mean(dim=1, keepdim=True) #把channel的三通道的mask做平均。不影响是否批量读入

        input_frames_i = frames_warpped.clone()
        mask_frames_i = frames_mask

        if generated is not None:
            try:
                input_frames_i[:cur_overlap] = generated[-cur_overlap:]
            except Exception as e:
                print(e)
                print(
                    f"i: {i}, cur_i: {cur_i}, cur_overlap: {cur_overlap}, input_frames_i: {input_frames_i.shape}, generated: {generated.shape}")

        video_latents = spatial_tiled_process(
            input_frames_i, mask_frames_i, pipeline, tile_num,
            spatial_n_compress=8, min_guidance_scale=1.01, max_guidance_scale=1.01,
            decode_chunk_size=8, fps=7, motion_bucket_id=127, noise_aug_strength=0.0,
            num_inference_steps=8
        ).unsqueeze(0)

        if video_latents.dtype == torch.float16:
            pipeline.vae.to(dtype=torch.float16)

        video_frames = pipeline.decode_latents(
            video_latents, num_frames=video_latents.shape[1], decode_chunk_size=2)
        video_frames = tensor2vid(video_frames, pipeline.image_processor, output_type="pil")[0]

        video_tensor = torch.stack([
            torch.tensor(np.array(img)).permute(2, 0, 1).to(dtype=torch.float32) / 255.0
            for img in video_frames
        ])

        generated = video_tensor
        if i != 0: #如果不是首批生成,就有重叠，要去掉重叠项
            generated = generated[cur_overlap:]
            frames_left = frames_left[cur_overlap:]

        for k in range(generated.shape[0]):
            frame_left = frames_left[k] * 255
            frame_right = generated[k] * 255
            frame_left = frame_left.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            frame_right = frame_right.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            frame_sbs = np.concatenate([frame_left, frame_right], axis=1)
            # 不调用write_video_opencv，直接写入每一帧
            writer.write(frame_sbs.astype(np.uint8)[:, :, ::-1])  # Convert RGB to BGR and write

        del frames, frames_left, frames_mask, frames_warpped, video_frames, video_tensor
        torch.cuda.empty_cache()

    writer.release()


if __name__ == "__main__":
    Fire(main)
