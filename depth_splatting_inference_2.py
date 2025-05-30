import time
import sys
import gc
import cv2
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.io import write_video

from diffusers.training_utils import set_seed
from fire import Fire
from decord import VideoReader, cpu

from dependency.DepthCrafter.depthcrafter.depth_crafter_ppl import DepthCrafterPipeline
from dependency.DepthCrafter.depthcrafter.unet import DiffusersUNetSpatioTemporalConditionModelDepthCrafter
from dependency.DepthCrafter.depthcrafter.utils import vis_sequence_depth

from Forward_Warp import forward_warp


def read_video_frames(video_path, process_length, target_fps, max_res, dataset="open"):
    t0 = time.time()
    if dataset == "open":
        print("==> processing video: ", video_path)
        vid = VideoReader(video_path, ctx=cpu())
        print("==> original video shape: ", (len(vid), *vid.get_batch([0]).shape[1:]))
        original_height, original_width = vid.get_batch([0]).shape[1:3]
        height = round(original_height / 64) * 64
        width = round(original_width / 64) * 64
        if max(height, width) > max_res:
            scale = max_res / max(original_height, original_width)
            height = round(original_height * scale / 64) * 64
            width = round(original_width * scale / 64) * 64
    else:
        height = dataset_res_dict[dataset][0]
        width = dataset_res_dict[dataset][1]

    vid = VideoReader(video_path, ctx=cpu(), width=width, height=height)
    vid_orig = VideoReader(video_path, ctx=cpu())
    fps = vid.get_avg_fps() if target_fps == -1 else target_fps
    orig_fps = vid_orig.get_avg_fps()
    stride = round(vid.get_avg_fps() / fps)
    stride = max(stride, 1)
    frames_idx = list(range(0, len(vid), stride))
    print(
        f"==> downsampled shape: {len(frames_idx), *vid.get_batch([0]).shape[1:]}, with stride: {stride}"
    )
    if process_length != -1 and process_length < len(frames_idx):
        frames_idx = frames_idx[:process_length]
    print(
        f"==> final processing shape: {len(frames_idx), *vid.get_batch([0]).shape[1:]}.Now begin read in frames by vid.get_batch()")
    frames = []
    frames_orig = []
    batch_size = 10  # readin each batch
    total_frames = len(frames_idx)
    # 提前申请空间（注意大小写）
    print("Now i will try to allocate mem for all frames . ")
    t1 = time.time()
    # 计算内存大小（以 MB 为单位）
    frame_mem_GB = height * width * 3 * total_frames * 4 / 1024 / 1024 /1024 # float32 每个数值占 4 字节
    frame_orig_mem_GB = original_height * original_width * 3 * total_frames * 4 / 1024 / 1024 / 1024

    print(f"==> Allocating memory for downsampled frames: {frame_mem_GB:.2f} GB , Allocating memory for original frames: {frame_orig_mem_GB:.2f} GB")
    frames = np.empty((total_frames, height, width, 3), dtype=np.float32)
    frames_orig = np.empty((total_frames, original_height, original_width, 3), dtype=np.float32)
    t2 = time.time()
    print(f"==> in read_video_frames() , allocat space success.total costs: {t2 - t1:.2f} seconds")
    total_batches = (len(frames_idx) + batch_size - 1) // batch_size
    start_read = time.time()

    for i in range(0, len(frames_idx), batch_size):
        batch_start_time = time.time()
        batch_indices = frames_idx[i:i + batch_size]

        batch_frames = vid.get_batch(batch_indices).asnumpy().astype("float32") / 255.0
        batch_frames_orig = vid_orig.get_batch(batch_indices).asnumpy().astype("float32") / 255.0

        frames[i:i + len(batch_indices)] = batch_frames
        frames_orig[i:i + len(batch_indices)] = batch_frames_orig

        batch_time = time.time() - batch_start_time
        elapsed_time = time.time() - start_read
        percent = (i + batch_size) / len(frames_idx)
        percent = min(percent, 1.0)

        sys.stdout.write(
            f"\rReading video frames: [{int(percent * 100):3d}%] "
            f"Batch {i // batch_size + 1}/{total_batches}, "
            f"Batch time: {batch_time:.2f}s, "
            f"Elapsed: {elapsed_time:.2f}s"
        )
        sys.stdout.flush()

    print()  # 换行
    t3 = time.time()
    print(
        f"==> read in frames successful. read_video_frames() finished. total: {t3 - t0:.2f}s | "
        f"alloc: {t2 - t1:.2f}s | read: {t3 - t2:.2f}s")
    return frames, fps, frames_orig, orig_fps, original_height, original_width


class DepthCrafterDemo:
    def __init__(
            self,
            unet_path: str,
            pre_trained_path: str,
            cpu_offload: str = "model",
    ):
        # DiffusersUNetSpatioTemporalConditionModelDepthCrafter来自DepthCrafter
        unet = DiffusersUNetSpatioTemporalConditionModelDepthCrafter.from_pretrained(
            unet_path,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
        )
        # load weights of other components from the provided checkpoint
        self.pipe = DepthCrafterPipeline.from_pretrained(
            pre_trained_path,
            unet=unet,
            torch_dtype=torch.float16,
            variant="fp16",
        )

        # for saving memory, we can offload the model to CPU, or even run the model sequentially to save more memory
        # 缺省 cpu_offload = "model" 不写的话，pipe.to("cuda")，不知道是否会变快。
        if cpu_offload is not None:
            if cpu_offload == "sequential":
                # This will slow, but save more memory
                self.pipe.enable_sequential_cpu_offload()
            elif cpu_offload == "model":
                self.pipe.enable_model_cpu_offload()
            else:
                raise ValueError(f"Unknown cpu offload option: {cpu_offload}")
        else:
            self.pipe.to("cuda")
        # enable attention slicing and xformers memory efficient attention
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
        except Exception as e:
            print(e)
            print("Xformers is not enabled")
        self.pipe.enable_attention_slicing()

    def infer(
            self,
            input_video_path: str,
            output_video_path: str,
            process_length: int = -1,
            num_denoising_steps: int = 8,  # 每一个轮次，会有8次循环，到进度条100%。就是这里设置的。
            guidance_scale: float = 1.2,  # 这个不知道是什么
            window_size: int = 70,
            overlap: int = 25,  # 这个看起来是两个任务之间的重叠帧?
            max_res: int = 1024,  # 这个是什么作用？看起来是限制最大资源数
            dataset: str = "open",
            target_fps: int = -1,
            seed: int = 42,
            track_time: bool = False,
            save_depth: bool = False,
    ):
        set_seed(seed)

        # 注意：这里的frames也是全量读入的，也就是说：如果一个视频足够长，那么在这一个环节应该也会遇到内存不足导致失败的。
        frames, target_fps, frames_orig, orig_fps, original_height, original_width = read_video_frames(input_video_path,
                                                                                                       process_length,
                                                                                                       target_fps,
                                                                                                       max_res,
                                                                                                       dataset, )
        # inference the depth map using the DepthCrafter pipeline
        # 这里用到了over_lap. 有个问题是：frames是整个视频的全量frames，那么能否实现frames分批读入？

        print("now i will try to infer the depth...")
        t0 = time.time()
        with torch.inference_mode():
            res = self.pipe(frames, height=frames.shape[1], width=frames.shape[2], output_type="np",
                            guidance_scale=guidance_scale, num_inference_steps=num_denoising_steps,
                            window_size=window_size, overlap=overlap, track_time=track_time,
                            ).frames[0]

        # convert the three-channel output to a single channel depth map
        res = res.sum(-1) / res.shape[-1]

        # resize the depth to the original size
        tensor_res = torch.tensor(res).unsqueeze(1).float().contiguous().cuda()
        res = F.interpolate(tensor_res, size=(original_height, original_width), mode='bilinear', align_corners=False)
        res = res.cpu().numpy()[:, 0, :, :]

        # normalize the depth map to [0, 1] across the whole video
        res = (res - res.min()) / (res.max() - res.min())
        # visualize the depth map and save the results
        vis = vis_sequence_depth(res)
        # save the depth map and visualization with the target FPS
        save_path = os.path.join(
            os.path.dirname(output_video_path), os.path.splitext(os.path.basename(output_video_path))[0]
        )

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if save_depth:
            np.savez_compressed(save_path + ".npz", depth=res)
            write_video(save_path + "_depth_vis.mp4", vis * 255.0, fps=target_fps, video_codec="h264",
                        options={"crf": "16"})

        t1 = time.time()
        print(f"finished infer the depth.it costs {t1 - t0:.2f} seconds.")
        return frames_orig, orig_fps, res, vis


class ForwardWarpStereo(nn.Module):
    def __init__(self, eps=1e-6, occlu_map=False):
        super(ForwardWarpStereo, self).__init__()
        self.eps = eps
        self.occlu_map = occlu_map
        self.fw = forward_warp()

    def forward(self, im, disp):
        """
        :param im: BCHW
        :param disp: B1HW
        :return: BCHW
        detach will lead to unconverge!!
        """
        im = im.contiguous()
        disp = disp.contiguous()
        # weights_map = torch.abs(disp)
        weights_map = disp - disp.min()
        weights_map = (
                          1.414
                      ) ** weights_map  # using 1.414 instead of EXP for avoding numerical overflow.
        flow = -disp.squeeze(1)
        dummy_flow = torch.zeros_like(flow, requires_grad=False)
        flow = torch.stack((flow, dummy_flow), dim=-1)
        res_accum = self.fw(im * weights_map, flow)
        # mask = self.fw(weights_map, flow.detach())
        mask = self.fw(weights_map, flow)
        mask.clamp_(min=self.eps)
        res = res_accum / mask
        if not self.occlu_map:
            return res
        else:
            ones = torch.ones_like(disp, requires_grad=False)
            occlu_map = self.fw(ones, flow)
            occlu_map.clamp_(0.0, 1.0)
            occlu_map = 1.0 - occlu_map
            return res, occlu_map


def DepthSplatting(input_video_path, output_video_path, frames_orig, orig_fps, video_depth, depth_vis, max_disp,
                   process_length, batch_size):
    '''
    Depth-Based Video Splatting Using the Video Depth.
    Args:
        input_video_path: Path to the input video.
        output_video_path: Path to the output video.
        video_depth: Video depth with shape of [T, H, W] in [0, 1].
        depth_vis: Visualized video depth with shape of [T, H, W, 3] in [0, 1].
        process_length: The length of video to process.
        batch_size: The batch size for splatting to save GPU memory.
    '''
    t0 = time.time()
    # vid_reader = VideoReader(input_video_path, ctx=cpu(0))
    # original_fps = vid_reader.get_avg_fps()
    # input_frames = vid_reader[:].asnumpy().astype("float32") / 255.0
    original_fps = orig_fps
    input_frames = frames_orig
    if process_length != -1 and process_length < len(input_frames):
        input_frames = input_frames[:process_length]
        video_depth = video_depth[:process_length]
        depth_vis = depth_vis[:process_length]

    stereo_projector = ForwardWarpStereo(occlu_map=True).cuda()

    num_frames = len(input_frames)
    height, width, _ = input_frames[0].shape

    # Initialize OpenCV VideoWriter
    out = cv2.VideoWriter(
        output_video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        original_fps,
        (width * 2, height * 2)
    )

    total_batches = (num_frames + batch_size - 1) // batch_size
    start_time = time.time()

    for i in range(0, num_frames, batch_size):
        batch_start_time = time.time()
        batch_frames = input_frames[i:i + batch_size]
        batch_depth = video_depth[i:i + batch_size]
        batch_depth_vis = depth_vis[i:i + batch_size]

        left_video = torch.from_numpy(batch_frames).permute(0, 3, 1, 2).float().cuda()
        disp_map = torch.from_numpy(batch_depth).unsqueeze(1).float().cuda()

        disp_map = disp_map * 2.0 - 1.0
        disp_map = disp_map * max_disp

        with torch.no_grad():
            right_video, occlusion_mask = stereo_projector(left_video, disp_map)

        right_video = right_video.cpu().permute(0, 2, 3, 1).numpy()
        occlusion_mask = occlusion_mask.cpu().permute(0, 2, 3, 1).numpy().repeat(3, axis=-1)

        for j in range(len(batch_frames)):
            video_grid_top = np.concatenate([batch_frames[j], batch_depth_vis[j]], axis=1)
            video_grid_bottom = np.concatenate([occlusion_mask[j], right_video[j]], axis=1)
            video_grid = np.concatenate([video_grid_top, video_grid_bottom], axis=0)

            video_grid_uint8 = np.clip(video_grid * 255.0, 0, 255).astype(np.uint8)
            video_grid_bgr = cv2.cvtColor(video_grid_uint8, cv2.COLOR_RGB2BGR)
            out.write(video_grid_bgr)

        # 清理显存
        del left_video, disp_map, right_video, occlusion_mask
        torch.cuda.empty_cache()
        gc.collect()

        # 进度信息
        batch_time = time.time() - batch_start_time
        elapsed_time = time.time() - start_time
        percent = (i + batch_size) / num_frames
        percent = min(percent, 1.0)
        sys.stdout.write(
            f"\rDepth splatting: [{int(percent * 100):3d}%] "
            f"Batch {i // batch_size + 1}/{total_batches}, "
            f"Batch time: {batch_time:.2f}s, "
            f"Elapsed: {elapsed_time:.2f}s"
        )
        sys.stdout.flush()

    print()  # 换行

    t2 = time.time()
    print(f"in DepthSplatting . finished all work. all the loops costs {t2 - t0:.2f} seconds.")
    out.release()


def main(input_video_path: str, output_video_path: str, unet_path: str, pre_trained_path: str, max_disp: float = 20.0,
         process_length=-1, batch_size=20):
    depthcrafter_demo = DepthCrafterDemo(unet_path=unet_path, pre_trained_path=pre_trained_path)
    print("Starting depth inference...")  # 打印进度，开始depth infer...
    frames_orig, orig_fps, video_depth, depth_vis = depthcrafter_demo.infer(input_video_path, output_video_path,
                                                                            process_length)

    print("depth inference finished. Starting DepthSplatting...")
    DepthSplatting(input_video_path, output_video_path, frames_orig, orig_fps, video_depth, depth_vis, max_disp,
                   process_length, batch_size)
    print("depth splatting finished. ")


if __name__ == "__main__":
    Fire(main)
