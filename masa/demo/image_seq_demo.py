"""
MASA tracking demo for image sequences (e.g., drone construction site images).
Processes a directory of images sorted by filename and runs open-vocabulary
detection + tracking using the unified MASA-GroundingDINO model.
"""
import os
import sys
os.environ["TOKENIZERS_PARALLELISM"] = "false"
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import gc
import json
import glob
import argparse
import resource

import subprocess
import cv2
import numpy as np
import torch
from torch.multiprocessing import Pool, set_start_method

from mmdet.registry import VISUALIZERS

import masa
from masa.apis import inference_masa, init_masa, build_test_pipeline
from utils import filter_and_update_tracks

import warnings
warnings.filterwarnings('ignore')

try:
    set_start_method('spawn')
except RuntimeError:
    pass


def set_file_descriptor_limit(limit):
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (limit, hard))

set_file_descriptor_limit(65536)


def visualize_frame(args, visualizer, frame, track_result, frame_idx):
    visualizer.add_datasample(
        name='frame_' + str(frame_idx),
        image=frame[:, :, ::-1],
        data_sample=track_result[0],
        draw_gt=False,
        show=False,
        out_file=None,
        pred_score_thr=args.score_thr)
    frame = visualizer.get_image()
    gc.collect()
    return frame


def parse_args():
    parser = argparse.ArgumentParser(description='MASA image sequence tracking demo')
    parser.add_argument('image_dir', help='Directory containing images')
    parser.add_argument('--masa_config', required=True, help='MASA config file')
    parser.add_argument('--masa_checkpoint', required=True, help='MASA checkpoint file')
    parser.add_argument('--device', default='cuda:0', help='Device for inference')
    parser.add_argument('--score-thr', type=float, default=0.3, help='Score threshold')
    parser.add_argument('--out', type=str, required=True, help='Output video file')
    parser.add_argument('--save_dir', type=str, default=None, help='Save individual frames to dir')
    parser.add_argument('--texts', type=str,
                        default='vehicle . truck . excavator . crane . person . car',
                        help='Text prompt for detection (categories separated by " . ")')
    parser.add_argument('--line_width', type=int, default=5, help='Line width for visualization')
    parser.add_argument('--fp16', action='store_true', help='Use FP16 mode')
    parser.add_argument('--no-post', action='store_true', help='Skip post-processing')
    parser.add_argument('--max_frames', type=int, default=None, help='Limit number of frames to process')
    parser.add_argument('--resize', type=int, default=None,
                        help='Resize longest edge to this value before processing (saves GPU memory)')
    parser.add_argument('--save_json', type=str, default=None, help='Save tracking results as JSON')
    parser.add_argument('--fps', type=int, default=2, help='FPS for output video')
    args = parser.parse_args()
    return args


def load_images(image_dir, max_frames=None):
    """Load and sort images from a directory."""
    extensions = ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG']
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(image_dir, ext)))
    image_paths = sorted(image_paths)

    if max_frames is not None:
        image_paths = image_paths[:max_frames]

    print(f'Found {len(image_paths)} images in {image_dir}')
    return image_paths


def main():
    args = parse_args()

    # Load images
    image_paths = load_images(args.image_dir, args.max_frames)
    if len(image_paths) == 0:
        print(f'No images found in {args.image_dir}')
        return

    # Initialize the unified MASA model
    print('Loading MASA model...')
    masa_model = init_masa(args.masa_config, args.masa_checkpoint, device=args.device)

    # Build test pipeline with text support
    texts = args.texts
    masa_test_pipeline = build_test_pipeline(masa_model.cfg, with_text=True)

    # Setup visualizer
    masa_model.cfg.visualizer['texts'] = texts
    masa_model.cfg.visualizer['save_dir'] = args.save_dir
    masa_model.cfg.visualizer['line_width'] = args.line_width
    visualizer = VISUALIZERS.build(masa_model.cfg.visualizer)

    # Read first image to get dimensions
    first_img = cv2.imread(image_paths[0])
    if args.resize:
        h, w = first_img.shape[:2]
        scale = args.resize / max(h, w)
        new_w, new_h = int(w * scale) & ~1, int(h * scale) & ~1  # ensure even dims
    else:
        h, w = first_img.shape[:2]
        new_h, new_w = h & ~1, w & ~1  # ensure even dims

    # Setup frame output directory (always needed for ffmpeg video creation)
    frame_dir = args.save_dir if args.save_dir else os.path.join(
        os.path.dirname(args.out) if os.path.dirname(args.out) else '.', '_tmp_frames')
    os.makedirs(frame_dir, exist_ok=True)

    if args.out:
        os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else '.', exist_ok=True)

    # Process each frame
    print(f'Processing {len(image_paths)} images...')
    print(f'Text prompt: "{texts}"')
    instances_list = []
    frames = []
    tracking_results = []

    for frame_idx, img_path in enumerate(image_paths):
        print(f'  [{frame_idx + 1}/{len(image_paths)}] {os.path.basename(img_path)}', end='')

        frame = cv2.imread(img_path)
        if frame is None:
            print(' - SKIPPED (could not read)')
            continue

        if args.resize:
            frame = cv2.resize(frame, (new_w, new_h))

        # Run unified MASA inference (detection + tracking)
        track_result = inference_masa(
            masa_model, frame,
            frame_id=frame_idx,
            video_len=len(image_paths),
            test_pipeline=masa_test_pipeline,
            text_prompt=texts,
            fp16=args.fp16,
            detector_type='mmdet')

        # Process results
        if 'masks' in track_result[0].pred_track_instances:
            if len(track_result[0].pred_track_instances.masks) > 0:
                track_result[0].pred_track_instances.masks = torch.stack(
                    track_result[0].pred_track_instances.masks, dim=0)
                track_result[0].pred_track_instances.masks = \
                    track_result[0].pred_track_instances.masks.cpu().numpy()

        track_result[0].pred_track_instances.bboxes = \
            track_result[0].pred_track_instances.bboxes.to(torch.float32)

        n_tracks = len(track_result[0].pred_track_instances.instances_id)
        print(f' - {n_tracks} tracks')

        instances_list.append(track_result.to('cpu'))
        frames.append(frame)

        # Collect tracking data for JSON export
        if args.save_json:
            frame_tracks = []
            pti = track_result[0].pred_track_instances
            for i in range(len(pti.instances_id)):
                frame_tracks.append({
                    'instance_id': int(pti.instances_id[i].item()),
                    'bbox': pti.bboxes[i].cpu().tolist(),
                    'score': float(pti.scores[i].cpu().item()),
                    'label': int(pti.labels[i].cpu().item()),
                })
            tracking_results.append({
                'frame_idx': frame_idx,
                'image': os.path.basename(img_path),
                'tracks': frame_tracks,
            })

    # Post-processing
    if not args.no_post and len(instances_list) > 0:
        print('Post-processing tracks...')
        instances_list = filter_and_update_tracks(
            instances_list, (frames[0].shape[1], frames[0].shape[0]))

    # Visualization and frame saving
    if len(frames) > 0:
        print('Visualizing results...')
        num_cores = max(1, min(os.cpu_count() - 1, 16))
        print(f'Using {num_cores} cores for visualization')

        with Pool(processes=num_cores) as pool:
            vis_frames = pool.starmap(
                visualize_frame,
                [(args, visualizer, frame, track_result.to('cpu'), idx)
                 for idx, (frame, track_result) in enumerate(zip(frames, instances_list))])

        for idx, vis_frame in enumerate(vis_frames):
            out_path = os.path.join(frame_dir, f'frame_{idx:06d}.jpg')
            cv2.imwrite(out_path, vis_frame[:, :, ::-1])

        # Create video from frames using ffmpeg
        if args.out:
            print('Creating video with ffmpeg...')
            ffmpeg_cmd = [
                'ffmpeg', '-y',
                '-framerate', str(args.fps),
                '-i', os.path.join(frame_dir, 'frame_%06d.jpg'),
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                '-crf', '18',
                args.out
            ]
            subprocess.run(ffmpeg_cmd, capture_output=True)
            print(f'Output video saved to: {args.out}')

            # Clean up temp frames if no save_dir was requested
            if not args.save_dir:
                import shutil
                shutil.rmtree(frame_dir)
                print('Cleaned up temporary frames.')

    # Save JSON results
    if args.save_json and tracking_results:
        with open(args.save_json, 'w') as f:
            json.dump({
                'text_prompt': texts,
                'num_frames': len(image_paths),
                'frames': tracking_results,
            }, f, indent=2)
        print(f'Tracking results saved to: {args.save_json}')

    print('Done!')


if __name__ == '__main__':
    main()
