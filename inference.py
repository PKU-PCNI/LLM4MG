import torch
import argparse
import numpy as np

from models.LLM4MG import LLM4MG as Model
from utils import IMG_TRANSFORMS, build_custom_batch, validate_custom_paths


def parse_args():
    parser = argparse.ArgumentParser(description="LLM4MG single-sample inference")
    # data
    parser.add_argument('--rgb_bs', type=str, default="sample/bs/RGB/rgb_image_80.png", help='Base-station RGB image path')
    parser.add_argument('--depth_bs', type=str, default="sample/bs/Depth/depth_image_80.png", help='Base-station depth image path (optional)')
    parser.add_argument('--lidar_bs', type=str, default="sample/bs/Lidar/point_cloud_80.ply", help='Base-station LiDAR point cloud path')
    parser.add_argument('--radar_bs', type=str, default="sample/bs/Radar/RSF_1_radarpoint_snapshot_80.mat", help='Base-station radar data path')

    parser.add_argument('--rgb_vh', type=str, default="sample/veh/RGB/front_rgb_image_80.png", help='Vehicle-view RGB image path (optional)')
    parser.add_argument('--depth_vh', type=str, default="sample/veh/Depth/front_depth_image_80.png", help='Vehicle-view depth image path (optional)')
    parser.add_argument('--lidar_vh', type=str, default="sample/veh/Lidar/point_cloud_80.ply", help='Vehicle-view LiDAR point cloud path (optional)')
    parser.add_argument('--radar_vh', type=str, default="sample/veh/Radar/Car_0_radarpoint_snapshot_80.mat", help='Vehicle-view radar data path (optional)')

    parser.add_argument('--dis', type=float, default=49.75, help='Transceiver distance (m)')
    parser.add_argument('--phi', type=float, default=81.76, help='Azimuth angle (deg)')
    parser.add_argument('--theta', type=float, default=-0.81, help='Elevation angle (deg)')

    # model
    parser.add_argument('--categories', type=str, nargs='+', default=['Depth', 'Lidar', 'RGB', 'Radar'], help='Sensor categories (dataset mode)')
    parser.add_argument('--fc', type=str, default='60GHz', help='Frequency band')
    parser.add_argument('--decoder_num', type=int, default=2, help='Number of Llama decoder layers')
    parser.add_argument('--llama_path', type=str, default='Llama-3.2-1B', help='Path to the Llama model')
    parser.add_argument('--input_text', type=str, default='Frequency=60GHz, Bandwidth=2GHz, Distance=', help='LLM text prefix')
    parser.add_argument('--pretrained_dir', type=str, default='LLM4MG.pth', help='Path to pretrained weights')
    parser.add_argument('--device', type=str, default='cuda', help='Inference device (cuda / cpu)')
    return parser.parse_args()


def run_inference(model, batch, device):
    """Run inference for a single batch."""
    batch = {k: v.to(device) for k, v in batch.items()}
    model.eval()
    outputs = model(batch)
    return outputs


def print_results(outputs):
    """Print inference results for a single sample."""
    LoS_logits = outputs['LoS']
    LoS_pred = LoS_logits.argmax(dim=1).item()
    power = outputs['power'].squeeze(0).cpu().detach().numpy()
    tau = outputs['tau'].squeeze(0).cpu().detach().numpy()

    print("\n" + "=" * 60)
    print("Inference Results")
    print("=" * 60)
    print(f"LoS/NLoS: {'LoS' if LoS_pred == 1 else 'NLoS'} ")
    print(f"power: {np.round(power, 4)}")
    print(f"tau  : {np.round(tau, 4)}")
    print("=" * 60 + "\n")


def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")

    validate_custom_paths(args)
    batch = build_custom_batch(args, IMG_TRANSFORMS)

    model = Model(llama_path=args.llama_path,
                  input_text=args.input_text,
                  decoder_num=args.decoder_num).to(device)
    model.load_state_dict(torch.load(args.pretrained_dir, map_location=device))
    print(f"Loaded pretrained weights from {args.pretrained_dir}")

    outputs = run_inference(model, batch, device)
    print_results(outputs)


if __name__ == '__main__':
    main()
