import json
import os

import cv2
import numpy as np


OPENGL2OPENCV = np.array([
    1, 0, 0, 0,
    0, -1, 0, 0,
    0, 0, -1, 0,
    0, 0, 0, 1,
], dtype=np.float32).reshape(4, 4)


def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--bd_min', type=float, default=0.1)
    parser.add_argument('--bd_max', type=float, default=10.)
    return parser


def main():
    parser = config_parser()
    args = parser.parse_args()
    data_path = args.data_path
    input_path = os.path.join(data_path, 'images')
    json_path  = os.path.join(data_path, 'transforms.json')
    out_path   = os.path.join(data_path, 'images_1')
    pose_out_path = os.path.join(data_path, 'poses_bounds.npy')
    if not os.path.exists(out_path):
        os.makedirs(out_path, exist_ok=True)

    with open(json_path, 'r') as metaf:
        meta = json.load(metaf)
        frames_data = meta['frames']
        h, w, f = meta['h'], meta['w'], meta['fl_x']

    hwf = np.array([h, w, f]).reshape([3, 1])
    pose_arr = []

    for i, frame_data in enumerate(frames_data):
        pose = frame_data['transform_matrix']
        pose = np.array(pose, dtype=np.float32)
        pose = OPENGL2OPENCV @ np.linalg.inv(pose)
        pose = np.linalg.inv(pose)[:3]
        pose = np.concatenate([pose, hwf], 1)
        pose = np.concatenate(
            [pose[:, 1:2], pose[:, 0:1], -pose[:, 2:3], pose[:, 3:4], pose[:, 4:5]], 1,
        )
        pose = np.concatenate(
            [pose.reshape(-1), np.array((args.bd_min, args.bd_max), dtype=np.float32)]
        )
        pose_arr.append(pose)

        img = cv2.imread(f'{input_path}/{i:04d}.png')
        cv2.imwrite(f'{out_path}/{i:04d}.png', img)
        print(f'Frame {i} saved!')
    
    pose_arr = np.stack(pose_arr)
    np.save(pose_out_path, pose_arr)
    return True


if __name__ == '__main__':
    main()
