"""从本地模型读取一块权重，查看它的尺寸和整体大小。"""

import argparse

from weight_programming import WeightStore


def main():
    """读取原始权重，并打印最常用的三个信息。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, help='Local safetensors model directory')
    parser.add_argument('--edit-dir', required=True, help='Separate directory for edit records')
    parser.add_argument('--name', default='model.layers.0.mlp.down_proj.weight')
    args = parser.parse_args()

    # WeightStore 将记录与原模型绑定。读取出的权重是一份独立副本。
    store = WeightStore(args.model, args.edit_dir)
    weight = store.load_original(args.name)
    print(f'Weight: {args.name}')
    print(f'Shape: {tuple(weight.shape)}')
    print(f'Frobenius norm: {weight.norm().item():.6f}')


if __name__ == '__main__':
    main()
