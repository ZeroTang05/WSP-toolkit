"""Read a block of weights from the local model and see its dimensions and overall size."""

import argparse

from weight_programming import WeightStore


def main():
    """Read the original weights and print three common properties."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, help='Local safetensors model directory')
    parser.add_argument('--edit-dir', required=True, help='Separate directory for edit records')
    parser.add_argument('--name', default='model.layers.0.mlp.down_proj.weight')
    args = parser.parse_args()

    # WeightStore binds the record to the original model. The weights read out are an independent copy.
    store = WeightStore(args.model, args.edit_dir)
    weight = store.load_original(args.name)
    print(f'Weight: {args.name}')
    print(f'Shape: {tuple(weight.shape)}')
    print(f'Frobenius norm: {weight.norm().item():.6f}')


if __name__ == '__main__':
    main()
