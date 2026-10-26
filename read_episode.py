"""Read an episode's arrays and metadata, and optionally view or save a frame."""
import argparse
import json
from pathlib import Path

import numpy as np


def load_episode(path):
    """Load independent arrays and decode metadata without enabling pickle."""
    with np.load(path, allow_pickle=False) as archive:
        episode = {key: archive[key] for key in archive.files}
    episode['metadata'] = json.loads(episode.pop('metadata_json').item())
    return episode


def transitions(episode):
    """Yield image, action, reward, next image, terminated, truncated."""
    for step in range(len(episode['action'])):
        yield (episode['rgb'][step], episode['action'][step],
               episode['reward'][step], episode['rgb'][step + 1],
               episode['terminated'][step], episode['truncated'][step])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', nargs='?', type=Path,
                        default=Path(__file__).resolve().parent / 'data/train/train_00000.npz')
    parser.add_argument('--frame', type=int, default=0)
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--save', type=Path, help='Save the selected frame as an image.')
    args = parser.parse_args()
    episode = load_episode(args.path)
    for key, value in episode.items():
        if key != 'metadata':
            print(f'{key:12} shape={value.shape!s:20} dtype={value.dtype}')
    print(json.dumps(episode['metadata'], indent=2))
    if not 0 <= args.frame < len(episode['rgb']):
        parser.error(f'--frame must be between 0 and {len(episode["rgb"]) - 1}')
    if args.show or args.save:
        import matplotlib.pyplot as plt
        frame = episode['rgb'][args.frame]
        if args.save:
            plt.imsave(args.save, frame)
        if args.show:
            plt.imshow(frame)
            plt.title(f'{args.path.name} | frame {args.frame}')
            plt.axis('off')
            plt.show()


if __name__ == '__main__':
    main()
