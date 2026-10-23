"""A small NumPy dataset, also usable with PyTorch's DataLoader.

Install: pip install numpy
Run:     python duckietown_dataset.py --split train
"""
import argparse
from bisect import bisect_right
from pathlib import Path

from read_episode import load_episode


class DuckietownDataset:
    """Index overlapping T-step windows; cache only the last episode.

    Images have shape (T, H, W, C), or (T, C, H, W) with channels_first.
    Actions have shape (T, 2). Windows never cross episode boundaries.
    With normalize=True images are float32 in [0, 1], otherwise uint8.
    training_only=True loads and returns only images and actions.
    """

    def __init__(self, root=None, split='train', normalize=False,
                 channels_first=False, T=1, training_only=False):
        if isinstance(T, bool) or not isinstance(T, int) or T < 1:
            raise ValueError('T must be a positive integer')
        self.T = T
        if split not in ('train', 'validation', 'test'):
            raise ValueError('split must be train, validation, or test')
        root = Path(root) if root is not None else Path(__file__).resolve().parents[1] / 'data'
        self.files = sorted((root / split).glob('*.npz'))
        if not self.files:
            raise FileNotFoundError(f'No .npz files found in {root / split}')
        self.normalize = normalize
        self.channels_first = channels_first
        self.training_only = training_only
        self._ends = []
        self._cached_file = None
        self._cached_episode = None
        total = 0
        # Read only the small action array while building the index.
        import numpy as np
        for path in self.files:
            with np.load(path, allow_pickle=False) as archive:
                total += max(0, len(archive['action']) - T + 1)
            self._ends.append(total)
        if total == 0:
            raise ValueError(f'No episodes contain at least T={T} actions')

    def __len__(self):
        return self._ends[-1]

    def _image(self, image):
        # Copy so callers cannot modify the cached episode accidentally.
        image = image.astype('float32') / 255.0 if self.normalize else image.copy()
        if self.channels_first:
            image = image.transpose(0, 3, 1, 2).copy()
        return image

    def __getitem__(self, index):
        from operator import index as integer_index
        index = integer_index(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError('dataset index out of range')
        episode_index = bisect_right(self._ends, index)
        start = self._ends[episode_index - 1] if episode_index else 0
        step = index - start
        path = self.files[episode_index]
        if path != self._cached_file:
            # Release the previous episode before decompressing the next one.
            self._cached_episode = None
            self._cached_file = None
            if self.training_only:
                import numpy as np
                with np.load(path, allow_pickle=False) as archive:
                    self._cached_episode = {key: archive[key] for key in ('rgb', 'action')}
            else:
                self._cached_episode = load_episode(path)
            self._cached_file = path
        episode = self._cached_episode
        end = step + self.T
        sample = {
            'image': self._image(episode['rgb'][step:end]),
            'action': episode['action'][step:end].copy(),
        }
        if self.training_only:
            return sample
        sample.update({
            'next_image': self._image(episode['rgb'][step + 1:end + 1]),
            'reward': episode['reward'][step:end].copy(),
            'terminated': episode['terminated'][step:end].copy(),
            'truncated': episode['truncated'][step:end].copy(),
            'episode': path.stem,
            'step': step,
        })
        return sample


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=None)
    parser.add_argument('--T', type=int, default=5)
    parser.add_argument('--split', default='train', choices=['train', 'validation', 'test'])
    args = parser.parse_args()
    dataset = DuckietownDataset(args.root, args.split, T=args.T)
    print(f'{len(dataset)} windows of T={dataset.T} across {len(dataset.files)} episodes')
    sample = dataset[0]
    for key, value in sample.items():
        print(f'{key}: {value.shape if hasattr(value, "shape") else value}')
