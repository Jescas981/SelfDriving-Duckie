"""Shuffle episodes without scattering their windows across the dataset."""
import random


class EpisodeShuffleSampler:
    """Yield every window once, in sequential order within shuffled episodes.

    episode_ends contains cumulative window counts, including empty episodes.
    Lightning calls set_epoch before each epoch for reproducible reshuffling.
    Only episode boundaries are stored; window indices are generated lazily.
    """

    def __init__(self, episode_ends, seed=0):
        self.episode_ends = tuple(episode_ends)
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.episode_ends[-1] if self.episode_ends else 0

    def __iter__(self):
        order = list(range(len(self.episode_ends)))
        random.Random(self.seed + self.epoch).shuffle(order)
        for episode in order:
            start = self.episode_ends[episode - 1] if episode else 0
            yield from range(start, self.episode_ends[episode])
