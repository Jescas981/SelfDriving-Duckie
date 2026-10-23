"""Browse Duckietown images and their two recorded action values.

python preview_dataset.py --split train
python preview_dataset.py --file data/test/test_00000.npz
Controls: Space = play/pause, Left/Right = step, buttons = change episode.
Action column semantics/units are not confirmed by the supplied dataset schema.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider

from read_episode import load_episode


class DatasetPreviewer:
    def __init__(self, files):
        self.files = files
        self.episode_index = 0
        self.playing = False
        self.fig = plt.figure(figsize=(12, 8))
        grid = self.fig.add_gridspec(
            3, 2, height_ratios=[3, 1, 1], width_ratios=[3, 2],
            left=0.08, right=0.96, bottom=0.24, top=0.90, hspace=0.65,
        )
        self.image_ax = self.fig.add_subplot(grid[0, 0])
        self.info_ax = self.fig.add_subplot(grid[0, 1])
        self.info_ax.axis('off')
        self.info = self.info_ax.text(0, 1, '', va='top', family='monospace')
        self.action_axes = [self.fig.add_subplot(grid[i, :]) for i in (1, 2)]
        self.fig.text(
            0.08, 0.02,
            'Recorded actions: column meanings and units are unconfirmed. '
            'Space: play/pause | Left/Right: frame', fontsize=9,
        )
        self.slider_ax = self.fig.add_axes([0.14, 0.14, 0.76, 0.03])
        self.buttons = []
        for left, label, callback in [
            (0.15, 'Previous episode', lambda event: self.change_episode(-1)),
            (0.41, 'Play / Pause', self.toggle_play),
            (0.67, 'Next episode', lambda event: self.change_episode(1)),
        ]:
            button = Button(self.fig.add_axes([left, 0.065, 0.20, 0.045]), label)
            button.on_clicked(callback)
            self.buttons.append(button)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.canvas.mpl_connect('close_event', lambda event: self.timer.stop())
        self.timer = self.fig.canvas.new_timer(interval=100)
        self.timer.add_callback(self.tick)
        self.load(0)
        self.timer.start()

    def load(self, index):
        self.playing = False
        self.episode_index = index
        self.episode = load_episode(self.files[index])
        frames = self.episode['rgb']
        actions = self.episode['action']
        if len(frames) != len(actions) + 1 or actions.ndim != 2 or actions.shape[1] != 2:
            raise ValueError('Expected N+1 camera frames and action shape (N, 2)')
        self.image_ax.clear()
        self.image = self.image_ax.imshow(frames[0], interpolation='nearest')
        self.image_ax.axis('off')
        self.cursors = []
        self.points = []
        times = self.episode['time_s']
        for column, ax in enumerate(self.action_axes):
            ax.clear()
            ax.plot(times[:-1], actions[:, column], linewidth=1)
            ax.set_title(f'Action[{column}] — recorded command', fontsize=10)
            ax.set_xlabel('Simulated time (s)')
            ax.set_ylabel('Value (raw)')
            ax.set_xlim(times[0], times[-1])
            ax.grid(alpha=0.2)
            self.cursors.append(ax.axvline(times[0], color='black', linewidth=1))
            point, = ax.plot([], [], 'o', color='black', markersize=5)
            self.points.append(point)
        if hasattr(self, 'slider'):
            self.slider.disconnect_events()
        self.slider_ax.clear()
        self.slider = Slider(self.slider_ax, 'Frame', 0, len(frames) - 1,
                             valinit=0, valstep=1, valfmt='%0.0f')
        self.slider.on_changed(self.update)
        meta = self.episode['metadata']
        hz = meta.get('control_hz', 10)
        self.timer.interval = max(1, round(1000 / hz))
        self.fig.suptitle(
            f'{self.files[index].name} | episode {index + 1}/{len(self.files)} | '
            f'map: {meta.get("map", "unknown")}', fontsize=13,
        )
        self.update(0)

    def update(self, value):
        t = int(value)
        ep = self.episode
        time = float(ep['time_s'][t])
        self.image.set_data(ep['rgb'][t])
        self.image_ax.set_title(f'Frame {t}/{len(ep["rgb"]) - 1} | {time:.2f} s')
        lines = [f'Split: {ep["metadata"].get("split", "unknown")}',
                 f'Reverse: {ep["metadata"].get("reverse", "unknown")}',
                 f'Recovery: {ep["metadata"].get("recovery", "unknown")}', '']
        has_action = t < len(ep['action'])
        if has_action:
            a = ep['action'][t]
            lines += [f'Action[0]: {a[0]: .5f}', f'Action[1]: {a[1]: .5f}',
                      f'Reward:    {ep["reward"][t]: .5f}',
                      f'Terminated: {bool(ep["terminated"][t])}',
                      f'Truncated:  {bool(ep["truncated"][t])}']
        else:
            lines += ['Final observation', 'No action for this frame']
        self.info.set_text('\n'.join(lines))
        for column, (cursor, point) in enumerate(zip(self.cursors, self.points)):
            cursor.set_xdata([time, time])
            point.set_data([time] if has_action else [],
                           [ep['action'][t, column]] if has_action else [])
        self.fig.canvas.draw_idle()

    def change_episode(self, direction):
        self.load((self.episode_index + direction) % len(self.files))

    def toggle_play(self, event=None):
        if not self.playing and int(self.slider.val) == len(self.episode['rgb']) - 1:
            self.slider.set_val(0)
        self.playing = not self.playing

    def tick(self):
        if self.playing:
            frame = int(self.slider.val) + 1
            if frame < len(self.episode['rgb']):
                self.slider.set_val(frame)
            else:
                self.playing = False

    def on_key(self, event):
        if event.key == ' ':
            self.toggle_play()
        elif event.key in ('left', 'right'):
            self.playing = False
            frame = int(self.slider.val) + (1 if event.key == 'right' else -1)
            self.slider.set_val(max(0, min(frame, len(self.episode['rgb']) - 1)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent / 'data')
    parser.add_argument('--split', choices=['train', 'validation', 'test'], default='train')
    parser.add_argument('--file', type=Path, help='Open a single episode instead of a split')
    args = parser.parse_args()
    files = [args.file] if args.file else sorted((args.root / args.split).glob('*.npz'))
    if not files or any(not path.is_file() for path in files):
        parser.error('No episode files found; check --root, --split, or --file')
    viewer = DatasetPreviewer(files)  # Keep widget and timer references alive.
    plt.show()


if __name__ == '__main__':
    main()
