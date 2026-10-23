"""Compare a checkpoint's predictions against recorded episode targets.

python visualize_checkpoint.py --checkpoint model.ckpt --episode data/test/test_00000.npz --window-size 20
Requires a desktop Matplotlib backend. Space: play/pause; arrows: step.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
import numpy as np
import torch

from model.model import SelfDrivingModel


def load_model(path, device):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    state = checkpoint.get('state_dict', checkpoint)
    state = {key.removeprefix('model.'): value for key, value in state.items()}
    hidden = state['lstm.weight_hh_l0'].shape[1]
    if state['head.2.weight'].shape[0] != 1:
        raise ValueError('This viewer expects a model with one inverse-radius output.')
    model = SelfDrivingModel(lstm_hidden=hidden)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise ValueError('Checkpoint architecture does not match model/model.py. '
                         'Use the model definition that created this checkpoint.') from error
    return model.to(device).eval()


@torch.inference_mode()
def predict(model, frames, count, window_size, batch_size, device):
    predictions = np.full(count, np.nan, dtype=np.float32)
    first = window_size - 1
    for start in range(first, count, batch_size):
        ends = range(start, min(start + batch_size, count))
        windows = np.stack([frames[end - window_size + 1:end + 1] for end in ends])
        images = torch.from_numpy(windows).permute(0, 1, 4, 2, 3).contiguous()
        images = images.to(device=device, dtype=torch.float32).div_(255)
        output = model(images)
        output = output[0] if isinstance(output, tuple) else output
        predictions[start:start + len(windows)] = output[:, -1, 0].cpu().numpy()
        print(f'Inference: {min(start + batch_size, count)}/{count} frames', end='\r', flush=True)
    print()
    return predictions


class PredictionViewer:
    def __init__(self, frames, actions, predictions, target, title, interval):
        self.frames = frames
        self.actions = actions
        self.predictions = predictions
        self.target = target
        self.playing = False
        self.first = int(np.flatnonzero(np.isfinite(predictions))[0])
        valid = np.isfinite(predictions)
        mae = np.abs(predictions[valid] - target[valid]).mean()
        self.fig = plt.figure(figsize=(12, 8))
        grid = self.fig.add_gridspec(2, 2, height_ratios=[2, 1], bottom=0.23, hspace=0.4)
        image_ax = self.fig.add_subplot(grid[0, 0])
        self.image = image_ax.imshow(frames[self.first])
        image_ax.axis('off')
        info_ax = self.fig.add_subplot(grid[0, 1])
        info_ax.axis('off')
        self.info = info_ax.text(0, 0.9, '', va='top', family='monospace')
        ax = self.fig.add_subplot(grid[1, :])
        ax.plot(target, label='Recorded target', linewidth=1)
        ax.plot(predictions, label='Model prediction', linewidth=1)
        ax.set(xlabel='Episode frame index', ylabel='Inverse-radius target (training scale)')
        ax.legend()
        ax.grid(alpha=0.2)
        self.cursor = ax.axvline(self.first, color='black', linewidth=1)
        self.fig.suptitle(f'{title}\nRolling-window last-frame predictions | MAE: {mae:.4f}')
        self.fig.text(0.08, 0.02,
                      'Target = clip(action[1] / (abs(action[0]) + 1e-6), -10, 10).\n'
                      'Each prediction uses only its preceding window. First window_size−1 frames have no prediction.',
                      fontsize=9)
        self.slider = Slider(self.fig.add_axes([0.15, 0.13, 0.7, 0.03]),
                             'Frame', 0, len(actions) - 1, valinit=self.first, valstep=1)
        self.slider.on_changed(self.update)
        self.button = Button(self.fig.add_axes([0.4, 0.07, 0.2, 0.04]), 'Play / Pause')
        self.button.on_clicked(self.toggle)
        self.timer = self.fig.canvas.new_timer(interval=interval)
        self.timer.add_callback(self.tick)
        self.fig.canvas.mpl_connect('key_press_event', self.key)
        self.fig.canvas.mpl_connect('close_event', lambda event: self.timer.stop())
        self.update(self.first)
        self.timer.start()

    def update(self, value):
        frame = int(value)
        self.image.set_data(self.frames[frame])
        prediction = self.predictions[frame]
        predicted = f'{prediction:.5f}' if np.isfinite(prediction) else 'Unavailable (window warmup)'
        error = f'{abs(prediction - self.target[frame]):.5f}' if np.isfinite(prediction) else 'N/A'
        self.info.set_text(f'Frame: {frame}\n\nRecorded action[0]: {self.actions[frame, 0]:.5f}\n'
                           f'Recorded action[1]: {self.actions[frame, 1]:.5f}\n\n'
                           f'Target:     {self.target[frame]:.5f}\nPrediction: {predicted}\n'
                           f'Abs. error: {error}')
        self.cursor.set_xdata([frame, frame])
        self.fig.canvas.draw_idle()

    def toggle(self, event=None):
        if not self.playing and int(self.slider.val) == len(self.actions) - 1:
            self.slider.set_val(0)
        self.playing = not self.playing

    def tick(self):
        if self.playing:
            frame = int(self.slider.val) + 1
            if frame < len(self.actions):
                self.slider.set_val(frame)
            else:
                self.playing = False

    def key(self, event):
        if event.key == ' ':
            self.toggle()
        elif event.key in ('left', 'right'):
            self.playing = False
            frame = int(self.slider.val) + (1 if event.key == 'right' else -1)
            self.slider.set_val(np.clip(frame, 0, len(self.actions) - 1))


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError('must be at least 1')
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--episode', type=Path, required=True)
    parser.add_argument('--window-size', type=positive_int, required=True,
                        help='Use the same window size as training.')
    parser.add_argument('--batch-size', type=positive_int, default=8)
    parser.add_argument('--device', default='auto', help='auto, cpu, cuda, etc.')
    parser.add_argument('--interval', type=positive_int, default=100, help='Playback frame interval (ms).')
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else torch.device(args.device)
    with np.load(args.episode, allow_pickle=False) as archive:
        frames, actions = archive['rgb'], archive['action']
    if actions.ndim != 2 or actions.shape[1] != 2:
        parser.error('Expected actions of shape (N, 2).')
    if frames.ndim != 4 or frames.shape[-1] != 3 or frames.dtype != np.uint8:
        parser.error('Expected uint8 RGB frames of shape (N+1, H, W, 3).')
    if len(frames) != len(actions) + 1 or len(actions) < max(2, args.window_size):
        parser.error('Expected N+1 frames and at least max(2, window_size) actions.')
    if not np.isfinite(actions).all():
        parser.error('Recorded actions contain non-finite values.')
    model = load_model(args.checkpoint, device)
    print(f'Running inference on {device}...', flush=True)
    predictions = predict(model, frames, len(actions), args.window_size, args.batch_size, device)
    target = np.clip(actions[:, 1] / (np.abs(actions[:, 0]) + 1e-6), -10, 10)
    viewer = PredictionViewer(frames, actions, predictions, target,
                              f'{args.checkpoint.name} | {args.episode.name}', args.interval)
    plt.show()


if __name__ == '__main__':
    main()
