"""Run checkpoint inference and inspect attention and spatiotemporal saliency.

Space: play/pause. Left/right: step. Only RGB data is loaded. --clustering enables DBSCAN and occlusion scoring;
otherwise only attention is displayed. Occlusion measures prediction change, not ground-truth error.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.widgets import Button, Slider
import numpy as np
import torch

from visualize_checkpoint import load_model, positive_int


class AttentionViewer:
    def __init__(self, model, frames, device, window_size, interval,
                 particles_per_frame=100, eps=0.08, min_samples=10,
                 temporal_weight=1.0, seed=0, clustering=False):
        self.model, self.frames, self.device = model, frames, device
        self.window_size = window_size
        self.first = window_size - 1
        self.playing = False
        self.clustering = clustering
        self.particles_per_frame = particles_per_frame
        self.eps = eps
        self.min_samples = min_samples
        self.temporal_weight = temporal_weight
        self.rng = torch.Generator(device='cpu').manual_seed(seed)
        self.hull_artists = []
        self.selected_cluster = None
        self.last_patch = None

        self.fig, axes = plt.subplots(1, 4 if clustering else 3,
                                      figsize=(18 if clustering else 14, 5))
        self.fig.subplots_adjust(bottom=0.25, top=0.8)
        frame = frames[self.first]
        self.camera = axes[0].imshow(frame)
        self.heat = axes[1].imshow(np.zeros(frame.shape[:2]), cmap='inferno', vmin=0, vmax=1)
        self.overlay_camera = axes[2].imshow(frame)
        self.overlay = axes[2].imshow(np.zeros(frame.shape[:2]), cmap='inferno', alpha=0.45, vmin=0, vmax=1)
        self.overlay_ax = axes[2]
        if clustering:
            self.patch_image = axes[3].imshow(np.zeros((64, 64, 3), dtype=np.uint8))
        for ax, title in zip(axes, ('Camera', 'Attention (frame min–max normalized)',
                                    'Attention + DBSCAN convex hulls', 'Top-impact crop (64×64)')):
            ax.set_title(title, fontsize=10)
            ax.axis('off')
        if not clustering:
            axes[2].set_title('Attention overlay', fontsize=10)
        self.status = self.fig.text(0.08, 0.17, '')
        self.fig.text(0.08, 0.02,
                      'Occlusion score = absolute final prediction change (not ground-truth error). Space: play/pause.',
                      fontsize=9)
        self.slider = Slider(self.fig.add_axes([0.15, 0.11, 0.7, 0.03]), 'Frame',
                             self.first, len(frames) - 1, valinit=self.first, valstep=1)
        self.slider.on_changed(self.update)
        self.button = Button(self.fig.add_axes([0.4, 0.05, 0.2, 0.04]), 'Play / Pause')
        self.button.on_clicked(self.toggle)
        self.timer = self.fig.canvas.new_timer(interval=interval)
        self.timer.add_callback(self.tick)
        self.fig.canvas.mpl_connect('key_press_event', self.key)
        self.fig.canvas.mpl_connect('close_event', lambda event: self.timer.stop())
        self.update(self.first)
        self.timer.start()

    @torch.inference_mode()
    def update(self, value):
        index = int(value)
        window = self.frames[index - self.window_size + 1:index + 1]
        images = torch.from_numpy(window).permute(0, 3, 1, 2).unsqueeze(0).contiguous()
        images = images.to(device=self.device, dtype=torch.float32).div_(255)
        predictions, _, att_map = self.model(images)
        B, T, C, H, W = att_map.shape
        if B != 1 or C != 1:
            raise ValueError(f'Expected attention shape [1,T,1,H,W], got {tuple(att_map.shape)}')

        frame_h, frame_w = self.frames[index].shape[:2]
        if self.clustering:
            import cv2
            from scipy.spatial import ConvexHull, QhullError
            from sklearn.cluster import DBSCAN

            # Convert nonnegative attention into a valid sampling distribution.
            # Uniform fallback avoids torch.multinomial failing on all-zero maps.
            probs = att_map.detach().float().reshape(T, H * W).clamp_min(0).cpu()
            sums = probs.sum(dim=-1, keepdim=True)
            probs = torch.where(sums > 0, probs / sums.clamp_min(1e-12),
                                torch.full_like(probs, 1.0 / (H * W)))
            indices = torch.multinomial(probs, self.particles_per_frame,
                                        replacement=True, generator=self.rng)
            y = indices // W
            x = indices % W
            t = torch.arange(T).view(T, 1).expand_as(x)
            points = torch.stack((x, y, t), dim=-1).reshape(-1, 3).numpy()

            # Normalize x/y/t so the neighborhood radius is resolution independent.
            points_scaled = points.astype(np.float32).copy()
            points_scaled[:, 0] /= W
            points_scaled[:, 1] /= H
            points_scaled[:, 2] = points_scaled[:, 2] / T * self.temporal_weight
            labels = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit_predict(points_scaled)
            clusters = sorted(set(labels) - {-1})

            # Build per-cluster, per-timestep convex hulls on the attention grid.
            hulls = {}
            for cluster_id in clusters:
                for frame_t in range(T):
                    mask = (labels == cluster_id) & (points[:, 2] == frame_t)
                    xy = np.unique(points[mask, :2], axis=0).astype(float)
                    if len(xy) < 3:
                        continue
                    try:
                        hull = ConvexHull(xy)
                    except QhullError:
                        continue
                    hulls[(int(cluster_id), frame_t)] = xy[hull.vertices].copy()

            frame_h, frame_w = self.frames[index].shape[:2]
            scale_x, scale_y = frame_w / W, frame_h / H
            original_prediction = predictions[0, -1, 0].item()
            scores = {}
            for cluster_id in clusters:
                masked_images = images.clone()
                has_hull = False
                for frame_t in range(T):
                    polygon = hulls.get((int(cluster_id), frame_t))
                    if polygon is None:
                        continue
                    vertices = polygon.copy()
                    vertices[:, 0] = (vertices[:, 0] + 0.5) * scale_x - 0.5
                    vertices[:, 1] = (vertices[:, 1] + 0.5) * scale_y - 0.5
                    pixel_mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
                    cv2.fillConvexPoly(pixel_mask, np.rint(vertices).astype(np.int32), 1)
                    mask_tensor = torch.from_numpy(pixel_mask).to(self.device, dtype=images.dtype)
                    masked_images[:, frame_t] *= (1 - mask_tensor)[None, None]
                    has_hull = True
                if has_hull:
                    masked_predictions, _, _ = self.model(masked_images)
                    changed_prediction = masked_predictions[0, -1, 0].item()
                    scores[int(cluster_id)] = abs(original_prediction - changed_prediction)

            # Show the cluster with the largest prediction change that also
            # has a valid convex hull on the displayed (last) frame.
            candidates = [c for c in scores if (c, T - 1) in hulls]
            self.selected_cluster = max(candidates, key=lambda c: scores[c]) if candidates else None
            for artist in self.hull_artists:
                artist.remove()
            self.hull_artists.clear()
            visible_hulls = 0
            for cluster_id in clusters:
                polygon = hulls.get((int(cluster_id), T - 1))
                if polygon is None:
                    continue
                vertices = polygon.copy()
                vertices[:, 0] = (vertices[:, 0] + 0.5) * scale_x - 0.5
                vertices[:, 1] = (vertices[:, 1] + 0.5) * scale_y - 0.5
                color = 'cyan' if int(cluster_id) == self.selected_cluster else plt.get_cmap('tab10')(int(cluster_id) % 10)
                patch = Polygon(vertices, closed=True, fill=False, edgecolor=color,
                                linewidth=3 if int(cluster_id) == self.selected_cluster else 1.5)
                self.overlay_ax.add_patch(patch)
                self.hull_artists.append(patch)
                visible_hulls += 1

            # Standardize a bounding-box crop of the highest-impact region to 64x64.
            # This is a practical approximation: the excerpt does not define the warp.
            crop = np.zeros((64, 64, 3), dtype=np.uint8)
            if self.selected_cluster is not None:
                vertices = hulls[(self.selected_cluster, T - 1)].copy()
                vertices[:, 0] = (vertices[:, 0] + 0.5) * scale_x - 0.5
                vertices[:, 1] = (vertices[:, 1] + 0.5) * scale_y - 0.5
                x0 = max(0, int(np.floor(vertices[:, 0].min())))
                x1 = min(frame_w, int(np.ceil(vertices[:, 0].max())) + 1)
                y0 = max(0, int(np.floor(vertices[:, 1].min())))
                y1 = min(frame_h, int(np.ceil(vertices[:, 1].max())) + 1)
                if x1 > x0 and y1 > y0:
                    crop = cv2.resize(self.frames[index][y0:y1, x0:x1], (64, 64),
                                      interpolation=cv2.INTER_LINEAR)
            self.patch_image.set_data(crop)
        raw = att_map[0, -1, 0].detach().float().cpu().numpy()
        low, high = float(raw.min()), float(raw.max())
        normalized = (raw - low) / (high - low) if high > low else np.zeros_like(raw)
        # Attention may be at a different resolution than the input frames.
        # imshow maps it into the image extent without altering the sampling grid.
        self.heat.set_data(normalized)
        self.overlay.set_data(normalized)
        self.heat.set_extent((-0.5, frame_w - 0.5, frame_h - 0.5, -0.5))
        self.overlay.set_extent((-0.5, frame_w - 0.5, frame_h - 0.5, -0.5))
        self.camera.set_data(self.frames[index])
        self.overlay_camera.set_data(self.frames[index])
        prediction = predictions[0, -1, 0].item()
        if self.clustering:
            noise = int(np.sum(labels == -1))
            self.status.set_text(
                f'Frame {index} | Prediction: {prediction:.5f} | Attention: {low:.6f}–{high:.6f} '
                f'| Clusters: {len(clusters)} | Hulls now: {visible_hulls} | Noise: {noise} '
                f'| Top cluster: {self.selected_cluster} | Δprediction: {scores.get(self.selected_cluster, 0.0):.5f}'
            )
        else:
            self.status.set_text(
                f'Frame {index} | Prediction: {prediction:.5f} | '
                f'Attention: {low:.6f}–{high:.6f} | Clustering: off'
            )
        self.fig.canvas.draw_idle()

    def toggle(self, event=None):
        if not self.playing and int(self.slider.val) == len(self.frames) - 1:
            self.slider.set_val(self.first)
        self.playing = not self.playing

    def tick(self):
        if self.playing:
            index = int(self.slider.val) + 1
            if index < len(self.frames):
                self.slider.set_val(index)
            else:
                self.playing = False

    def key(self, event):
        if event.key == ' ':
            self.toggle()
        elif event.key in ('left', 'right'):
            self.playing = False
            index = int(self.slider.val) + (1 if event.key == 'right' else -1)
            self.slider.set_val(int(np.clip(index, self.first, len(self.frames) - 1)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--episode', type=Path, required=True)
    parser.add_argument('--window-size', type=positive_int, default=30)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--interval', type=positive_int, default=100, help='Playback interval (ms).')
    parser.add_argument('--clustering', action='store_true',
                        help='Enable DBSCAN clusters, convex hulls and occlusion scoring (default: off).')
    parser.add_argument('--particles-per-frame', type=positive_int, default=100)
    parser.add_argument('--eps', type=float, default=0.08)
    parser.add_argument('--min-samples', type=positive_int, default=10)
    parser.add_argument('--temporal-weight', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    if args.eps <= 0 or args.temporal_weight <= 0:
        parser.error('--eps and --temporal-weight must be positive.')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else torch.device(args.device)
    with np.load(args.episode, allow_pickle=False) as archive:
        frames = archive['rgb']
    if frames.ndim != 4 or frames.shape[-1] != 3 or frames.dtype != np.uint8:
        parser.error('Expected uint8 RGB frames of shape (N, H, W, 3).')
    if len(frames) < args.window_size:
        parser.error('Episode must contain at least --window-size frames.')
    model = load_model(args.checkpoint, device, ret_attfmaps=True)
    print(f'Running inference on {device}; ret_attfmaps=True', flush=True)
    viewer = AttentionViewer(model, frames, device, args.window_size, args.interval,
                             args.particles_per_frame, args.eps, args.min_samples,
                             args.temporal_weight, args.seed, clustering=args.clustering)
    viewer.fig.suptitle(f'{args.checkpoint.name} | {args.episode.name}')
    plt.show()


if __name__ == '__main__':
    main()