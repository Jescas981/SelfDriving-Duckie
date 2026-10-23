import torch
from torch import nn
import torch.nn.functional as F


class SelfDrivingModel(nn.Module):
    def __init__(self,lstm_hidden=256, n_actions=1):
        super(SelfDrivingModel, self).__init__()
        self.encoder = nn.Sequential(
            # First block
            nn.Conv2d(in_channels=3, out_channels=24,
                      kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(num_groups=4, num_channels=24),
            nn.ReLU(),
            # Second block
            nn.Conv2d(in_channels=24, out_channels=36,
                      kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(num_groups=4, num_channels=36),
            nn.ReLU(),
            # Third block
            nn.Conv2d(in_channels=36, out_channels=48,
                      kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(num_groups=4, num_channels=48),
            nn.ReLU(),
            # Fourth block
            nn.Conv2d(in_channels=48, out_channels= 64,
                      kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=4, num_channels=64),
            nn.ReLU(),
            # Final block
            nn.Conv2d(in_channels=64, out_channels=64,
                      kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=4, num_channels=64),
            nn.ReLU()
        )

        self.lstm = nn.LSTM(input_size=64, hidden_size=lstm_hidden,
                            num_layers=1, batch_first=True)

        self.head = nn.Sequential(
            nn.Linear(in_features=64 + lstm_hidden, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=n_actions)
        )

        # Attention
        self.query = nn.Linear(in_features=lstm_hidden, out_features=lstm_hidden)
        self.key = nn.Linear(in_features=64, out_features=lstm_hidden)

        # Initialize h_t
        self.f_init_c = nn.Linear(64, lstm_hidden)
        self.f_init_h = nn.Linear(64, lstm_hidden)

        # Gated contextual vector
        self.gate_lin = nn.Linear(lstm_hidden, 64)

    def forward(self, x):
        # Input shape
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        embeddings = self.encoder(x)

        # Embedings shape: (B*T, C, H, W)
        _, C, H, W = embeddings.shape
        L = H * W
        feat_vecs = embeddings.view(B, T, C, H, W).flatten(start_dim=3).permute(0, 1, 3, 2) # (B, T, L, D): (32, 5, 200, 64)

        # Initial state using feat vecs mean [encoder]
        feat_vecs_mean = feat_vecs[:, 0, :, :].mean(dim=1)
        h_t = self.f_init_h(feat_vecs_mean).unsqueeze(0)
        c_t = self.f_init_c(feat_vecs_mean).unsqueeze(0)
        
        outputs = torch.zeros(B, T, 1, device=x.device)
        attentions = torch.zeros(B, T, L, device=x.device)

        for i in range(T):
            feat_vecs_t = feat_vecs[:, i, :, :] # 32, 200, 64
            h_t_tr = h_t.transpose(0, 1)
            Q = self.query(h_t_tr) # 32, 1, 128
            K = self.key(feat_vecs_t) # 32, 200, 128
            scores = torch.bmm(Q, K.transpose(1, 2)) / (K.size(-1) ** 0.5)
            attention_weights = F.softmax(scores, dim=-1) # 32, 1, 200
            attentions[:, i, :] = attention_weights.squeeze(1) # Save
            # Get attention matrix for visualization
            # att_matrix = attention_weights.view(B, 1, H, W) 
            # Weighted average (context)
            ctx_at = torch.bmm(attention_weights, feat_vecs_t) # 32 x 1 x 200 @ 32, 200, 64 -> 32, 1, 64 # a_i
            beta = torch.sigmoid(self.gate_lin(h_t_tr))
            ctx_gate = beta * ctx_at
            # Update LSTM state
            _, (h_t, c_t) = self.lstm(ctx_gate, (h_t, c_t))
            
            # Predict action
            ctx_gate_flat = ctx_gate.view(B, -1)
            h_t_flat = h_t.view(B, -1)

            ctxh_cat = torch.cat([ctx_gate_flat, h_t_flat], dim=-1) # 64 + 128 = 192
            out =  self.head(ctxh_cat)
            outputs[:, i, :] = out

        return outputs, attentions