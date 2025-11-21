import torch.nn as nn
import torch
from utils import pad
import math
class MedVidNet(nn.Module):
    def __init__(self, encoder, num_heads=16, num_out=1, pooling_method='transformer', drop_rate=0.0, 
                 debug=False, attn_hidden_size=64):
        super(MedVidNet, self).__init__()
        
        # ResNet encoder
        self.encoder = encoder
        self.num_features = encoder.num_features
        
        assert self.num_features % num_heads == 0, "The number of encoder features must be divisible by the number of attention heads."
        self.num_heads = num_heads
        self.subspace_size = self.num_features // num_heads
        self._scale = math.sqrt(self.subspace_size)
        self.num_out = num_out  # 1 for binary quality classification
        self.drop_rate = drop_rate
        self.debug = debug
        self.pool = pooling_method
        self.attn_hidden_size = attn_hidden_size
        
        # Only create projection layer if using transformer pooling
        if self.pool == 'transformer':
            self.projection = nn.Linear(self.num_features, 16)
            out_dim = 16
        else:
            out_dim = self.num_features
        
        # FC for final classification
        self.fc_out = nn.Sequential(
            nn.Dropout(p=self.drop_rate),
            nn.Linear(out_dim, self.num_out)
        )
        
        if self.pool == 'transformer':
            self.pool_func = self.transformer_pool
            # Add transformer encoder layer
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=16,
                nhead=num_heads,
                dim_feedforward=128,
                dropout=drop_rate,
                batch_first=True
            )
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=1  
            )
            # Learnable [CLS] token
            self.cls_token = nn.Parameter(torch.randn(1, 1, 16))
        elif self.pool == 'attn':
            self.pool_func = self.attention_pool
            self.attn_query_vecs = nn.Parameter(torch.randn(self.num_heads, self.subspace_size))
        elif self.pool == 'tanh_attn':
            self.pool_func = self.tanh_attention_pool
            self.V = nn.Parameter(torch.randn(self.num_heads, self.attn_hidden_size, self.subspace_size))
            self.w = nn.Parameter(torch.randn(self.num_heads, self.attn_hidden_size))
        elif self.pool == 'max':
            self.pool_func = self.max_pool
        elif self.pool == 'avg':
            self.pool_func = self.avg_pool
        else:
            raise NotImplementedError(f"{self.pool} pooling method has not been implemented. Use one of 'transformer', 'attn', 'max', or 'avg'")
    
    def forward(self, videos, videos_per_study, video_markers):
        """
        Args:
            videos: tensor of concatenated frames [sum(total_T), C, H, W]
            videos_per_study: list of number of videos in each study
            video_markers: list of lists, each sublist contains [start,end] frame indices for each video
        """
        h = self.encoder(videos)
        
        # Only apply projection for transformer pooling
        if self.pool == 'transformer':
            h = self.projection(h)
            
        h_study, attn = self.pool_func(h, videos_per_study, video_markers)
        output = self.fc_out(h_study)
        return output, attn
    
    def attention_pool(self, h, videos_per_study, video_markers):
        study_features = []
        
        for study_idx, study_markers in enumerate(video_markers):
            # Get all frames for this study
            study_start = study_markers[0]
            study_end = study_markers[-1]
            study_h = h[study_start:study_end]  # Get all frames for this study
            
            # Reshape for attention
            h_query = study_h.view(-1, self.num_heads, self.subspace_size)
            
            # Compute attention scores
            alpha = (h_query * self.attn_query_vecs).sum(axis=-1) / self._scale
            attn = torch.softmax(alpha, dim=0)
            
            # Apply attention
            h_study = torch.sum(h_query * attn[..., None], dim=0)
            study_features.append(h_study.view(-1))
            
        h_study = torch.stack(study_features)  # [num_studies, num_features]
        return h_study, None
        
    def tanh_attention_pool(self, h, videos_per_study, video_markers):
        study_features = []
        
        for study_idx, study_markers in enumerate(video_markers):
            # Get all frames for this study
            study_start = study_markers[0]
            study_end = study_markers[-1]
            study_h = h[study_start:study_end]  # Get all frames for this study
            
            # Reshape for attention
            h_query = study_h.view(-1, self.num_heads, self.subspace_size)
            
            # Compute attention scores
            alpha = torch.einsum('ijk,jlk->ijl', h_query, self.V).tanh()
            lamb = torch.einsum('ijl,jl->ij', alpha, self.w)
            attn = torch.softmax(lamb, dim=0)
            
            # Apply attention
            h_study = torch.sum(h_query * attn[..., None] / self._scale, dim=0)
            study_features.append(h_study.view(-1))
            
        h_study = torch.stack(study_features)  # [num_studies, num_features]
        return h_study, None
        
    def max_pool(self, h, videos_per_study, video_markers):
        study_features = []
        
        for study_idx, study_markers in enumerate(video_markers):
            # Get all frames for this study
            study_start = study_markers[0]
            study_end = study_markers[-1]
            study_h = h[study_start:study_end]  # Get all frames for this study
            
            # Max pool across all frames in study
            h_study = torch.max(study_h, dim=0)[0]  # [num_features]
            study_features.append(h_study)
            
        h_study = torch.stack(study_features)  # [num_studies, num_features]
        return h_study, None
        
    def avg_pool(self, h, videos_per_study, video_markers):
        study_features = []
        
        for study_idx, study_markers in enumerate(video_markers):
            # Get all frames for this study
            study_start = study_markers[0]
            study_end = study_markers[-1]
            study_h = h[study_start:study_end]  # Get all frames for this study
            
            # Average pool across all frames in study
            h_study = torch.mean(study_h, dim=0)  # [num_features]
            study_features.append(h_study)
            
        h_study = torch.stack(study_features)  # [num_studies, num_features]
        return h_study, None
    
    def transformer_pool(self, h, videos_per_study, video_markers):
        # First group frames by study using video_markers
        study_frames = []
        max_frames = 0
        
        for study_idx, study_markers in enumerate(video_markers):
            # Get all frames for this study's videos
            study_start = study_markers[0]  # First frame of first video
            study_end = study_markers[-1]   # Last frame of last video
            study_frames.append(h[study_start:study_end])
            max_frames = max(max_frames, study_end - study_start)
            
        # Pad each study's frames to max length
        batch_size = len(study_frames)
        padded_frames = torch.zeros(batch_size, max_frames, h.size(-1), device=h.device)
        attention_mask = torch.zeros(batch_size, max_frames + 1, dtype=torch.bool, device=h.device)
        
        for i, frames in enumerate(study_frames):
            n_frames = frames.size(0)
            padded_frames[i, :n_frames] = frames
            attention_mask[i, :n_frames + 1] = True  # +1 for CLS token
            
        # Add CLS token and apply transformer
        cls_tokens = self.cls_token.to(h.device).expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, padded_frames), dim=1)
        x = self.transformer_encoder(x, src_key_padding_mask=~attention_mask)

        # Remove CLS token and get frame features
        frame_features = x[:, 1:]  # [N, L, 16]
        real_frames_mask = attention_mask[:, 1:].unsqueeze(-1)  # [N, L, 1]

        # Zero out padded frames and mean pool
        masked_features = frame_features * real_frames_mask
        study_features = []
        
        for i in range(batch_size):
            valid_frames = masked_features[i][real_frames_mask[i].squeeze(-1)]
            study_mean = valid_frames.mean(dim=0)  # [16]
            study_features.append(study_mean)

        # Stack all study representations
        h_study = torch.stack(study_features)  # [num_studies, 16]
        
        return h_study, None