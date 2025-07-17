import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pycbc
from pycbc import noise, psd
from pycbc.waveform import get_td_waveform
from pycbc.filter import matched_filter
from pycbc.types import TimeSeries
import h5py
import os
from pathlib import Path
import math
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, _LRScheduler

# ==================== Loss Functions ====================

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance in GW detection"""
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# ==================== Learning Rate Schedulers ====================

class CosineAnnealingWithWarmup(_LRScheduler):
    """Cosine Annealing with linear warmup"""
    def __init__(self, optimizer, warmup_steps, total_steps, eta_min=1e-6):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.eta_min = eta_min
        super(CosineAnnealingWithWarmup, self).__init__(optimizer)
        
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            return [base_lr * self.last_epoch / self.warmup_steps 
                    for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            return [self.eta_min + (base_lr - self.eta_min) * 
                    (1 + math.cos(math.pi * progress)) / 2
                    for base_lr in self.base_lrs]

# ==================== Activation Functions ====================

class Swish(nn.Module):
    """Swish activation function for better gradient flow"""
    def forward(self, x):
        return x * torch.sigmoid(x)

# ==================== Normalization Modules ====================

class RobustNormalization(nn.Module):
    """Robust normalization using median and IQR"""
    def __init__(self, dim=-1, eps=1e-8):
        super(RobustNormalization, self).__init__()
        self.dim = dim
        self.eps = eps
        
    def forward(self, x):
        median = x.median(dim=self.dim, keepdim=True)[0]
        q75 = x.quantile(0.75, dim=self.dim, keepdim=True)
        q25 = x.quantile(0.25, dim=self.dim, keepdim=True)
        iqr = q75 - q25 + self.eps
        
        # Prevent division by zero
        iqr = torch.clamp(iqr, min=self.eps)
        
        return (x - median) / iqr

class WhiteningTransform(nn.Module):
    """Whitening transform for GW data"""
    def __init__(self, sample_rate=4096, segment_duration=4):
        super(WhiteningTransform, self).__init__()
        self.sample_rate = sample_rate
        self.segment_duration = segment_duration
        
    def forward(self, x, psd_data=None):
        """Apply whitening in frequency domain"""
        # x shape: (batch_size, sequence_length)
        batch_size, seq_len = x.shape
        
        # FFT to frequency domain
        x_fft = torch.fft.rfft(x, dim=-1)
        freqs = torch.fft.rfftfreq(seq_len, d=1/self.sample_rate)
        
        # Apply whitening
        if psd_data is not None:
            # Interpolate PSD to match frequency bins
            asd = torch.sqrt(psd_data)
            x_white = x_fft / (asd + 1e-10)
        else:
            # Simple whitening without PSD
            x_white = x_fft / (torch.abs(x_fft).mean(dim=0, keepdim=True) + 1e-10)
        
        # Back to time domain
        x_whitened = torch.fft.irfft(x_white, n=seq_len, dim=-1)
        return x_whitened

# ==================== CNN Components with Residual Connections ====================

class ResidualConvBlock(nn.Module):
    """Residual CNN block with proper initialization"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, 
                 activation='elu', use_dropout=True, dropout_rate=0.2):
        super(ResidualConvBlock, self).__init__()
        
        padding = kernel_size // 2
        
        # Main path
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.ln1 = nn.LayerNorm(out_channels)  # Layer norm instead of batch norm
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding)
        self.ln2 = nn.LayerNorm(out_channels)
        
        # Activation
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'swish':
            self.activation = Swish()
        
        # Dropout
        self.use_dropout = use_dropout
        if use_dropout:
            self.dropout = nn.Dropout(dropout_rate)
        
        # Skip connection
        self.skip = nn.Conv1d(in_channels, out_channels, 1, stride) if in_channels != out_channels or stride != 1 else nn.Identity()
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        # He initialization for conv layers
        for m in [self.conv1, self.conv2]:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        identity = self.skip(x)
        
        # Main path with proper normalization
        out = self.conv1(x)
        out = out.transpose(1, 2)  # (B, L, C) for LayerNorm
        out = self.ln1(out)
        out = out.transpose(1, 2)  # Back to (B, C, L)
        out = self.activation(out)
        
        if self.use_dropout:
            out = self.dropout(out)
        
        out = self.conv2(out)
        out = out.transpose(1, 2)
        out = self.ln2(out)
        out = out.transpose(1, 2)
        
        # Residual connection
        out = out + identity
        out = self.activation(out)
        
        return out

# ==================== LSTM Components with Attention ====================

class ResidualLSTM(nn.Module):
    """LSTM with residual connections and layer normalization"""
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.2):
        super(ResidualLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        for i in range(num_layers):
            input_dim = input_size if i == 0 else hidden_size
            self.lstm_layers.append(
                nn.LSTM(input_dim, hidden_size, 1, batch_first=True)
            )
            self.layer_norms.append(nn.LayerNorm(hidden_size))
            if i < num_layers - 1:  # No dropout on last layer
                self.dropouts.append(nn.Dropout(dropout))
                
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        for lstm in self.lstm_layers:
            # Xavier initialization for input weights
            for name, param in lstm.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param)
                elif 'weight_hh' in name:
                    # Orthogonal initialization for recurrent weights
                    nn.init.orthogonal_(param)
                elif 'bias' in name:
                    nn.init.constant_(param, 0)
                    # Set forget gate bias to 1
                    n = param.size(0)
                    param.data[n//4:n//2].fill_(1.0)
                    
    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        out = x
        
        for i in range(self.num_layers):
            # LSTM forward
            lstm_out, _ = self.lstm_layers[i](out)
            
            # Layer normalization
            lstm_out = self.layer_norms[i](lstm_out)
            
            # Residual connection (if dimensions match)
            if out.size(-1) == lstm_out.size(-1):
                lstm_out = lstm_out + out
                
            # Dropout
            if i < len(self.dropouts):
                lstm_out = self.dropouts[i](lstm_out)
                
            out = lstm_out
            
        return out

class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention for temporal dependencies"""
    def __init__(self, hidden_size, num_heads=8, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.k_linear = nn.Linear(hidden_size, hidden_size)
        self.v_linear = nn.Linear(hidden_size, hidden_size)
        self.out_linear = nn.Linear(hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        # Linear transformations and split into heads
        Q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        
        # Output projection
        output = self.out_linear(context)
        output = self.dropout(output)
        
        # Residual connection and layer norm
        output = self.layer_norm(output + x)
        
        return output

# ==================== Main Architecture ====================

class OptimizedGWDetectionNetwork(nn.Module):
    """Optimized CNN-LSTM architecture for gravitational wave detection"""
    def __init__(self, input_length=16384, segment_length=1024, overlap=0.5,
                 conv_channels=[64, 128, 256, 512], kernel_sizes=[7, 5, 3, 3],
                 lstm_hidden_size=256, num_lstm_layers=3, num_attention_heads=8,
                 dense_features=512, num_classes=2, dropout=0.3,
                 use_whitening=True, activation='elu'):
        super(OptimizedGWDetectionNetwork, self).__init__()
        
        self.input_length = input_length
        self.segment_length = segment_length
        self.overlap = overlap
        self.use_whitening = use_whitening
        
        # Calculate segments
        stride = int(segment_length * (1 - overlap))
        self.num_segments = (input_length - segment_length) // stride + 1
        
        # Preprocessing layers
        if use_whitening:
            self.whitening = WhiteningTransform()
        self.robust_norm = RobustNormalization()
        
        # CNN Encoder with residual blocks
        self.conv_blocks = nn.ModuleList()
        in_channels = 1
        
        for i, (out_channels, kernel_size) in enumerate(zip(conv_channels, kernel_sizes)):
            block = ResidualConvBlock(
                in_channels, out_channels, kernel_size,
                stride=2,  # Downsample
                activation=activation,
                dropout_rate=dropout
            )
            self.conv_blocks.append(block)
            in_channels = out_channels
            
        # Calculate CNN output size dynamically
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, segment_length)
            dummy_output = dummy_input
            for conv_block in self.conv_blocks:
                dummy_output = conv_block(dummy_output)
            self.cnn_output_size = dummy_output.numel()
            
        # Transition layer
        self.transition = nn.Sequential(
            nn.Linear(self.cnn_output_size, dense_features),
            nn.LayerNorm(dense_features),
            nn.ELU(),
            nn.Dropout(dropout)
        )
        
        # Aggregate segments
        self.segment_aggregator = nn.Linear(dense_features * self.num_segments, dense_features)
        
        # LSTM with residual connections
        self.lstm = ResidualLSTM(
            dense_features, lstm_hidden_size, 
            num_layers=num_lstm_layers, dropout=dropout
        )
        
        # Self-attention
        self.attention = MultiHeadSelfAttention(
            lstm_hidden_size, num_heads=num_attention_heads, dropout=dropout
        )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden_size, dense_features // 2),
            nn.LayerNorm(dense_features // 2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(dense_features // 2, num_classes)
        )
        
        # Initialize classifier weights
        self._initialize_classifier()
        
    def _initialize_classifier(self):
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
                
    def segment_data(self, x):
        """Segment input data with overlap"""
        batch_size = x.size(0)
        stride = int(self.segment_length * (1 - self.overlap))
        
        segments = []
        for i in range(self.num_segments):
            start = i * stride
            end = start + self.segment_length
            segment = x[:, start:end]
            segments.append(segment)
            
        return torch.stack(segments, dim=1)  # (batch, num_segments, segment_length)
        
    def forward(self, x, psd_data=None):
        batch_size = x.size(0)
        
        # Preprocessing
        if self.use_whitening and psd_data is not None:
            x = self.whitening(x, psd_data)
            
        # Segment data
        segmented = self.segment_data(x)  # (batch, num_segments, segment_length)
        
        # Process each segment through CNN
        segment_features = []
        for i in range(self.num_segments):
            segment = segmented[:, i, :].unsqueeze(1)  # (batch, 1, segment_length)
            
            # Robust normalization
            segment = self.robust_norm(segment)
            
            # CNN encoding
            for conv_block in self.conv_blocks:
                segment = conv_block(segment)
                
            # Flatten and transition with proper shape handling
            segment_flat = segment.contiguous().view(batch_size, -1)
            segment_feat = self.transition(segment_flat)
            segment_features.append(segment_feat)
            
        # Aggregate segments
        all_segments = torch.cat(segment_features, dim=1)  # (batch, dense_features * num_segments)
        aggregated = self.segment_aggregator(all_segments)  # (batch, dense_features)
        
        # Prepare for LSTM (add sequence dimension)
        lstm_input = aggregated.unsqueeze(1).repeat(1, 16, 1)  # (batch, 16, dense_features)
        
        # LSTM processing
        lstm_out = self.lstm(lstm_input)  # (batch, 16, lstm_hidden)
        
        # Self-attention
        attended = self.attention(lstm_out)  # (batch, 16, lstm_hidden)
        
        # Global average pooling
        pooled = attended.mean(dim=1)  # (batch, lstm_hidden)
        
        # Classification
        logits = self.classifier(pooled)
        
        return logits

# ==================== Dataset with Curriculum Learning ====================

class CurriculumGWDataset(Dataset):
    """Dataset with curriculum learning for progressive difficulty"""
    def __init__(self, num_samples=10000, sample_rate=4096, duration=4,
                 curriculum_stage=0, noise_type='aLIGOZeroDetHighPower'):
        self.num_samples = num_samples
        self.sample_rate = sample_rate
        self.duration = duration
        self.curriculum_stage = curriculum_stage
        self.noise_type = noise_type
        
        # Define SNR ranges for curriculum stages
        self.snr_ranges = {
            0: (20, 50),    # Easy: High SNR
            1: (10, 30),    # Medium: Medium SNR
            2: (5, 20),     # Hard: Low SNR
            3: (3, 15)      # Expert: Near threshold
        }
        
        # Generate dataset
        self.data, self.labels = self._generate_dataset()
        
    def _generate_dataset(self):
        """Generate dataset with randomized coalescence times"""
        data = []
        labels = []
        
        # Get SNR range for current curriculum stage
        snr_range = self.snr_ranges.get(self.curriculum_stage, (5, 20))
        
        # Generate noise PSD
        delta_f = 1.0 / self.duration
        flen = int(self.sample_rate / (2 * delta_f)) + 1
        psd_data = psd.aLIGOZeroDetHighPower(flen, delta_f, 20)
        
        target_len = int(self.duration * self.sample_rate)
        delta_t = 1.0 / self.sample_rate
        
        for i in range(self.num_samples):
            has_signal = np.random.random() > 0.5
            
            if has_signal:
                # Use realistic mass distributions
                mass1 = np.random.lognormal(np.log(30), 0.5)
                mass1 = np.clip(mass1, 5, 100)
                mass2 = np.random.uniform(5, mass1)  # Ensure mass1 >= mass2
                
                # Realistic spin distributions
                spin1z = np.random.normal(0, 0.3)
                spin1z = np.clip(spin1z, -0.99, 0.99)
                spin2z = np.random.normal(0, 0.3)
                spin2z = np.clip(spin2z, -0.99, 0.99)
                
                try:
                    # Generate waveform
                    hp, hc = get_td_waveform(
                        approximant="SEOBNRv4_opt",
                        mass1=mass1, mass2=mass2,
                        spin1z=spin1z, spin2z=spin2z,
                        delta_t=delta_t,
                        f_lower=20
                    )
                    
                    # Reset epoch
                    hp._epoch = 0
                    hc._epoch = 0
                    
                    # Randomize coalescence time
                    max_shift = target_len - len(hp) - 100  # Leave some buffer
                    if max_shift > 0:
                        shift = np.random.randint(0, max_shift)
                        hp_data = np.zeros(target_len)
                        hp_data[shift:shift+len(hp)] = hp.data
                    else:
                        hp_data = np.zeros(target_len)
                        hp_data[-len(hp):] = hp.data
                        
                    hp_fixed = TimeSeries(hp_data, delta_t=delta_t, epoch=0)
                    
                    # Generate noise
                    noise_ts = noise.noise_from_psd(
                        target_len, delta_t, psd_data,
                        seed=np.random.randint(0, 1000000)
                    )
                    noise_ts._epoch = 0
                    
                    # Scale to target SNR
                    target_snr = np.random.uniform(*snr_range)
                    signal_power = np.sqrt(np.mean(hp_fixed.data**2))
                    noise_power = np.sqrt(np.mean(noise_ts.data**2))
                    
                    if signal_power > 0 and noise_power > 0:
                        scale_factor = target_snr * noise_power / signal_power
                        hp_scaled = hp_fixed * scale_factor
                    else:
                        hp_scaled = hp_fixed
                        
                    combined = hp_scaled + noise_ts
                    labels.append(1)
                    
                except Exception as e:
                    # Fallback to noise
                    combined = noise.noise_from_psd(
                        target_len, delta_t, psd_data,
                        seed=np.random.randint(0, 1000000)
                    )
                    combined._epoch = 0
                    labels.append(0)
            else:
                # Noise only
                combined = noise.noise_from_psd(
                    target_len, delta_t, psd_data,
                    seed=np.random.randint(0, 1000000)
                )
                combined._epoch = 0
                labels.append(0)
                
            data.append(combined.data)
            
            if (i + 1) % 100 == 0:
                print(f"Generated {i + 1}/{self.num_samples} samples (Stage {self.curriculum_stage})")
                
        return np.array(data), np.array(labels)
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx]), torch.LongTensor([self.labels[idx]])

# ==================== Training Function ====================

def train_optimized_model(model, train_loader, val_loader, num_epochs=50,
                         initial_lr=1e-3, warmup_steps=1000, device='cuda',
                         use_mixed_precision=True, save_path='best_gw_optimized.pth'):
    """Train with all optimizations"""
    
    # Save directly in current directory - no subdirectories needed
    pass
    
    # Move model to device
    model = model.to(device)
    
    # Loss function - Focal Loss for class imbalance
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    
    # Optimizer with differential learning rates
    optimizer = torch.optim.AdamW([
        {'params': model.conv_blocks.parameters(), 'lr': initial_lr},
        {'params': model.lstm.parameters(), 'lr': initial_lr * 0.5},
        {'params': model.attention.parameters(), 'lr': initial_lr * 0.5},
        {'params': model.classifier.parameters(), 'lr': initial_lr}
    ], weight_decay=0.01)
    
    # Learning rate scheduler with warmup
    total_steps = num_epochs * len(train_loader)
    scheduler = CosineAnnealingWithWarmup(
        optimizer, warmup_steps=warmup_steps, 
        total_steps=total_steps, eta_min=1e-6
    )
    
    # Mixed precision training
    scaler = GradScaler() if use_mixed_precision else None
    
    # Training metrics
    best_val_acc = 0.0
    train_losses = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.squeeze().to(device)
            
            # Debug: Check for NaN/Inf in input data
            if torch.isnan(data).any() or torch.isinf(data).any():
                print(f"Warning: NaN/Inf detected in input data at batch {batch_idx}")
                continue
                
            optimizer.zero_grad()
            
            if use_mixed_precision:
                with autocast():
                    output = model(data)
                    
                    # Debug: Check for NaN/Inf in output
                    if torch.isnan(output).any() or torch.isinf(output).any():
                        print(f"Warning: NaN/Inf detected in model output at batch {batch_idx}")
                        continue
                        
                    loss = criterion(output, target)
                    
                    # Debug: Check for NaN loss
                    if torch.isnan(loss):
                        print(f"NaN loss detected at batch {batch_idx}")
                        print(f"Output stats: min={output.min():.6f}, max={output.max():.6f}")
                        print(f"Target stats: {target.unique()}")
                        continue
                    
                scaler.scale(loss).backward()
                
                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
            else:
                output = model(data)
                
                # Debug: Check for NaN/Inf in output
                if torch.isnan(output).any() or torch.isinf(output).any():
                    print(f"Warning: NaN/Inf detected in model output at batch {batch_idx}")
                    continue
                    
                loss = criterion(output, target)
                
                # Debug: Check for NaN loss
                if torch.isnan(loss):
                    print(f"NaN loss detected at batch {batch_idx}")
                    print(f"Output stats: min={output.min():.6f}, max={output.max():.6f}")
                    print(f"Target stats: {target.unique()}")
                    continue
                    
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
            scheduler.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()
            
            if batch_idx % 50 == 0:
                current_lr = scheduler.get_lr()[0]
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.6f}, LR: {current_lr:.6f}')
                
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.squeeze().to(device)
                
                if use_mixed_precision:
                    with autocast():
                        output = model(data)
                        loss = criterion(output, target)
                else:
                    output = model(data)
                    loss = criterion(output, target)
                    
                val_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()
                
        # Calculate metrics
        train_accuracy = 100 * train_correct / train_total
        val_accuracy = 100 * val_correct / val_total
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_accuracies.append(val_accuracy)
        
        # Save best model
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            try:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_accuracy': val_accuracy,
                    'train_loss': avg_train_loss
                }, save_path)
                print(f"✓ New best model saved! Val Acc: {val_accuracy:.2f}%")
            except Exception as e:
                print(f"⚠ Failed to save model: {e}")
                print(f"Continuing training without saving...")
                # Try alternative save location
                try:
                    alt_path = f"backup_model_{epoch}.pth"
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_accuracy': val_accuracy,
                        'train_loss': avg_train_loss
                    }, alt_path)
                    print(f"✓ Backup saved as {alt_path}")
                except:
                    print("⚠ Backup save also failed")
            
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%')
        print(f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')
        print(f'Best Val Acc: {best_val_acc:.2f}%')
        print('-' * 70)
        
    return train_losses, val_accuracies

# ==================== Main Function ====================

def main():
    """Main training pipeline with curriculum learning"""
    print("Starting Optimized Gravitational Wave Detection Training")
    print("=" * 70)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = OptimizedGWDetectionNetwork(
        input_length=16384,
        segment_length=1024,
        overlap=0.5,
        conv_channels=[64, 128, 256, 512],
        kernel_sizes=[7, 5, 3, 3],
        lstm_hidden_size=256,
        num_lstm_layers=3,
        num_attention_heads=8,
        dense_features=512,
        num_classes=2,
        dropout=0.3,
        use_whitening=True,
        activation='elu'
    )
    
    print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Curriculum learning stages
    curriculum_stages = [0, 1, 2, 3]  # Easy to Hard
    epochs_per_stage = 10
    
    # Initialize optimized parameters
    best_overall_acc = 0.0
    all_train_losses = []
    all_val_accuracies = []
    
    # Train through curriculum stages
    for stage in curriculum_stages:
        print(f"\n{'='*70}")
        print(f"CURRICULUM STAGE {stage} - {['Easy', 'Medium', 'Hard', 'Expert'][stage]}")
        print(f"{'='*70}")
        
        # Create datasets for current stage
        train_dataset = CurriculumGWDataset(
            num_samples=5000, 
            curriculum_stage=stage
        )
        val_dataset = CurriculumGWDataset(
            num_samples=1000, 
            curriculum_stage=stage
        )
        
        # Create data loaders with optimal batch size
        train_loader = DataLoader(
            train_dataset, 
            batch_size=64,  # Optimal for LSTM
            shuffle=True, 
            num_workers=4,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=64, 
            shuffle=False, 
            num_workers=4,
            pin_memory=True
        )
        
        # Adjust learning rate for later stages
        initial_lr = 1e-3 * (0.5 ** stage)
        
        # Train for this stage
        train_losses, val_accuracies = train_optimized_model(
            model, train_loader, val_loader,
            num_epochs=epochs_per_stage,
            initial_lr=initial_lr,
            warmup_steps=500 if stage == 0 else 100,
            device=device,
            use_mixed_precision=True,
            save_path=f'gw_optimized_stage_{stage}.pth'
        )
        
        all_train_losses.extend(train_losses)
        all_val_accuracies.extend(val_accuracies)
        
        # Update best accuracy
        stage_best_acc = max(val_accuracies)
        if stage_best_acc > best_overall_acc:
            best_overall_acc = stage_best_acc
            # Save as best overall model
            try:
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'stage': stage,
                    'val_accuracy': stage_best_acc
                }, 'best_gw_optimized_overall.pth')
                print(f"✓ Best overall model saved! Accuracy: {stage_best_acc:.2f}%")
            except Exception as e:
                print(f"⚠ Failed to save best overall model: {e}")
                print("Continuing training...")
    
    print(f"\n{'='*70}")
    print("TRAINING COMPLETED!")
    print(f"Best Overall Validation Accuracy: {best_overall_acc:.2f}%")
    print(f"{'='*70}")
    
    # Test on final challenging dataset
    print("\nTesting on challenging dataset...")
    test_dataset = CurriculumGWDataset(
        num_samples=500,
        curriculum_stage=3  # Expert level
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4
    )
    
    # Evaluate
    model.eval()
    test_correct = 0
    test_total = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.squeeze().to(device)
            output = model(data)
            
            _, predicted = torch.max(output.data, 1)
            test_total += target.size(0)
            test_correct += (predicted == target).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    test_accuracy = 100 * test_correct / test_total
    
    print(f"\nTest Accuracy on Expert Level: {test_accuracy:.2f}%")
    
    # Classification report
    from sklearn.metrics import classification_report, confusion_matrix
    print("\nClassification Report:")
    print(classification_report(all_targets, all_predictions, 
                              target_names=['Noise', 'Signal']))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(all_targets, all_predictions)
    print(cm)
    
    return model, all_train_losses, all_val_accuracies

# ==================== Additional Utilities ====================

def test_gradient_flow(model, data_loader, device='cuda'):
    """Monitor gradient flow through the network"""
    model.train()
    model.to(device)
    
    # Get one batch
    data, target = next(iter(data_loader))
    data, target = data.to(device), target.squeeze().to(device)
    
    # Forward and backward
    output = model(data)
    loss = F.cross_entropy(output, target)
    loss.backward()
    
    # Check gradients
    gradient_norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.data.norm(2).item()
            gradient_norms[name] = grad_norm
    
    # Identify problematic layers
    print("\nGradient Flow Analysis:")
    print("-" * 50)
    for name, norm in sorted(gradient_norms.items()):
        status = "OK" if norm > 1e-7 else "WARNING: Vanishing gradient!"
        print(f"{name}: {norm:.6f} - {status}")
    
    return gradient_norms

def analyze_dead_neurons(model, data_loader, device='cuda'):
    """Check for dead ReLU neurons"""
    model.eval()
    model.to(device)
    
    activation_counts = {}
    
    def hook_fn(module, input, output):
        if isinstance(module, (nn.ReLU, nn.ELU)):
            dead_neurons = (output == 0).sum(dim=0).float().mean().item()
            activation_counts[module] = dead_neurons
    
    # Register hooks
    hooks = []
    for module in model.modules():
        if isinstance(module, (nn.ReLU, nn.ELU)):
            hooks.append(module.register_forward_hook(hook_fn))
    
    # Run one batch
    with torch.no_grad():
        data, _ = next(iter(data_loader))
        data = data.to(device)
        _ = model(data)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    print("\nDead Neuron Analysis:")
    print("-" * 50)
    for i, (module, dead_pct) in enumerate(activation_counts.items()):
        print(f"Layer {i}: {dead_pct*100:.1f}% dead neurons")
    
    return activation_counts

def create_visualization_plots(train_losses, val_accuracies, save_path='training_plots.png'):
    """Create training visualization plots"""
    try:
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Training loss
        ax1.plot(train_losses, 'b-', label='Training Loss')
        ax1.set_title('Training Loss Over Time', fontsize=14)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Focal Loss')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Add curriculum stage markers
        stages = len(train_losses) // 4
        for i in range(1, 4):
            ax1.axvline(x=i*stages, color='r', linestyle='--', alpha=0.5)
            ax1.text(i*stages, max(train_losses)*0.9, f'Stage {i}', 
                    rotation=90, verticalalignment='center')
        
        # Validation accuracy
        ax2.plot(val_accuracies, 'g-', label='Validation Accuracy')
        ax2.set_title('Validation Accuracy Over Time', fontsize=14)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Add curriculum stage markers
        for i in range(1, 4):
            ax2.axvline(x=i*stages, color='r', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.show()
        
        print(f"✓ Training plots saved to {save_path}")
        
    except ImportError:
        print("⚠ matplotlib not available, skipping visualization")

def inference_pipeline(model_path, data_sample, device='cuda'):
    """Production inference pipeline"""
    # Load model
    model = OptimizedGWDetectionNetwork()
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Prepare data
    if isinstance(data_sample, np.ndarray):
        data_tensor = torch.FloatTensor(data_sample).unsqueeze(0)
    else:
        data_tensor = data_sample.unsqueeze(0)
    
    data_tensor = data_tensor.to(device)
    
    with torch.no_grad():
        # Get predictions
        logits = model(data_tensor)
        probabilities = F.softmax(logits, dim=1)
        prediction = torch.argmax(probabilities, dim=1)
        
        # Extract scores
        noise_prob = probabilities[0, 0].item()
        signal_prob = probabilities[0, 1].item()
        
    results = {
        'prediction': 'Signal' if prediction.item() == 1 else 'Noise',
        'confidence': max(noise_prob, signal_prob),
        'noise_probability': noise_prob,
        'signal_probability': signal_prob,
        'logits': logits.cpu().numpy()
    }
    
    return results

# ==================== Run Everything ====================

if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run main training
    model, train_losses, val_accuracies = main()
    
    # Additional analyses
    print("\n" + "="*70)
    print("PERFORMING POST-TRAINING ANALYSIS")
    print("="*70)
    
    # Test gradient flow
    test_loader = DataLoader(
        CurriculumGWDataset(num_samples=100, curriculum_stage=0),
        batch_size=32
    )
    gradient_norms = test_gradient_flow(model, test_loader)
    
    # Check for dead neurons
    dead_neurons = analyze_dead_neurons(model, test_loader)
    
    # Create visualization
    create_visualization_plots(train_losses, val_accuracies)
    
    # Example inference
    print("\n" + "="*70)
    print("EXAMPLE INFERENCE")
    print("="*70)
    
    # Generate a test sample
    test_data = np.random.randn(16384)  # Random data for demo
    results = inference_pipeline('best_gw_optimized_overall.pth', test_data)
    
    print(f"Prediction: {results['prediction']}")
    print(f"Confidence: {results['confidence']:.2%}")
    print(f"Signal Probability: {results['signal_probability']:.3f}")
    print(f"Noise Probability: {results['noise_probability']:.3f}")
    
    print("\n✅ Training and analysis complete!")
    print("Model is ready for gravitational wave detection!")