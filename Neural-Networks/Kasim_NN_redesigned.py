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

class FractalTanimotoLoss(nn.Module):
    """Fractal Tanimoto Loss as described in the paper"""
    def __init__(self, d=0):
        super(FractalTanimotoLoss, self).__init__()
        self.d = d
        
    def forward(self, output, target):
        # MSE term
        mse = torch.mean((target - output) ** 2)
        
        # Fractal Tanimoto similarity coefficient
        numerator = torch.sum(output * target)
        denominator = (2**self.d) * torch.sum(output**2 + target**2) - ((2**(self.d+1)) - 1) * torch.sum(output * target)
        
        # Avoid division by zero
        denominator = torch.clamp(denominator, min=1e-8)
        
        fractal_tanimoto = numerator / denominator
        
        # Combined loss
        loss = mse - fractal_tanimoto
        
        return loss

# ==================== Dense Block ====================

class DenseBlock(nn.Module):
    """Dense block inspired by DenseNet for feature reuse"""
    def __init__(self, in_channels, growth_rate, num_layers, dropout_rate=0.2):
        super(DenseBlock, self).__init__()
        self.num_layers = num_layers
        self.growth_rate = growth_rate
        
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer_in_channels = in_channels + i * growth_rate
            self.layers.append(self._make_layer(layer_in_channels, growth_rate, dropout_rate))
    
    def _make_layer(self, in_channels, out_channels, dropout_rate):
        return nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.ReLU(),
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.Dropout(dropout_rate)
        )
    
    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_feature = layer(torch.cat(features, dim=1))
            features.append(new_feature)
        return torch.cat(features, dim=1)

class TransitionLayer(nn.Module):
    """Transition layer between dense blocks"""
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        self.transition = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.ReLU(),
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            # Remove pooling to avoid dimension collapse
            # nn.AvgPool1d(kernel_size=2, stride=2)
        )
    
    def forward(self, x):
        return self.transition(x)

# ==================== DenseNet Encoder ====================

class DenseNetEncoder(nn.Module):
    """DenseNet-based encoder as described in the paper"""
    def __init__(self, input_channels=1, growth_rate=32, block_config=(6, 12, 24, 16), 
                 num_init_features=64, dropout_rate=0.2):
        super(DenseNetEncoder, self).__init__()
        
        # Initial convolution - less aggressive downsampling
        self.features = nn.Sequential(
            nn.Conv1d(input_channels, num_init_features, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(num_init_features),
            nn.ReLU(),
            # Reduce pooling aggressiveness
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0)  # Changed from 3,2,1 to 2,2,0
        )
        
        # Dense blocks and transitions
        num_features = num_init_features
        self.dense_blocks = nn.ModuleList()
        self.transitions = nn.ModuleList()
        
        for i, num_layers in enumerate(block_config):
            # Add dense block
            block = DenseBlock(num_features, growth_rate, num_layers, dropout_rate)
            self.dense_blocks.append(block)
            num_features += num_layers * growth_rate
            
            # Add transition layer (except for the last block)
            if i != len(block_config) - 1:
                transition = TransitionLayer(num_features, num_features // 2)
                self.transitions.append(transition)
                num_features = num_features // 2
        
        # Final batch norm
        self.final_bn = nn.BatchNorm1d(num_features)
        self.final_features = num_features
    
    def forward(self, x):
        # Initial convolution
        x = self.features(x)
        
        # Dense blocks with transitions
        for i, block in enumerate(self.dense_blocks):
            x = block(x)
            if i < len(self.transitions):
                x = self.transitions[i](x)
        
        # Final batch norm and global average pooling
        x = self.final_bn(x)
        x = F.relu(x)
        x = F.adaptive_avg_pool1d(x, 1)
        
        return x.squeeze(-1)  # Remove the last dimension

# ==================== Bidirectional LSTM Decoder ====================

class BidirectionalLSTMDecoder(nn.Module):
    """Bidirectional LSTM decoder as described in the paper"""
    def __init__(self, feature_dim, hidden_dim=256, num_layers=3, dropout=0.2, output_length=16000):
        super(BidirectionalLSTMDecoder, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_length = output_length
        
        # LSTM layers
        self.lstm_layers = nn.ModuleList()
        for i in range(num_layers):
            input_dim = feature_dim if i == 0 else hidden_dim * 2  # *2 for bidirectional
            self.lstm_layers.append(
                nn.LSTM(input_dim, hidden_dim, batch_first=True, 
                       bidirectional=True, dropout=dropout if i < num_layers-1 else 0)
            )
        
        # Dense output layer
        self.dense_output = nn.Linear(hidden_dim * 2, 1)  # *2 for bidirectional
        
    def forward(self, x, sequence_length=16):
        # x shape: (batch_size, feature_dim)
        batch_size = x.size(0)
        
        # Repeat features for sequence length (as mentioned in paper)
        x = x.unsqueeze(1).repeat(1, sequence_length, 1)  # (batch, seq_len, feature_dim)
        
        # Pass through LSTM layers
        for lstm in self.lstm_layers:
            x, _ = lstm(x)
        
        # Generate output sequence
        outputs = []
        for t in range(sequence_length):
            # Use LSTM output to predict next time step
            step_output = self.dense_output(x[:, t, :])  # (batch, 1)
            outputs.append(step_output)
        
        # Concatenate all outputs and interpolate to target length
        output = torch.cat(outputs, dim=1)  # (batch, sequence_length)
        
        # Interpolate to target output length
        if output.size(1) != self.output_length:
            output = F.interpolate(output.unsqueeze(1), size=self.output_length, mode='linear', align_corners=False)
            output = output.squeeze(1)
        
        return output

# ==================== Complete Dense-LSTM Model ====================

class DenseLSTMGWDetector(nn.Module):
    """Dense-LSTM model for gravitational wave signal extraction as described in the paper"""
    def __init__(self, input_length=16000, subsequence_length=4, overlap=0.75,
                 growth_rate=32, block_config=(6, 12, 24, 16), 
                 lstm_hidden_dim=256, lstm_layers=3, dropout=0.2):
        super(DenseLSTMGWDetector, self).__init__()
        
        self.input_length = input_length
        self.subsequence_length = subsequence_length
        self.overlap = overlap
        
        # Calculate number of subsequences with overlap
        stride = int(subsequence_length * (1 - overlap))
        self.num_subsequences = (input_length - subsequence_length) // stride + 1
        
        # DenseNet encoder for feature extraction
        self.encoder = DenseNetEncoder(
            input_channels=1,
            growth_rate=growth_rate,
            block_config=block_config,
            dropout_rate=dropout
        )
        
        # Bidirectional LSTM decoder
        self.decoder = BidirectionalLSTMDecoder(
            feature_dim=self.encoder.final_features,
            hidden_dim=lstm_hidden_dim,
            num_layers=lstm_layers,
            dropout=dropout,
            output_length=input_length
        )
        
    def segment_data(self, x):
        """Segment input data into overlapping subsequences as described in paper"""
        batch_size = x.size(0)
        stride = int(self.subsequence_length * (1 - self.overlap))
        
        segments = []
        for i in range(self.num_subsequences):
            start = i * stride
            end = start + self.subsequence_length
            if end <= x.size(1):
                segment = x[:, start:end]
                segments.append(segment)
        
        return torch.stack(segments, dim=1)  # (batch, num_segments, segment_length)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Normalize data between -1 and 1 as mentioned in paper
        x_min = x.min(dim=1, keepdim=True)[0]
        x_max = x.max(dim=1, keepdim=True)[0]
        x_range = x_max - x_min + 1e-8  # Avoid division by zero
        x = 2 * (x - x_min) / x_range - 1
        
        # Segment data into overlapping subsequences
        segmented = self.segment_data(x)  # (batch, num_segments, segment_length)
        
        # Process each segment through DenseNet encoder
        segment_features = []
        for i in range(segmented.size(1)):
            segment = segmented[:, i, :].unsqueeze(1)  # (batch, 1, segment_length)
            features = self.encoder(segment)  # (batch, feature_dim)
            segment_features.append(features)
        
        # Average features from all segments (feature aggregation)
        aggregated_features = torch.stack(segment_features, dim=1).mean(dim=1)  # (batch, feature_dim)
        
        # Pass through bidirectional LSTM decoder
        output = self.decoder(aggregated_features)  # (batch, input_length)
        
        return output

# ==================== Dataset for Space-based GW Detection ====================

class SpaceBasedGWDataset(Dataset):
    """Dataset for space-based gravitational wave detection as described in the paper"""
    def __init__(self, num_samples=10000, sample_rate=0.1, duration=160000,
                 snr_range=(30, 70), add_anomalies=False):
        self.num_samples = num_samples
        self.sample_rate = sample_rate  # 0.1 Hz as in paper
        self.duration = duration  # 160,000 seconds as in paper
        self.snr_range = snr_range
        self.add_anomalies = add_anomalies
        
        # Generate dataset
        self.data, self.clean_signals = self._generate_dataset()
        
    def _generate_dataset(self):
        """Generate space-based GW dataset"""
        data = []
        clean_signals = []
        
        target_len = int(self.duration * self.sample_rate)  # 16,000 data points
        delta_t = 1.0 / self.sample_rate
        
        print(f"Generating {self.num_samples} samples with {target_len} data points each...")
        
        for i in range(self.num_samples):
            try:
                # Generate massive black hole binary parameters (as in paper Table 1)
                total_mass = np.random.uniform(1e6, 1e8)  # Solar masses
                mass_ratio = np.random.uniform(0.01, 1.0)
                mass1 = total_mass / (1 + mass_ratio)
                mass2 = total_mass - mass1
                
                # Spin parameters
                spin1z = np.random.uniform(-0.99, 0.99)
                spin2z = np.random.uniform(-0.99, 0.99)
                
                # Generate waveform using SEOBNRv4 (as mentioned in paper)
                hp, hc = get_td_waveform(
                    approximant="SEOBNRv4_opt",
                    mass1=mass1, mass2=mass2,
                    spin1z=spin1z, spin2z=spin2z,
                    delta_t=delta_t,
                    f_lower=3e-5  # 3×10^-5 Hz as mentioned in paper
                )
                
                # Resize waveform to target length
                if len(hp) > target_len:
                    # Take the last part (merger and ringdown)
                    hp_data = hp.data[-target_len:]
                else:
                    # Pad with zeros at the beginning
                    hp_data = np.zeros(target_len)
                    hp_data[-len(hp):] = hp.data
                
                # Generate noise (simplified - in real case would use LISA noise curves)
                noise_data = np.random.normal(0, 1e-20, target_len)
                
                # Add signal to noise with target SNR
                target_snr = np.random.uniform(*self.snr_range)
                signal_power = np.sqrt(np.mean(hp_data**2))
                noise_power = np.sqrt(np.mean(noise_data**2))
                
                if signal_power > 0 and noise_power > 0:
                    scale_factor = target_snr * noise_power / signal_power
                    hp_scaled = hp_data * scale_factor
                else:
                    hp_scaled = hp_data
                
                combined = hp_scaled + noise_data
                
                data.append(combined)
                clean_signals.append(hp_scaled)
                
            except Exception as e:
                # Fallback to noise only
                noise_data = np.random.normal(0, 1e-20, target_len)
                data.append(noise_data)
                clean_signals.append(np.zeros(target_len))
                
            if (i + 1) % 100 == 0:
                print(f"Generated {i + 1}/{self.num_samples} samples")
        
        return np.array(data), np.array(clean_signals)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx]), torch.FloatTensor(self.clean_signals[idx])

# ==================== Training Function ====================

def train_dense_lstm_model(model, train_loader, val_loader, num_epochs=200,
                          initial_lr=1e-3, device='cuda', use_mixed_precision=True,
                          save_path='best_dense_lstm_gw.pth'):
    """Train Dense-LSTM model with fractal Tanimoto loss"""
    
    model = model.to(device)
    
    # Fractal Tanimoto loss (start with d=0, increase during training)
    criterion = FractalTanimotoLoss(d=0)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=10
    )
    
    # Mixed precision training
    scaler = GradScaler() if use_mixed_precision else None
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Update fractal parameter d during training (as mentioned in paper)
        if epoch == 67:  # After 1/3 of training
            criterion.d = 5
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
        elif epoch == 134:  # After 2/3 of training
            criterion.d = 10
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
        
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            
            if use_mixed_precision:
                with autocast():
                    output = model(data)
                    loss = criterion(output, target)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 50 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.6f}')
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                
                if use_mixed_precision:
                    with autocast():
                        output = model(data)
                        loss = criterion(output, target)
                else:
                    output = model(data)
                    loss = criterion(output, target)
                
                val_loss += loss.item()
        
        # Calculate average losses
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            try:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': avg_val_loss,
                    'train_loss': avg_train_loss
                }, save_path)
                print(f"✓ New best model saved! Val Loss: {avg_val_loss:.6f}")
            except Exception as e:
                print(f"⚠ Failed to save model: {e}")
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')
        print(f'Best Val Loss: {best_val_loss:.6f}')
        print('-' * 70)
    
    return train_losses, val_losses

# ==================== Main Function ====================

def main():
    """Main training pipeline for Dense-LSTM GW detector"""
    print("Starting Dense-LSTM Gravitational Wave Detection Training")
    print("Based on: 'Gravitational wave signal extraction against non-stationary instrumental noises with deep neural network'")
    print("=" * 70)
    
    # Set device - USE YOUR RTX 4060!
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create model with RTX 4060 optimized settings
    model = DenseLSTMGWDetector(
        input_length=8000,   # Reduced from 16000 but still substantial
        subsequence_length=4,
        overlap=0.75,
        growth_rate=24,      # Reduced from 32 to save memory
        block_config=(4, 8, 12, 8),  # Smaller but still powerful
        lstm_hidden_dim=192, # Reduced from 256
        lstm_layers=3,       # Keep 3 layers as in paper
        dropout=0.2
    )
    
    print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create datasets - PROPER TESTING SIZE FOR RTX 4060
    print("\nGenerating training dataset...")
    train_dataset = SpaceBasedGWDataset(
        num_samples=500,     # Decent size - not tiny, not massive
        snr_range=(50, 50)   # Fixed SNR=50 for training as in paper
    )
    
    print("Generating validation dataset...")
    val_dataset = SpaceBasedGWDataset(
        num_samples=100,     # Good validation set
        snr_range=(30, 70)   # Variable SNR for validation
    )
    
    # Create data loaders - OPTIMIZED FOR RTX 4060
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,       # Good batch size for RTX 4060
        shuffle=True,
        num_workers=4,       # Use your CPU cores
        pin_memory=True      # Speed up GPU transfer
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=16,       # Same batch size
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Train the model - REASONABLE EPOCHS FOR RTX 4060
    train_losses, val_losses = train_dense_lstm_model(
        model, train_loader, val_loader,
        num_epochs=20,       # Solid training - not too short, not too long
        initial_lr=1e-3,
        device=device,
        use_mixed_precision=True,  # Use tensor cores on RTX 4060
        save_path='best_dense_lstm_gw.pth'
    )
    
    print(f"\n{'='*70}")
    print("TRAINING COMPLETED!")
    print(f"Best Validation Loss: {min(val_losses):.6f}")
    print(f"{'='*70}")
    
    return model, train_losses, val_losses

# ==================== Run Everything ====================

if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run main training
    model, train_losses, val_losses = main()
    
    print("\n✅ Training complete!")
    print("Model is ready for gravitational wave signal extraction!")
