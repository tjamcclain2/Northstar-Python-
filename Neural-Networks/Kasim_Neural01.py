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

class NormalizationSegmentation(nn.Module):
    """
    Normalization & Segmentation block as shown in the diagram
    """
    def __init__(self, segment_length=1024, overlap=0.5, normalization_type='zscore'):
        super(NormalizationSegmentation, self).__init__()
        self.segment_length = segment_length
        self.overlap = overlap
        self.normalization_type = normalization_type
        self.stride = int(segment_length * (1 - overlap))
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, sequence_length) or (batch_size, 1, sequence_length)
        Returns:
            Segmented and normalized tensor of shape (batch_size, num_segments, segment_length)
        """
        if x.dim() == 3:
            x = x.squeeze(1)  # Remove channel dimension if present
        
        batch_size, seq_len = x.shape
        
        # Calculate number of segments
        num_segments = (seq_len - self.segment_length) // self.stride + 1
        
        # Create segments
        segments = []
        for i in range(num_segments):
            start_idx = i * self.stride
            end_idx = start_idx + self.segment_length
            segment = x[:, start_idx:end_idx]
            segments.append(segment)
        
        # Stack segments
        segmented = torch.stack(segments, dim=1)  # (batch_size, num_segments, segment_length)
        
        # Normalize each segment
        if self.normalization_type == 'zscore':
            # Z-score normalization per segment
            mean = segmented.mean(dim=2, keepdim=True)
            std = segmented.std(dim=2, keepdim=True) + 1e-8
            normalized = (segmented - mean) / std
        elif self.normalization_type == 'minmax':
            # Min-max normalization per segment
            min_val = segmented.min(dim=2, keepdim=True)[0]
            max_val = segmented.max(dim=2, keepdim=True)[0]
            normalized = (segmented - min_val) / (max_val - min_val + 1e-8)
        elif self.normalization_type == 'robust':
            # Robust normalization using median and IQR
            median = segmented.median(dim=2, keepdim=True)[0]
            q75 = segmented.quantile(0.75, dim=2, keepdim=True)
            q25 = segmented.quantile(0.25, dim=2, keepdim=True)
            iqr = q75 - q25 + 1e-8
            normalized = (segmented - median) / iqr
        else:
            normalized = segmented
        
        return normalized


class GravitationalWaveDataset(Dataset):
    """
    FIXED Dataset class for gravitational wave detection using pyCBC
    """
    def __init__(self, num_samples=10000, sample_rate=4096, duration=4, 
                 snr_range=(5, 50), noise_type='aLIGOZeroDetHighPower'):
        self.num_samples = num_samples
        self.sample_rate = sample_rate
        self.duration = duration
        self.snr_range = snr_range
        self.noise_type = noise_type
        
        # Generate dataset
        self.data, self.labels = self._generate_dataset()
    
    def _generate_dataset(self):
        """Generate synthetic gravitational wave data with noise - FIXED VERSION"""
        data = []
        labels = []
        
        # Generate noise PSD
        delta_f = 1.0 / self.duration
        flen = int(self.sample_rate / (2 * delta_f)) + 1
        psd_data = psd.aLIGOZeroDetHighPower(flen, delta_f, 20)
        
        # Define time parameters consistently
        target_len = int(self.duration * self.sample_rate)
        delta_t = 1.0 / self.sample_rate
        
        for i in range(self.num_samples):
            # 50% signal + noise, 50% noise only
            has_signal = np.random.random() > 0.5
            
            if has_signal:
                # Generate random binary black hole parameters
                mass1 = np.random.uniform(10, 80)
                mass2 = np.random.uniform(10, 80)
                spin1z = np.random.uniform(-0.8, 0.8)
                spin2z = np.random.uniform(-0.8, 0.8)
                
                try:
                    # Generate waveform
                    hp, hc = get_td_waveform(approximant="SEOBNRv4_opt",
                                           mass1=mass1, mass2=mass2,
                                           spin1z=spin1z, spin2z=spin2z,
                                           delta_t=delta_t,
                                           f_lower=20)
                    
                    # FIX 1: Reset epoch to 0 immediately after generation
                    hp._epoch = 0
                    hc._epoch = 0
                    
                    # FIX 2: Create a new TimeSeries with exact target length and zero epoch
                    if len(hp) > target_len:
                        # Take the last part of the waveform (merger and ringdown)
                        hp_data = hp.data[-target_len:]
                    else:
                        # Pad with zeros at the beginning
                        hp_data = np.zeros(target_len)
                        hp_data[-len(hp):] = hp.data
                    
                    # Create new TimeSeries with consistent parameters
                    hp_fixed = TimeSeries(hp_data, delta_t=delta_t, epoch=0)
                    
                    # Generate noise with matching parameters
                    noise_ts = noise.noise_from_psd(target_len, delta_t, psd_data, 
                                                   seed=np.random.randint(0, 1000000))
                    # FIX 3: Ensure noise also has zero epoch
                    noise_ts._epoch = 0
                    
                    # FIX 4: Scale signal to desired SNR using proper calculation
                    target_snr = np.random.uniform(*self.snr_range)
                    
                    # Calculate signal and noise power more robustly
                    signal_power = np.sqrt(np.mean(hp_fixed.data**2))
                    noise_power = np.sqrt(np.mean(noise_ts.data**2))
                    
                    if signal_power > 0 and noise_power > 0:
                        scale_factor = target_snr * noise_power / signal_power
                        hp_scaled = hp_fixed * scale_factor
                    else:
                        hp_scaled = hp_fixed
                    
                    # FIX 5: Ensure both time series have same epoch before combining
                    assert hp_scaled._epoch == noise_ts._epoch, f"Epoch mismatch: {hp_scaled._epoch} vs {noise_ts._epoch}"
                    
                    # Combine signal and noise
                    combined = hp_scaled + noise_ts
                    labels.append(1)
                    
                except Exception as e:
                    print(f"Warning: Failed to generate waveform for sample {i}: {e}")
                    # Fall back to noise only
                    combined = noise.noise_from_psd(target_len, delta_t, psd_data, 
                                                   seed=np.random.randint(0, 1000000))
                    combined._epoch = 0
                    labels.append(0)
                
            else:
                # Generate noise only
                combined = noise.noise_from_psd(target_len, delta_t, psd_data, 
                                               seed=np.random.randint(0, 1000000))
                combined._epoch = 0
                labels.append(0)
            
            # Store raw data (normalization will be done in the network)
            data.append(combined.data)
            
            # Progress indicator
            if (i + 1) % 100 == 0:
                print(f"Generated {i + 1}/{self.num_samples} samples")
        
        return np.array(data), np.array(labels)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx]), torch.LongTensor([self.labels[idx]])


class ConvLSTMBlock(nn.Module):
    """
    Convolutional block from the encoder part
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvLSTMBlock, self).__init__()
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.batch_norm = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.avg_pool = nn.AvgPool1d(kernel_size=2, stride=2)
        
    def forward(self, x):
        x = self.conv1d(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.avg_pool(x)
        return x


class DenseBlock(nn.Module):
    """
    Dense block for feature extraction
    """
    def __init__(self, in_features, out_features):
        super(DenseBlock, self).__init__()
        self.dense = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        return self.dropout(self.relu(self.dense(x)))


class GWDetectionNetwork(nn.Module):
    """
    Main neural network architecture for gravitational wave detection
    Includes Normalization & Segmentation as shown in diagram
    """
    def __init__(self, input_length=16384, segment_length=1024, overlap=0.5,
                 num_conv_layers=3, conv_channels=[64, 128, 256],
                 lstm_hidden_size=128, num_lstm_layers=3, dense_features=512, 
                 num_classes=2, dropout=0.3):
        super(GWDetectionNetwork, self).__init__()
        
        self.input_length = input_length
        self.segment_length = segment_length
        self.num_conv_layers = num_conv_layers
        self.lstm_hidden_size = lstm_hidden_size
        self.num_lstm_layers = num_lstm_layers
        
        # Normalization & Segmentation block (as shown in diagram)
        self.norm_segment = NormalizationSegmentation(
            segment_length=segment_length, 
            overlap=overlap,
            normalization_type='zscore'
        )
        
        # Calculate number of segments
        stride = int(segment_length * (1 - overlap))
        self.num_segments = (input_length - segment_length) // stride + 1
        
        # Encoder blocks (left side of diagram)
        self.encoder_blocks = nn.ModuleList()
        in_channels = 1  # Single channel input per segment
        
        for i in range(num_conv_layers):
            out_channels = conv_channels[i] if i < len(conv_channels) else conv_channels[-1]
            block = ConvLSTMBlock(in_channels, out_channels)
            self.encoder_blocks.append(block)
            in_channels = out_channels
        
        # Calculate the length after convolutions and pooling
        conv_output_length = segment_length
        for _ in range(num_conv_layers):
            conv_output_length = conv_output_length // 2  # Due to AvgPool1d with stride=2
        
        # Ensure we have valid dimensions
        if conv_output_length <= 0:
            raise ValueError(f"Segment length {segment_length} is too small for {num_conv_layers} conv layers")
        
        # Transition layer - process all segments
        segment_features = conv_channels[-1] * conv_output_length
        self.transition = nn.Linear(segment_features, dense_features)
        
        # Dense block
        self.dense_block = DenseBlock(dense_features, dense_features)
        
        # Aggregate features from all segments
        self.segment_aggregator = nn.Linear(dense_features * self.num_segments, dense_features)
        
        # Create multiple vectors for LSTM input
        self.num_vectors = 16
        self.vector_projection = nn.Linear(dense_features, self.num_vectors * lstm_hidden_size)
        
        # LSTM layers (3 layers as shown in diagram)
        self.lstm_layers = nn.ModuleList()
        for i in range(num_lstm_layers):
            lstm = nn.LSTM(lstm_hidden_size, lstm_hidden_size, batch_first=True, 
                          dropout=dropout if i < num_lstm_layers-1 else 0)
            self.lstm_layers.append(lstm)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Final classifier
        self.classifier = nn.Linear(lstm_hidden_size, num_classes)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Step 1: Normalization & Segmentation (as shown in diagram)
        segmented = self.norm_segment(x)  # (batch_size, num_segments, segment_length)
        
        # Step 2: Process each segment through encoder
        segment_features = []
        for i in range(self.num_segments):
            segment = segmented[:, i, :].unsqueeze(1)  # (batch_size, 1, segment_length)
            
            # Encoder path for this segment
            current = segment
            for block in self.encoder_blocks:
                current = block(current)
            
            # Flatten segment features
            segment_flat = current.view(batch_size, -1)
            segment_feat = self.transition(segment_flat)
            segment_feat = self.dense_block(segment_feat)
            segment_features.append(segment_feat)
        
        # Step 3: Aggregate features from all segments
        all_features = torch.cat(segment_features, dim=1)  # (batch_size, dense_features * num_segments)
        aggregated = self.segment_aggregator(all_features)
        
        # Step 4: Project to multiple vectors for LSTM (Vec1, Vec2, ..., Vec16000)
        lstm_input = self.vector_projection(aggregated)
        lstm_input = lstm_input.view(batch_size, self.num_vectors, self.lstm_hidden_size)
        
        # Step 5: LSTM processing (3 layers as shown in diagram)
        lstm_out = lstm_input
        for lstm_layer in self.lstm_layers:
            lstm_out, _ = lstm_layer(lstm_out)
            lstm_out = self.dropout(lstm_out)
        
        # Step 6: Use the last output for classification
        final_output = lstm_out[:, -1, :]  # (batch_size, lstm_hidden_size)
        
        # Step 7: Classification
        logits = self.classifier(final_output)
        
        return logits


def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=0.001):
    """
    Training function for the gravitational wave detection model
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    train_losses = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.squeeze().to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.6f}')
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.squeeze().to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        val_accuracy = 100 * correct / total
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_accuracies.append(val_accuracy)
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')
        print('-' * 60)
        
        scheduler.step()
    
    return train_losses, val_accuracies


def test_network_functionality():
    """
    Test network functionality with normalization & segmentation
    """
    print("=" * 60)
    print("TESTING NETWORK WITH NORMALIZATION & SEGMENTATION")
    print("=" * 60)
    
    # Test parameters
    input_length = 16384
    segment_length = 1024
    overlap = 0.5
    
    # 1. Test normalization & segmentation
    print("\n1. Testing Normalization & Segmentation...")
    try:
        norm_seg = NormalizationSegmentation(segment_length=segment_length, overlap=overlap)
        test_input = torch.randn(4, input_length)
        
        segmented = norm_seg(test_input)
        print(f"✓ Input shape: {test_input.shape}")
        print(f"✓ Segmented shape: {segmented.shape}")
        
        # Check normalization
        segment_means = segmented.mean(dim=2)
        segment_stds = segmented.std(dim=2)
        print(f"✓ Segment means range: [{segment_means.min():.3f}, {segment_means.max():.3f}]")
        print(f"✓ Segment stds range: [{segment_stds.min():.3f}, {segment_stds.max():.3f}]")
        
    except Exception as e:
        print(f"✗ Normalization & Segmentation failed: {e}")
        return False
    
    # 2. Test full model
    print("\n2. Testing full model...")
    try:
        model = GWDetectionNetwork(
            input_length=input_length,
            segment_length=segment_length,
            overlap=overlap,
            conv_channels=[64, 128, 256],
            lstm_hidden_size=128,
            num_lstm_layers=3,
            dense_features=512,
            num_classes=2
        )
        
        print(f"✓ Model created successfully")
        print(f"✓ Number of segments: {model.num_segments}")
        print(f"✓ Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test forward pass
        with torch.no_grad():
            output = model(test_input)
        
        print(f"✓ Forward pass successful")
        print(f"✓ Output shape: {output.shape}")
        
    except Exception as e:
        print(f"✗ Full model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 3. Test with real data
    print("\n3. Testing with real data...")
    try:
        dataset = GravitationalWaveDataset(num_samples=10, sample_rate=4096, duration=4)
        real_data, real_labels = dataset[0]
        
        real_batch = real_data.unsqueeze(0)
        with torch.no_grad():
            real_output = model(real_batch)
        
        print(f"✓ Real data test successful")
        print(f"✓ Real output shape: {real_output.shape}")
        
        # Check predictions
        probabilities = torch.softmax(real_output, dim=1)
        prediction = torch.argmax(probabilities, dim=1)
        print(f"✓ Prediction: {prediction.item()}")
        print(f"✓ True label: {real_labels.item()}")
        
    except Exception as e:
        print(f"✗ Real data test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED! ✓")
    print("Network with Normalization & Segmentation is ready!")
    print("=" * 60)
    return True


def main():
    """
    Main function to run the complete training pipeline
    """
    print("Starting Gravitational Wave Detection Training Pipeline")
    print("=" * 60)
    
    # Test functionality first
    if not test_network_functionality():
        print("Tests failed! Please fix errors before training.")
        return
    
    # Create datasets
    print("\nCreating training and validation datasets...")
    train_dataset = GravitationalWaveDataset(num_samples=5000, sample_rate=4096, duration=4)
    val_dataset = GravitationalWaveDataset(num_samples=1000, sample_rate=4096, duration=4)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Create model
    print("\nCreating model...")
    model = GWDetectionNetwork(
        input_length=16384,
        segment_length=1024,
        overlap=0.5,
        conv_channels=[64, 128, 256],
        lstm_hidden_size=128,
        num_lstm_layers=3,
        dense_features=512,
        num_classes=2
    )
    
    print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Train model
    print("\nStarting training...")
    train_losses, val_accuracies = train_model(
        model, train_loader, val_loader, 
        num_epochs=20, learning_rate=0.001
    )
    
    print("\nTraining completed!")
    print(f"Final validation accuracy: {val_accuracies[-1]:.2f}%")


# Example usage
if __name__ == "__main__":
    main()
