import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.interpolate import interp1d
import sys
import os

# Try to import pycWB components
import pycwb
from pycwb.modules.cwb_conversions import *
from pycwb.modules.read_data import *
PYCWB_AVAILABLE = True

class LIGOChirpExtractor:
    def __init__(self, sample_rate=4096, low_freq=35, high_freq=350):
        """
        Initialize the chirp extractor
        
        Parameters:
        -----------
        sample_rate : int
            Sample rate of the data (Hz)
        low_freq : float
            Low frequency cutoff (Hz)
        high_freq : float
            High frequency cutoff (Hz)
        """
        self.sample_rate = sample_rate
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.data = None
        self.time = None
        self.filtered_data = None
        
    def load_ligo_txt(self, filename):
        """
        Load LIGO strain data from text file
        
        Expected format:
        Column 1: Time (GPS or relative)
        Column 2: Strain data
        """
        try:
            # Try different delimiters
            data = np.loadtxt(filename)
            
            if data.shape[1] >= 2:
                self.time = data[:, 0]
                self.data = data[:, 1]
            else:
                # Single column - assume uniform time spacing
                self.data = data.flatten()
                self.time = np.arange(len(self.data)) / self.sample_rate
                
            print(f"Loaded {len(self.data)} data points from {filename}")
            print(f"Time range: {self.time[0]:.6f} to {self.time[-1]:.6f} seconds")
            print(f"Data range: {np.min(self.data):.2e} to {np.max(self.data):.2e}")
            
            return True
            
        except Exception as e:
            print(f"Error loading file {filename}: {e}")
            return False
    
    def preprocess_data(self):
        """
        Preprocess the data: whitening and bandpass filtering
        """
        if self.data is None:
            print("No data loaded!")
            return False
            
        # Ensure uniform time spacing
        dt = np.median(np.diff(self.time))
        if not np.allclose(np.diff(self.time), dt, rtol=1e-3):
            print("Resampling data to uniform time grid...")
            time_uniform = np.arange(self.time[0], self.time[-1], dt)
            interp_func = interp1d(self.time, self.data, kind='linear')
            self.data = interp_func(time_uniform)
            self.time = time_uniform
        
        # Update sample rate based on actual data
        self.sample_rate = 1.0 / dt
        print(f"Sample rate: {self.sample_rate:.1f} Hz")
        
        # Bandpass filter
        nyquist = self.sample_rate / 2
        low = self.low_freq / nyquist
        high = min(self.high_freq / nyquist, 0.95)
        
        sos = signal.butter(4, [low, high], btype='band', output='sos')
        self.filtered_data = signal.sosfilt(sos, self.data)
        
        # Simple whitening (normalize by local RMS)
        window_size = int(0.5 * self.sample_rate)  # 0.5 second window
        rms = np.sqrt(np.convolve(self.filtered_data**2, 
                                 np.ones(window_size)/window_size, mode='same'))
        rms = np.maximum(rms, np.std(self.filtered_data) * 0.1)  # Avoid division by zero
        self.filtered_data = self.filtered_data / rms
        
        print("Data preprocessing completed")
        return True
    
    def extract_chirps_pycwb(self):
        """
        Extract chirps using pycWB
        """
        if not PYCWB_AVAILABLE:
            print("Error: pycWB not available. Cannot extract chirps.")
            return []
        
        try:
            # Configure pycWB
            config = pycwb.Config()
            config.inRate = int(self.sample_rate)
            config.fLow = self.low_freq
            config.fHigh = self.high_freq
            config.levelR = 8  # Wavelet resolution levels
            config.l_low = 3   # Low frequency resolution level
            config.l_high = 8  # High frequency resolution level
            
            # Prepare data in pycWB format
            detector_data = {
                'H1': self.filtered_data  # Assume single detector for now
            }
            
            # Run coherent WaveBurst analysis
            print("Running pycWB coherent WaveBurst analysis...")
            results = pycwb.cwb_analysis(detector_data, config)
            
            # Extract chirp events
            events = results.get_events()
            
            chirps = []
            for event in events:
                if event.snr > 5.0:  # SNR threshold
                    chirp_info = {
                        'time': event.time,
                        'frequency': event.central_freq,
                        'snr': event.snr,
                        'duration': event.duration,
                        'bandwidth': event.bandwidth
                    }
                    chirps.append(chirp_info)
            
            print(f"Found {len(chirps)} chirp candidates using pycWB")
            return chirps
            
        except Exception as e:
            print(f"Error in pycWB analysis: {e}")
            return []
    
    def plot_results(self, chirps, save_plot=True):
        """
        Plot the results
        """
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Plot 1: Raw and filtered data
        axes[0].plot(self.time, self.data, 'b-', alpha=0.7, label='Raw data')
        axes[0].plot(self.time, self.filtered_data, 'r-', label='Filtered data')
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Strain')
        axes[0].set_title('LIGO Strain Data')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot 2: Spectrogram
        f, t, Sxx = signal.spectrogram(self.filtered_data, 
                                      fs=self.sample_rate,
                                      window='hann',
                                      nperseg=int(0.25 * self.sample_rate))
        
        axes[1].pcolormesh(t + self.time[0], f, 10*np.log10(Sxx), 
                          shading='gouraud', cmap='viridis')
        axes[1].set_ylabel('Frequency (Hz)')
        axes[1].set_xlabel('Time (s)')
        axes[1].set_title('Spectrogram')
        axes[1].set_ylim([self.low_freq, self.high_freq])
        
        # Mark chirps on spectrogram
        for chirp in chirps[:10]:  # Show top 10 chirps
            axes[1].plot(chirp['time'], chirp['frequency'], 'ro', markersize=8)
            axes[1].annotate(f"SNR: {chirp['snr']:.1f}", 
                           (chirp['time'], chirp['frequency']),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, color='white')
        
        # Plot 3: Chirp parameters
        if chirps:
            times = [c['time'] for c in chirps]
            snrs = [c['snr'] for c in chirps]
            freqs = [c['frequency'] for c in chirps]
            
            scatter = axes[2].scatter(times, snrs, c=freqs, cmap='plasma', s=60)
            axes[2].set_xlabel('Time (s)')
            axes[2].set_ylabel('SNR')
            axes[2].set_title('Detected Chirps')
            axes[2].grid(True)
            
            cbar = plt.colorbar(scatter, ax=axes[2])
            cbar.set_label('Frequency (Hz)')
        
        plt.tight_layout()
        
        if save_plot:
            plt.savefig('ligo_chirp_analysis.png', dpi=300, bbox_inches='tight')
            print("Plot saved as 'ligo_chirp_analysis.png'")
        
        plt.show()
    
    def save_results(self, chirps, filename='chirp_results.txt'):
        """
        Save chirp extraction results to file
        """
        with open(filename, 'w') as f:
            f.write("# LIGO Chirp Extraction Results\n")
            f.write("# Time(s)\tFrequency(Hz)\tSNR\tMethod\n")
            
            for chirp in chirps:
                method = 'pycwb'
                f.write(f"{chirp['time']:.6f}\t{chirp['frequency']:.2f}\t{chirp['snr']:.2f}\t{method}\n")
        
        print(f"Results saved to {filename}")

def main():
    """
    Main function to run the chirp extraction
    """
    # Check if pycWB is available
    if not PYCWB_AVAILABLE:
        print("Error: pycWB is required for this script to function.")
        print("Please install pycWB using: pip install pycwb")
        sys.exit(1)
    
    # Use the specified file path
    input_file = "/content/H-H1_GWOSC_16KHZ_R1-1268903496-32.txt"
    
    if not os.path.exists(input_file):
        print(f"Error: File {input_file} not found!")
        sys.exit(1)
    
    # Initialize extractor
    extractor = LIGOChirpExtractor(sample_rate=16384, low_freq=35, high_freq=350)
    
    # Load data
    print(f"Loading LIGO data from {input_file}...")
    if not extractor.load_ligo_txt(input_file):
        sys.exit(1)
    
    # Preprocess data
    print("Preprocessing data...")
    if not extractor.preprocess_data():
        sys.exit(1)
    
    # Extract chirps
    print("Extracting chirps...")
    chirps = extractor.extract_chirps_pycwb()
    
    # Display results
    print(f"\n=== CHIRP EXTRACTION RESULTS ===")
    print(f"Total chirps found: {len(chirps)}")
    
    if chirps:
        print("\nTop 10 chirps:")
        for i, chirp in enumerate(chirps[:10]):
            print(f"{i+1:2d}. Time: {chirp['time']:8.3f}s, "
                  f"Freq: {chirp['frequency']:6.1f}Hz, "
                  f"SNR: {chirp['snr']:5.1f}")
    
    # Save results and plot
    if chirps:
        extractor.save_results(chirps)
        extractor.plot_results(chirps)
    else:
        print("No chirps detected. Try adjusting the frequency range or SNR threshold.")

if __name__ == "__main__":
    main()