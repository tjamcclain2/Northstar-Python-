#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GW_CAE.py - Gravitational Wave Autoencoder Neural Network Cascade
Converted from Colab notebook for local AWS EC2 execution
https://drive.google.com/file/d/1rEJX0FvoY2KAROK6vYHAlyOO3fqHs2aF/view?usp=sharing
link to download bg_list.txt as file is too big to be uploaded to github
"""

import os
import sys
import pickle
import numpy as np
import tensorflow as tf
import tensorflow_io as tfio
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for AWS
import matplotlib.pyplot as plt
import soundfile as sf

# PyCBC imports
from pycbc.detector import Detector, get_available_detectors
from pycbc.waveform import get_td_waveform

# Signal processing imports
from scipy.signal import butter, filtfilt, iirnotch
from scipy import stats
import librosa

# Set base paths for AWS
BASE_DIR = "/home/sshamsi"
DATA_DIR = f"{BASE_DIR}/GW_Data/x"
MODEL_DIR = f"{BASE_DIR}/models"
BG_LIST_PATH = f"{BASE_DIR}/bg_list.txt"

# Check if directories exist
if not os.path.exists(DATA_DIR):
    print(f"ERROR: Data directory {DATA_DIR} does not exist. Please create it manually.")
    sys.exit(1)

if not os.path.exists(MODEL_DIR):
    print(f"ERROR: Model directory {MODEL_DIR} does not exist. Please create it manually.")
    sys.exit(1)

print("Loading background noise...")
if not os.path.exists(BG_LIST_PATH):
    print(f"ERROR: Background noise file {BG_LIST_PATH} not found!")
    sys.exit(1)

with open(BG_LIST_PATH, 'rb') as file:
    bg_list = pickle.load(file)
    print(f"Loaded background noise: {len(bg_list)} entries")

# ===== GW Signal Generation Functions =====
def generate_clean_signal(mass1, mass2, sample_rate=4096, duration=18):
    """Generate a single clean GW signal exactly like the original code expects"""
    try:
        # Generate waveform
        hp, hc = get_td_waveform(
            approximant='IMRPhenomPv2',
            mass1=mass1,
            mass2=mass2,
            distance=400,  # Fixed distance
            f_lower=30,
            delta_t=1.0/sample_rate
        )
        
        # Convert to numpy
        signal = np.array(hp.data)
        
        # Target length
        target_length = duration * sample_rate
        
        # Resize properly
        if len(signal) > target_length:
            signal = signal[-target_length:]  # Take end (merger part)
        else:
            # Pad beginning with zeros
            padding = target_length - len(signal)
            signal = np.pad(signal, (padding, 0), mode='constant')
        
        # Normalize to max amplitude 1.0 (exactly like original)
        max_amp = np.max(np.abs(signal))
        if max_amp > 0:
            signal = signal / max_amp
            
        return signal
        
    except Exception as e:
        print(f"Failed to generate mass1={mass1}, mass2={mass2}: {e}")
        return None

def save_wav_simple(signal, filename, sample_rate=4096):
    """Save as WAV exactly like TensorFlow expects"""
    if signal is None:
        return False
        
    try:
        # Convert to 16-bit int exactly like TensorFlow decode_wav expects
        signal_int16 = np.clip(signal * 32767, -32767, 32767).astype(np.int16)
        sf.write(filename, signal_int16, sample_rate)
        return True
    except:
        return False

def generate_systematic_dataset():
    """Generate exactly 400 GW signals as specified in the research paper"""
    
    print("Generating 400 GW signals (as per research paper)...")
    
    # Generate 400 signals with random parameters (5-99 solar masses as per paper)
    mass_configs = []
    
    # Famous events first (5 signals)
    famous_events = [
        (36, 29, 'GW150914'),    # Original detection
        (1.5, 1.3, 'GW170817'),  # Neutron star merger  
        (31, 25, 'GW170814'),    # Three detector event
        (85, 66, 'GW190521'),    # Intermediate mass
        (23, 13, 'GW151226'),    # Boxing day event
    ]
    
    # Add famous events
    for event in famous_events:
        mass_configs.append(event)
    
    # Generate remaining 395 signals with random masses (5-99 solar masses)
    np.random.seed(42)  # For reproducibility
    
    for i in range(395):
        # Random masses between 5 and 99 solar masses (as per paper)
        mass1 = np.random.uniform(5, 99)
        mass2 = np.random.uniform(5, 99)
        
        # Ensure mass1 >= mass2 (convention)
        if mass2 > mass1:
            mass1, mass2 = mass2, mass1
        
        # Categorize by total mass
        total_mass = mass1 + mass2
        if total_mass <= 12:
            category = 'light'
        elif total_mass <= 20:
            category = 'medium'
        else:
            category = 'heavy'
            
        mass_configs.append((mass1, mass2, category))
    
    print(f"Will generate exactly 400 signals (5 famous + 395 random)...")
    
    successful = 0
    failed = 0
    
    for i, config in enumerate(tqdm(mass_configs)):
        if len(config) == 3 and isinstance(config[2], str) and config[2].startswith('GW'):  # Famous event
            m1, m2, event_name = config[0], config[1], config[2]
            filename = f"{event_name}.wav"
        else:  # Regular event
            m1, m2, category = config[0], config[1], config[2]
            filename = f"gw_{category}_{i:04d}.wav"  # 4-digit padding for 400 files
        
        # Generate signal
        signal = generate_clean_signal(m1, m2)
        
        if signal is not None:
            filepath = os.path.join(DATA_DIR, filename)
            if save_wav_simple(signal, filepath):
                successful += 1
            else:
                failed += 1
        else:
            failed += 1
    
    print(f"\n✅ Generation complete!")
    print(f"✅ Successful: {successful}/400 files")
    print(f"❌ Failed: {failed}/400 files")
    print(f"📊 Success rate: {successful/400*100:.1f}%")
    
    return successful

# ===== Preprocessing Functions =====
def spectros(audio):
    wave = audio/tf.math.reduce_max(audio)
    audio_y = tf.squeeze(wave, axis=-1)
    spectrogram = tfio.audio.spectrogram(audio_y, nfft=512, window=512, stride=256)
    
    img = tf.image.rot90(tf.expand_dims(spectrogram, axis=-1), k=1)
    spectrogram = tf.squeeze(img, axis=-1)
    spectrogram = spectrogram/tf.math.reduce_max(spectrogram) #NORMALIZE!!!!
    spectrogram = tf.clip_by_value(spectrogram, 0, 1)
    y = tf.where(tf.math.is_nan(spectrogram), tf.zeros_like(spectrogram), spectrogram)
    
    return y

def crop_upper(img, length):
    img = tf.image.crop_to_bounding_box(tf.expand_dims(img, axis=-1), 257-96, 0, 96, length)
    spectrogram = tf.squeeze(img, axis=-1)
    spectrogram = spectrogram/tf.math.reduce_max(spectrogram)
    spectrogram = tf.clip_by_value(spectrogram, 0, 1)
    final = tf.where(tf.math.is_nan(spectrogram), tf.zeros_like(spectrogram), spectrogram)
    
    return final

def window(img, offset):
    img = tf.image.crop_to_bounding_box(tf.expand_dims(img, axis=-1), 0, offset, 96, 96)
    return tf.squeeze(img, axis=-1)

def normalize(wav):
    max = tf.math.reduce_max(wav, axis=0)
    wav /= max
    return wav, max

def bp(signal, low=60, high=250):
    fs = 4096
    nyq = fs/2
    low /= nyq
    high /= nyq
    order = 2
    
    b,a = butter(order, [low, high], 'bandpass', analog=False)
    y = filtfilt(b,a, signal, axis=0)
    return y

def notch(signal, delete=350):
    fs = 4096
    Q = 30
    
    b,a = iirnotch(delete, Q, fs)
    y = filtfilt(b,a, signal, axis=0)
    return y

def clip(img, p):
    bottom50 = np.percentile(img, p)
    img = tf.clip_by_value(img, bottom50, 100)
    return img

def imgNorm(img):
    max = tf.math.reduce_max(img)
    min = tf.math.reduce_min(img)
    avg = (max+min)/2.0
    final = tf.math.add(img, -avg*tf.ones_like(img))
    final = final/(max-min)
    final = tf.math.add(final, 0.5*tf.ones_like(final))
    return final

def to_3d(x):
    final = tf.concat([x, x, x], axis=-1)
    return final

def piece_together(x):
    x = tf.cast(x, 'float32')
    top = tf.zeros((35,96), dtype='float32')
    side = tf.zeros((90-35, 96-20), dtype='float32')
    bottom = tf.zeros((96-90, 96), dtype='float32')
    final = tf.concat([side, x], axis=-1)
    final = tf.concat([top, final, bottom], axis=0)
    return final

def focus(x):
    if x.shape == (96,96):
        x = x[35:90,-20:]
    elif x.shape == (1,96,96):
        x = x[:,35:90,-20:]
    elif x.shape == (1,96,96,1):
        x = x[:,35:90,-20:,:]
    return x

# ===== Map Functions =====
def decoder_map(x,y):
    x, y = x_y(x,y)
    return tf.expand_dims(y, axis=-1), tf.expand_dims(y, axis=-1)

def x_y(x, y):
    offset = 100
    
    # Input
    a = np.random.randint(20)
    b = np.random.randint(2)
    bg = tf.constant((bg_list[a][b]).value, dtype='float32')
    bg, _ = normalize(bg[-18*4096:])
    
    silencer = 0.000175
    x = tf.math.add(bg, silencer*y)
    audio0 = tf.py_function(func=bp, inp=[x], Tout=tf.float32)
    audio = tf.py_function(func=bp, inp=[audio0], Tout=tf.float32)
    
    x_spect = spectros(tf.expand_dims(audio, axis=-1))
    x_spect = crop_upper(x_spect, 288)
    x_spect = window(x_spect, offset)
    x_spect = focus(x_spect)
    x_spect = piece_together(x_spect)
    x_spect = tf.py_function(func=clip, inp=[x_spect, 97.5], Tout=tf.float32)
    
    # Target
    y_spect = spectros(tf.expand_dims(y, axis=-1))
    y_spect = crop_upper(y_spect, 288)
    y_spect = window(y_spect, offset)
    
    return tf.expand_dims(imgNorm(x_spect), axis=-1), tf.expand_dims(y_spect, axis=-1)

def x_yForTest(x, y, silencer):
    offset = 100
    
    # Input
    a = np.random.randint(20)
    b = np.random.randint(2)
    bg = tf.constant((bg_list[a][b]).value, dtype='float32')
    bg, _ = normalize(bg[-18*4096:])
    
    x = tf.math.add(bg, silencer*y)
    audio0 = tf.py_function(func=bp, inp=[x], Tout=tf.float32)
    audio = tf.py_function(func=bp, inp=[audio0], Tout=tf.float32)
    
    x_spect = spectros(tf.expand_dims(audio, axis=-1))
    x_spect = crop_upper(x_spect, 288)
    x_spect = window(x_spect, offset)
    x_spect = focus(x_spect)
    x_spect = piece_together(x_spect)
    x_spect = tf.py_function(func=clip, inp=[x_spect, 92], Tout=tf.float32)
    
    # Target
    y_spect = spectros(tf.expand_dims(y, axis=-1))
    y_spect = crop_upper(y_spect, 288)
    y_spect = window(y_spect, offset)
    
    return tf.expand_dims(imgNorm(x_spect), axis=-1), y_spect, y, x

# ===== Model Definition =====
def SSIMLoss(y_true, y_pred):
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))

def create_autoencoder():
    """Create the primary autoencoder model"""
    import tensorflow.keras.layers as lay
    
    input = tf.keras.Input(shape=(96,96,1))
    add_back = lay.SeparableConv2D(32, (3,3), padding="same", name="branch_off_conv")(input)
    add_back = lay.BatchNormalization(name="branch_off_batch_norm")(add_back)
    add_back = lay.Activation("sigmoid", name="branch_off_sigmoid")(add_back)
    
    # Encoder
    x = lay.SeparableConv2D(32, (3,3), padding="same")(input)
    x = lay.BatchNormalization()(x)
    x = lay.LeakyReLU(alpha=0.2)(x)
    x = lay.MaxPool2D(2)(x)
    
    x = lay.SeparableConv2D(64, (3,3), padding="same")(x)
    x = lay.BatchNormalization()(x)
    x = lay.LeakyReLU(alpha=0.2)(x)
    x = lay.MaxPool2D(2)(x)
    
    x = lay.Flatten()(x)
    x = lay.Dense(128, activation='sigmoid')(x)
    x = lay.Dense(24*24*64, activation='sigmoid')(x)
    x = lay.Reshape((24, 24, 64))(x)
    
    # Decoder
    x = lay.Conv2DTranspose(32, (4,4), strides=2, padding="same")(x)
    x = lay.BatchNormalization()(x)
    x = lay.LeakyReLU(alpha=0.2)(x)
    x = lay.Conv2DTranspose(32, (4,4), strides=2, padding="same")(x)
    x = lay.BatchNormalization()(x)
    x = lay.LeakyReLU(alpha=0.2)(x)
    
    x = lay.Add()([x, add_back])
    
    x = lay.Conv2DTranspose(1, (4,4), strides=1, padding="same")(x)
    x = lay.BatchNormalization()(x)
    
    output = lay.Activation("sigmoid")(x)
    
    autoencoder = tf.keras.Model(inputs=input, outputs=output)
    return autoencoder

# ===== Main Execution =====
def main():
    # Check if data exists, if not generate it
    try:
        files = os.listdir(DATA_DIR)
    except:
        print(f"ERROR: Cannot access data directory {DATA_DIR}")
        sys.exit(1)
        
    if len(files) < 400:
        print("Generating GW signals...")
        generate_systematic_dataset()
    
    # Load data
    print("Loading GW data...")
    files = os.listdir(DATA_DIR)
    
    inputs = []
    targets = []
    
    for filename in tqdm(files):
        exact_path = os.path.join(DATA_DIR, filename)
        if exact_path.endswith('.wav'):
            try:
                wav_x = tf.io.read_file(exact_path)
                audio, sr = tf.audio.decode_wav(wav_x)
                audio_x = tf.squeeze(audio, axis=-1)
                inputs.append(audio_x)
                targets.append(audio_x)
            except:
                print(f"Issue with {exact_path}")
    
    print(f"Loaded {len(inputs)} signals")
    
    # Create datasets
    gw170817 = [inputs[3], inputs[27], inputs[5], inputs[21], inputs[9]]
    
    train_ds_light = tf.data.Dataset.from_tensor_slices((
        inputs[6:21]+inputs[22:27]+inputs[28:35], 
        targets[6:21]+targets[22:27]+targets[28:35]
    ))
    train_ds_light = train_ds_light.map(x_y).batch(1).prefetch(25).shuffle(20)
    
    val_ds_light = tf.data.Dataset.from_tensor_slices((
        inputs[:4]+inputs[35:40], 
        targets[:4]+targets[35:40]
    ))
    val_ds_light = val_ds_light.map(x_y).batch(1).prefetch(25).shuffle(20)
    
    train_ds_heavy = tf.data.Dataset.from_tensor_slices((inputs[40:280], targets[40:280]))
    train_ds_heavy = train_ds_heavy.map(x_y).batch(1).prefetch(25).shuffle(20)
    
    val_ds_heavy = tf.data.Dataset.from_tensor_slices((inputs[280:400], targets[280:400]))
    val_ds_heavy = val_ds_heavy.map(x_y).batch(1).prefetch(25).shuffle(20)
    
    train_ds_xs = tf.data.Dataset.from_tensor_slices((gw170817, gw170817))
    train_ds_xs = train_ds_xs.map(x_y).batch(1).prefetch(25).shuffle(10)
    
    train_ds_full = tf.data.Dataset.from_tensor_slices((
        inputs[10:85]+gw170817+gw170817+gw170817, 
        targets[10:85]+gw170817+gw170817+gw170817
    ))
    train_ds_full = train_ds_full.map(x_y).batch(1).prefetch(25).shuffle(20)
    
    val_ds_full = tf.data.Dataset.from_tensor_slices((
        inputs[:10]+inputs[300:], 
        targets[:10]+targets[300:]
    ))
    val_ds_full = val_ds_full.map(x_y).batch(1).prefetch(25).shuffle(20)
    
    # Create models
    print("Creating models...")
    autoencoder = create_autoencoder()
    
    very_lightAE = tf.keras.models.clone_model(autoencoder)
    very_lightAE._name = "Very_Light_AE"
    
    lightAE = tf.keras.models.clone_model(autoencoder)
    lightAE._name = "Light_AE"
    
    heavyAE = tf.keras.models.clone_model(autoencoder)
    heavyAE._name = "Heavy_AE"
    
    # Train models
    from tensorflow.keras.optimizers import Adam
    
    print("Training Light AE...")
    lightAE.compile(optimizer=Adam(learning_rate=0.00005), loss=SSIMLoss)
    lightAE.fit(train_ds_light, validation_data=val_ds_light, epochs=300)
    lightAE.save(f"{MODEL_DIR}/hwg_initial_light")
    
    print("Training Heavy AE...")
    heavyAE.compile(optimizer=Adam(learning_rate=0.001), loss=SSIMLoss)
    heavyAE.fit(train_ds_heavy, validation_data=val_ds_heavy, epochs=300)
    heavyAE.save(f"{MODEL_DIR}/hwg_initial_heavy")
    
    print("Training Very Light AE...")
    very_lightAE.compile(optimizer=Adam(learning_rate=0.001), loss=SSIMLoss)
    very_lightAE.fit(train_ds_xs, epochs=300)
    very_lightAE.save(f"{MODEL_DIR}/hwg_initial_veryLight")
    
    print("Training complete!")

if __name__ == "__main__":
    main()