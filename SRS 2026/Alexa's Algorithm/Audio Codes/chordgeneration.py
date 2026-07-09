import numpy as np
import soundfile as sf
import os
from music21 import chord, note, pitch

# -------------------------------------------------------
# SETTINGS
# -------------------------------------------------------
CLEAN_PATH  = r"C:\Users\alexa\WnL Docs\Signal Processing\audio signals dataset\clean_input"
SAMPLE_RATE = 22050
DURATION    = 5.0        # seconds per chord
AMPLITUDE   = 0.6        # master volume (0-1) — headroom to avoid clipping when summing notes

# ADSR envelope shape
ATTACK  = 0.15           # seconds to ramp up from silence to full volume
DECAY   = 0.25           # seconds to fall from full volume to sustain level
SUSTAIN = 0.75           # sustain level as a fraction of full volume (0-1)
RELEASE = 0.6            # seconds to fade out at the end

# Harmonic content — each note is not a pure sine but a mix of overtones
# Format: (harmonic_number, relative_amplitude)
# Harmonic 1 = fundamental, 2 = octave above, 3 = fifth above that, etc.
HARMONICS = [
    (1, 1.00),
    (2, 0.45),
    (3, 0.20),
    (4, 0.10),
    (5, 0.05),
]

# -------------------------------------------------------
# CHORD DEFINITIONS
# -------------------------------------------------------
# Each entry: (filename_stem, list of pitch strings in music21 format)
# music21 pitch format: note name + accidental + octave
#   C4 = middle C, # = sharp, - = flat
#   Examples: 'C4', 'F#3', 'B-4'

CHORDS = [
    ("C_major",          ['C4',  'E4',  'G4']),
    ("A_minor",          ['A3',  'C4',  'E4']),
    ("G_major",          ['G3',  'B3',  'D4']),
    ("F_major",          ['F3',  'A3',  'C4']),
    ("D_minor",          ['D4',  'F4',  'A4']),
    ("E_major",          ['E3',  'G#3', 'B3']),
    ("B_diminished",     ['B3',  'D4',  'F4']),
    ("C_major_seventh",  ['C4',  'E4',  'G4',  'B4']),
    ("G_dominant_seventh",['G3', 'B3',  'D4',  'F4']),
    ("A_major",          ['A3',  'C#4', 'E4']),
]

# -------------------------------------------------------
# ADSR ENVELOPE
# -------------------------------------------------------
def make_envelope(num_samples, sample_rate, attack, decay, sustain, release):
    envelope = np.ones(num_samples)
    a = int(attack  * sample_rate)
    d = int(decay   * sample_rate)
    r = int(release * sample_rate)

    # Attack: ramp from 0 to 1
    if a > 0:
        envelope[:a] = np.linspace(0.0, 1.0, a)

    # Decay: ramp from 1 down to sustain level
    if d > 0:
        end = min(a + d, num_samples)
        envelope[a:end] = np.linspace(1.0, sustain, end - a)

    # Sustain: hold at sustain level
    s_start = a + d
    s_end   = num_samples - r
    if s_end > s_start:
        envelope[s_start:s_end] = sustain

    # Release: ramp from sustain level to 0
    if r > 0 and s_end < num_samples:
        envelope[s_end:] = np.linspace(sustain, 0.0, num_samples - s_end)

    return envelope


# -------------------------------------------------------
# SYNTHESIZE ONE NOTE WITH HARMONICS
# -------------------------------------------------------
def synthesize_note(frequency, num_samples, sample_rate):
    t = np.linspace(0, num_samples / sample_rate, num_samples, endpoint=False)
    wave = np.zeros(num_samples)
    total_weight = sum(amp for _, amp in HARMONICS)
    for harmonic, amp in HARMONICS:
        wave += (amp / total_weight) * np.sin(2 * np.pi * frequency * harmonic * t)
    return wave


# -------------------------------------------------------
# SYNTHESIZE ONE CHORD
# -------------------------------------------------------
def synthesize_chord(pitch_strings, sample_rate, duration):
    num_samples = int(duration * sample_rate)

    # Build music21 chord to extract frequencies
    notes    = [note.Note(p) for p in pitch_strings]
    m21chord = chord.Chord(notes)

    print(f"  Notes: {[str(n.pitch) for n in m21chord.notes]}")
    print(f"  Frequencies (Hz): {[round(n.pitch.frequency, 2) for n in m21chord.notes]}")

    # Sum sine waves for each note in the chord
    signal = np.zeros(num_samples)
    for n in m21chord.notes:
        freq = n.pitch.frequency
        signal += synthesize_note(freq, num_samples, sample_rate)

    # Normalize so the sum of notes doesn't clip
    max_val = np.max(np.abs(signal))
    if max_val > 0:
        signal = signal / max_val

    # Apply ADSR envelope
    envelope = make_envelope(num_samples, sample_rate, ATTACK, DECAY, SUSTAIN, RELEASE)
    signal   = signal * envelope

    # Apply master amplitude
    signal = signal * AMPLITUDE

    return signal.astype(np.float32)


# -------------------------------------------------------
# GENERATE AND SAVE ALL CHORDS
# -------------------------------------------------------
os.makedirs(CLEAN_PATH, exist_ok=True)
print(f"Saving chord files to: {CLEAN_PATH}\n")

generated = []

for name, pitches in CHORDS:
    print(f"Generating: {name}")
    signal    = synthesize_chord(pitches, SAMPLE_RATE, DURATION)
    filename  = f"{name}.wav"
    filepath  = os.path.join(CLEAN_PATH, filename)
    sf.write(filepath, signal, SAMPLE_RATE)
    generated.append(filename)
    print(f"  Saved: {filename} ({len(signal)} samples, {DURATION}s)\n")

print(f"Done! {len(generated)} chord files saved to {CLEAN_PATH}")
print("\nNext steps:")
print("  1. Run audiosignalsphase1.py  — adds noise to the chord files")
print("  2. Run loadsignals2.py        — rebuilds clean_cache and noisy_cache")
print("  3. Run audiosignalsphase2b.py — denoises the noisy chords")
print("  4. Run loadsignals2.py again  — rebuilds denoised_cache")
print("  5. Run audiosignalsevaluateb.py — evaluate and plot results")
