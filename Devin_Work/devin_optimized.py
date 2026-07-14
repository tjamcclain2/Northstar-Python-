#Let's fix this code like a computer scientist!

"""
Northstar Algorithm — Original Implementation and Optimization. This is the most readable version of the code, with all the optimizations applied as of July 14, 2025.

Author: Dr. Tom McClain  
Optimized by: Devin Upadhyay
Date: 2025-07-14
"""

import time
import numpy as np
import cupy as cp 
from cupyx.profiler import benchmark
DTYPE = cp.float32  # if needed to switch to fp64 
abs_diff_sum = cp.ReductionKernel( # needed in the get_best_fit_angles_deltas method to reduce the diff abs and sum into 1 operation
    'T wh0, T wh1, T wh2, T wh3, T wl0, T wl1, T wl2, T wl3, T a0, T a1, T a2, T a3, T realH, T realL',   # input params
    'T out',             # one output
    'abs(realH - (wh0*a0+wh1*a1+wh2*a2+wh3*a3)) + abs(realL - (wl0*a0+wl1*a1+wl2*a2+wl3*a3))', # what to compute per element
    'a + b',             # how to combine them (add)
    'out = a',           # write the final total
    '0',                 # starting value
    'abs_diff_sum'
)
# Constants (SI units unless noted otherwise)
NUMBER_DETECTORS = 2
NUMBER_GW_POLARIZATIONS = 2
NUMBER_GW_MODES = 4
NUMBER_SOURCE_ANGLES = 3
LIGO_DETECTOR_SAMPLING_RATE = 16_384  # Hz
EARTH_RADIUS = 6_371_000  # meters
SPEED_OF_LIGHT = 299_792_458  # m/s
MAX_HANFORD_LIVINGSTON_DELAY = 0.010002567302556083  # seconds
WEIGHTING_POWER = 2

# Helper to convert (deg, min, sec) to radians
def dms_to_rad(deg, minutes, seconds):
    return cp.deg2rad(deg + minutes / 60 + seconds / 3600)

# Detector angles: [latitude, longitude, orientation]
hanford_detector_angles = cp.array([
    dms_to_rad(46, 27, 18.528),
    dms_to_rad(240, 35, 32.4343),
    cp.deg2rad(125.9994) + cp.pi / 2
], dtype=DTYPE)

livingston_detector_angles = cp.array([
    dms_to_rad(30, 33, 46.4196),
    dms_to_rad(269, 13, 32.7346),
    cp.deg2rad(197.7165) + cp.pi / 2
], dtype=DTYPE)

#=========================================================================

# transforms rank-2 contravariant tensor under a change of basis
def transform_2_0_tensor(matrix, change_basis_matrix) :
    contravariant_transformation_matrix = cp.linalg.inv(change_basis_matrix)
    partial_transformation = cp.einsum('...ki,...kl->...il', contravariant_transformation_matrix, matrix)
    return cp.einsum('...lj,...il->...ij', contravariant_transformation_matrix, partial_transformation)

# transforms mixed tensor
# Note: This is not used in the current code, but provided for completeness
def transform_1_1_tensor(matrix, change_basis_matrix) :
    contravariant_transformation_matrix = np.linalg.inv(change_basis_matrix)
    partial_transformation = np.einsum('ik,kl->il', change_basis_matrix, matrix)
    return np.einsum('lj,il->ij', contravariant_transformation_matrix, partial_transformation)

# transforms rank-2 covariant tensor under a change of basis
# Note: This is not used in the current code, but provided for completeness
def transform_0_2_tensor(matrix, change_basis_matrix) :
    partial_transformation = cp.einsum('aik,akl->ail', change_basis_matrix, matrix)
    return cp.einsum('ajl,ail->aij', change_basis_matrix, partial_transformation)

#=========================================================================

def source_vector_from_angles(angle_grid) :
    
    """
        Compute a Cartesian unit vector from spherical coordinates. This function takes a list with three angles in it -- either the declination, right ascension, and polarization angles of a gravitational wave source, or the latitute, longitude, and orientation a gravitational wave detector 
        -- and returns a 1D NumPy array representing the unit-length vector from the center the Earth to the source/detector. Note that only the first two angles are actually needed to compute the unit vector.

        Parameters
        ----------
        angles : array-like of float, shape (3,)
            Three angles in radians:
            - For a gravitational–wave **source**: (declination δ, right ascension α, polarization ψ)
            - For a **detector**:            (latitude θ, longitude ϕ,  orientation γ)
            Only the first two angles (angles[0], angles[1]) are used to compute the unit vector.

        Returns
        -------
        vector : ndarray of float, shape (3,)
            Unit-length vector in Cartesian (x, y, z) coordinates pointing from the Earth's center
            toward the specified direction.
    """
    #[first, second, third] = angles
    all_first = angle_grid[..., 0]
    all_second = angle_grid[..., 1]
    initial_source_vector = cp.array([cp.cos(all_first)*cp.cos(all_second), cp.cos(all_first)*cp.sin(all_second), cp.sin(all_first)], dtype=DTYPE).T # .T because we need (n_ang,3) 
    return initial_source_vector # returns whole angle grid converted to source vector

#=========================================================================

def change_basis_gw_to_ec(angle_grid) :
    
    """
        Compute the covariant change-of-basis matrix from the gravitational-wave frame to the Earth-centered frame. This function takes a list with the declination, right ascension, and polarization angles of a gravitational wave source in the Earth-centered 
        coordinate system and returns a NumPy array that effects the change-of-basis (covariant transformation matrix) from the gravitational wave frame to the Earth-centered frame. 
        The inverse of this matrix inverse effects the change-of-basis from the Earth-centered frame to the gravitational wave frame, and is also the contravariant transformation matrix 
        (i.e., changes the components vectors) from the gravitational wave frame to the Earth-centered frame.

        Parameters
        ----------
        source_angles : array-like of float, shape (3,)
            Three angles (in radians) defining the source orientation in Earth-centered coordinates:
            - declination δ (elevation above the celestial equator)
            - right ascension α (azimuthal angle around Earth’s axis)
            - polarization ψ (rotation about the line of sight)

        Returns
        -------
        change_basis_matrix : ndarray of float, shape (3, 3)
            Covariant transformation matrix that, when applied to a vector expressed in the
            gravitational-wave frame, yields its components in the Earth-centered frame.
    """
    initial_source_vector = source_vector_from_angles(angle_grid)
    initial_gw_z_vector_earth_centered = -1 * initial_source_vector
    all_declination = angle_grid[...,0]
    all_right_ascension = angle_grid[...,1]
    all_polarization = angle_grid[...,2]
    initial_gw_y_vector_earth_centered = cp.array([
            -cp.sin(all_declination) * cp.cos(all_right_ascension),
            -cp.sin(all_declination) * cp.sin(all_right_ascension),
            cp.cos(all_declination) 
        ], dtype=DTYPE).T  # .T because we need (n_ang,3)
    initial_gw_x_vector_earth_centered = cp.cross(
            initial_gw_z_vector_earth_centered,
            initial_gw_y_vector_earth_centered,
            axis=1)
    # stack (with -1) adds a new axis at the end which makes it (angles, component, which_vector) (n,3,3) 
    initial_gw_vecs_ec = cp.stack([
            initial_gw_x_vector_earth_centered,
            initial_gw_y_vector_earth_centered,
            initial_gw_z_vector_earth_centered
        ], axis=-1)
        # Rotate by polarization about z_gw
    cos_p = cp.cos(all_polarization)
    sin_p = cp.sin(all_polarization)
    # Create 1D tracking arrays filled with 0s and 1s matching the length of n_ang
    zeros = cp.zeros_like(all_polarization)
    ones = cp.ones_like(all_polarization)
    
    polarization_rotation_matrices = cp.array([
            [cos_p, -sin_p, zeros],
            [sin_p,  cos_p, zeros],
            [zeros,  zeros,  ones]
        ], dtype=DTYPE)
    # to get the shape (n_ang, 3, 3) so that the matmul can be applied to the 3x3 matrix
    polarization_rotation_matrices = cp.transpose(polarization_rotation_matrices, (2,0,1))

    contravariant_transformation_matrices = polarization_rotation_matrices @ initial_gw_vecs_ec
    change_basis_matrices = cp.linalg.inv(contravariant_transformation_matrices)
    return change_basis_matrices

#=========================================================================

def change_basis_detector_to_ec(detector_angles) :
    
    """
    Compute the covariant change-of-basis matrix from the detector frame to the Earth-centered frame. 
    This function takes a list containing the latitude, longitude, and orientation a gravitational wave detector 
    and returns a NumPy array that effects the change-of-basis (covariant transformation matrix) from the detector frame 
    to the Earth-centered frame.

    Parameters
    ----------
    detector_angles : array-like of float, shape (3,)
        Three angles (in radians) defining the detector orientation in Earth-centered coordinates:
        - latitude θ (elevation above the equatorial plane)
        - longitude ϕ (azimuthal angle around Earth’s axis)
        - orientation γ (rotation about the local vertical axis)

    Returns
    -------
    change_basis_matrix : ndarray of float, shape (3, 3)
        Covariant transformation matrix that, when applied to a tensor in the detector frame,
        yields its components in the Earth-centered frame.

    """
    
    latitude, longitude, orientation = detector_angles

    # ẑ_det points from Earth's center to detector
    z_det_ec = source_vector_from_angles(detector_angles)

    # x̂_det is tangent eastward
    x_det_ec = cp.array([-cp.sin(longitude), cp.cos(longitude), cp.array(0.0)], dtype=DTYPE) # array because 0-d array and python float mismatch

    # ŷ_det completes right-handed set
    y_det_ec = cp.cross(z_det_ec, x_det_ec)

    # Stack as columns to form the GW-frame basis matrix in EC coords
    det_vecs_ec = cp.vstack([x_det_ec, y_det_ec, z_det_ec]).T


    # Rotate about local vertical (z_det_ec) by orientation γ
    orientation_rotation = cp.array([
        [cp.cos(orientation), -cp.sin(orientation),  cp.array(0.0)],
        [cp.sin(orientation),  cp.cos(orientation),  cp.array(0.0)],
        [cp.array(0.0),              cp.array(0.0),  cp.array(1.0)]
    ], dtype=DTYPE)

    T_contravariant = orientation_rotation @ det_vecs_ec
    
    # Directly inverted the contravariant matrix in one line
    change_basis_matrix = cp.linalg.inv(T_contravariant)
    return change_basis_matrix

#=========================================================================

def beam_pattern_response_functions(detector_angles, angle_grid) :
    """
    Compute the beam-pattern (antenna-pattern) response functions F₊ and Fₓ for a gravitational-wave detector. This function takes two lists -- the first containing the latitude, 
    longitude, and orientation angles of a gravitational wave detector, and the second containing the declination, right ascensions, 
    and polarization angles of a gravitational wave source -- and returns a list with the beam pattern response functions F_+ and F_x of the detector for that source.

    Parameters
    ----------
    detector_angles : array-like of float, shape (3,)
        Detector orientation in Earth-centered coordinates (radians):
        - latitude θ  
        - longitude ϕ  
        - orientation γ (rotation about the local vertical axis)
    source_angles : array-like of float, shape (3,)
        Source orientation in Earth-centered coordinates (radians):
        - declination δ  
        - right ascension α  
        - polarization ψ

    Returns
    -------
    F_plus, F_cross : float
        Beam-pattern response functions:
        - F₊ (“plus” polarization response)
        - Fₓ (“cross” polarization response)

    """
    
    # Detector‐frame response tensor (2-0)
    arm_response_tensor = cp.array([
        [0.5,  0.0, 0.0],
        [0.0, -0.5, 0.0],
        [0.0,  0.0, 0.0]
    ], dtype=DTYPE)

    # Change-of-basis: detector frame → Earth-centered
    det_to_ec_basis = change_basis_detector_to_ec(detector_angles)
    arm_response_tensor_ec = transform_2_0_tensor(arm_response_tensor, det_to_ec_basis)

    # Change-of-basis: Earth-centered → GW frame
    gw_to_ec_basis = change_basis_gw_to_ec(angle_grid)
    ec_to_gw_basis = cp.linalg.inv(gw_to_ec_basis)
    arm_response_tensor_gw = transform_2_0_tensor(arm_response_tensor_ec, ec_to_gw_basis) #cupy ndarray of size(n_ang, 3, 3)

    # Extract plus and cross responses
    F_plus  = arm_response_tensor_gw[:, 0, 0] - arm_response_tensor_gw[:, 1, 1]
    F_cross = arm_response_tensor_gw[:, 0, 1] + arm_response_tensor_gw[:, 1, 0]

    return F_plus, F_cross
    

#=========================================================================

def time_delay_hanford_to_livingston(angle_grid) :
    
    """
    Compute the gravitational-wave arrival time delay between the Hanford and Livingston detectors. This function take a list of the declination, right ascension, 
    and polarization angles of a gravitational wave source and returns the time delay between when the signal will arrive at the Hanford detector and 
    when it will arrive at the Livingston detector. Negative values indicate that the signal arrives at the Livingston detector first.

    Parameters
    ----------
    source_angles : array-like of float, shape (3,)
        GW source orientation angles in Earth-centered coordinates (radians):
        - declination δ
        - right ascension α
        - polarization ψ

    Returns
    -------
    delay : float
        Time difference Δt = t_Hanford – t_Livingston in seconds.
        Negative values indicate the wavefront reaches Livingston before Hanford.

    """
    
    # Earth-centered position vectors (m) of each site
    r_H = EARTH_RADIUS * source_vector_from_angles(hanford_detector_angles)
    r_L = EARTH_RADIUS * source_vector_from_angles(livingston_detector_angles)

    # Baseline from Hanford to Livingston
    baseline = r_L - r_H

    # GW propagation direction (unit vector) in Earth frame
    propagation_dir = -source_vector_from_angles(angle_grid) # (n_ang, 3)

    # Return time delay (s)
    return cp.dot(propagation_dir, baseline) / SPEED_OF_LIGHT

#=========================================================================

def generate_network_time_array(signal_lifetime, detector_sampling_rate, maximum_time_delay) :
    
    """
    Create a symmetric time-sample array spanning the signal duration plus network delays. This function takes a gravitational wave signal lifetime, 
    a detector sampling rate, and the maximum possible time delay between the detectors in a network, and returns a 
    NumPy array with absolute detector strain response times appropriate for all the detectors in a network. 
    Note that all responses times are actual sampled times, assuming correct time synchorization between sites.

    Parameters
    ----------
    signal_lifetime : float
        Duration of the gravitational-wave signal in seconds.
    detector_sampling_rate : float
        Detector sampling rate in Hz (samples per second).
    maximum_time_delay : float
        Maximum inter-detector time delay in seconds.

    Returns
    -------
    time_array : ndarray of float
        1D array of time samples (in seconds), from
        –T_max to +T_max (exclusive), where
        T_max = signal_lifetime + maximum_time_delay,
        sampled at 1/detector_sampling_rate intervals.

    """

    # Total half-window in seconds
    T_max = signal_lifetime + maximum_time_delay

    # Number of samples on each side of zero
    half_samples = int(cp.ceil(T_max * detector_sampling_rate))

    # Generate symmetric time array around zero
    time_array = (cp.arange(-half_samples, half_samples) / detector_sampling_rate).astype(DTYPE)
    return time_array

#=========================================================================


def generate_oscillatory_terms(signal_lifetime, signal_frequency, time_array, time_delay) :
    
    """
    Generate Gaussian-modulated sinusoidal terms for all GW modes over a time grid. This function takes the lifetime and frequency of a gravitional wave, a NumPy array with the detector strain response times of a gravitational wave detector network, 
    and the time delay between when the gravitational wave arrives at the specific detector where the terms are being evaluated compared to when it arrived at a fixed reference detector (Hanford, in this code), 
    and returns a NumPy array with the appropriate sine-Gaussian gravitational wave strain amplitudes for 
    the detector at each network time. Note that the order of the output array is 1) time sample 2) [A_+, B_x, A_+, B_x] for each of the four gravitational wave modes.

    Parameters
    ----------
    signal_lifetime : float
        1/ e-folding time of the Gaussian envelope in seconds.
    signal_frequency : float
        Central frequency of the GW signal in Hz.
    time_array : ndarray of float, shape (N,)
        Array of time samples (s) at which to evaluate the waveform.
    time_delay : float
        Arrival-time offset (s) of the wavefront at this detector relative to a reference.

    Returns
    -------
    oscillatory_terms : ndarray of float, shape (N, 4)
        For each time in `time_array`, the four mode amplitudes
        [A₊, Bₓ, A₊, Bₓ], where
        A₊(t) = e^(−Q² (t−τ)²) · cos(2π f (t−τ)),
        Bₓ(t) = e^(−Q² (t−τ)²) · sin(2π f (t−τ)).
    """
    
    # Gaussian Q-factor
    Q = (cp.sqrt(cp.log(2)) / signal_lifetime).astype(DTYPE)

    # Time relative to arrival at this detector
    time_delay = cp.atleast_1d(time_delay).astype(DTYPE)
    dt = time_array[cp.newaxis, :] - time_delay[:, cp.newaxis] # dt = (n_ang, n_times)
    
    # Envelope and phase arrays
    envelope = cp.exp(-Q**2 * dt**2)
    phase    = 2 * cp.pi * signal_frequency * dt 

    # Cosine- and sine-modulated terms
    A_plus = envelope * cp.cos(phase)
    B_cross = envelope * cp.sin(phase)

    # Stack into (N,4) for the four modes [A₊, Bₓ, A₊, Bₓ]
    oscillatory_terms = cp.stack([A_plus, B_cross, A_plus, B_cross], axis=-1)
    '''
    (n_ang, n_times, 4)
    time 0:
    angle0: [a00, b00, a00, b00]   ← the 4 modes, cleanly grouped
    angle1: [a01, b01, a01, b01]
    angle2: [a02, b02, a02, b02]
    time 1:
    angle0: [a10, b10, a10, b10]
    angle1: [a11, b11, a11, b11]
    angle2: [a12, b12, a12, b12]
    '''
    return oscillatory_terms

#=========================================================================

def generate_model_angles_array(number_angular_samples) :
    
    """
    Generate randomized source-angle sets [declination, right ascension, polarization]. This functions takes the number of desired model angle sets [S, phi, psi] 
    and returns a NumPy array with that many randomized angle sets. Note that the first angle in each set is bewteen -π/2 and π/2, 
    while the other two angles are between 0 and 2π. The first angle is the declination, the second is the right ascension, 
    and the third is the polarization angle of a gravitational wave source.

    Parameters
    ----------
    number_angular_samples : int
        Number of angle triples to generate.

    Returns
    -------
    angle_grid : ndarray of float, shape (number_angular_samples, 3)
        Array of angle sets:
        - Column 0: declination δ ∈ [–π/2, π/2]
        - Column 1: right ascension α ∈ [0, 2π)
        - Column 2: polarization ψ ∈ [0, 2π)
    """
    
    # Declinations: uniform in [–π/2, π/2]
    dec = ((cp.random.rand(number_angular_samples) - 0.5) * cp.pi).astype(DTYPE)
    # Right ascension & polarization: uniform in [0, 2π)
    ra_psi = (cp.random.rand(number_angular_samples, 2) * 2 * cp.pi).astype(DTYPE)
    # Stack into shape (N, 3)
    angle_grid = cp.column_stack((dec, ra_psi))
    return angle_grid

#=========================================================================

def generate_model_amplitudes_array(number_amplitude_combinations, gw_max_amps) :
    """
    Generates a NumPy array of random gravitational wave amplitude combinations. This function takes the number of desired model amplitude combinations 
    [A_+, B_x, A_+, B_x] and the maximum allowed value for any amplitude and returns a NumPy array with the desired amplitude combinations.


    Parameters:
        number_amplitude_combinations (int): The number of random amplitude vectors to generate.
        gw_max_amps (float): The maximum possible value for any individual amplitude component.

    Returns:
        np.ndarray: A 2D NumPy array of shape (number_amplitude_combinations, number_gw_modes),
                    where each row is a random amplitude vector [A_+, B_x, A_+, B_x],
                    with each component sampled uniformly from [0, gw_max_amps).
    """
    return (cp.random.rand(number_amplitude_combinations, NUMBER_GW_MODES) * gw_max_amps).astype(DTYPE)


#=========================================================================

def generate_model_detector_responses(
    signal_frequency,
    signal_lifetime,
    detector_sampling_rate,
    gw_max_amps,
    number_amplitude_combinations,
    number_angular_samples,
):
    """
    Predict network detector responses over amplitude and angle models.

    Parameters
    ----------
    signal_frequency : float
        Central frequency f of the GW monochromatic mode (Hz).
    signal_lifetime : float
        Lifetime τ (time to half-maximum amplitude) of the GW mode (s).
    detector_sampling_rate : float
        Sampling rate (Hz) of all detectors (samples per second).
    gw_max_amps : float
        Maximum GW amplitude magnitude to sample.
    number_amplitude_combinations : int
        Number of amplitude combinations to generate (each combination is a 4-vector).
    number_angular_samples : int
        Number of source-angle combinations to generate (each is [δ, α, ψ]).

    Returns
    -------
    responses : ndarray, shape (number_angular_samples, number_amplitude_combinations, number_time_samples, 2)
        Predicted strain responses for each angle/amp/time/model:
        - axis 0: index of angle combination
        - axis 1: index of amplitude combination
        - axis 2: time sample
        - axis 3: detector index (0 = Hanford, 1 = Livingston)
    angle_grid : ndarray, shape (number_angular_samples, 3)
        Array of source angle triples [declination δ, right ascension α, polarization ψ].
    
    """
    # Abbreviate the input sizes
    n_amp = number_amplitude_combinations
    n_ang = number_angular_samples

    # Time grid for the network (reference detector delay = 0)
    time_array = generate_network_time_array(
        signal_lifetime,
        detector_sampling_rate,
        MAX_HANFORD_LIVINGSTON_DELAY,
    )
    n_times = time_array.size

    # Precompute Hanford oscillatory terms (zero delay)
    hanford_terms = generate_oscillatory_terms(
        signal_lifetime, signal_frequency, time_array, time_delay=0.0
    )

    # Generate model amplitude and angle grids
    amplitude_grid = generate_model_amplitudes_array(n_amp, gw_max_amps)  # shape (n_amp, 4) (100,4)
    angle_grid = generate_model_angles_array(n_ang)                     # shape (n_ang, 3) (100,3)

    # Initialize output array: detectors=2 (0=H1, 1=L1)
    # responses = cp.empty((n_ang, n_amp, n_times, 2), dtype=DTYPE)

    Fp_H, Fx_H = beam_pattern_response_functions(hanford_detector_angles, angle_grid)  
    Fp_L, Fx_L = beam_pattern_response_functions(livingston_detector_angles, angle_grid)
    pattern_H = cp.array([Fp_H, Fx_H, Fp_H, Fx_H]).T  # (n_ang, 4)
    weighted_H = hanford_terms * pattern_H[:, cp.newaxis, :] # (n_ang, n_times, 4)
    delay_L = time_delay_hanford_to_livingston(angle_grid)
    liv_terms = generate_oscillatory_terms(
            signal_lifetime, signal_frequency, time_array, delay_L
        )   
    pattern_L = cp.array([Fp_L, Fx_L, Fp_L, Fx_L]).T # (n_ang, 4)
    weighted_L = liv_terms * pattern_L[:, cp.newaxis, :]
    # einsum
    '''
        weighted_H.shape = (n_ang, n_times, 4)
        amplitude_grid.shape = (n_amplitudes, 4) 
        needed shape = (n_ang, n_amp, n_times)
        # computes all (weighted_H @ amps) at once instead of looping
    # '''
    # responses[:, :, :, 0] = cp.einsum('atm,pm->apt', weighted_H, amplitude_grid)
    # responses[:, :, :, 1] = cp.einsum('atm,pm->apt', weighted_L, amplitude_grid)
    weights_and_amps = [weighted_H, weighted_L, amplitude_grid]
    return weights_and_amps, angle_grid


#=========================================================================

def generate_noise_array(max_noise_amp,number_time_samples) :
    
    """
    Generates a 1D NumPy array of random non-Gaussian noise values. This function takes a maximum noise amplitude and a number of time samples 
    and returns random (non-Gaussian) noise between zero and the appropriate maximum for each time sample.

    Parameters:
        max_noise_amp (float): The maximum possible amplitude of the noise.
        number_time_samples (int): The total number of discrete time samples.

    Returns:
        np.ndarray: A 1D NumPy array of shape (number_time_samples,) containing 
                    uniformly distributed random noise values in the range [0, max_noise_amp).
    """
    
    noise_array = (cp.random.rand(number_time_samples)*max_noise_amp).astype(DTYPE)
    return noise_array

#=========================================================================

def generate_real_detector_responses(
        signal_frequency,
        signal_lifetime,
        detector_sampling_rate,
        gw_max_amps, number_amplitude_combinations,
        number_angular_samples, 
        max_noise_amp
        ):
    """
    Efficiently generates a simulated detector response for one gravitational wave signal,
    duplicating it across model parameter combinations to match later processing.

    Returns:
        real_detector_response_array: shape (n_angles, n_amps, n_times, n_detectors)
        real_angles_array: shape (n_angles, 3)
    """
    # 1. Setup
    time_array = generate_network_time_array(signal_lifetime, detector_sampling_rate, MAX_HANFORD_LIVINGSTON_DELAY)
    number_time_samples = time_array.size
    print("time_array:", time_array.dtype)
    # 2. Sample 1 true amplitude and angle
    real_amplitudes = generate_model_amplitudes_array(1, gw_max_amps)[0]
    real_angles = generate_model_angles_array(1)[0] # returns (3,)
    real_angle_batch = real_angles[cp.newaxis, :] # makes it (1,3)

    # 3. Detector beam pattern responses
    fplus_hanford, fcross_hanford = beam_pattern_response_functions(hanford_detector_angles, real_angle_batch)
    fplus_livingston, fcross_livingston = beam_pattern_response_functions(livingston_detector_angles, real_angle_batch)

    # 4. Time delay and oscillatory terms
    time_delay = time_delay_hanford_to_livingston(real_angles)
    osc_hanford = generate_oscillatory_terms(signal_lifetime, signal_frequency, time_array, 0) # (1, n_times, 4)
    osc_livingston = generate_oscillatory_terms(signal_lifetime, signal_frequency, time_array, time_delay)
    osc_hanford = cp.squeeze(osc_hanford, axis=0)
    osc_livingston = cp.squeeze(osc_livingston, axis=0)

    # 5. Noise arrays
    noise_h = generate_noise_array(max_noise_amp, number_time_samples)
    noise_l = generate_noise_array(max_noise_amp, number_time_samples)
    # 6. Combine signal + noise for both detectors using broadcasting
    weights_h = cp.array([fplus_hanford[0], fcross_hanford[0], fplus_hanford[0], fcross_hanford[0]]) # (4,)
    weights_l = cp.array([fplus_livingston[0], fcross_livingston[0], fplus_livingston[0], fcross_livingston[0]]) 

    signal_h = cp.dot(osc_hanford * weights_h, real_amplitudes) # (n_times,)
    signal_l = cp.dot(osc_livingston * weights_l, real_amplitudes)
    print("osc_h:", osc_hanford.dtype, "weights_h:", weights_h.dtype, "signal_h:", signal_h.dtype)

    # Shape: (time_samples, detectors)
    small_response = cp.stack([signal_h + noise_h, signal_l + noise_l], axis=1)

    # 7. Efficient duplication using broadcasting
    real_detector_response_array = cp.broadcast_to(
        small_response[cp.newaxis, cp.newaxis, :, :],
        (number_angular_samples, number_amplitude_combinations, number_time_samples, NUMBER_DETECTORS)
    )

    real_angles_array = cp.broadcast_to(
        cp.array(real_angles)[None, :],
        (number_angular_samples, NUMBER_SOURCE_ANGLES)
    )

    return real_detector_response_array, real_angles_array 

#=========================================================================
def get_best_fit_angles_deltas(real_detector_responses, real_angles_array,
                                model_detector_responses, model_angles_array):
    """
    Compares real (simulated) source angles with:
    1. The closest angles in the model (oracle best).
    2. The angles from the best-matching model detector response (single best fit).
    3. The angles from a weighted average over all model responses (weighted fit).

    Parameters:
        real_detector_responses (np.ndarray): Shape (n_angles, n_amps, n_times, n_detectors).
        real_angles_array (np.ndarray): Shape (n_angles, 3). Simulated source angles.
        model_detector_responses (np.ndarray): Shape (n_angles, n_amps, n_times, n_detectors).
        model_angles_array (np.ndarray): Shape (n_angles, 3). Modeled source angles.

    Returns:
        list: [
            sum_real_minimum_angle_deltas,
            sum_real_minimum_response_angle_deltas,
            sum_real_maximum_weighted_response_angle_deltas,
            single_best_fit_time,
            weighted_best_fit_time
        ]
    """
    weighted_H = model_detector_responses[0]
    weighted_L = model_detector_responses[1]
    amplitude_grid = model_detector_responses[2]
    a0 = amplitude_grid[:, 0][cp.newaxis, :, cp.newaxis]
    a1 = amplitude_grid[:, 1][cp.newaxis, :, cp.newaxis]
    a2 = amplitude_grid[:, 2][cp.newaxis, :, cp.newaxis]
    a3 = amplitude_grid[:, 3][cp.newaxis, :, cp.newaxis]
    wh0 = weighted_H[:, :, 0][:, cp.newaxis, :]
    wh1 = weighted_H[:, :, 1][:, cp.newaxis, :]
    wh2 = weighted_H[:, :, 2][:, cp.newaxis, :]
    wh3 = weighted_H[:, :, 3][:, cp.newaxis, :]
    wl0 = weighted_L[:, :, 0][:, cp.newaxis, :]
    wl1 = weighted_L[:, :, 1][:, cp.newaxis, :]
    wl2 = weighted_L[:, :, 2][:, cp.newaxis, :]
    wl3 = weighted_L[:, :, 3][:, cp.newaxis, :]
    realH = real_detector_responses[...,0]
    realL = real_detector_responses[...,1]

    # 1. Oracle best: angle delta to closest model angle
    model_angles_array = cp.array(model_angles_array)
    angle_deltas = cp.abs(real_angles_array[0] - model_angles_array)
    summed_angle_deltas = cp.sum(angle_deltas, axis=1)
    min_angle_idx = cp.argmin(summed_angle_deltas)
    sum_real_minimum_angle_deltas = cp.sum(cp.abs(real_angles_array[0] - model_angles_array[min_angle_idx]))
    s = cp.cuda.Event(); e = cp.cuda.Event()
    s.record()
    # response_deltas = cp.abs(real_detector_responses - model_detector_responses)
    # summed_response_deltas = cp.sum(response_deltas, axis=(-1, -2))  # shape: (n_angles, n_amps)
    summed_response_deltas = abs_diff_sum( 
        wh0, wh1, wh2,   wh3,   wl0,   wl1,   wl2,   wl3,   a0,   a1,   a2,   a3,   realH,   realL,
        axis = -1
    )
    e.record(); e.synchronize()
    t = cp.cuda.get_elapsed_time(s,e)/1000
    print("time taken for diff+abs+sum",t)
    # 2. Single best fit: minimum total difference in detector responses
    def single_best_fit():
        min_response_idx = cp.unravel_index(cp.argmin(summed_response_deltas), summed_response_deltas.shape)
        best_fit_angles = model_angles_array[min_response_idx[0]]
        return cp.sum(cp.abs(real_angles_array[0] - best_fit_angles))
   
    sum_real_minimum_response_angle_deltas = single_best_fit()
   
    # 3. Weighted best fit
    def weighted_best_fit():
        fractional_deltas = summed_response_deltas / cp.min(summed_response_deltas)
        weights = cp.exp(1 - fractional_deltas**WEIGHTING_POWER)
        summed_weights = cp.sum(weights, axis=1)
        weighted_idx = cp.argmax(summed_weights)
        weighted_angles = model_angles_array[weighted_idx]
        return cp.sum(cp.abs(real_angles_array[0] - weighted_angles))

    sum_real_maximum_weighted_response_angle_deltas = weighted_best_fit()
    deltas = [
        sum_real_minimum_angle_deltas,
        sum_real_minimum_response_angle_deltas,
        sum_real_maximum_weighted_response_angle_deltas
        ]
    return [
        deltas, single_best_fit, weighted_best_fit
    ]

#========================================================================= START OF DRIVER FUNCTIONS =========================================================================

def run_northstar_pipeline(
    gw_frequency=100,
    gw_lifetime=0.03,
    detector_sampling_rate=LIGO_DETECTOR_SAMPLING_RATE,
    gw_max_amps=1,
    max_noise_amp=0.1,
    number_angular_samples=500,
    number_amplitude_combinations=500
):
    # cp.random.seed(0)
    start = cp.cuda.Event()
    end = cp.cuda.Event()
    # calculating runtime on a perfectly warmed up GPU i.e runtime minus the context initialization cost.
    start.record()
    # Generate synthetic model and noisy real detector responses
    s = cp.cuda.Event(); e = cp.cuda.Event()
    s.record()

    model_responses, model_angles = generate_model_detector_responses(
        gw_frequency,
        gw_lifetime,
        detector_sampling_rate,
        gw_max_amps,
        number_amplitude_combinations,
        number_angular_samples
    )
    e.record(); e.synchronize()
    t_model = cp.cuda.get_elapsed_time(s, e) / 1000
    s.record()

    real_responses, real_angles = generate_real_detector_responses(
        gw_frequency,
        gw_lifetime,
        detector_sampling_rate,
        gw_max_amps,
        number_amplitude_combinations,
        number_angular_samples,
        max_noise_amp
    )
    e.record(); e.synchronize()
    t_real = cp.cuda.get_elapsed_time(s, e) / 1000

    # Run the angle comparison algorithms
    s.record()
    # print("real:", real_responses.dtype, "model:", model_responses.dtype)
    deltas, single_func, weighted_func = get_best_fit_angles_deltas(
        real_responses,
        real_angles,
        model_responses,
        model_angles
    )
    e.record(); e.synchronize()
    t_bestfit = cp.cuda.get_elapsed_time(s, e) / 1000

    end.record()
    end.synchronize()  # ensure all GPU work is done before stopping the clock
    total_runtime = cp.cuda.get_elapsed_time(start,end)/1000
    bench_single = benchmark(single_func, (), n_repeat=50, n_warmup=10)
    bench_weighted = benchmark(weighted_func, (), n_repeat=50, n_warmup=10)
    t_single_fit = (bench_single.gpu_times.mean())
    t_weighted_fit = (bench_weighted.gpu_times.mean())
    best_fit_data = [
        deltas[0], deltas[1], deltas[2],
        t_single_fit,
        t_weighted_fit,
    ]
    # Create a human-readable timestamped filename
    timestamp = time.strftime("%d_%b_%Y_%H-%M-%S")
    filename = f"OPTIMIZED_northstar_output_{timestamp}.txt"

    # Format results
    results = [
        f"The best possible fit angle delta (in radians) was: {best_fit_data[0]:.6f}",
        f"The single best fit algorithm angle delta (in radians) was: {best_fit_data[1]:.6f}",
        f"The weighted best fit algorithm angle delta (in radians) was: {best_fit_data[2]:.6f}",
        f"The full process run time (in seconds) was: {total_runtime:.6f}",
        f"The single best fit algorithm run time (in seconds) was: {best_fit_data[3]:.6f}",
        f"The weighted best fit algorithm run time (in seconds) was: {best_fit_data[4]:.6f}"
        ]

    # Print to terminal
    print("\n[Optimized Northstar Run Summary]")
    for line in results:
        print(line)

    print("\n[Per-Stage GPU Time]")
    print(f"  Model generation : {t_model:.4f} s")
    print(f"  Real generation  : {t_real:.4f} s")
    print(f"  Best-fit compare : {t_bestfit:.4f} s")
    print(f"  {'-'*30}")
    print(f"  Total            : {t_model + t_real + t_bestfit:.4f} s")

    # Write to file

    # with open(filename, "w") as f:
    #     for line in results:
    #         f.write(line + "\n")

    # print(f"\n[OK] Output also written to: {filename}")
    print(f"ang x amp: {number_angular_samples} x {number_amplitude_combinations}")

if __name__=="__main__":
    run_northstar_pipeline()
    run_northstar_pipeline()
    run_northstar_pipeline()
    run_northstar_pipeline()

#LATEST