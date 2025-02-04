"""
Gravitational Wave Detector Response Simulation
=================================================

This module simulates the responses of a network of gravitational wave detectors
to an incoming gravitational wave signal. It includes functions to:

  - Transform tensors between different coordinate systems.
  - Generate unit source vectors from angular parameters.
  - Compute change-of-basis matrices between the gravitational wave (GW) frame,
    detector frame, and the Earth-centered (EC) frame.
  - Construct the gravitational wave strain tensor in the Earth-centered frame.
  - Compute detector responses including beam pattern functions and time delays.
  - Generate model detector responses and simulated (real) detector responses with noise.
  - Compare the simulated (real) responses with a model to obtain best-fit source angles.

All angles are in radians and all dimensionful quantities are in SI units.
"""

import math as m
import numpy as np
import matplotlib.pyplot as plt
from numpy import random

# ------------------------------------------------------------------------------
# Global Constants and Detector Parameters
# ------------------------------------------------------------------------------
number_detectors = 2
number_gw_polarizations = 2
number_gw_modes = 4          # e.g., [A_+, B_+, A_x, B_x]
number_source_angles = 3     # e.g., [declination/latitude, right ascension/longitude, polarization/orientation]

# Detector locations and orientations (angles in radians)
hanford_detector_angles = [
    (46 + 27/60 + 18.528/3600) * np.pi/180,         # Latitude or declination
    (240 + 35/60 + 32.4343/3600) * np.pi/180,         # Longitude or right ascension
    np.pi/2 + 125.9994 * np.pi/180                     # Orientation (with an offset)
]
livingston_detector_angles = [
    (30 + 33/60 + 46.4196/3600) * np.pi/180,
    (269 + 13/60 + 32.7346/3600) * np.pi/180,
    np.pi/2 + 197.7165 * np.pi/180
]

ligo_detector_sampling_rate = 16384  # Hz
earth_radius = 6371000               # meters
speed_light = 299792458              # m/s

# Maximum possible time delay between Hanford and Livingston (seconds)
maximum_hanford_livingston_time_delay = 0.010002567302556083

# Weighting power used in the best-fit analysis function
weighting_power = 2

# ------------------------------------------------------------------------------
# Tensor Transformation Functions
# ------------------------------------------------------------------------------

def transform_2_0_tensor(matrix, change_basis_matrix):
    """
    Transforms a (2,0) tensor (with two contravariant indices) to a new basis.
    
    The transformation is given by:
        T'^{ij} = Λ^i_k Λ^j_l T^{kl},
    where Λ is the inverse of the change_basis_matrix.

    Parameters:
        matrix (np.ndarray): The original (2,0) tensor.
        change_basis_matrix (np.ndarray): The matrix that changes the basis.

    Returns:
        np.ndarray: The transformed (2,0) tensor.
    """
    # Compute the contravariant transformation matrix as the inverse.
    contravariant_transformation_matrix = np.linalg.inv(change_basis_matrix)
    # First contraction: contract over the first index.
    partial_transformation = np.einsum('ki,kl->il', contravariant_transformation_matrix, matrix)
    # Second contraction: contract over the second index.
    return np.einsum('lj,il->ij', contravariant_transformation_matrix, partial_transformation)


def transform_1_1_tensor(matrix, change_basis_matrix):
    """
    Transforms a (1,1) tensor (mixed tensor) to a new basis.
    
    The transformation is given by:
        T'^i_j = Λ^i_k T^k_l (Λ^{-1})^l_j.

    Parameters:
        matrix (np.ndarray): The original (1,1) tensor.
        change_basis_matrix (np.ndarray): The matrix that changes the basis.

    Returns:
        np.ndarray: The transformed (1,1) tensor.
    """
    contravariant_transformation_matrix = np.linalg.inv(change_basis_matrix)
    partial_transformation = np.einsum('ik,kl->il', change_basis_matrix, matrix)
    return np.einsum('lj,il->ij', contravariant_transformation_matrix, partial_transformation)


def transform_0_2_tensor(matrix, change_basis_matrix):
    """
    Transforms a (0,2) tensor (with two covariant indices) to a new basis.
    
    The transformation is given by:
        T'_{ij} = (Λ^{-1})^k_i (Λ^{-1})^l_j T_{kl}.
    In this implementation the change_basis_matrix is applied directly.

    Parameters:
        matrix (np.ndarray): The original (0,2) tensor.
        change_basis_matrix (np.ndarray): The matrix that changes the basis.

    Returns:
        np.ndarray: The transformed (0,2) tensor.
    """
    partial_transformation = np.einsum('ik,kl->il', change_basis_matrix, matrix)
    return np.einsum('jl,il->ij', change_basis_matrix, partial_transformation)

# ------------------------------------------------------------------------------
# Source and Detector Vector / Basis Functions
# ------------------------------------------------------------------------------

def source_vector_from_angles(angles):
    """
    Computes a unit vector from the center of the Earth toward the source or detector.
    
    The input list should contain three angles. For both gravitational wave sources and detectors,
    only the first two angles are used:
      - first angle: declination (or latitude)
      - second angle: right ascension (or longitude)
      - third angle: polarization (or orientation) [not used for computing the vector]

    The unit vector is defined by:
        [cos(first)*cos(second), cos(first)*sin(second), sin(first)]

    Parameters:
        angles (list or array-like): [angle1, angle2, angle3]

    Returns:
        np.ndarray: A 1D array (vector) of length 3.
    """
    [first, second, third] = angles
    initial_source_vector = np.array([
        m.cos(first) * m.cos(second),
        m.cos(first) * m.sin(second),
        m.sin(first)
    ])
    return initial_source_vector


def change_basis_gw_to_ec(source_angles):
    """
    Computes the change-of-basis (covariant transformation) matrix from the gravitational wave (GW)
    frame to the Earth-centered (EC) frame.
    
    The input source_angles is a list [declination, right ascension, polarization]. In the GW frame,
    the z-axis is typically chosen along the direction of wave propagation. In the Earth-centered frame,
    we define the following:
      - The GW z-axis in EC coordinates is -1 times the source vector.
      - The y-axis is constructed to be orthogonal to the source vector.
      - The x-axis is given by the cross product of z and y.
      
    Finally, a rotation by the polarization angle is applied.

    Parameters:
        source_angles (list or array-like): [declination, right ascension, polarization]

    Returns:
        np.ndarray: The change-of-basis matrix (EC to GW or vice versa, depending on inversion).
    """
    [declination, right_ascension, polarization] = source_angles

    # Compute the unit vector pointing from Earth's center to the source.
    initial_source_vector = source_vector_from_angles(source_angles)
    
    # Define the GW frame z-axis in EC coordinates (points opposite to the source vector).
    initial_gw_z_vector_earth_centered = -1 * initial_source_vector
    
    # Define a temporary y-axis orthogonal to the source vector.
    initial_gw_y_vector_earth_centered = np.array([
        -1 * m.sin(declination) * m.cos(right_ascension),
        -1 * m.sin(declination) * m.sin(right_ascension),
        m.cos(declination)
    ])
    
    # Define the x-axis as the cross product (ensuring an orthogonal basis).
    initial_gw_x_vector_earth_centered = np.cross(initial_gw_z_vector_earth_centered, initial_gw_y_vector_earth_centered)
    
    # Combine the basis vectors into a matrix; each column represents one basis vector.
    transpose_gw_vecs_ec = np.array([
        initial_gw_x_vector_earth_centered,
        initial_gw_y_vector_earth_centered,
        initial_gw_z_vector_earth_centered
    ])
    # Transpose to get the correct orientation.
    initial_gw_vecs_ec = np.transpose(transpose_gw_vecs_ec)
    
    # Create a rotation matrix that accounts for the polarization angle.
    polarization_rotation_matrix = np.array([
        [m.cos(polarization), -m.sin(polarization), 0],
        [m.sin(polarization),  m.cos(polarization), 0],
        [0,                   0,                  1]
    ])
    
    # Apply the polarization rotation.
    contravariant_transformation_matrix = np.matmul(polarization_rotation_matrix, initial_gw_vecs_ec)
    
    # The inverse of this matrix transforms components from the GW frame to the EC frame.
    change_basis_matrix = np.linalg.inv(contravariant_transformation_matrix)
    return change_basis_matrix


def gravitational_wave_ec_frame(source_angles, tt_amplitudes):
    """
    Computes the gravitational wave strain tensor in the Earth-centered (EC) frame.
    
    The gravitational wave tensor in the transverse-traceless (TT) gauge is defined as:
        h_tt = [ [h_plus, h_cross, 0],
                 [h_cross, -h_plus, 0],
                 [0, 0, 0] ]
    This tensor is then transformed into the Earth-centered frame using the change-of-basis matrix
    computed from the source angles.

    Parameters:
        source_angles (list or array-like): [declination, right ascension, polarization]
        tt_amplitudes (list or array-like): [h_plus, h_cross]

    Returns:
        np.ndarray: The gravitational wave strain tensor in the Earth-centered frame.
    """
    [hplus, hcross] = tt_amplitudes
    # Construct the TT gauge gravitational wave tensor.
    gwtt = np.array([
        [hplus, hcross, 0],
        [hcross, -hplus, 0],
        [0, 0, 0]
    ])
    # Get the transformation matrix from the GW frame to EC frame.
    transformation = change_basis_gw_to_ec(source_angles)
    # Transform the (0,2) tensor to the EC frame.
    return transform_0_2_tensor(gwtt, transformation)


def change_basis_detector_to_ec(detector_angles):
    """
    Computes the change-of-basis (covariant transformation) matrix from a detector's local frame
    to the Earth-centered (EC) frame.
    
    The detector_angles is a list [latitude, longitude, orientation]. The detector's local z-axis is
    determined by the unit vector from Earth's center to the detector, computed via source_vector_from_angles.
    The x-axis is chosen to be perpendicular to the local vertical (here, using the longitude) and
    the y-axis is obtained via the cross product.

    Parameters:
        detector_angles (list or array-like): [latitude, longitude, orientation]

    Returns:
        np.ndarray: The change-of-basis matrix from the detector frame to the Earth-centered frame.
    """
    [latitude, longitude, orientation] = detector_angles

    # Get the detector's unit vector from Earth's center.
    initial_detector_z_vector_earth_centered = source_vector_from_angles(detector_angles)
    
    # Define the local x-axis in EC coordinates (perpendicular to the meridian).
    initial_detector_x_vector_earth_centered = np.array([
        -m.sin(longitude),
         m.cos(longitude),
         0
    ])
    
    # Define the local y-axis as the cross product of z and x.
    initial_detector_y_vector_earth_centered = np.cross(initial_detector_z_vector_earth_centered, initial_detector_x_vector_earth_centered)
    
    # Combine the basis vectors into a matrix.
    transpose_detector_vecs_ec = np.array([
        initial_detector_x_vector_earth_centered,
        initial_detector_y_vector_earth_centered,
        initial_detector_z_vector_earth_centered
    ])
    initial_detector_vecs_ec = np.transpose(transpose_detector_vecs_ec)
    
    # Create a rotation matrix to account for the detector's orientation.
    orientation_rotation_matrix = np.array([
        [m.cos(orientation), -m.sin(orientation), 0],
        [m.sin(orientation),  m.cos(orientation), 0],
        [0,                  0,                  1]
    ])
    
    # Apply the orientation rotation.
    contravariant_transformation_matrix = np.matmul(orientation_rotation_matrix, initial_detector_vecs_ec)
    # Invert the matrix to get the change-of-basis (covariant transformation).
    change_basis_matrix = np.linalg.inv(contravariant_transformation_matrix)
    return change_basis_matrix

# ------------------------------------------------------------------------------
# Detector Response and Beam Pattern Functions
# ------------------------------------------------------------------------------

def detector_response(detector_angles, source_angles, tt_amplitudes):
    """
    Computes the scalar detector response (i.e. strain) to a gravitational wave.
    
    The detector is modeled by its detector response tensor in its own frame,
    given by:
          D = [ [ 1/2,  0,   0],
                [ 0,  -1/2,  0],
                [ 0,    0,   0] ]
    This tensor is transformed to the Earth-centered frame and then contracted
    with the gravitational wave strain tensor (also in the Earth-centered frame).

    Parameters:
        detector_angles (list or array-like): [latitude, longitude, orientation] for the detector.
        source_angles (list or array-like): [declination, right ascension, polarization] for the source.
        tt_amplitudes (list or array-like): [h_plus, h_cross] strain amplitudes in the TT gauge.

    Returns:
        float: The detector response (a scalar).
    """
    # Detector response tensor in the detector's own frame.
    detector_response_tensor_detector_frame = np.array([
        [1/2, 0, 0],
        [0, -1/2, 0],
        [0, 0, 0]
    ])
    # Transform the detector response tensor to the Earth-centered frame.
    transform_detector_to_ec = change_basis_detector_to_ec(detector_angles)
    detector_response_tensor_earth_centered = transform_2_0_tensor(detector_response_tensor_detector_frame, transform_detector_to_ec)
    # Compute the gravitational wave tensor in the Earth-centered frame.
    gw_earth_centered = gravitational_wave_ec_frame(source_angles, tt_amplitudes)
    # Contract the two tensors to obtain a scalar response.
    detector_response_val = np.tensordot(gw_earth_centered, detector_response_tensor_earth_centered)
    return detector_response_val


def beam_pattern_response_functions(detector_angles, source_angles):
    """
    Computes the beam pattern response functions F_+ and F_× of a gravitational wave detector.
    
    These functions describe the detector's sensitivity to the two gravitational wave polarizations.
    The procedure is as follows:
      1. Start with the detector response tensor in the detector frame.
      2. Transform it to the Earth-centered frame.
      3. Then, change the basis from the Earth-centered frame to the gravitational wave frame.
      4. In the gravitational wave frame, the plus and cross responses are given by:
           F_+ = D'_11 - D'_22,
           F_× = D'_12 + D'_21,
         where D' is the detector response tensor in the GW frame.

    Parameters:
        detector_angles (list or array-like): [latitude, longitude, orientation] for the detector.
        source_angles (list or array-like): [declination, right ascension, polarization] for the source.

    Returns:
        list: [F_+, F_×] the beam pattern response functions.
    """
    # Get the detector response tensor in the detector frame.
    detector_response_tensor_detector_frame = np.array([
        [1/2, 0, 0],
        [0, -1/2, 0],
        [0, 0, 0]
    ])
    # Transform from detector frame to Earth-centered frame.
    transform_detector_ec = change_basis_detector_to_ec(detector_angles)
    detector_response_tensor_earth_centered = transform_2_0_tensor(detector_response_tensor_detector_frame, transform_detector_ec)
    # Compute the transformation from GW frame to Earth-centered frame.
    transform_gw_ec = change_basis_gw_to_ec(source_angles)
    # Invert to get transformation from EC frame to GW frame.
    transform_ec_gw = np.linalg.inv(transform_gw_ec)
    # Transform the detector response tensor from the Earth-centered frame to the GW frame.
    detector_response_tensor_gw_frame = transform_2_0_tensor(detector_response_tensor_earth_centered, transform_ec_gw)
    # Extract the beam pattern functions.
    fplus = detector_response_tensor_gw_frame[0, 0] - detector_response_tensor_gw_frame[1, 1]
    fcross = detector_response_tensor_gw_frame[0, 1] + detector_response_tensor_gw_frame[1, 0]
    return [fplus, fcross]

# ------------------------------------------------------------------------------
# Time Delay and Oscillatory Terms Generation
# ------------------------------------------------------------------------------

def time_delay_hanford_to_livingston(source_angles):
    """
    Computes the time delay (in seconds) between the arrival of a gravitational wave signal
    at the Hanford and Livingston detectors.
    
    The delay is determined by:
      1. Computing the unit vectors (in the Earth-centered frame) for both detectors.
      2. Forming the position vector between detectors (scaled by Earth's radius).
      3. Computing the projection of this vector along the gravitational wave propagation direction.
      4. Dividing by the speed of light to obtain the time delay.
    
    A negative delay indicates that the signal arrives at Livingston first.

    Parameters:
        source_angles (list or array-like): [declination, right ascension, polarization] for the source.

    Returns:
        float: The time delay between the detectors (seconds).
    """
    # Compute unit vectors for each detector.
    hanford_z_vector_earth_centered = source_vector_from_angles(hanford_detector_angles)
    livingston_z_vector_earth_centered = source_vector_from_angles(livingston_detector_angles)
    # Compute the displacement vector between detectors.
    position_vector_hanford_to_livingston = earth_radius * (livingston_z_vector_earth_centered - hanford_z_vector_earth_centered)
    # Get the gravitational wave source vector.
    gw_source_vector = source_vector_from_angles(source_angles)
    # In the Earth-centered frame, the wave propagates in the direction opposite to the source vector.
    gw_z_vector_earth_centered = -1 * gw_source_vector
    # Return the time delay as the projection divided by the speed of light.
    return 1 / speed_light * (np.dot(gw_z_vector_earth_centered, position_vector_hanford_to_livingston))


def generate_network_time_array(signal_lifetime, detector_sampling_rate, maximum_time_delay):
    """
    Generates an array of network sampling times for the gravitational wave signal.
    
    The time array spans from -T to +T where T = (signal_lifetime + maximum_time_delay)
    sampled at the detector's sampling rate.

    Parameters:
        signal_lifetime (float): The characteristic lifetime of the signal.
        detector_sampling_rate (int): The number of samples per second.
        maximum_time_delay (float): Maximum expected time delay between detectors.

    Returns:
        np.ndarray: 1D array of time samples (seconds).
    """
    # Calculate the number of time samples (rounded to the nearest integer).
    time_sample_width = round((signal_lifetime + maximum_time_delay) * detector_sampling_rate, 0)
    # Create a time array that goes from -time_sample_width to time_sample_width (in seconds).
    all_times = (1 / detector_sampling_rate) * np.arange(-time_sample_width, time_sample_width, 1)
    return all_times


def generate_oscillatory_terms(signal_lifetime, signal_frequency, time_array, time_delay):
    """
    Generates oscillatory (sine and cosine) terms modulated by a Gaussian envelope.
    
    These terms model a sine-Gaussian gravitational wave signal.
    For each time sample the following is computed:
      - A Gaussian term: exp[-(sqrt(ln2)/signal_lifetime)^2 * (t - time_delay)^2]
      - Cosine and sine terms at the given signal frequency.
    
    The output is a NumPy array with the following order per time sample:
      [A_+, B_+, A_×, B_×],
    where A and B represent the cosine and sine modulated components respectively.

    Parameters:
        signal_lifetime (float): The time required for the signal amplitude to drop to half its maximum.
        signal_frequency (float): The frequency of the gravitational wave signal.
        time_array (np.ndarray): Array of time samples.
        time_delay (float): Time delay for the signal at the detector relative to a reference.

    Returns:
        np.ndarray: 2D array of oscillatory terms with shape (number_time_samples, number_gw_modes).
    """
    number_time_samples = time_array.size
    # Compute a Q-value that characterizes the Gaussian envelope.
    signal_q_value = m.sqrt(m.log(2)) / signal_lifetime
    oscillatory_terms = np.empty((number_time_samples, number_gw_modes))
    
    # Loop over time samples to compute the modulated sine and cosine components.
    for this_time_sample in range(number_time_samples):
        this_time = time_array[this_time_sample]
        # Gaussian envelope centered at the time delay.
        this_gaussian_term = m.exp(-1 * signal_q_value**2 * (this_time - time_delay)**2)
        # Compute cosine and sine oscillations.
        this_cos_term = m.cos(2 * np.pi * signal_frequency * (this_time - time_delay))
        this_sin_term = m.sin(2 * np.pi * signal_frequency * (this_time - time_delay))
        # Multiply by the Gaussian envelope.
        this_cos_gauss_term = this_gaussian_term * this_cos_term
        this_sin_gauss_term = this_gaussian_term * this_sin_term
        # Construct the oscillatory terms for both polarizations.
        these_oscillations = np.array([
            this_cos_gauss_term,  # A_+ component
            this_sin_gauss_term,  # B_+ component
            this_cos_gauss_term,  # A_× component (here assumed same as A_+)
            this_sin_gauss_term   # B_× component (here assumed same as B_+)
        ])
        oscillatory_terms[this_time_sample] = these_oscillations
    return oscillatory_terms

# ------------------------------------------------------------------------------
# Model Generation Functions (Angles, Amplitudes, and Detector Responses)
# ------------------------------------------------------------------------------

def generate_model_angles_array(number_angular_samples):
    """
    Generates an array of randomized source angle sets.
    
    Each angle set contains three angles:
      - The first angle is sampled between -pi/2 and pi/2 (e.g., declination or latitude).
      - The second and third angles are sampled uniformly between 0 and 2*pi.
    
    Parameters:
        number_angular_samples (int): The number of different source angle sets to generate.

    Returns:
        np.ndarray: Array with shape (number_angular_samples, number_source_angles).
    """
    model_angles_array = np.empty((number_angular_samples, number_source_angles))
    for this_angle_set in range(number_angular_samples):
        for this_source_angle in range(number_source_angles):
            if this_source_angle == 0:
                # Sample between -pi/2 and pi/2.
                model_angles_array[this_angle_set, this_source_angle] = (np.random.rand(1) - 0.5) * np.pi
            else:
                # Sample uniformly between 0 and 2*pi.
                model_angles_array[this_angle_set, this_source_angle] = np.random.rand(1) * 2 * np.pi
    return model_angles_array


def generate_model_amplitudes_array(number_amplitude_combinations, gw_max_amps):
    """
    Generates an array of randomized amplitude combinations for gravitational wave modes.
    
    Each amplitude is sampled uniformly between 0 and gw_max_amps.
    
    Parameters:
        number_amplitude_combinations (int): Number of amplitude combinations to generate.
        gw_max_amps (float): Maximum allowed amplitude value.

    Returns:
        np.ndarray: Array with shape (number_amplitude_combinations, number_gw_modes).
    """
    model_amplitudes_array = np.empty((number_amplitude_combinations, number_gw_modes))
    for this_amplitude_combination in range(number_amplitude_combinations):
        new_amplitudes = np.random.rand(number_gw_modes) * gw_max_amps
        model_amplitudes_array[this_amplitude_combination] = new_amplitudes
    return model_amplitudes_array


def generate_model_detector_responses(signal_frequency, signal_lifetime,
                                      detector_sampling_rate, gw_max_amps,
                                      number_amplitude_combinations, number_angular_samples):
    """
    Generates a model of detector responses for a network of gravitational wave detectors.
    
    For each combination of source angles and amplitude combinations, the function:
      1. Generates a network time array.
      2. Computes oscillatory terms (sine-Gaussian modulation) for the Hanford detector.
      3. Generates a set of randomized source angle sets and amplitude combinations.
      4. Computes the beam pattern response functions (F_+ and F_×) for both detectors.
      5. Computes the time delay between Hanford and Livingston.
      6. Computes the detector response (via a dot product of amplitudes and modulated oscillatory terms)
         for both detectors at each time sample.

    Parameters:
        signal_frequency (float): Frequency of the gravitational wave signal.
        signal_lifetime (float): Lifetime of the gravitational wave signal.
        detector_sampling_rate (int): Sampling rate of the detectors.
        gw_max_amps (float): Maximum gravitational wave amplitude.
        number_amplitude_combinations (int): Number of amplitude combinations to model.
        number_angular_samples (int): Number of source angle sets to model.

    Returns:
        tuple: (model_detector_response_array, model_angles_array)
          - model_detector_response_array: Array of shape (n_angles, n_amplitudes, n_time_samples, n_detectors).
          - model_angles_array: Array of source angle sets.
    """
    # Generate the time array for the network.
    time_array = generate_network_time_array(signal_lifetime, detector_sampling_rate,
                                             maximum_hanford_livingston_time_delay)
    number_time_samples = time_array.size
    
    # Generate oscillatory terms for Hanford (time delay = 0).
    hanford_oscillatory_terms = generate_oscillatory_terms(signal_lifetime, signal_frequency, time_array, 0)
    
    # Generate randomized model amplitude and angle arrays.
    model_amplitudes_array = generate_model_amplitudes_array(number_amplitude_combinations, gw_max_amps)
    model_angles_array = generate_model_angles_array(number_angular_samples)
    
    # Prepare an empty array to hold the detector responses.
    model_detector_response_array = np.empty((number_angular_samples, number_amplitude_combinations,
                                                number_time_samples, number_detectors))
    
    # Loop over each angle set.
    for this_angle_set in range(number_angular_samples):
        these_angles = model_angles_array[this_angle_set]
        # Get beam pattern functions for each detector.
        [fplus_hanford, fcross_hanford] = beam_pattern_response_functions(hanford_detector_angles, these_angles)
        [fplus_livingston, fcross_livingston] = beam_pattern_response_functions(livingston_detector_angles, these_angles)
        # Compute the time delay between detectors.
        hanford_livingston_time_delay = time_delay_hanford_to_livingston(these_angles)
        # Generate oscillatory terms for Livingston with the appropriate time delay.
        livingston_oscillatory_terms = generate_oscillatory_terms(signal_lifetime, signal_frequency,
                                                                  time_array, hanford_livingston_time_delay)
        # Loop over amplitude combinations.
        for this_amplitude_combination in range(number_amplitude_combinations):
            these_amplitudes = model_amplitudes_array[this_amplitude_combination]
            # For each time sample, compute the detector response as a dot product.
            for this_sample_time in range(number_time_samples):
                # Hanford response: dot(product of amplitudes and oscillatory terms weighted by beam pattern functions).
                model_detector_response_array[this_angle_set, this_amplitude_combination, this_sample_time, 0] = \
                    np.dot(these_amplitudes, hanford_oscillatory_terms[this_sample_time] *
                           [fplus_hanford, fplus_hanford, fcross_hanford, fcross_hanford])
                # Livingston response:
                model_detector_response_array[this_angle_set, this_amplitude_combination, this_sample_time, 1] = \
                    np.dot(these_amplitudes, livingston_oscillatory_terms[this_sample_time] *
                           [fplus_livingston, fplus_livingston, fcross_livingston, fcross_livingston])
    return model_detector_response_array, model_angles_array

# ------------------------------------------------------------------------------
# Noise and Simulated "Real" Detector Responses
# ------------------------------------------------------------------------------

def generate_noise_array(max_noise_amp, number_time_samples):
    """
    Generates a noise array for the detector response.
    
    The noise is sampled uniformly between 0 and max_noise_amp for each time sample.

    Parameters:
        max_noise_amp (float): The maximum amplitude of the noise.
        number_time_samples (int): The number of time samples.

    Returns:
        np.ndarray: 1D array of noise values.
    """
    noise_array = np.random.rand(number_time_samples) * max_noise_amp
    return noise_array


def generate_real_detector_responses(signal_frequency, signal_lifetime, detector_sampling_rate,
                                     gw_max_amps, number_amplitude_combinations,
                                     number_angular_samples, max_noise_amp):
    """
    Generates simulated ("real") detector responses, including noise.
    
    A single set of "real" amplitudes and angles is generated and then used to compute
    the detector responses for both detectors (Hanford and Livingston) over the network time array.
    Noise is added to each detector response.
    
    The output arrays are structured to match those from generate_model_detector_responses.

    Parameters:
        signal_frequency (float): Frequency of the gravitational wave signal.
        signal_lifetime (float): Lifetime of the gravitational wave signal.
        detector_sampling_rate (int): Sampling rate of the detectors.
        gw_max_amps (float): Maximum gravitational wave amplitude.
        number_amplitude_combinations (int): Number of amplitude combinations in the model.
        number_angular_samples (int): Number of angle combinations in the model.
        max_noise_amp (float): Maximum amplitude of the added noise.

    Returns:
        tuple: (real_detector_response_array, real_angles_array)
          - real_detector_response_array: Simulated responses with shape (n_angles, n_amplitudes, n_time_samples, n_detectors).
          - real_angles_array: Array of simulated source angles (repeated for each angle set).
    """
    time_array = generate_network_time_array(signal_lifetime, detector_sampling_rate,
                                             maximum_hanford_livingston_time_delay)
    number_time_samples = time_array.size
    
    # Generate one set of "real" amplitudes and angles.
    real_amplitudes = generate_model_amplitudes_array(1, gw_max_amps)[0]
    real_angles = generate_model_angles_array(1)[0]
    
    # Get beam pattern functions for the real source at each detector.
    [fplus_hanford, fcross_hanford] = beam_pattern_response_functions(hanford_detector_angles, real_angles)
    [fplus_livingston, fcross_livingston] = beam_pattern_response_functions(livingston_detector_angles, real_angles)
    
    # Compute time delay for the real source.
    hanford_livingston_time_delay = time_delay_hanford_to_livingston(real_angles)
    
    # Generate oscillatory terms for each detector.
    hanford_oscillatory_terms = generate_oscillatory_terms(signal_lifetime, signal_frequency, time_array, 0)
    livingston_oscillatory_terms = generate_oscillatory_terms(signal_lifetime, signal_frequency, time_array,
                                                               hanford_livingston_time_delay)
    
    # Initialize the response array for a single set.
    small_detector_response_array = np.empty((number_time_samples, number_detectors))
    
    # Generate noise arrays for each detector.
    hanford_noise_array = generate_noise_array(max_noise_amp, number_time_samples)
    livingston_noise_array = generate_noise_array(max_noise_amp, number_time_samples)
    
    # Compute the "real" detector responses (including noise) at each time sample.
    for this_sample_time in range(number_time_samples):
        small_detector_response_array[this_sample_time, 0] = np.dot(real_amplitudes,
            hanford_oscillatory_terms[this_sample_time] * [fplus_hanford, fplus_hanford, fcross_hanford, fcross_hanford]
        ) + hanford_noise_array[this_sample_time]
        
        small_detector_response_array[this_sample_time, 1] = np.dot(real_amplitudes,
            livingston_oscillatory_terms[this_sample_time] * [fplus_livingston, fplus_livingston, fcross_livingston, fcross_livingston]
        ) + livingston_noise_array[this_sample_time]
    
    # Prepare arrays that match the dimensions of the model responses.
    real_angles_array = np.empty((number_angular_samples, number_source_angles))
    real_detector_response_array = np.empty((number_angular_samples, number_amplitude_combinations,
                                               number_time_samples, number_detectors))
    
    # Fill the arrays with the same "real" values (repeated for each model angle/amplitude combination).
    for this_angle_set in range(number_angular_samples):
        real_angles_array[this_angle_set] = real_angles
        for this_amplitude_combination in range(number_amplitude_combinations):
            real_detector_response_array[this_angle_set, this_amplitude_combination] = small_detector_response_array
            
    return real_detector_response_array, real_angles_array

# ------------------------------------------------------------------------------
# Best-Fit Angle Delta Calculation
# ------------------------------------------------------------------------------

def get_best_fit_angles_deltas(real_detector_responses, real_angles_array,
                               model_detector_responses, model_angles_array):
    """
    Compares the "real" (simulated) detector responses and source angles with those from the model
    to determine how close the model comes to reproducing the real source angles.
    
    Three measures are computed:
      1. The sum of the absolute differences between the "real" angles and the model angles that are
         closest to the "real" angles.
      2. The sum of the absolute differences between the "real" angles and the angles corresponding to
         the single best (minimum summed) model detector response.
      3. The sum of the absolute differences between the "real" angles and the angles corresponding to
         the weighted maximum response (using an exponential weighting based on the summed response differences).

    Parameters:
        real_detector_responses (np.ndarray): Simulated detector responses.
        real_angles_array (np.ndarray): Simulated source angles.
        model_detector_responses (np.ndarray): Model detector responses.
        model_angles_array (np.ndarray): Model source angles.

    Returns:
        list: A list containing three sums of absolute angle differences corresponding to the three methods.
    """
    # Compute the absolute differences between the real and model angles.
    real_model_angle_deltas = np.absolute(real_angles_array - model_angles_array)
    summed_real_model_angle_deltas = np.sum(real_model_angle_deltas, axis=-1)
    
    # Method 1: Find the model angle set that minimizes the summed angle delta.
    minimum_summed_angle_delta = np.min(summed_real_model_angle_deltas)
    position_minimum_angles_delta = np.where(summed_real_model_angle_deltas == minimum_summed_angle_delta)
    angles_minimum_angles_delta = model_angles_array[position_minimum_angles_delta[0]]
    real_minimum_angles_deltas = np.absolute(real_angles_array[0] - angles_minimum_angles_delta)
    sum_real_minimum_angle_deltas = np.sum(real_minimum_angles_deltas)
    
    # Method 2: Compute differences in the detector responses.
    real_model_response_deltas = np.absolute(real_detector_responses - model_detector_responses)
    # Sum over time samples and detectors.
    summed_real_model_response_deltas = np.sum(real_model_response_deltas, axis=(-1, -2))
    minimum_summed_response_delta = np.min(summed_real_model_response_deltas)
    position_minimum_response_delta = np.where(summed_real_model_response_deltas == minimum_summed_response_delta)
    angles_minimum_response_delta = model_angles_array[position_minimum_response_delta[0]]
    real_minimum_response_angle_deltas = np.absolute(real_angles_array[0] - angles_minimum_response_delta)
    sum_real_minimum_response_angle_deltas = np.sum(real_minimum_response_angle_deltas)
    
    # Method 3: Apply an exponential weighting to the summed response deltas.
    offset_matrix = np.ones(summed_real_model_response_deltas.shape)
    fractional_summed_real_model_response_deltas = 1 / minimum_summed_response_delta * summed_real_model_response_deltas
    weighted_summed_real_model_response_deltas = np.exp(offset_matrix - fractional_summed_real_model_response_deltas**weighting_power)
    summed_weighted_summed_real_model_response_deltas = np.sum(weighted_summed_real_model_response_deltas, axis=-1)
    maximum_summed_weighted_response_delta = np.max(summed_weighted_summed_real_model_response_deltas)
    position_maximum_summed_weighted_response_delta = np.where(summed_weighted_summed_real_model_response_deltas == maximum_summed_weighted_response_delta)
    angles_maximum_summed_weighted_response_delta = model_angles_array[position_maximum_summed_weighted_response_delta[0]]
    real_maximum_weighted_response_angle_deltas = np.absolute(real_angles_array[0] - angles_maximum_summed_weighted_response_delta)
    sum_real_maximum_weighted_response_angle_deltas = np.sum(real_maximum_weighted_response_angle_deltas)
    
    return [sum_real_minimum_angle_deltas,
            sum_real_minimum_response_angle_deltas,
            sum_real_maximum_weighted_response_angle_deltas]

# ------------------------------------------------------------------------------
# Example Run (Test the functions)
# ------------------------------------------------------------------------------

if __name__ == '__main__':
    # Set simulation parameters.
    gw_frequency = 100                   # Frequency of the gravitational wave (Hz)
    gw_lifetime = 0.03                   # Signal lifetime (seconds)
    detector_sampling_rate = 10000       # Sampling rate (Hz)
    gw_max_amps = 1                      # Maximum gravitational wave amplitude
    max_noise_amp = 0                    # No noise for this example
    number_angular_samples = 5           # Number of source angle sets in the model
    number_amplitude_combinations = 10   # Number of amplitude combinations in the model

    # Generate model detector responses.
    model_detector_responses, model_angles_array = generate_model_detector_responses(
        gw_frequency, gw_lifetime, detector_sampling_rate, gw_max_amps,
        number_amplitude_combinations, number_angular_samples
    )

    # Generate simulated ("real") detector responses.
    real_detector_responses, real_angles_array = generate_real_detector_responses(
        gw_frequency, gw_lifetime, detector_sampling_rate, gw_max_amps,
        number_amplitude_combinations, number_angular_samples, max_noise_amp
    )

    # Compute the best-fit angle differences between model and real responses.
    best_fit_data = get_best_fit_angles_deltas(
        real_detector_responses, real_angles_array,
        model_detector_responses, model_angles_array
    )
    print(best_fit_data)
