"""
Gravitational Wave Detector Response Simulation Optimized with Numba
====================================================================

Author: Abid Jeem

This module simulates the responses of a network of gravitational wave detectors
to an incoming gravitational wave signal. This version has been optimized with Numba.
It includes functions to:

  - Transform tensors between different coordinate systems.
  - Generate unit source vectors from angular parameters.
  - Compute change-of-basis matrices between the gravitational wave (GW) frame,
    detector frame, and the Earth-centered (EC) frame.
  - Construct the gravitational wave strain tensor in the Earth-centered frame.
  - Compute detector responses including beam pattern functions and time delays.
  - Generate model detector responses and simulated (real) detector responses with noise.
  - Compare the simulated (real) responses with a model to obtain best-fit source angles.

All angles are in radians and all dimensional quantities are in SI units.
"""

import math as m
import numpy as np
import matplotlib.pyplot as plt
from numba import njit

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

# Convert detector angles to numpy arrays for use in Numba functions.
hanford_detector_angles = np.array(hanford_detector_angles)
livingston_detector_angles = np.array(livingston_detector_angles)

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

@njit
def transform_2_0_tensor(matrix, change_basis_matrix):
    """
    Transforms a (2,0) tensor (with two contravariant indices) to a new basis.
    
    Instead of using np.einsum, we compute:
        T' = (A^T) dot matrix dot (A^T)^T,
    where A = inv(change_basis_matrix) and (A^T) denotes its transpose.
    """
    A = np.linalg.inv(change_basis_matrix)
    B = A.T
    return np.dot(B, np.dot(matrix, B.T))


@njit
def transform_1_1_tensor(matrix, change_basis_matrix):
    """
    Transforms a (1,1) tensor (mixed tensor) to a new basis.
    
    The transformation is:
        T' = change_basis_matrix dot matrix dot inv(change_basis_matrix)
    """
    A_inv = np.linalg.inv(change_basis_matrix)
    return np.dot(change_basis_matrix, np.dot(matrix, A_inv))


@njit
def transform_0_2_tensor(matrix, change_basis_matrix):
    """
    Transforms a (0,2) tensor (with two covariant indices) to a new basis.
    
    The transformation is computed as:
        T' = change_basis_matrix dot matrix dot (change_basis_matrix)^T
    """
    return np.dot(change_basis_matrix, np.dot(matrix, change_basis_matrix.T))


# ------------------------------------------------------------------------------
# Source and Detector Vector / Basis Functions
# ------------------------------------------------------------------------------

@njit
def source_vector_from_angles(angles):
    """
    Computes a unit vector from Earth's center toward the source/detector.
    Only the first two angles (declination/latitude and right ascension/longitude) are used.
    """
    first = angles[0]
    second = angles[1]
    return np.array([m.cos(first)*m.cos(second),
                     m.cos(first)*m.sin(second),
                     m.sin(first)])


@njit
def change_basis_gw_to_ec(source_angles):
    """
    Computes the change-of-basis matrix from the gravitational wave (GW) frame
    to the Earth-centered (EC) frame.
    """
    declination = source_angles[0]
    right_ascension = source_angles[1]
    polarization = source_angles[2]
    
    initial_source_vector = source_vector_from_angles(source_angles)
    initial_gw_z = -1.0 * initial_source_vector
    initial_gw_y = np.array([-m.sin(declination)*m.cos(right_ascension),
                             -m.sin(declination)*m.sin(right_ascension),
                             m.cos(declination)])
    initial_gw_x = np.cross(initial_gw_z, initial_gw_y)
    
    # Assemble the Earth-centered basis matrix (each column is a basis vector)
    ec_matrix = np.empty((3, 3))
    ec_matrix[:, 0] = initial_gw_x
    ec_matrix[:, 1] = initial_gw_y
    ec_matrix[:, 2] = initial_gw_z
    
    # Polarization rotation matrix
    polarization_rotation_matrix = np.array([[m.cos(polarization), -m.sin(polarization), 0.0],
                                             [m.sin(polarization),  m.cos(polarization), 0.0],
                                             [0.0, 0.0, 1.0]])
    temp = np.dot(polarization_rotation_matrix, ec_matrix)
    return np.linalg.inv(temp)


@njit
def gravitational_wave_ec_frame(source_angles, tt_amplitudes):
    """
    Computes the gravitational wave strain tensor in the Earth-centered frame.
    """
    hplus = tt_amplitudes[0]
    hcross = tt_amplitudes[1]
    gwtt = np.array([[hplus, hcross, 0.0],
                     [hcross, -hplus, 0.0],
                     [0.0, 0.0, 0.0]])
    transformation = change_basis_gw_to_ec(source_angles)
    return transform_0_2_tensor(gwtt, transformation)


@njit
def change_basis_detector_to_ec(detector_angles):
    """
    Computes the change-of-basis matrix from a detector's local frame to the Earth-centered frame.
    """
    latitude = detector_angles[0]
    longitude = detector_angles[1]
    orientation = detector_angles[2]
    
    initial_detector_z = source_vector_from_angles(detector_angles)
    initial_detector_x = np.array([-m.sin(longitude), m.cos(longitude), 0.0])
    initial_detector_y = np.cross(initial_detector_z, initial_detector_x)
    
    ec_matrix = np.empty((3, 3))
    ec_matrix[:, 0] = initial_detector_x
    ec_matrix[:, 1] = initial_detector_y
    ec_matrix[:, 2] = initial_detector_z
    
    orientation_rotation_matrix = np.array([[m.cos(orientation), -m.sin(orientation), 0.0],
                                             [m.sin(orientation),  m.cos(orientation), 0.0],
                                             [0.0, 0.0, 1.0]])
    temp = np.dot(orientation_rotation_matrix, ec_matrix)
    return np.linalg.inv(temp)


# ------------------------------------------------------------------------------
# Detector Response and Beam Pattern Functions
# ------------------------------------------------------------------------------

@njit
def detector_response(detector_angles, source_angles, tt_amplitudes):
    """
    Computes the scalar detector response (strain) to a gravitational wave.
    """
    detector_tensor = np.array([[0.5, 0.0, 0.0],
                                [0.0, -0.5, 0.0],
                                [0.0, 0.0, 0.0]])
    transform_detector_to_ec = change_basis_detector_to_ec(detector_angles)
    detector_tensor_ec = transform_2_0_tensor(detector_tensor, transform_detector_to_ec)
    gw_tensor_ec = gravitational_wave_ec_frame(source_angles, tt_amplitudes)
    # Instead of np.tensordot, compute the sum of element-wise product.
    return np.sum(gw_tensor_ec * detector_tensor_ec)


@njit
def beam_pattern_response_functions(detector_angles, source_angles):
    """
    Computes the beam pattern functions [F_+, F_×] of a gravitational wave detector.
    """
    detector_tensor = np.array([[0.5, 0.0, 0.0],
                                [0.0, -0.5, 0.0],
                                [0.0, 0.0, 0.0]])
    transform_detector_ec = change_basis_detector_to_ec(detector_angles)
    detector_tensor_ec = transform_2_0_tensor(detector_tensor, transform_detector_ec)
    transform_gw_ec = change_basis_gw_to_ec(source_angles)
    transform_ec_gw = np.linalg.inv(transform_gw_ec)
    detector_tensor_gw = transform_2_0_tensor(detector_tensor_ec, transform_ec_gw)
    fplus = detector_tensor_gw[0, 0] - detector_tensor_gw[1, 1]
    fcross = detector_tensor_gw[0, 1] + detector_tensor_gw[1, 0]
    return np.array([fplus, fcross])


# ------------------------------------------------------------------------------
# Time Delay and Oscillatory Terms Generation
# ------------------------------------------------------------------------------

@njit
def time_delay_hanford_to_livingston(source_angles):
    """
    Computes the time delay (in seconds) between the Hanford and Livingston detectors.
    """
    hanford_z = source_vector_from_angles(hanford_detector_angles)
    livingston_z = source_vector_from_angles(livingston_detector_angles)
    pos_vector = earth_radius * (livingston_z - hanford_z)
    gw_source = source_vector_from_angles(source_angles)
    gw_z = -1.0 * gw_source
    return 1.0 / speed_light * np.dot(gw_z, pos_vector)


@njit
def generate_network_time_array(signal_lifetime, detector_sampling_rate, maximum_time_delay):
    """
    Generates an array of time samples for the gravitational wave signal.
    """
    time_sample_width = round((signal_lifetime + maximum_time_delay) * detector_sampling_rate)
    all_times = (1.0 / detector_sampling_rate) * np.arange(-time_sample_width, time_sample_width)
    return all_times


@njit
def generate_oscillatory_terms(signal_lifetime, signal_frequency, time_array, time_delay):
    """
    Generates sine and cosine terms modulated by a Gaussian envelope to model a sine-Gaussian signal.
    The output order per time sample is: [A_+, B_+, A_×, B_×].
    """
    number_time_samples = time_array.size
    signal_q_value = m.sqrt(m.log(2)) / signal_lifetime
    oscillatory_terms = np.empty((number_time_samples, number_gw_modes))
    
    for ts in range(number_time_samples):
        t = time_array[ts]
        gaussian = m.exp(-1.0 * signal_q_value**2 * (t - time_delay)**2)
        cos_term = m.cos(2.0 * np.pi * signal_frequency * (t - time_delay))
        sin_term = m.sin(2.0 * np.pi * signal_frequency * (t - time_delay))
        oscillatory_terms[ts, 0] = gaussian * cos_term  # A_+
        oscillatory_terms[ts, 1] = gaussian * sin_term  # B_+
        oscillatory_terms[ts, 2] = gaussian * cos_term  # A_× (assumed same)
        oscillatory_terms[ts, 3] = gaussian * sin_term  # B_× (assumed same)
    return oscillatory_terms


# ------------------------------------------------------------------------------
# Model Generation Functions (Angles, Amplitudes, and Detector Responses)
# ------------------------------------------------------------------------------

@njit
def generate_model_angles_array(number_angular_samples):
    """
    Generates an array of randomized source angle sets.
    """
    model_angles_array = np.empty((number_angular_samples, number_source_angles))
    for a in range(number_angular_samples):
        for s in range(number_source_angles):
            if s == 0:
                model_angles_array[a, s] = (np.random.random() - 0.5) * np.pi
            else:
                model_angles_array[a, s] = np.random.random() * 2.0 * np.pi
    return model_angles_array


@njit
def generate_model_amplitudes_array(number_amplitude_combinations, gw_max_amps):
    """
    Generates an array of randomized amplitude combinations for GW modes.
    """
    model_amplitudes_array = np.empty((number_amplitude_combinations, number_gw_modes))
    for a in range(number_amplitude_combinations):
        for i in range(number_gw_modes):
            model_amplitudes_array[a, i] = np.random.random() * gw_max_amps
    return model_amplitudes_array


@njit
def generate_model_detector_responses(signal_frequency, signal_lifetime,
                                      detector_sampling_rate, gw_max_amps,
                                      number_amplitude_combinations, number_angular_samples):
    """
    Generates a model of detector responses for a network of GW detectors.
    """
    time_array = generate_network_time_array(signal_lifetime, detector_sampling_rate,
                                             maximum_hanford_livingston_time_delay)
    number_time_samples = time_array.size
    hanford_terms = generate_oscillatory_terms(signal_lifetime, signal_frequency, time_array, 0.0)
    model_amplitudes_array = generate_model_amplitudes_array(number_amplitude_combinations, gw_max_amps)
    model_angles_array = generate_model_angles_array(number_angular_samples)
    model_detector_response_array = np.empty((number_angular_samples, number_amplitude_combinations,
                                              number_time_samples, number_detectors))
    
    for a in range(number_angular_samples):
        angles = model_angles_array[a]
        bp_hanford = beam_pattern_response_functions(hanford_detector_angles, angles)
        bp_livingston = beam_pattern_response_functions(livingston_detector_angles, angles)
        fplus_hanford = bp_hanford[0]
        fcross_hanford = bp_hanford[1]
        fplus_livingston = bp_livingston[0]
        fcross_livingston = bp_livingston[1]
        t_delay = time_delay_hanford_to_livingston(angles)
        livingston_terms = generate_oscillatory_terms(signal_lifetime, signal_frequency, time_array, t_delay)
        for amp in range(number_amplitude_combinations):
            amplitudes = model_amplitudes_array[amp]
            for t in range(number_time_samples):
                model_detector_response_array[a, amp, t, 0] = np.dot(amplitudes, hanford_terms[t] * np.array([fplus_hanford, fplus_hanford, fcross_hanford, fcross_hanford]))
                model_detector_response_array[a, amp, t, 1] = np.dot(amplitudes, livingston_terms[t] * np.array([fplus_livingston, fplus_livingston, fcross_livingston, fcross_livingston]))
    return model_detector_response_array, model_angles_array


@njit
def generate_noise_array(max_noise_amp, number_time_samples):
    """
    Generates a noise array with uniform random noise between 0 and max_noise_amp.
    """
    noise_array = np.empty(number_time_samples)
    for i in range(number_time_samples):
        noise_array[i] = np.random.random() * max_noise_amp
    return noise_array


@njit
def generate_real_detector_responses(signal_frequency, signal_lifetime, detector_sampling_rate,
                                     gw_max_amps, number_amplitude_combinations,
                                     number_angular_samples, max_noise_amp):
    """
    Generates simulated ("real") detector responses, including noise.
    """
    time_array = generate_network_time_array(signal_lifetime, detector_sampling_rate,
                                             maximum_hanford_livingston_time_delay)
    number_time_samples = time_array.size
    real_amplitudes = generate_model_amplitudes_array(1, gw_max_amps)[0]
    real_angles = generate_model_angles_array(1)[0]
    bp_hanford = beam_pattern_response_functions(hanford_detector_angles, real_angles)
    bp_livingston = beam_pattern_response_functions(livingston_detector_angles, real_angles)
    fplus_hanford = bp_hanford[0]
    fcross_hanford = bp_hanford[1]
    fplus_livingston = bp_livingston[0]
    fcross_livingston = bp_livingston[1]
    t_delay = time_delay_hanford_to_livingston(real_angles)
    hanford_terms = generate_oscillatory_terms(signal_lifetime, signal_frequency, time_array, 0.0)
    livingston_terms = generate_oscillatory_terms(signal_lifetime, signal_frequency, time_array, t_delay)
    small_response = np.empty((number_time_samples, number_detectors))
    hanford_noise = generate_noise_array(max_noise_amp, number_time_samples)
    livingston_noise = generate_noise_array(max_noise_amp, number_time_samples)
    
    for t in range(number_time_samples):
        small_response[t, 0] = np.dot(real_amplitudes, hanford_terms[t] * np.array([fplus_hanford, fplus_hanford, fcross_hanford, fcross_hanford])) + hanford_noise[t]
        small_response[t, 1] = np.dot(real_amplitudes, livingston_terms[t] * np.array([fplus_livingston, fplus_livingston, fcross_livingston, fcross_livingston])) + livingston_noise[t]
    
    real_angles_array = np.empty((number_angular_samples, number_source_angles))
    real_detector_response_array = np.empty((number_angular_samples, number_amplitude_combinations,
                                               number_time_samples, number_detectors))
    for a in range(number_angular_samples):
        for s in range(number_source_angles):
            real_angles_array[a, s] = real_angles[s]
        for amp in range(number_amplitude_combinations):
            for t in range(number_time_samples):
                for d in range(number_detectors):
                    real_detector_response_array[a, amp, t, d] = small_response[t, d]
    return real_detector_response_array, real_angles_array


@njit
def get_best_fit_angles_deltas(real_detector_responses, real_angles_array,
                               model_detector_responses, model_angles_array):
    """
    Compares the "real" detector responses and source angles with the model to compute three measures:
      1. Sum of absolute differences between the real angles and the model angles closest to the real angles.
      2. Sum of absolute differences for the model angles corresponding to the minimum detector response difference.
      3. Sum of absolute differences using an exponential weighting of the response differences.
    """
    diff_angles = np.abs(real_angles_array - model_angles_array)
    summed_diff = np.sum(diff_angles, axis=-1)
    min_sum = np.min(summed_diff)
    pos_min = np.where(summed_diff == min_sum)[0]
    angles_min = model_angles_array[pos_min[0]]
    real_diff_min = np.abs(real_angles_array[0] - angles_min)
    sum_real_diff_min = np.sum(real_diff_min)
    
    resp_diff = np.abs(real_detector_responses - model_detector_responses)
    summed_resp_diff = np.sum(resp_diff, axis=(-1, -2))
    min_resp_sum = np.min(summed_resp_diff)
    pos_min_resp = np.where(summed_resp_diff == min_resp_sum)[0]
    angles_min_resp = model_angles_array[pos_min_resp[0]]
    real_diff_min_resp = np.abs(real_angles_array[0] - angles_min_resp)
    sum_real_diff_min_resp = np.sum(real_diff_min_resp)
    
    offset = np.ones(summed_resp_diff.shape)
    fractional = 1.0 / min_resp_sum * summed_resp_diff
    weighted = np.exp(offset - fractional**weighting_power)
    summed_weighted = np.sum(weighted, axis=-1)
    max_weighted = np.max(summed_weighted)
    pos_max_weighted = np.where(summed_weighted == max_weighted)[0]
    angles_max_weighted = model_angles_array[pos_max_weighted[0]]
    real_diff_max_weighted = np.abs(real_angles_array[0] - angles_max_weighted)
    sum_real_diff_max_weighted = np.sum(real_diff_max_weighted)
    
    result = np.empty(3)
    result[0] = sum_real_diff_min
    result[1] = sum_real_diff_min_resp
    result[2] = sum_real_diff_max_weighted
    return result


# ------------------------------------------------------------------------------
# Example Run (Test the functions)
# ------------------------------------------------------------------------------

if __name__ == '__main__':
    # Set simulation parameters.
    gw_frequency = 100                   # Frequency (Hz)
    gw_lifetime = 0.03                   # Signal lifetime (seconds)
    detector_sampling_rate = 10000       # Sampling rate (Hz)
    gw_max_amps = 1                      # Maximum amplitude
    max_noise_amp = 0                    # No noise in this example
    number_angular_samples = 5           # Number of source angle sets
    number_amplitude_combinations = 10   # Number of amplitude combinations

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

    # Compute best-fit angle differences.
    best_fit_data = get_best_fit_angles_deltas(
        real_detector_responses, real_angles_array,
        model_detector_responses, model_angles_array
    )
    print(best_fit_data)
