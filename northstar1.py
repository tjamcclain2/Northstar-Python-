import math as m
import numpy as np
import time

# Constants
NUMBER_DETECTORS = 2
NUMBER_GW_POLARIZATIONS = 2
NUMBER_GW_MODES = 4
NUMBER_SOURCE_ANGLES = 3
LIGO_DETECTOR_SAMPLING_RATE = 16384
EARTH_RADIUS = 6371000
SPEED_OF_LIGHT = 299792458
MAX_TIME_DELAY = 0.010002567302556083
WEIGHTING_POWER = 2

# Detector Angles
HANFORD_DETECTOR_ANGLES = [
    (46 + 27 / 60 + 18.528 / 3600) * np.pi / 180,
    (240 + 35 / 60 + 32.4343 / 3600) * np.pi / 180,
    np.pi / 2 + 125.9994 * np.pi / 180
]
LIVINGSTON_DETECTOR_ANGLES = [
    (30 + 33 / 60 + 46.4196 / 3600) * np.pi / 180,
    (269 + 13 / 60 + 32.7346 / 3600) * np.pi / 180,
    np.pi / 2 + 197.7165 * np.pi / 180
]

def transform_tensor(matrix, change_basis_matrix, tensor_type):
    """
    Transforms a tensor to a new basis.

    Args:
        matrix (np.ndarray): The tensor to be transformed.
        change_basis_matrix (np.ndarray): The change of basis matrix.
        tensor_type (str): Type of the tensor ('2_0', '1_1', or '0_2').

    Returns:
        np.ndarray: Transformed tensor.
    """
    contravariant_transformation = np.linalg.inv(change_basis_matrix)

    if tensor_type == '2_0':
        partial = np.einsum('ki,kl->il', contravariant_transformation, matrix)
        return np.einsum('lj,il->ij', contravariant_transformation, partial)
    elif tensor_type == '1_1':
        partial = np.einsum('ik,kl->il', change_basis_matrix, matrix)
        return np.einsum('lj,il->ij', contravariant_transformation, partial)
    elif tensor_type == '0_2':
        partial = np.einsum('ik,kl->il', change_basis_matrix, matrix)
        return np.einsum('jl,il->ij', change_basis_matrix, partial)
    else:
        raise ValueError("Invalid tensor type. Must be '2_0', '1_1', or '0_2'.")

def source_vector_from_angles(angles):
    """
    Computes the source vector from angular coordinates.

    Args:
        angles (list): List of angles [declination, right ascension, polarization].

    Returns:
        np.ndarray: Source vector in Cartesian coordinates.
    """
    declination, right_ascension, _ = angles
    return np.array([
        m.cos(declination) * m.cos(right_ascension),
        m.cos(declination) * m.sin(right_ascension),
        m.sin(declination)
    ])

def change_basis_gw_to_ec(source_angles):
    """
    Computes the change of basis matrix from the gravitational wave frame to the Earth-centered frame.

    Args:
        source_angles (list): List of source angles [declination, right ascension, polarization].

    Returns:
        np.ndarray: Change of basis matrix.
    """
    declination, right_ascension, polarization = source_angles

    # Define vectors in the Earth-centered frame
    source_vector = source_vector_from_angles(source_angles)
    z_vector = -1 * source_vector
    y_vector = np.array([
        -m.sin(declination) * m.cos(right_ascension),
        -m.sin(declination) * m.sin(right_ascension),
        m.cos(declination)
    ])
    x_vector = np.cross(z_vector, y_vector)

    # Combine vectors into a matrix
    ec_matrix = np.array([x_vector, y_vector, z_vector]).T

    # Apply polarization rotation
    rotation_matrix = np.array([
        [m.cos(polarization), -m.sin(polarization), 0],
        [m.sin(polarization), m.cos(polarization), 0],
        [0, 0, 1]
    ])

    return np.linalg.inv(np.matmul(rotation_matrix, ec_matrix))

def gravitational_wave_ec_frame(source_angles, tt_amplitudes):
    """
    Computes the gravitational wave tensor in the Earth-centered frame.

    Args:
        source_angles (list): List of source angles [declination, right ascension, polarization].
        tt_amplitudes (list): Tensor amplitudes [hplus, hcross].

    Returns:
        np.ndarray: Gravitational wave tensor in the Earth-centered frame.
    """
    hplus, hcross = tt_amplitudes
    gw_tensor = np.array([
        [hplus, hcross, 0],
        [hcross, -hplus, 0],
        [0, 0, 0]
    ])
    transformation = change_basis_gw_to_ec(source_angles)
    return transform_tensor(gw_tensor, transformation, '0_2')

def change_basis_detector_to_ec(detector_angles):
    """
    Computes the change of basis matrix from the detector frame to the Earth-centered frame.

    Args:
        detector_angles (list): Detector angles [latitude, longitude, orientation].

    Returns:
        np.ndarray: Change of basis matrix.
    """
    latitude, longitude, orientation = detector_angles

    # Define vectors in the Earth-centered frame
    z_vector = source_vector_from_angles(detector_angles)
    x_vector = np.array([-m.sin(longitude), m.cos(longitude), 0])
    y_vector = np.cross(z_vector, x_vector)

    # Combine vectors into a matrix
    ec_matrix = np.array([x_vector, y_vector, z_vector]).T

    # Apply orientation rotation
    rotation_matrix = np.array([
        [m.cos(orientation), -m.sin(orientation), 0],
        [m.sin(orientation), m.cos(orientation), 0],
        [0, 0, 1]
    ])

    return np.linalg.inv(np.matmul(rotation_matrix, ec_matrix))

def detector_response(detector_angles, source_angles, tt_amplitudes):
    """
    Computes the detector response to gravitational waves.

    Args:
        detector_angles (list): Detector angles [latitude, longitude, orientation].
        source_angles (list): Source angles [declination, right ascension, polarization].
        tt_amplitudes (list): Tensor amplitudes [hplus, hcross].

    Returns:
        float: Detector response.
    """
    detector_tensor = np.array([
        [1 / 2, 0, 0],
        [0, -1 / 2, 0],
        [0, 0, 0]
    ])
    transform_detector_to_ec = change_basis_detector_to_ec(detector_angles)
    detector_tensor_ec = transform_tensor(detector_tensor, transform_detector_to_ec, '2_0')
    gw_tensor_ec = gravitational_wave_ec_frame(source_angles, tt_amplitudes)
    return np.tensordot(gw_tensor_ec, detector_tensor_ec, axes=2)