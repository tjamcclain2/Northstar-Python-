#Let's fix this code like a computer scientist!

import math as m
import numpy as np
import time
# NOTE: all angles are in radians and all dimension-ful quantities are in SI units 
# (meters, seconds, kilograms, and combinations thereof) unless explicitly indicated otherwise.
number_detectors = 2
number_gw_polarizations = 2
number_gw_modes = 4
number_source_angles = 3
hanford_detector_angles = [(46+27/60+18.528/3600)*np.pi/180, (240+35/60+32.4343/3600)*np.pi/180,np.pi/2+125.9994*np.pi/180]
livingston_detector_angles = [(30+33/60+46.4196/3600)*np.pi/180, (269+13/60+32.7346/3600)*np.pi/180,np.pi/2+197.7165*np.pi/180]
ligo_detector_sampling_rate = 16384
earth_radius = 6371000
speed_light = 299792458
maximum_hanford_livingston_time_delay = 0.010002567302556083
weighting_power = 2

#=========================================================================

# Why use np.einsum?
    # => It gives precise control over tensor index manipulation.
    # => Clear, readable summation notation: 'ik,kl->il' mimics Einstein summation.

"""
    These functions take an arbitrary matrix -- a NumPy array -- and the change-of-basis matrix -- another NumPy array -- 
    that effects a change-of-basis -- i.e., transforms the basis vectors -- from the coordinate system in which the matrix is currently 
    expressed to the coordinate system in which you would like the matrix to be expressed. 
    The first function is for (2,0) tensors (i.e., two contravariant/upper indices), the second is for (1,1) tensors (true matrices), 
    and the third is for (0,2) tensors (two covariant/lower indices).

"""

# transforms rank-2 contravariant tensor under a change of basis
def transform_2_0_tensor(matrix, change_basis_matrix) :
    contravariant_transformation_matrix = np.linalg.inv(change_basis_matrix)
    partial_transformation = np.einsum('ki,kl->il', contravariant_transformation_matrix, matrix)
    return np.einsum('lj,il->ij', contravariant_transformation_matrix, partial_transformation)

# transforms mixed tensor
def transform_1_1_tensor(matrix, change_basis_matrix) :
    contravariant_transformation_matrix = np.linalg.inv(change_basis_matrix)
    partial_transformation = np.einsum('ik,kl->il', change_basis_matrix, matrix)
    return np.einsum('lj,il->ij', contravariant_transformation_matrix, partial_transformation)

# transforms rank-2 covariant tensor under a change of basis
# Note: This is not used in the current code, but provided for completeness
def transform_0_2_tensor(matrix, change_basis_matrix) :
    partial_transformation = np.einsum('ik,kl->il', change_basis_matrix, matrix)
    return np.einsum('jl,il->ij', change_basis_matrix, partial_transformation)

#=========================================================================

def source_vector_from_angles(angles) :
    
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

        Notes
        -----
        - Assumes:
            x = cos(elevation) * cos(azimuth)  
            y = cos(elevation) * sin(azimuth)  
            z = sin(elevation)  
        where “elevation” is declination (or latitude) and “azimuth” is right ascension (or longitude).
        - The third angle (polarization/orientation) is provided for API consistency but not used here.

        Examples
        --------
        >>> angles = [0.1, 1.2, 0.0]  # δ=0.1 rad, α=1.2 rad, ψ unused
        >>> vec = source_vector_from_angles(angles)
        >>> np.linalg.norm(vec)
        1.0
    """
    [first, second, third] = angles
    initial_source_vector = np.array([m.cos(first)*m.cos(second),m.cos(first)*m.sin(second),m.sin(first)])
    return initial_source_vector

#=========================================================================

def change_basis_gw_to_ec(source_angles) :
    
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

        Notes
        -----
        1. We first build an orthonormal triad of basis vectors (x̂₍gw₎, ŷ₍gw₎, ẑ₍gw₎) in Earth coordinates:
        - ẑ₍gw₎ points from Earth’s center toward the source (–unit source vector).
        - ŷ₍gw₎ lies in the local tangent plane, orthogonal to ẑ₍gw₎.
        - x̂₍gw₎ = ŷ₍gw₎ × ẑ₍gw₎ completes the right-handed set.
        2. We then apply a rotation by the polarization angle ψ about ẑ₍gw₎ to form the contravariant
        transformation matrix Tᵢⱼ = R_z(ψ) · [x̂₍gw₎, ŷ₍gw₎, ẑ₍gw₎]₍EC₎.
        3. The returned matrix is the inverse of Tᵢⱼ, effecting the covariant change of basis
        (i.e., transforming basis vectors rather than vector components).

        Examples
        --------
        >>> angles = [0.5236, 1.0472, 0.7854]  # δ=30°, α=60°, ψ=45° in radians
        >>> M = change_basis_gw_to_ec(angles)
        >>> M.shape
        (3, 3)
        >>> # Verify inverse relationship
        >>> T = np.linalg.inv(M)
        >>> np.allclose(T @ M, np.eye(3))
        True
    """
#Original Implementation:

    # [declination,right_ascension,polarization] = source_angles
    # initial_source_vector = source_vector_from_angles(source_angles)
    # initial_gw_z_vector_earth_centered = -1*initial_source_vector
    # initial_gw_y_vector_earth_centered = np.array([-1*m.sin(declination)*m.cos(right_ascension),-1*m.sin(declination)*m.sin(right_ascension),m.cos(declination)])
    # initial_gw_x_vector_earth_centered = np.cross(initial_gw_z_vector_earth_centered,initial_gw_y_vector_earth_centered)
    # transpose_gw_vecs_ec = np.array([initial_gw_x_vector_earth_centered,initial_gw_y_vector_earth_centered,initial_gw_z_vector_earth_centered])
    # initial_gw_vecs_ec = np.transpose(transpose_gw_vecs_ec)
    # polarization_rotation_matrix = np.array([[m.cos(polarization),-1*m.sin(polarization),0],[m.sin(polarization),m.cos(polarization),0],[0,0,1]])
    # contravariant_transformation_matrix = np.matmul(polarization_rotation_matrix,initial_gw_vecs_ec)
    # change_basis_matrix = np.linalg.inv(contravariant_transformation_matrix)
    # return change_basis_matrix

#Abid's optimized implementation:

    [declination, right_ascension, polarization] = source_angles
    initial_source_vector = source_vector_from_angles(source_angles)
    initial_gw_z_vector_earth_centered = -1 * initial_source_vector
    initial_gw_y_vector_earth_centered = np.array([
            -np.sin(declination) * np.cos(right_ascension),
            -np.sin(declination) * np.sin(right_ascension),
            np.cos(declination)
        ])
    initial_gw_x_vector_earth_centered = np.cross(
            initial_gw_z_vector_earth_centered,
            initial_gw_y_vector_earth_centered
        )
        # Stack as columns: each column is a GW‐frame basis vector in EC coords
        # Stacking the three GW‐frame basis vectors into a single 2D array (and then transposing) gives you a clean matrix whose columns are exactly the basis vectors expressed in Earth-centered coordinates. The benefits are:
            # 1. Clear intent: it’s obvious that you’re building a matrix [x_gw, y_gw, z_gw]—just oriented so each vector becomes a column.
            # 2. Correct shape without juggling axes: If you stacked them as rows (the default), you’d get a 3×3 matrix where each row is a basis vector; transposing immediately turns those into columns, which is what you need for a change‐of‐basis matrix.
            # 3. Conciseness and readability: One call to vstack plus .T replaces manually constructing an empty array and assigning each column. It’s self-documenting and less prone to indexing mistakes.
            # 4. Leverage NumPy’s vectorized routines: vstack is implemented in optimized C, so you get both clarity and performance in one line.

    initial_gw_vecs_ec = np.vstack([
            initial_gw_x_vector_earth_centered,
            initial_gw_y_vector_earth_centered,
            initial_gw_z_vector_earth_centered
        ]).T
        # Rotate by polarization about z_gw
    polarization_rotation_matrix = np.array([
            [np.cos(polarization), -np.sin(polarization), 0],
            [np.sin(polarization),  np.cos(polarization), 0],
            [0,                    0,                     1]
        ])
    contravariant_transformation_matrix = polarization_rotation_matrix @ initial_gw_vecs_ec
    change_basis_matrix = np.linalg.inv(contravariant_transformation_matrix)
    return change_basis_matrix

#=========================================================================


def gravitational_wave_ec_frame(source_angles,tt_amplitudes) :
    
    """
    Compute the gravitational-wave strain tensor in Earth-centered coordinates. This function takes two lists -- the first containing the declination, right ascension, and polarization angles of the source, the second containing the "plus" and "cross" strain amplitudes of 
    the gravitational wave in the transverse, traceless ("TT") gauge of the gravitational wave frame -- and returns a NumPy array characterizing the gravitational wave's strain amplitudes in 
    the Earth-centered frame. Note that the strain tensor is a (0-2) tensor.
    
    Parameters
    ----------
    source_angles : array-like of float, shape (3,)
        Three angles (in radians) defining the source orientation in Earth-centered frame:
        - declination δ
        - right ascension α
        - polarization ψ
    tt_amplitudes : array-like of float, shape (2,)
        Transverse-traceless ("TT") strain amplitudes in the GW frame:
        - h_plus (h₊)
        - h_cross (hₓ)

    Returns
    -------
    strain_ec : ndarray of float, shape (3, 3)
        The (0,2) GW strain tensor components expressed in Earth-centered coordinates.

    Notes
    -----
    1. We first form the GW-frame strain tensor in TT gauge:
       ```
       h_tt = [[ h₊,   hₓ, 0 ],
               [ hₓ,  -h₊, 0 ],
               [  0,    0, 0 ]]
       ```
    2. `change_basis_gw_to_ec(source_angles)` returns the covariant change-of-basis matrix M
       that maps GW-frame basis vectors to Earth-centered basis vectors.
    3. `transform_0_2_tensor(h_tt, M)` applies the (0-2) tensor transformation
       \(h_{EC} = M \, h_{TT} \, M^T\), yielding the strain tensor in Earth frame.

    Examples
    --------
    >>> angles = [0.1, 1.2, 0.3]    # δ=0.1 rad, α=1.2 rad, ψ=0.3 rad
    >>> amps   = [1e-21, 2e-21]    # typical GW amplitudes
    >>> h_ec = gravitational_wave_ec_frame(angles, amps)
    >>> h_ec.shape
    (3, 3)
    
    """
#Original Implementation:

    # [hplus,hcross] = tt_amplitudes
    # gwtt = np.array([[hplus,hcross,0],[hcross,-hplus,0],[0,0,0]])
    # transformation = change_basis_gw_to_ec(source_angles)
    # return transform_0_2_tensor(gwtt,transformation)
    
#Abid's optimized implementation:

    # Changed variable names for readability
    h_plus, h_cross = tt_amplitudes
    gw_tt = np.array([
        [h_plus,  h_cross, 0],
        [h_cross, -h_plus, 0],
        [0,       0,       0]
    ])
    change_mat = change_basis_gw_to_ec(source_angles)
    return transform_0_2_tensor(gw_tt, change_mat)

#=========================================================================


def change_basis_detector_to_ec(detector_angles) :
    [latitude,longitude,orientation] = detector_angles
    initial_detector_z_vector_earth_centered = source_vector_from_angles(detector_angles)
    initial_detector_x_vector_earth_centered = np.array([-1*m.sin(longitude),m.cos(longitude),0])
    initial_detector_y_vector_earth_centered = np.cross(initial_detector_z_vector_earth_centered,initial_detector_x_vector_earth_centered)
    transpose_detector_vecs_ec = np.array([initial_detector_x_vector_earth_centered,initial_detector_y_vector_earth_centered,initial_detector_z_vector_earth_centered])
    initial_detector_vecs_ec = np.transpose(transpose_detector_vecs_ec)
    orientation_rotation_matrix = np.array([[m.cos(orientation),-1*m.sin(orientation),0],[m.sin(orientation),m.cos(orientation),0],[0,0,1]])
    contravariant_transformation_matrix = np.matmul(orientation_rotation_matrix,initial_detector_vecs_ec)
    change_basis_matrix = np.linalg.inv(contravariant_transformation_matrix)
    return change_basis_matrix

"""This function takes a list containing the latitude, longitude, and orientation a gravitational wave detector 
    and returns a NumPy array that effects the change-of-basis (covariant transformation matrix) from the detector frame 
    to the Earth-centered frame.

"""
#=========================================================================

"""
    This function takes three lists -- the first containing the latitude, longitude, and orientation angles of a gravitational wave detector, 
    the second containing the declination, right ascension, and polarization angles of the source, and the third containing the "plus" and "cross" 
    strain amplitudes of the gravitational wave in the transverse, traceless ("TT") gauge -- and returns a scalar representing the strain measured by the gravitational wave detector. 
    Note that the detector response tensor is a (2-0) tensor.
"""

def detector_response(detector_angles, source_angles, tt_amplitudes) :
    detector_response_tensor_detector_frame = np.array([[1/2,0,0],[0,-1/2,0],[0,0,0]])
    transform_detector_to_ec = change_basis_detector_to_ec(detector_angles)
    detector_response_tensor_earth_centered = transform_2_0_tensor(detector_response_tensor_detector_frame,transform_detector_to_ec)
    gw_earth_centered = gravitational_wave_ec_frame(source_angles,tt_amplitudes)
    detector_response = np.tensordot(gw_earth_centered,detector_response_tensor_earth_centered)
    return detector_response


#=========================================================================

"""This function takes two lists -- the first containing the latitude, longitude, and orientation angles of a gravitational wave detector, and the second containing the declination, 
right ascensions, and polarization angles of a gravitational wave source -- and returns a list with the beam pattern response functions F_+ and F_x of the detector for that source."""

def beam_pattern_response_functions(detector_angles,source_angles) :
    detector_response_tensor_detector_frame = np.array([[1/2,0,0],[0,-1/2,0],[0,0,0]])
    transform_detector_ec = change_basis_detector_to_ec(detector_angles)
    detector_response_tensor_earth_centered = transform_2_0_tensor(detector_response_tensor_detector_frame,transform_detector_ec)
    transform_gw_ec = change_basis_gw_to_ec(source_angles)
    transform_ec_gw = np.linalg.inv(transform_gw_ec)
    detector_response_tensor_gw_frame = transform_2_0_tensor(detector_response_tensor_earth_centered,transform_ec_gw)
    fplus = detector_response_tensor_gw_frame[0,0]-detector_response_tensor_gw_frame[1,1]
    fcross = detector_response_tensor_gw_frame[0,1]+detector_response_tensor_gw_frame[1,0]
    return [fplus, fcross]

#=========================================================================

"""This function take a list of the declination, right ascension, and polarization angles of a gravitational wave source and returns the time delay between when the signal will arrive at the Hanford detector 
and when it will arrive at the Livingston detector. Negative values indicate that the signal arrives at the Livingston detector first."""

def time_delay_hanford_to_livingston(source_angles) :
    hanford_z_vector_earth_centered = source_vector_from_angles(hanford_detector_angles)
    livingston_z_vector_earth_centered = source_vector_from_angles(livingston_detector_angles)
    position_vector_hanford_to_livingston = earth_radius * (livingston_z_vector_earth_centered - hanford_z_vector_earth_centered)
    gw_source_vector = source_vector_from_angles(source_angles)
    gw_z_vector_earth_centered = -1*gw_source_vector
    return 1/speed_light*(np.dot(gw_z_vector_earth_centered,position_vector_hanford_to_livingston))

#=========================================================================

""".This function takes a gravitational wave signal lifetime, a detector sampling rate, and the maximum possible time delay between the detectors in a network, and returns a 
NumPy array with absolute detector strain response times appropriate for all the detectors in a network. 
Note that all responses times are actual sampled times, assuming correct time synchorization between sites."""

def generate_network_time_array(signal_lifetime, detector_sampling_rate, maximum_time_delay) :
    time_sample_width = round((signal_lifetime + maximum_time_delay)*detector_sampling_rate,0)
    all_times = 1/detector_sampling_rate*(np.arange(-time_sample_width,time_sample_width,1))
    return all_times

#=========================================================================

""".This function takes the lifetime and frequency of a gravitional wave, a NumPy array with the detector strain response times of a gravitational wave detector network, 
and the time delay between when the gravitational wave arrives at the specific detector where the terms are being evaluated compared to when it arrived at a fixed reference detector (Hanford, in this code), 
and returns a NumPy array with the appropriate sine-Gaussian gravitational wave strain amplitudes for 
the detector at each network time. Note that the order of the output array is 1) time sample 2) [A_+, B_x, A_+, B_x] for each of the four gravitational wave modes."""

#Space complexity issue

"""3. Inefficient Array Operations

a) oscillatory_terms creates a full time×modes matrix when vectorized operations could be used
b) Temporary arrays created in tensor transformations aren't reused"""
def generate_oscillatory_terms(signal_lifetime, signal_frequency, time_array, time_delay) :
    number_time_samples = time_array.size
    signal_q_value = m.sqrt(m.log(2))/signal_lifetime
    oscillatory_terms = np.empty((number_time_samples,number_gw_modes))
    for this_time_sample in range(number_time_samples) :
        this_time = time_array[this_time_sample]
        this_gaussian_term = m.exp(-1*signal_q_value**2*(this_time - time_delay)**2)
        this_cos_term = m.cos(2*np.pi*signal_frequency*(this_time - time_delay))
        this_sin_term = m.sin(2*np.pi*signal_frequency*(this_time - time_delay))
        this_cos_gauss_term = this_gaussian_term*this_cos_term
        this_sin_gauss_term = this_gaussian_term*this_sin_term
        these_oscillations = np.array([this_cos_gauss_term,this_sin_gauss_term,this_cos_gauss_term,this_sin_gauss_term])
        oscillatory_terms[this_time_sample] = these_oscillations
    return oscillatory_terms

#=========================================================================
"""This functions takes the number of desired model angle sets [S, phi, psi] and returns a NumPy array with that many randomized angle sets. Note that the first angle in each set is bewteen 
-π/2 and π/2, while the other two angles are between 0 and 2π. The first angle is the declination, the second is the right ascension, and the third is the polarization angle of a gravitational wave source.
"""
def generate_model_angles_array(number_angular_samples) :
    model_angles_array = np.empty((number_angular_samples,number_source_angles))
    for this_angle_set in range(number_angular_samples) :
        for this_source_angle in range(number_source_angles) :
            if this_source_angle == 0 :
                model_angles_array[this_angle_set,this_source_angle] = (np.random.rand(1) - 1/2)*np.pi
            else :
                model_angles_array[this_angle_set,this_source_angle] = np.random.rand(1)*2*np.pi
    return model_angles_array

#=========================================================================

"""This function takes the number of desired model amplitude combinations [A_+, B_x, A_+, B_x] and the maximum allowed value for any amplitude and returns a NumPy array with the desired amplitude combinations."""
def generate_model_amplitudes_array(number_amplitude_combinations, gw_max_amps) :
    model_amplitudes_array = np.empty((number_amplitude_combinations,number_gw_modes))
    for this_amplitude_combination in range(number_amplitude_combinations) :
        new_amplitudes = np.random.rand(number_gw_modes)*gw_max_amps
        model_amplitudes_array[this_amplitude_combination] = new_amplitudes
    return model_amplitudes_array

#=========================================================================

"""This function takes three source/detector parameters -- the frequency and lifetime (here defined as the time required to drop to one-half of maximum amplitude) of (one monochromatic Fourier mode of) a gravitational wave, and the sampling rate of the detectors 
-- and three model parameters -- the maximum model graviational wave amplitude to consider and the number of amplitude and angle combinations to generate -- 
and returns a pair of NumPy arrays that give the expected response of each gravitational wave detector at each time for each amplitude and
angle combination -- indexed by 1) angle combination 2) amplitude combination 3) time sample and 4) detector -- and the array of angle combinations referenced by the detector response array."""

def generate_model_detector_responses(signal_frequency,signal_lifetime,detector_sampling_rate,gw_max_amps,number_amplitude_combinations,number_angular_samples) :
    time_array = generate_network_time_array(signal_lifetime,detector_sampling_rate,maximum_hanford_livingston_time_delay)
    number_time_samples = time_array.size
    hanford_oscillatory_terms = generate_oscillatory_terms(signal_lifetime,signal_frequency,time_array,0)
    model_amplitudes_array = generate_model_amplitudes_array(number_amplitude_combinations, gw_max_amps)
    model_angles_array = generate_model_angles_array(number_angular_samples)
    
    #Space Complexity issue
    
    model_detector_response_array = np.empty((number_angular_samples,number_amplitude_combinations,number_time_samples,number_detectors))
    
    
    for this_angle_set in range(number_angular_samples) :
        these_angles = model_angles_array[this_angle_set]
        [fplus_hanford,fcross_hanford] = beam_pattern_response_functions(hanford_detector_angles,these_angles)
        [fplus_livingston,fcross_livingston] = beam_pattern_response_functions(livingston_detector_angles,these_angles)
        hanford_livingston_time_delay = time_delay_hanford_to_livingston(these_angles)
        livingston_oscillatory_terms = generate_oscillatory_terms(signal_lifetime,signal_frequency,time_array,hanford_livingston_time_delay)
        for this_amplitude_combination in range(number_amplitude_combinations) :
            these_amplitudes = model_amplitudes_array[this_amplitude_combination]
            for this_sample_time in range(number_time_samples) :
                model_detector_response_array[this_angle_set,this_amplitude_combination,this_sample_time,0] = np.dot(these_amplitudes,hanford_oscillatory_terms[this_sample_time]*[fplus_hanford,fplus_hanford,fcross_hanford,fcross_hanford])
                model_detector_response_array[this_angle_set,this_amplitude_combination,this_sample_time,1] = np.dot(these_amplitudes,livingston_oscillatory_terms[this_sample_time]*[fplus_livingston,fplus_livingston,fcross_livingston,fcross_livingston])
    return [model_detector_response_array,model_angles_array]


# # AFTER: Optimized version that eliminates the massive array
# class ModelResponseGenerator:
#     """
#     Virtual array that generates model responses on-demand.
#     Eliminates the need to store the massive 4D array.
#     """
#     def __init__(self, signal_frequency, signal_lifetime, detector_sampling_rate, gw_max_amps, number_amplitude_combinations, number_angular_samples):
#         # Store parameters instead of pre-computing everything
#         self.signal_frequency = signal_frequency
#         self.signal_lifetime = signal_lifetime
#         self.detector_sampling_rate = detector_sampling_rate
        
#         # Generate the smaller arrays we actually need
#         self.time_array = generate_network_time_array(signal_lifetime, detector_sampling_rate, maximum_hanford_livingston_time_delay)
#         self.number_time_samples = self.time_array.size
#         self.model_amplitudes_array = generate_model_amplitudes_array(number_amplitude_combinations, gw_max_amps)
#         self.model_angles_array = generate_model_angles_array(number_angular_samples)
        
#         # Pre-compute Hanford oscillatory terms (no time delay)
#         self.hanford_oscillatory_terms = generate_oscillatory_terms(signal_lifetime, signal_frequency, self.time_array, 0)
        
#         # Cache for expensive calculations
#         self.beam_pattern_cache = {}
#         self.time_delay_cache = {}
#         self.livingston_oscillatory_cache = {}
        
#         # Store dimensions for array-like interface
#         self.shape = (number_angular_samples, number_amplitude_combinations, self.number_time_samples, number_detectors)
    
#     def _get_beam_patterns(self, angle_idx):
#         """Get beam patterns with caching"""
#         if angle_idx not in self.beam_pattern_cache:
#             angles = self.model_angles_array[angle_idx]
#             hanford_patterns = beam_pattern_response_functions(hanford_detector_angles, angles)
#             livingston_patterns = beam_pattern_response_functions(livingston_detector_angles, angles)
#             self.beam_pattern_cache[angle_idx] = (hanford_patterns, livingston_patterns)
#         return self.beam_pattern_cache[angle_idx]
    
#     def _get_time_delay(self, angle_idx):
#         """Get time delay with caching"""
#         if angle_idx not in self.time_delay_cache:
#             angles = self.model_angles_array[angle_idx]
#             delay = time_delay_hanford_to_livingston(angles)
#             self.time_delay_cache[angle_idx] = delay
#         return self.time_delay_cache[angle_idx]
    
#     def _get_livingston_oscillatory(self, angle_idx):
#         """Get Livingston oscillatory terms with caching"""
#         if angle_idx not in self.livingston_oscillatory_cache:
#             delay = self._get_time_delay(angle_idx)
#             livingston_terms = generate_oscillatory_terms(self.signal_lifetime, self.signal_frequency, self.time_array, delay)
#             self.livingston_oscillatory_cache[angle_idx] = livingston_terms
#         return self.livingston_oscillatory_cache[angle_idx]
    
#     def get_response(self, angle_idx, amplitude_idx, time_idx, detector_idx):
#         """Generate a single response value on-demand"""
#         # Get cached expensive calculations
#         (hanford_patterns, livingston_patterns) = self._get_beam_patterns(angle_idx)
#         [fplus_hanford, fcross_hanford] = hanford_patterns
#         [fplus_livingston, fcross_livingston] = livingston_patterns
        
#         amplitudes = self.model_amplitudes_array[amplitude_idx]
        
#         if detector_idx == 0:  # Hanford
#             oscillatory_terms = self.hanford_oscillatory_terms[time_idx]
#             pattern_multipliers = [fplus_hanford, fplus_hanford, fcross_hanford, fcross_hanford]
#         else:  # Livingston
#             livingston_oscillatory = self._get_livingston_oscillatory(angle_idx)
#             oscillatory_terms = livingston_oscillatory[time_idx]
#             pattern_multipliers = [fplus_livingston, fplus_livingston, fcross_livingston, fcross_livingston]
        
#         return np.dot(amplitudes, oscillatory_terms * pattern_multipliers)
    
#     def get_full_response_for_params(self, angle_idx, amplitude_idx):
#         """Get full time series for specific angle/amplitude combination"""
#         # This is more efficient than calling get_response() for each time point
#         (hanford_patterns, livingston_patterns) = self._get_beam_patterns(angle_idx)
#         [fplus_hanford, fcross_hanford] = hanford_patterns
#         [fplus_livingston, fcross_livingston] = livingston_patterns
        
#         amplitudes = self.model_amplitudes_array[amplitude_idx]
#         livingston_oscillatory = self._get_livingston_oscillatory(angle_idx)
        
#         # Vectorized calculation for all time points
#         hanford_response = np.dot(
#             amplitudes[np.newaxis, :] * self.hanford_oscillatory_terms, 
#             [fplus_hanford, fplus_hanford, fcross_hanford, fcross_hanford]
#         )
#         livingston_response = np.dot(
#             amplitudes[np.newaxis, :] * livingston_oscillatory, 
#             [fplus_livingston, fplus_livingston, fcross_livingston, fcross_livingston]
#         )
        
#         return np.column_stack([hanford_response, livingston_response])
    
#     def __getitem__(self, idx):
#         """Allow array-like indexing"""
#         if len(idx) == 4:
#             angle_idx, amplitude_idx, time_idx, detector_idx = idx
#             return self.get_response(angle_idx, amplitude_idx, time_idx, detector_idx)
#         elif len(idx) == 2:
#             angle_idx, amplitude_idx = idx
#             return self.get_full_response_for_params(angle_idx, amplitude_idx)
#         else:
#             raise IndexError("Invalid indexing")

# # Updated function that returns the generator instead of massive array
# def generate_model_detector_responses(signal_frequency, signal_lifetime, detector_sampling_rate, gw_max_amps, number_amplitude_combinations, number_angular_samples):
#     """
#     OPTIMIZED VERSION: Returns a generator that creates responses on-demand
#     Memory usage: ~2MB instead of ~800MB
#     """
#     generator = ModelResponseGenerator(
#         signal_frequency, signal_lifetime, detector_sampling_rate, 
#         gw_max_amps, number_amplitude_combinations, number_angular_samples
#     )
    
#     return [generator, generator.model_angles_array]


# # USAGE EXAMPLE:
# # Before (creates 800MB array):
# # [model_detector_responses, model_angles_array] = generate_model_detector_responses_original(...)
# # response = model_detector_responses[10, 5, 100, 0]  # Access one value

# # After (uses ~2MB):
# # [model_generator, model_angles_array] = generate_model_detector_responses(...)  
# # response = model_generator[10, 5, 100, 0]  # Same interface, computed on-demand
# # full_response = model_generator[10, 5]  # Get full time series efficiently

#=========================================================================

"""This function takes a maximum noise amplitude and a number of time samples and returns random (non-Gaussian) noise between zero and the appropriate maximum for each time sample."""
def generate_noise_array(max_noise_amp,number_time_samples) :
    noise_array = np.random.rand(number_time_samples)*max_noise_amp
    return noise_array

#=========================================================================

""".This function takes three source/detector parameters -- the frequency and lifetime (here defined as the time required to drop to one-half of maximum amplitude) of (one monochromatic Fourier mode of) a gravitational wave, 
and the sampling rate of the detectors -- and three model parameters -- the maximum model graviational wave amplitude to consider and the number of amplitude and angle combinations in the model 
-- and returns a pair of NumPy arrays that give the "true" (simulated) response of each gravitational wave detector at each time for each amplitude and angle combination 
-- indexed by 1) angle combination 2) amplitude combination 3) time sample and 4) detector -- and the "true" simulated source angles for each angle combination referenced by the detector response array. 
Note that there is a great deal of repeated information in these arrays, 
but they are intentionally size-matched with the outputs of the function "generate_model_detector_responses" to ease later array computations."""
def generate_real_detector_responses(signal_frequency,signal_lifetime,detector_sampling_rate,gw_max_amps,number_amplitude_combinations,number_angular_samples,max_noise_amp) :
    time_array = generate_network_time_array(signal_lifetime,detector_sampling_rate,maximum_hanford_livingston_time_delay)
    number_time_samples = time_array.size
    real_amplitudes = generate_model_amplitudes_array(1, gw_max_amps)[0]
    real_angles = generate_model_angles_array(1)[0]
    [fplus_hanford,fcross_hanford] = beam_pattern_response_functions(hanford_detector_angles,real_angles)
    [fplus_livingston,fcross_livingston] = beam_pattern_response_functions(livingston_detector_angles,real_angles)
    hanford_livingston_time_delay = time_delay_hanford_to_livingston(real_angles)
    hanford_oscillatory_terms = generate_oscillatory_terms(signal_lifetime,signal_frequency,time_array,0)
    livingston_oscillatory_terms = generate_oscillatory_terms(signal_lifetime,signal_frequency,time_array,hanford_livingston_time_delay)
    small_detector_response_array = np.empty((number_time_samples,number_detectors))
    hanford_noise_array = generate_noise_array(max_noise_amp,number_time_samples)
    livingston_noise_array = generate_noise_array(max_noise_amp,number_time_samples)
    for this_sample_time in range(number_time_samples) :
        small_detector_response_array[this_sample_time,0] = np.dot(real_amplitudes,hanford_oscillatory_terms[this_sample_time]*[fplus_hanford,fplus_hanford,fcross_hanford,fcross_hanford]) + hanford_noise_array[this_sample_time]
        small_detector_response_array[this_sample_time,1] = np.dot(real_amplitudes,livingston_oscillatory_terms[this_sample_time]*[fplus_livingston,fplus_livingston,fcross_livingston,fcross_livingston]) + livingston_noise_array[this_sample_time]
    
     #Space Complexity issue
    real_angles_array = np.empty((number_angular_samples,number_source_angles))
    
    #Space Complexity issue
    real_detector_response_array = np.empty((number_angular_samples,number_amplitude_combinations,number_time_samples,number_detectors))
    
    """2. Redundant Data Storage

    real_detector_response_array stores identical copies of the same detector response across all angle/amplitude combinations
    real_angles_array repeats the same angles for every sample
    Multiple intermediate arrays are created unnecessarily"""
    
    
    
    for this_angle_set in range(number_angular_samples) :
        
        real_angles_array[this_angle_set] = real_angles
        for this_amplitude_combination in range(number_amplitude_combinations) :
            real_detector_response_array[this_angle_set,this_amplitude_combination] = small_detector_response_array
    return [real_detector_response_array,real_angles_array]

#=========================================================================

""".This function takes NumPy arrays containing the "real" (simulated) detector responses and source angles and the model detector responses and source angles 
and returns the sum of the absolute values of the differences between the "real" (simulated) angles and 1) the angles in model_angles_array that are closest to the "real" angles (i.e., 
the best the fitting procedure could have done given the model angles it used) 2) the angles produced by finding the single best time-and-site summed model detector response and using the angles 
that produce it and 3) the angles produced by weighting the time-and-site summed detector responses, summing them over all amplitude combinations, and using the angles produced by maximizing that sum.

"""

def get_best_fit_angles_deltas(real_detector_responses,real_angles_array,model_detector_responses,model_angles_array) :
    real_model_angle_deltas = np.absolute(real_angles_array - model_angles_array)
    summed_real_model_angle_deltas = np.sum(real_model_angle_deltas,-1)
    minimum_summed_angle_delta = np.min(summed_real_model_angle_deltas)
    position_minimum_angles_delta = np.where(summed_real_model_angle_deltas == minimum_summed_angle_delta)
    angles_minimum_angles_delta = model_angles_array[position_minimum_angles_delta[0]]
    real_minimum_angles_deltas = np.absolute(real_angles_array[0] - angles_minimum_angles_delta)
    sum_real_minimum_angle_deltas = np.sum(real_minimum_angles_deltas)

    single_best_fit_start_time = time.process_time()
    real_model_response_deltas = np.absolute(real_detector_responses - model_detector_responses)
    summed_real_model_response_deltas = np.sum(real_model_response_deltas,axis=(-1,-2))
    minimum_summed_response_delta = np.min(summed_real_model_response_deltas)
    position_minimum_response_delta = np.where(summed_real_model_response_deltas == minimum_summed_response_delta)
    angles_minimum_response_delta = model_angles_array[position_minimum_response_delta[0]]
    real_minimum_response_angle_deltas = np.absolute(real_angles_array[0] - angles_minimum_response_delta)
    sum_real_minimum_response_angle_deltas = np.sum(real_minimum_response_angle_deltas)
    single_best_fit_end_time = time.process_time()
    single_best_fit_time = single_best_fit_end_time - single_best_fit_start_time

    weighted_best_fit_start_time = time.process_time()
    offset_matrix = np.ones(summed_real_model_response_deltas.shape)
    fractional_summed_real_model_response_deltas = 1/minimum_summed_response_delta * summed_real_model_response_deltas
    weighted_summed_real_model_response_deltas = np.exp(offset_matrix - fractional_summed_real_model_response_deltas**weighting_power)
    summed_weighted_summed_real_model_response_deltas = np.sum(weighted_summed_real_model_response_deltas,axis=-1)
    maximum_summed_weighted_response_delta = np.max(summed_weighted_summed_real_model_response_deltas)
    position_maximum_summed_weighted_response_delta = np.where(summed_weighted_summed_real_model_response_deltas == maximum_summed_weighted_response_delta)
    angles_maximum_summed_weighted_response_delta = model_angles_array[position_maximum_summed_weighted_response_delta[0]]
    real_maximum_weighted_response_angle_deltas = np.absolute(real_angles_array[0] - angles_maximum_summed_weighted_response_delta)
    sum_real_maximum_weighted_response_angle_deltas = np.sum(real_maximum_weighted_response_angle_deltas)
    weighted_best_fit_end_time = time.process_time()
    weighted_best_fit_time = weighted_best_fit_end_time - weighted_best_fit_start_time

    return [sum_real_minimum_angle_deltas,sum_real_minimum_response_angle_deltas,sum_real_maximum_weighted_response_angle_deltas,single_best_fit_time,weighted_best_fit_time]

#=========================================================================

full_process_start_time = time.process_time()
gw_frequency = 100
gw_lifetime = 0.03
detector_sampling_rate = ligo_detector_sampling_rate
gw_max_amps = 1
max_noise_amp = 0.1

# Number of samples for angles and amplitudes
number_angular_samples = 105
number_amplitude_combinations = 10**5
#########################################


[model_detector_responses,model_angles_array] = generate_model_detector_responses(gw_frequency,gw_lifetime,detector_sampling_rate,gw_max_amps,number_amplitude_combinations,number_angular_samples)
[real_detector_responses,real_angles_array] = generate_real_detector_responses(gw_frequency,gw_lifetime,detector_sampling_rate,gw_max_amps,number_amplitude_combinations,number_angular_samples,max_noise_amp)
best_fit_data = get_best_fit_angles_deltas(real_detector_responses,real_angles_array,model_detector_responses,model_angles_array)
full_process_end_time = time.process_time()
full_process_time = full_process_end_time - full_process_start_time
end_time_string = time.strftime("%d_%b_%Y_%H%M%S",)
file_name = "northstar-ouput-" + end_time_string + ".txt"

with open(file_name,"w") as file:
    file.write("The best possible fit angle delta (in radians) was: " + str(best_fit_data[0]))
    file.write("\n")
    file.write("The single best fit algorithm angle delta (in radians) was: " + str(best_fit_data[1]))
    file.write("\n")
    file.write("The weighted best fit algorithm angle delta (in radians) was: " + str(best_fit_data[2]))
    file.write("\n")
    file.write("The full process run time (in seconds) was: " + str(full_process_time))
    file.write("\n")
    file.write("The single best fit algorithm run time (in seconds) was: " + str(best_fit_data[3]))
    file.write("\n")
    file.write("The weighted best fit algorithm run time (in seconds) was: " + str(best_fit_data[4]))
