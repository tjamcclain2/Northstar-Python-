import math as m
import numpy as np
import time
def main():
    
#Let's fix this code like a computer scientist!

    """
    Northstar Algorithm — Original Implementation and Optimization. This is the most readable version of the code, with all the optimizations applied as of July 14, 2025.

    Author: Dr. Tom McClain  
    Optimized by: Abid Jeem
    Date: 2025-07-14
    """

    import time
    import numpy as np

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
        return np.deg2rad(deg + minutes / 60 + seconds / 3600)

    # Detector angles: [latitude, longitude, orientation]
    hanford_detector_angles = np.array([
        dms_to_rad(46, 27, 18.528),
        dms_to_rad(240, 35, 32.4343),
        np.deg2rad(125.9994) + np.pi / 2
    ])

    livingston_detector_angles = np.array([
        dms_to_rad(30, 33, 46.4196),
        dms_to_rad(269, 13, 32.7346),
        np.deg2rad(197.7165) + np.pi / 2
    ])

    #=========================================================================

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
        """
        [first, second, third] = angles
        initial_source_vector = np.array([np.cos(first)*np.cos(second), np.cos(first)*np.sin(second), np.sin(first)])
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
        """

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
        
        """
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
        x_det_ec = np.array([-np.sin(longitude), np.cos(longitude), 0.0])

        # ŷ_det completes right-handed set
        y_det_ec = np.cross(z_det_ec, x_det_ec)

        # Stack as columns to form the GW-frame basis matrix in EC coords
        det_vecs_ec = np.vstack([x_det_ec, y_det_ec, z_det_ec]).T

        # Rotate about local vertical (z_det_ec) by orientation γ
        orientation_rotation = np.array([
            [np.cos(orientation), -np.sin(orientation), 0.0],
            [np.sin(orientation),  np.cos(orientation), 0.0],
            [0.0,                  0.0,                 1.0]
        ])

        T_contravariant = orientation_rotation @ det_vecs_ec
        
        # Directly inverted the contravariant matrix in one line
        change_basis_matrix = np.linalg.inv(T_contravariant)
        return change_basis_matrix
        

    #=========================================================================

    def detector_response(detector_angles, source_angles, tt_amplitudes) :
        
        """
        Compute the scalar strain measured by a gravitational-wave detector. This function takes three lists -- the first containing the latitude, longitude, and orientation angles of a gravitational wave detector, 
        the second containing the declination, right ascension, and polarization angles of the source, and the third containing the "plus" and "cross" 
        strain amplitudes of the gravitational wave in the transverse, traceless ("TT") gauge -- and returns a scalar representing the strain measured by the gravitational wave detector. 
        Note that the detector response tensor is a (2-0) tensor.

        Parameters
        ----------
        detector_angles : array-like of float, shape (3,)
            Detector orientation in Earth-centered coordinates (radians):
            - latitude θ
            - longitude ϕ
            - orientation γ (rotation about local vertical)
        source_angles : array-like of float, shape (3,)
            Source orientation in Earth-centered coordinates (radians):
            - declination δ
            - right ascension α
            - polarization ψ
        tt_amplitudes : array-like of float, shape (2,)
            Transverse-traceless strain amplitudes in GW frame:
            - h₊ (plus)
            - hₓ (cross)

        Returns
        -------
        response : float
            Scalar detector response: the double contraction of the Earth-frame
            GW strain tensor with the detector’s response tensor.
        """
        # Detector-frame response tensor (2-0)
        D_det = np.array([
            [0.5,  0.0, 0.0],
            [0.0, -0.5, 0.0],
            [0.0,  0.0, 0.0]
        ])

        # Covariant change-of-basis matrix for detector → Earth-centered
        R_det_ec = change_basis_detector_to_ec(detector_angles)

        # Transform detector response tensor into Earth-centered frame
        D_ec = transform_2_0_tensor(D_det, R_det_ec)

        # Get GW strain tensor in Earth-centered frame
        h_ec = gravitational_wave_ec_frame(source_angles, tt_amplitudes)

        # Double contraction over both tensor indices → scalar
        return np.tensordot(h_ec, D_ec, axes=([0, 1], [0, 1]))


    #=========================================================================



    def beam_pattern_response_functions(detector_angles,source_angles) :
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
        D_det = np.array([
            [0.5,  0.0, 0.0],
            [0.0, -0.5, 0.0],
            [0.0,  0.0, 0.0]
        ])

        # Change-of-basis: detector frame → Earth-centered
        R_det_ec = change_basis_detector_to_ec(detector_angles)
        D_ec    = transform_2_0_tensor(D_det, R_det_ec)

        # Change-of-basis: Earth-centered → GW frame
        R_gw_ec = change_basis_gw_to_ec(source_angles)
        R_ec_gw = np.linalg.inv(R_gw_ec)
        D_gw    = transform_2_0_tensor(D_ec, R_ec_gw)

        # Extract plus and cross responses
        F_plus  = D_gw[0, 0] - D_gw[1, 1]
        F_cross = D_gw[0, 1] + D_gw[1, 0]

        return F_plus, F_cross
        

    #=========================================================================

    def time_delay_hanford_to_livingston(source_angles) :
        
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
        propagation_dir = -source_vector_from_angles(source_angles)

        # Return time delay (s)
        return np.dot(propagation_dir, baseline) / SPEED_OF_LIGHT

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
        half_samples = int(np.ceil(T_max * detector_sampling_rate))

        # Generate symmetric time array around zero
        time_array = np.arange(-half_samples, half_samples) / detector_sampling_rate

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
        Q = np.sqrt(np.log(2)) / signal_lifetime

        # Time relative to arrival at this detector
        dt = time_array - time_delay

        # Envelope and phase arrays
        envelope = np.exp(-Q**2 * dt**2)
        phase    = 2 * np.pi * signal_frequency * dt

        # Cosine- and sine-modulated terms
        A_plus = envelope * np.cos(phase)
        B_cross = envelope * np.sin(phase)

        # Stack into (N,4) for the four modes [A₊, Bₓ, A₊, Bₓ]
        oscillatory_terms = np.column_stack([A_plus, B_cross, A_plus, B_cross])

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
        dec = (np.random.rand(number_angular_samples) - 0.5) * np.pi
        # Right ascension & polarization: uniform in [0, 2π)
        ra_psi = np.random.rand(number_angular_samples, 2) * 2 * np.pi
        # Stack into shape (N, 3)
        angle_grid = np.column_stack((dec, ra_psi))
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
        return np.random.rand(number_amplitude_combinations, NUMBER_GW_MODES) * gw_max_amps


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
        amplitude_grid = generate_model_amplitudes_array(n_amp, gw_max_amps)  # shape (n_amp, 4)
        angle_grid = generate_model_angles_array(n_ang)                     # shape (n_ang, 3)

        # Initialize output array: detectors=2 (0=H1, 1=L1)
        responses = np.empty((n_ang, n_amp, n_times, 2))

        # Loop over angle sets
        for i_ang, angles in enumerate(angle_grid):
            # Beam patterns for Hanford & Livingston
            Fp_H, Fx_H = beam_pattern_response_functions(hanford_detector_angles, angles)
            Fp_L, Fx_L = beam_pattern_response_functions(livingston_detector_angles, angles)

            # Weight Hanford oscillatory terms: shape (n_times, 4)
            pattern_H = np.array([Fp_H, Fx_H, Fp_H, Fx_H])
            weighted_H = hanford_terms * pattern_H[np.newaxis, :]

            # Compute Livingston oscillatory terms with its time delay
            delay_L = time_delay_hanford_to_livingston(angles)
            liv_terms = generate_oscillatory_terms(
                signal_lifetime, signal_frequency, time_array, delay_L
            )
            pattern_L = np.array([Fp_L, Fx_L, Fp_L, Fx_L])
            weighted_L = liv_terms * pattern_L[np.newaxis, :]

            # Loop over amplitude combinations and vector-dot over modes
            for j_amp, amps in enumerate(amplitude_grid):
                # responses for Hanford (detector 0) and Livingston (detector 1)
                responses[i_ang, j_amp, :, 0] = weighted_H @ amps
                responses[i_ang, j_amp, :, 1] = weighted_L @ amps

        return responses, angle_grid


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
        
        noise_array = np.random.rand(number_time_samples)*max_noise_amp
        return noise_array

    #=========================================================================

    def generate_real_detector_responses(signal_frequency, signal_lifetime, detector_sampling_rate,
                                        gw_max_amps, number_amplitude_combinations,
                                        number_angular_samples, max_noise_amp):
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

        # 2. Sample 1 true amplitude and angle
        real_amplitudes = generate_model_amplitudes_array(1, gw_max_amps)[0]
        real_angles = generate_model_angles_array(1)[0]

        # 3. Detector beam pattern responses
        fplus_hanford, fcross_hanford = beam_pattern_response_functions(hanford_detector_angles, real_angles)
        fplus_livingston, fcross_livingston = beam_pattern_response_functions(livingston_detector_angles, real_angles)

        # 4. Time delay and oscillatory terms
        time_delay = time_delay_hanford_to_livingston(real_angles)
        osc_hanford = generate_oscillatory_terms(signal_lifetime, signal_frequency, time_array, 0)
        osc_livingston = generate_oscillatory_terms(signal_lifetime, signal_frequency, time_array, time_delay)

        # 5. Noise arrays
        noise_h = generate_noise_array(max_noise_amp, number_time_samples)
        noise_l = generate_noise_array(max_noise_amp, number_time_samples)

        # 6. Combine signal + noise for both detectors using broadcasting
        weights_h = np.array([fplus_hanford, fplus_hanford, fcross_hanford, fcross_hanford])
        weights_l = np.array([fplus_livingston, fplus_livingston, fcross_livingston, fcross_livingston])

        signal_h = np.dot(osc_hanford * weights_h, real_amplitudes)
        signal_l = np.dot(osc_livingston * weights_l, real_amplitudes)

        # Shape: (time_samples, detectors)
        small_response = np.stack([signal_h + noise_h, signal_l + noise_l], axis=1)

        # 7. Efficient duplication using broadcasting
        real_detector_response_array = np.broadcast_to(
            small_response[None, None, :, :],
            (number_angular_samples, number_amplitude_combinations, number_time_samples, NUMBER_DETECTORS)
        ).copy()

        real_angles_array = np.broadcast_to(
            real_angles[None, :],
            (number_angular_samples, NUMBER_SOURCE_ANGLES)
        ).copy()

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
        # 1. Oracle best: angle delta to closest model angle
        angle_deltas = np.abs(real_angles_array[0] - model_angles_array)
        summed_angle_deltas = np.sum(angle_deltas, axis=1)
        min_angle_idx = np.argmin(summed_angle_deltas)
        sum_real_minimum_angle_deltas = np.sum(np.abs(real_angles_array[0] - model_angles_array[min_angle_idx]))

        # 2. Single best fit: minimum total difference in detector responses
        t_start = time.process_time()
        response_deltas = np.abs(real_detector_responses - model_detector_responses)
        summed_response_deltas = np.sum(response_deltas, axis=(-1, -2))  # shape: (n_angles, n_amps)
        min_response_idx = np.unravel_index(np.argmin(summed_response_deltas), summed_response_deltas.shape)
        best_fit_angles = model_angles_array[min_response_idx[0]]
        sum_real_minimum_response_angle_deltas = np.sum(np.abs(real_angles_array[0] - best_fit_angles))
        t_single_fit = time.process_time() - t_start

        # 3. Weighted best fit
        t_start = time.process_time()
        fractional_deltas = summed_response_deltas / np.min(summed_response_deltas)
        weights = np.exp(1 - fractional_deltas**WEIGHTING_POWER)
        summed_weights = np.sum(weights, axis=1)
        weighted_idx = np.argmax(summed_weights)
        weighted_angles = model_angles_array[weighted_idx]
        sum_real_maximum_weighted_response_angle_deltas = np.sum(np.abs(real_angles_array[0] - weighted_angles))
        t_weighted_fit = time.process_time() - t_start

        return [
            sum_real_minimum_angle_deltas,
            sum_real_minimum_response_angle_deltas,
            sum_real_maximum_weighted_response_angle_deltas,
            t_single_fit,
            t_weighted_fit
        ]


    #========================================================================= DRIVER SECTION — EXECUTION STARTS HERE =========================================================================

    full_process_start_time = time.process_time()

    # Define fixed inputs
    gw_frequency = 100
    gw_lifetime = 0.03
    detector_sampling_rate = LIGO_DETECTOR_SAMPLING_RATE
    gw_max_amps = 1
    max_noise_amp = 0.1
    number_angular_samples = 100
    number_amplitude_combinations = 100

    # Generate model responses
    model_detector_responses, model_angles_array = generate_model_detector_responses(
        gw_frequency,
        gw_lifetime,
        detector_sampling_rate,
        gw_max_amps,
        number_amplitude_combinations,
        number_angular_samples
    )

    # Generate noisy real responses
    real_detector_responses, real_angles_array = generate_real_detector_responses(
        gw_frequency,
        gw_lifetime,
        detector_sampling_rate,
        gw_max_amps,
        number_amplitude_combinations,
        number_angular_samples,
        max_noise_amp
    )

    # Compare real vs model
    best_fit_data = get_best_fit_angles_deltas(
        real_detector_responses,
        real_angles_array,
        model_detector_responses,
        model_angles_array
    )

    full_process_end_time = time.process_time()
    full_process_time = full_process_end_time - full_process_start_time

    # Generate timestamped filename
    timestamp = time.strftime("%d_%b_%Y_%H-%M-%S")
    file_name = f"northstar_output_{timestamp}.txt"

    # Prepare results
    results = [
        f"The best possible fit angle delta (in radians) was: {best_fit_data[0]:.6f}",
        f"The single best fit algorithm angle delta (in radians) was: {best_fit_data[1]:.6f}",
        f"The weighted best fit algorithm angle delta (in radians) was: {best_fit_data[2]:.6f}",
        f"The full process run time (in seconds) was: {full_process_time:.4f}",
        f"The single best fit algorithm run time (in seconds) was: {best_fit_data[3]:.4f}",
        f"The weighted best fit algorithm run time (in seconds) was: {best_fit_data[4]:.4f}"
    ]

    # Print to console
    print("\n[Northstar Run Summary]")
    for line in results:
        print(line)

    # Save to file
    with open(file_name, "w") as file:
        for line in results:
            file.write(line + "\n")

    print(f"\n[✔] Output also written to: {file_name}")



if __name__ == "__main__":
    import cProfile
    cProfile.run('main()', filename='Optimized_profile.prof')
    print("Profiling complete. View results with: snakeviz Original_profile.prof")
