import math as m
import numpy as np
import time
def main():
    
    #Let's fix this code like a computer scientist!

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

        Notes
        -----
        1. We build an orthonormal triad of basis vectors (x̂₍det₎, ŷ₍det₎, ẑ₍det₎) in Earth coords:
        - ẑ₍det₎ points from Earth’s center toward the detector (unit “source” vector).
        - x̂₍det₎ points eastward tangent to Earth’s surface, given by [−sinϕ, cosϕ, 0].
        - ŷ₍det₎ = ẑ₍det₎ × x̂₍det₎ completes the right-handed basis.
        2. We then rotate this triad by the orientation angle γ about ẑ₍det₎:
        Tᵢⱼ = R_z(γ) · [x̂₍det₎, ŷ₍det₎, ẑ₍det₎]₍EC₎.
        3. The returned matrix is the inverse of Tᵢⱼ, effecting the covariant change of basis
        (transforming basis vectors rather than components).

        Examples
        --------
        >>> angles = [0.7854, 1.0472, 0.5236]  # θ=45°, ϕ=60°, γ=30° in radians
        >>> M = change_basis_detector_to_ec(angles)
        >>> M.shape
        (3, 3)
        >>> # Verify inverse relationship
        >>> T = np.linalg.inv(M)
        >>> np.allclose(T @ M, np.eye(3))
        True
        """
        
        # Original Implementation:
        
        # [latitude,longitude,orientation] = detector_angles
        # initial_detector_z_vector_earth_centered = source_vector_from_angles(detector_angles)
        # initial_detector_x_vector_earth_centered = np.array([-1*m.sin(longitude),m.cos(longitude),0])
        # initial_detector_y_vector_earth_centered = np.cross(initial_detector_z_vector_earth_centered,initial_detector_x_vector_earth_centered)
        # transpose_detector_vecs_ec = np.array([initial_detector_x_vector_earth_centered,initial_detector_y_vector_earth_centered,initial_detector_z_vector_earth_centered])
        # initial_detector_vecs_ec = np.transpose(transpose_detector_vecs_ec)
        # orientation_rotation_matrix = np.array([[m.cos(orientation),-1*m.sin(orientation),0],[m.sin(orientation),m.cos(orientation),0],[0,0,1]])
        # contravariant_transformation_matrix = np.matmul(orientation_rotation_matrix,initial_detector_vecs_ec)
        # change_basis_matrix = np.linalg.inv(contravariant_transformation_matrix)
        # return change_basis_matrix
        
        # Abid's optimized implementation:
        
        # 1. Changed all math functions to use NumPy (vetorized operations) for consistency and performance
        # 2. Shorter variable names for readability
        # 3. I added explicit orthonormal-triad comments before each basis vector construction to label as hats in Earth coords
            # to help see the geometry being built
        # 4. Used np.vstack to stack vectors instead of two-step array. This is more concise, less chances of mixing up row vs col orientation, and 
            # it clearly shows stacks these 3 vectors as columns, which is what we want for a change-of-basis matrix.
        # 5. Streamlined rotation multiplication by using @ operator instead of np.matmul, which is more readable and idiomatic in modern NumPy
            # plus it reads like linear algebra notation! 
            # After comparing it using perfplot (https://github.com/nschloe/perfplot) it turns out the @ operator is actually faster than np.matmul for this case.
        
        
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

        Notes
        -----
        1. In the detector frame, the response tensor is 
        ```
        D_det = ½ diag(1, –1, 0)
        ```
        a (2-0) tensor.
        2. Transform D_det to Earth-centered frame via 
        `D_ec = transform_2_0_tensor(D_det, R_det_ec)`, where 
        `R_det_ec = change_basis_detector_to_ec(detector_angles)`.
        3. Compute the GW strain in Earth frame:
        `h_ec = gravitational_wave_ec_frame(source_angles, tt_amplitudes)`.
        4. The measured strain is 
        ```
        response = h_ec : D_ec = Σ_{i,j} h_ec[i,j] · D_ec[i,j].
        ```

        Examples
        --------
        >>> det_angles = [0.6, 1.2, 0.3]
        >>> src_angles = [0.1, 2.3, 0.4]
        >>> amps = [1e-22, 2e-22]
        >>> r = detector_response(det_angles, src_angles, amps)
        >>> isinstance(r, float)
        True
        """
        
        #Original Implementation:
            # detector_response_tensor_detector_frame = np.array([[1/2,0,0],[0,-1/2,0],[0,0,0]])
            # transform_detector_to_ec = change_basis_detector_to_ec(detector_angles)
            # detector_response_tensor_earth_centered = transform_2_0_tensor(detector_response_tensor_detector_frame,transform_detector_to_ec)
            # gw_earth_centered = gravitational_wave_ec_frame(source_angles,tt_amplitudes)
            # detector_response = np.tensordot(gw_earth_centered,detector_response_tensor_earth_centered)
            # return detector_response
            
        
        #Abid's optimized implementation:
        
        # Explanation of changes made:
            # 1. Changed variable names for clarity:
                #BEFORE: 
                    # detector_response_tensor_detector_frame
                    # transform_detector_to_ec
                    # detector_response_tensor_earth_centered
                    # gw_earth_centered
                
                #AFTER:
                    # D_det       # detector‐frame response tensor
                    # R_det_ec    # change‐of‐basis matrix
                    # D_ec        # Earth‐centered response tensor
                    # h_ec        # Earth‐centered GW strain tensor
            #2. Explicit double contraction: By specifying axes=([0,1],[0,1]), it’s crystal-clear that we’re summing over both tensor indices (a double contraction), 
                #   rather than relying on the default behavior.
            
            #3. Removed unnecessary intermediate variables: just return detector response tensorProduct directly instead of assigning it to another variable

        
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

        Notes
        -----
        1. In the detector frame, the response tensor is
        ```
        D_det = ½ · diag(1, –1, 0)
        ```
        a (2-0) tensor.
        2. Transform D_det to Earth-centered frame:
        ```
        R_det_ec = change_basis_detector_to_ec(detector_angles)
        D_ec     = transform_2_0_tensor(D_det, R_det_ec)
        ```
        3. Transform D_ec into the GW frame:
        ```
        R_gw_ec = change_basis_gw_to_ec(source_angles)
        R_ec_gw = inv(R_gw_ec)
        D_gw    = transform_2_0_tensor(D_ec, R_ec_gw)
        ```
        4. The antenna patterns are then extracted as
        ```
        F₊ = D_gw[0,0] – D_gw[1,1]
        Fₓ = D_gw[0,1] + D_gw[1,0]
        ```

        Examples
        --------
        >>> det = [0.6, 1.2, 0.3]
        >>> src = [0.1, 2.3, 0.4]
        >>> Fp, Fc = beam_pattern_response_functions(det, src)
        >>> isinstance(Fp, float)
        True
        """
        
        #Original Implementation:
            # detector_response_tensor_detector_frame = np.array([[1/2,0,0],[0,-1/2,0],[0,0,0]])
            # transform_detector_ec = change_basis_detector_to_ec(detector_angles)
            # detector_response_tensor_earth_centered = transform_2_0_tensor(detector_response_tensor_detector_frame,transform_detector_ec)
            # transform_gw_ec = change_basis_gw_to_ec(source_angles)
            # transform_ec_gw = np.linalg.inv(transform_gw_ec)
            # detector_response_tensor_gw_frame = transform_2_0_tensor(detector_response_tensor_earth_centered,transform_ec_gw)
            # fplus = detector_response_tensor_gw_frame[0,0]-detector_response_tensor_gw_frame[1,1]
            # fcross = detector_response_tensor_gw_frame[0,1]+detector_response_tensor_gw_frame[1,0]
            # return [fplus, fcross] # I dont like this list return
            
        #Abid's optimized implementation:
        
        # Explanation of changes made:
        # 1. Changed variable names for clarity:
            # Reference:
            
                # D_det    # detector-frame response tensor  
                # R_det_ec # change-of-basis matrix detector→EC  
                # D_ec     # EC-frame response tensor  
                # R_gw_ec  # GW→EC change-of-basis  
                # R_ec_gw  # EC→GW inverse  
                # D_gw     # GW-frame response tensor  
        #2. Returns Tuple instead of List
            # This is more idiomatic in Python for fixed-size collections,
            # and it makes it clear that the two values are related (F₊ and Fₓ).
            # It also allows unpacking directly into F_plus, F_cross.
            
        #3. regular numpy vectorizing routine
        
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

        Notes
        -----
        1. Detector positions (in meters) in the Earth-centered frame are
        r_H = Rₑ · u_H,  r_L = Rₑ · u_L,
        where u_H/L = source_vector_from_angles(hanford_detector_angles/livingston_detector_angles)
        and Rₑ = earth_radius.
        2. The baseline vector from Hanford to Livingston is 
        b = r_L – r_H.
        3. The GW propagation direction (unit vector) is 
        p = –source_vector_from_angles(source_angles).
        4. The time delay is Δt = (p · b) / c, with c = speed_light.

        Examples
        --------
        >>> angles = [0.1, 1.2, 0.0]            # δ, α, ψ in radians
        >>> dt = time_delay_hanford_to_livingston(angles)
        >>> isinstance(dt, float)
        True
        """
        # Original Implementation:
            # hanford_z_vector_earth_centered = source_vector_from_angles(hanford_detector_angles)
            # livingston_z_vector_earth_centered = source_vector_from_angles(livingston_detector_angles)
            # position_vector_hanford_to_livingston = earth_radius * (livingston_z_vector_earth_centered - hanford_z_vector_earth_centered)
            # gw_source_vector = source_vector_from_angles(source_angles)
            # gw_z_vector_earth_centered = -1*gw_source_vector
            # return 1/speed_light*(np.dot(gw_z_vector_earth_centered,position_vector_hanford_to_livingston))
        
        # Abid's optimized implementation
        
        # Explanation of changes made:
            # 1. Changed variable names for clarity:
                # Reference:
                    # r_H  # Earth-centered position vector of Hanford
                    # r_L  # Earth-centered position vector of Livingston
                    # baseline  # Vector from Hanford to Livingston
                    # propagation_dir  # Unit vector in GW propagation direction
                    
            # 2. Explicit division by speed of light 
        
        # Earth-centered position vectors (m) of each site
        r_H = earth_radius * source_vector_from_angles(hanford_detector_angles)
        r_L = earth_radius * source_vector_from_angles(livingston_detector_angles)

        # Baseline from Hanford to Livingston
        baseline = r_L - r_H

        # GW propagation direction (unit vector) in Earth frame
        propagation_dir = -source_vector_from_angles(source_angles)

        # Return time delay (s)
        return np.dot(propagation_dir, baseline) / speed_light

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

        Notes
        -----
        - The half-window sample count is computed as
        `N = ceil(T_max * detector_sampling_rate)`.
        - The full time span covers 2·N samples symmetrically around zero.
        - Assumes perfect synchronization across sites.

        Examples
        --------
        >>> fs = 4096  # Hz
        >>> tau = 1.0  # s
        >>> delay = 0.01  # s
        >>> t = generate_network_time_array(tau, fs, delay)
        >>> t[0], t[-1]
        (-1.01, 0.999755859375)
        >>> len(t) == 2 * int(np.ceil((tau + delay) * fs))
        True
        """
        #Original Implementation:
            # time_sample_width = round((signal_lifetime + maximum_time_delay)*detector_sampling_rate,0)
            # all_times = 1/detector_sampling_rate*(np.arange(-time_sample_width,time_sample_width,1))
            # return all_times
            
        # Abid's optimized implementation:
            # Explanation of changes made:
                # 1. Used np.arange for cleaner time array generation
                # 2. Removed unnecessary rounding: np.arange handles floating-point precision well
                # 3. Used np.ceil to ensure we cover the full half-window duration ==> Using np.ceil ensures we cover the full time window 
                # (rounding up to the next sample) rather than a potentially truncated window from round. Converting explicitly to an integer (half_samples) 
                # makes it crystal-clear that we’re indexing sample counts, 
                # avoids floating-point endpoints in arange, and prevents off-by-one errors.
                
                # 4. Clearer variable names: T_max instead of time_sample_width for clarity on what it represents
                
                    # signal_lifetime + maximum_time_delay → T_max
                    # time_sample_width → half_samples (an integer count)
                    # all_times → time_array
                
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

        Notes
        -----
        - Q = √(ln 2) / signal_lifetime sets the Gaussian width.
        - We reuse A₊ and Bₓ for the 3rd and 4th GW modes respectively.
        - Fully vectorized: no explicit Python loops or temporary storage per sample.

        Examples
        --------
        >>> τ = 0.01                  # s
        >>> fs = np.linspace(-0.1, 0.1, 1001)
        >>> H = generate_oscillatory_terms(0.05, 150.0, fs, τ)
        >>> H.shape
        (1001, 4)
        
        """
        
        # Original Implementation:
            # number_time_samples = time_array.size
            # signal_q_value = m.sqrt(m.log(2))/signal_lifetime
            # oscillatory_terms = np.empty((number_time_samples,number_gw_modes))
            # for this_time_sample in range(number_time_samples) :
            #     this_time = time_array[this_time_sample]
            #     this_gaussian_term = m.exp(-1*signal_q_value**2*(this_time - time_delay)**2)
            #     this_cos_term = m.cos(2*np.pi*signal_frequency*(this_time - time_delay))
            #     this_sin_term = m.sin(2*np.pi*signal_frequency*(this_time - time_delay))
            #     this_cos_gauss_term = this_gaussian_term*this_cos_term
            #     this_sin_gauss_term = this_gaussian_term*this_sin_term
            #     these_oscillations = np.array([this_cos_gauss_term,this_sin_gauss_term,this_cos_gauss_term,this_sin_gauss_term])
            #     oscillatory_terms[this_time_sample] = these_oscillations
            # return oscillatory_terms
            
        # Abid's optimized implementation:
        # Explanation of changes made:
            # 1. Removed explicit Python loop over time samples: now fully vectorized using NumPy operations.
            # 2. Used NumPy's vectorized operations for Gaussian envelope and phase
            # 3. COncise variable names: Q for the Gaussian factor, dt for time differences, envelope and phase for the two main components.
            # 4. Used np.column_stack to create the final oscillatory terms array in one go, which is more efficient and readable (eliminates redundant arrays)
                # Avoids repeated allocations and clarifies that the four modes simply reuse the two computed waveforms.
            
            #5. Got rid of math module and used NumPy for all math operations, which is more efficient and consistent with the rest of the code.
            
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

    #    Original Implementation:

    # def generate_model_detector_responses(signal_frequency,signal_lifetime,detector_sampling_rate,gw_max_amps,number_amplitude_combinations,number_angular_samples) :
    

    
    #     time_array = generate_network_time_array(signal_lifetime,detector_sampling_rate,maximum_hanford_livingston_time_delay)
    #     number_time_samples = time_array.size
    #     hanford_oscillatory_terms = generate_oscillatory_terms(signal_lifetime,signal_frequency,time_array,0)
    #     model_amplitudes_array = generate_model_amplitudes_array(number_amplitude_combinations, gw_max_amps)
    #     model_angles_array = generate_model_angles_array(number_angular_samples)
        
    #     #Space Complexity issue
        
    #     model_detector_response_array = np.empty((number_angular_samples,number_amplitude_combinations,number_time_samples,number_detectors))
        
        
    #     for this_angle_set in range(number_angular_samples) :
    #         these_angles = model_angles_array[this_angle_set]
    #         [fplus_hanford,fcross_hanford] = beam_pattern_response_functions(hanford_detector_angles,these_angles)
    #         [fplus_livingston,fcross_livingston] = beam_pattern_response_functions(livingston_detector_angles,these_angles)
    #         hanford_livingston_time_delay = time_delay_hanford_to_livingston(these_angles)
    #         livingston_oscillatory_terms = generate_oscillatory_terms(signal_lifetime,signal_frequency,time_array,hanford_livingston_time_delay)
    #         for this_amplitude_combination in range(number_amplitude_combinations) :
    #             these_amplitudes = model_amplitudes_array[this_amplitude_combination]
    #             for this_sample_time in range(number_time_samples) :
    #                 model_detector_response_array[this_angle_set,this_amplitude_combination,this_sample_time,0] = np.dot(these_amplitudes,hanford_oscillatory_terms[this_sample_time]*[fplus_hanford,fplus_hanford,fcross_hanford,fcross_hanford])
    #                 model_detector_response_array[this_angle_set,this_amplitude_combination,this_sample_time,1] = np.dot(these_amplitudes,livingston_oscillatory_terms[this_sample_time]*[fplus_livingston,fplus_livingston,fcross_livingston,fcross_livingston])
    #     return [model_detector_response_array,model_angles_array]
        
        
    # Abid's optimized implementation:

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

        Notes
        -----
        1. A reference Hanford detector (zero delay) defines the “network” time grid.
        2. For each angle set:
        - Compute beam patterns F₊, Fₓ for Hanford & Livingston.
        - Weight the precomputed Hanford oscillatory terms by [F₊, Fₓ, F₊, Fₓ].
        - Compute Livingston oscillatory terms with its time delay, then weight similarly.
        3. Inner time loop is eliminated by vectorized dot products:
        responses[i, j, :, d] = (weighted_terms_d @ amplitude_vector).

        Examples
        --------
        >>> R, angles = generate_model_detector_responses(150.0, 0.05, 4096, 1e-21, 10, 20)
        >>> R.shape  # (20 angles, 10 amps, N times, 2 detectors)
        (20, 10, R.shape[2], 2)
        
        #Explanation of changes made:
        1. Replaced innermost time-sample loop with Vectorizing the dot over the entire time axis at once as it vastly 
            reduces Python-level overhead and makes the code both faster and more concise.
        2. Built pattern_H = [F₊ₕ, Fₓₕ, F₊ₕ, Fₓₕ] and applied it in one shot via broadcasting.
            Avoids recomputing or re-allocating the same four-element array inside the loops, and clarifies that both Hanford and Livingston use the same pattern structure.
        
        """
        # Abbreviate the input sizes
        n_amp = number_amplitude_combinations
        n_ang = number_angular_samples

        # Time grid for the network (reference detector delay = 0)
        time_array = generate_network_time_array(
            signal_lifetime,
            detector_sampling_rate,
            maximum_hanford_livingston_time_delay,
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

    #========================================================================= START OF DRIVER FUNCTIONS =========================================================================


    full_process_start_time = time.process_time()
    gw_frequency = 100
    gw_lifetime = 0.03
    detector_sampling_rate = ligo_detector_sampling_rate
    gw_max_amps = 1
    max_noise_amp = 0.1

    # Number of samples for angles and amplitudes
    number_angular_samples = 100
    number_amplitude_combinations = 100
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


if __name__ == "__main__":
    import cProfile
    cProfile.run('main()', filename='Draft01_profile.prof')
    print("Profiling complete. View results with: snakeviz Draft01_profile.prof")
