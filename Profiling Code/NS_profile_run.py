import math as m
import numpy as np
import time
def main():
    

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

    def transform_2_0_tensor(matrix, change_basis_matrix) : 
        contravariant_transformation_matrix = np.linalg.inv(change_basis_matrix)
        partial_transformation = np.einsum('ki,kl->il', contravariant_transformation_matrix, matrix)
        return np.einsum('lj,il->ij', contravariant_transformation_matrix, partial_transformation)

    def transform_1_1_tensor(matrix, change_basis_matrix) : 
        contravariant_transformation_matrix = np.linalg.inv(change_basis_matrix)
        partial_transformation = np.einsum('ik,kl->il', change_basis_matrix, matrix)
        return np.einsum('lj,il->ij', contravariant_transformation_matrix, partial_transformation)

    def transform_0_2_tensor(matrix, change_basis_matrix) :
        partial_transformation = np.einsum('ik,kl->il', change_basis_matrix, matrix)
        return np.einsum('jl,il->ij', change_basis_matrix, partial_transformation)

    def source_vector_from_angles(angles) :
        [first, second, third] = angles
        initial_source_vector = np.array([m.cos(first)*m.cos(second),m.cos(first)*m.sin(second),m.sin(first)])
        return initial_source_vector

    def change_basis_gw_to_ec(source_angles) :
        [declination,right_ascension,polarization] = source_angles
        initial_source_vector = source_vector_from_angles(source_angles)
        initial_gw_z_vector_earth_centered = -1*initial_source_vector
        initial_gw_y_vector_earth_centered = np.array([-1*m.sin(declination)*m.cos(right_ascension),-1*m.sin(declination)*m.sin(right_ascension),m.cos(declination)])
        initial_gw_x_vector_earth_centered = np.cross(initial_gw_z_vector_earth_centered,initial_gw_y_vector_earth_centered)
        transpose_gw_vecs_ec = np.array([initial_gw_x_vector_earth_centered,initial_gw_y_vector_earth_centered,initial_gw_z_vector_earth_centered])
        initial_gw_vecs_ec = np.transpose(transpose_gw_vecs_ec)
        polarization_rotation_matrix = np.array([[m.cos(polarization),-1*m.sin(polarization),0],[m.sin(polarization),m.cos(polarization),0],[0,0,1]])
        contravariant_transformation_matrix = np.matmul(polarization_rotation_matrix,initial_gw_vecs_ec)
        change_basis_matrix = np.linalg.inv(contravariant_transformation_matrix)
        return change_basis_matrix

    def gravitational_wave_ec_frame(source_angles,tt_amplitudes) :
        [hplus,hcross] = tt_amplitudes
        gwtt = np.array([[hplus,hcross,0],[hcross,-hplus,0],[0,0,0]])
        transformation = change_basis_gw_to_ec(source_angles)
        return transform_0_2_tensor(gwtt,transformation)

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

    def detector_response(detector_angles, source_angles, tt_amplitudes) :
        detector_response_tensor_detector_frame = np.array([[1/2,0,0],[0,-1/2,0],[0,0,0]])
        transform_detector_to_ec = change_basis_detector_to_ec(detector_angles)
        detector_response_tensor_earth_centered = transform_2_0_tensor(detector_response_tensor_detector_frame,transform_detector_to_ec)
        gw_earth_centered = gravitational_wave_ec_frame(source_angles,tt_amplitudes)
        detector_response = np.tensordot(gw_earth_centered,detector_response_tensor_earth_centered)
        return detector_response

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

    def time_delay_hanford_to_livingston(source_angles) :
        hanford_z_vector_earth_centered = source_vector_from_angles(hanford_detector_angles)
        livingston_z_vector_earth_centered = source_vector_from_angles(livingston_detector_angles)
        position_vector_hanford_to_livingston = earth_radius * (livingston_z_vector_earth_centered - hanford_z_vector_earth_centered)
        gw_source_vector = source_vector_from_angles(source_angles)
        gw_z_vector_earth_centered = -1*gw_source_vector
        return 1/speed_light*(np.dot(gw_z_vector_earth_centered,position_vector_hanford_to_livingston))

    def generate_network_time_array(signal_lifetime, detector_sampling_rate, maximum_time_delay) :
        time_sample_width = round((signal_lifetime + maximum_time_delay)*detector_sampling_rate,0)  
        all_times = 1/detector_sampling_rate*(np.arange(-time_sample_width,time_sample_width,1))
        return all_times

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

    def generate_model_angles_array(number_angular_samples) : 
        model_angles_array = np.empty((number_angular_samples,number_source_angles))
        for this_angle_set in range(number_angular_samples) :
            for this_source_angle in range(number_source_angles) :
                if this_source_angle == 0 :
                    model_angles_array[this_angle_set,this_source_angle] = (np.random.rand(1) - 1/2)*np.pi
                else :
                    model_angles_array[this_angle_set,this_source_angle] = np.random.rand(1)*2*np.pi
        return model_angles_array

    def generate_model_amplitudes_array(number_amplitude_combinations, gw_max_amps) :
        model_amplitudes_array = np.empty((number_amplitude_combinations,number_gw_modes))
        for this_amplitude_combination in range(number_amplitude_combinations) :
            new_amplitudes = np.random.rand(number_gw_modes)*gw_max_amps
            model_amplitudes_array[this_amplitude_combination] = new_amplitudes 
        return model_amplitudes_array

    def generate_model_detector_responses(signal_frequency,signal_lifetime,detector_sampling_rate,gw_max_amps,number_amplitude_combinations,number_angular_samples) :
        time_array = generate_network_time_array(signal_lifetime,detector_sampling_rate,maximum_hanford_livingston_time_delay)
        number_time_samples = time_array.size
        hanford_oscillatory_terms = generate_oscillatory_terms(signal_lifetime,signal_frequency,time_array,0)
        model_amplitudes_array = generate_model_amplitudes_array(number_amplitude_combinations, gw_max_amps)
        model_angles_array = generate_model_angles_array(number_angular_samples)
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

    def generate_noise_array(max_noise_amp,number_time_samples) :
        noise_array = np.random.rand(number_time_samples)*max_noise_amp
        return noise_array

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
        real_angles_array = np.empty((number_angular_samples,number_source_angles))
        real_detector_response_array = np.empty((number_angular_samples,number_amplitude_combinations,number_time_samples,number_detectors))
        for this_angle_set in range(number_angular_samples) :
            real_angles_array[this_angle_set] = real_angles
            for this_amplitude_combination in range(number_amplitude_combinations) :
                real_detector_response_array[this_angle_set,this_amplitude_combination] = small_detector_response_array
        return [real_detector_response_array,real_angles_array]

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

    full_process_start_time = time.process_time()
    gw_frequency = 100
    gw_lifetime = 0.03
    detector_sampling_rate = ligo_detector_sampling_rate
    gw_max_amps = 1
    max_noise_amp = 0.1
    number_angular_samples = 100
    number_amplitude_combinations = 100
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
    cProfile.run('main()', filename='Original_profile.prof')
    print("Profiling complete. View results with: snakeviz Original_profile.prof")
