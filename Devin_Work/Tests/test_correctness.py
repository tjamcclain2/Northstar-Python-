import numpy as np
import cupy as cp
import devin_optimized_testing as op
import abid_testing as og

def compare(func_name, cpu_out, gpu_out, rtol = 1e-3, atol = 1e-5):
    gpu_on_host = cp.asnumpy(gpu_out)
    cpu = np.asarray(cpu_out)
    ok = np.allclose(cpu, gpu_on_host, rtol=rtol, atol=atol)
    max_diff = np.max(np.abs(cpu - gpu_on_host))
    print(f"{func_name:40s} {'PASS' if ok else 'FAIL'} max|diff| = {max_diff:.2e}")

angle_grid_cpu = np.array([
    [ 0.30,  1.20,  0.70],
    [-0.85,  4.50,  2.10],
    [ 0.10,  3.00,  5.20],
])
angle_grid_gpu = cp.asarray(angle_grid_cpu)

#source_vector_from_angles
initial_source_vec_cpu =np.stack([og.source_vector_from_angles(r) for r in angle_grid_cpu])
initial_source_vec_gpu = op.source_vector_from_angles(angle_grid_gpu)
compare("source_vector_from_angles", initial_source_vec_cpu, initial_source_vec_gpu)

#transform_2_0_tensor
arm_cpu = np.array([
    [0.5,  0.0, 0.0],
    [0.0, -0.5, 0.0],
    [0.0,  0.0, 0.0],
])
arm_gpu = cp.asarray(arm_cpu)
c, s = np.cos(0.6), np.sin(0.6)
cbm_cpu = np.array([
    [ c, -s, 0.0],
    [ s,  c, 0.0],
    [0.0, 0.0, 1.0],
])
cbm_gpu = cp.asarray(cbm_cpu)
op_out_2_0 = op.transform_2_0_tensor(arm_gpu,cbm_gpu)
og_out_2_0 = og.transform_2_0_tensor(arm_cpu,cbm_cpu)
compare("transform_2_0_tensor", og_out_2_0, op_out_2_0)

#generate_oscillatory_terms
signal_lifetime = 0.03
signal_f = 100.0
time_array_cpu = np.linspace(-0.05,0.05,10)
time_array_gpu = cp.asarray(time_array_cpu)
delay = 0.02
og_osc_terms = og.generate_oscillatory_terms(signal_lifetime, signal_f, time_array_cpu, delay)
op_osc_terms = op.generate_oscillatory_terms(signal_lifetime, signal_f, time_array_gpu, delay)
compare("generate_oscillatory_terms", og_osc_terms, op_osc_terms)

#change_basis_gw_to_ec
og_out_gw_to_ec = np.stack([og.change_basis_gw_to_ec(r) for r in angle_grid_cpu])
op_out_gw_to_ec = op.change_basis_gw_to_ec(angle_grid_gpu)
compare("change_basis_gw_to_ec", og_out_gw_to_ec, op_out_gw_to_ec)

#change_basis_detector_to_ec
compare("change_basis_detector_to_ec", og.change_basis_detector_to_ec(og.hanford_detector_angles), op.change_basis_detector_to_ec(op.hanford_detector_angles))
compare("change_basis_detector_to_ec",  og.change_basis_detector_to_ec(og.livingston_detector_angles), op.change_basis_detector_to_ec(op.livingston_detector_angles))

#time_delay_hanford_to_livingston
delay_cpu = np.stack([og.time_delay_hanford_to_livingston(i) for i in angle_grid_cpu])
delay_gpu = op.time_delay_hanford_to_livingston(angle_grid_gpu)
compare("time_delay_hanford_to_livingston", delay_cpu, delay_gpu)

#beam_pattern_response_functions 
og_cpu_pairs = [og.beam_pattern_response_functions(og.hanford_detector_angles, r) for r in angle_grid_cpu]
op_fp, op_fx = op.beam_pattern_response_functions(op.hanford_detector_angles,angle_grid_gpu)
og_fp = [i[0] for i in og_cpu_pairs]
og_fx = [i[1] for i in og_cpu_pairs] 
compare("beam pattern Fplus", og_fp, op_fp)
compare("beam pattern Fcross", og_fx, op_fx)

#generate_model_detector_responses
gw_frequency=100
gw_lifetime=0.03
detector_sampling_rate=op.LIGO_DETECTOR_SAMPLING_RATE
gw_max_amps=1
number_angular_samples=100
number_amplitude_combinations=100
ang_cpu = og.generate_model_angles_array(number_angular_samples)
amp_cpu = og.generate_model_amplitudes_array(number_amplitude_combinations, gw_max_amps)
amp_gpu = cp.asarray(amp_cpu, dtype=cp.float32)
ang_gpu = cp.asarray(ang_cpu, dtype=cp.float32)
m_resp_cpu, m_ang_cpu = og.generate_model_detector_responses(amp_cpu, ang_cpu, gw_frequency, gw_lifetime, detector_sampling_rate, number_amplitude_combinations, number_angular_samples)
m_resp_gpu, m_ang_gpu = op.generate_model_detector_responses(amp_gpu, ang_gpu, gw_frequency, gw_lifetime, detector_sampling_rate, number_amplitude_combinations, number_angular_samples)
compare("model response", m_resp_cpu, m_resp_gpu)
compare("model angles", m_ang_cpu, m_ang_gpu)

#generate_real_detector_responses
max_noise_amp=0
ang_r_cpu = og.generate_model_angles_array(1)
amp_r_cpu = og.generate_model_amplitudes_array(1, gw_max_amps)
ang_r_gpu = cp.asarray(ang_r_cpu, dtype=cp.float32)
amp_r_gpu = cp.asarray(amp_r_cpu, dtype=cp.float32)
r_resp_cpu, r_ang_cpu = og.generate_real_detector_responses(amp_r_cpu[0],ang_r_cpu[0],gw_frequency,gw_lifetime,detector_sampling_rate,number_amplitude_combinations,number_angular_samples,max_noise_amp)
r_resp_gpu, r_ang_gpu = op.generate_real_detector_responses(amp_r_gpu[0],ang_r_gpu[0],gw_frequency,gw_lifetime,detector_sampling_rate,number_amplitude_combinations,number_angular_samples,max_noise_amp)
compare("real response", r_resp_cpu, r_resp_gpu)
compare("real angles", r_ang_cpu, r_ang_gpu)

#get_best_fit_angle_deltas
res_cpu = og.get_best_fit_angles_deltas(r_resp_cpu, r_ang_cpu, m_resp_cpu, m_ang_cpu)
res_gpu = op.get_best_fit_angles_deltas(r_resp_gpu, r_ang_gpu, m_resp_gpu, m_ang_gpu)
deltas_cpu = res_cpu[0:3]
deltas_gpu = res_gpu[0]
for i, name in enumerate(["oracle","single","weighted"]):
    compare(f"[{name}]", deltas_cpu[i], deltas_gpu[i])


