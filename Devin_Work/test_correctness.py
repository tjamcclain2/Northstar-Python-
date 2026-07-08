import numpy as np
import cupy as cp
import devin_optimized as op
import northstar_og_abid as og

def compare(func_name, cpu_out, gpu_out, rtol = 1e-5, atol = 1e-8):
    gpu_on_host = cp.asnumpy(gpu_out)
    cpu = np.asarray(cpu_out)
    ok = np.allclose(cpu, gpu_on_host, rtol=rtol, atol=atol)
    max_diff = np.max(np.abs(cpu - gpu_on_host))
    print(f"{func_name:40s} {'PASS' if ok else 'FAIL'} max|diff| = {max_diff:.2e}")
'''
Layer 0 (leaves, test first):
    source_vector_from_angles
    transform_2_0_tensor
    generate_oscillatory_terms

Layer 1 (depend only on layer 0):
    change_basis_gw_to_ec          (needs source_vector)
    change_basis_detector_to_ec    (needs source_vector)
    time_delay_hanford_to_livingston (needs source_vector)

Layer 2 (depend on layer 1):
    beam_pattern_response_functions (needs both change_basis fns + transform_2_0)

Layer 3 (top):
    generate_model_detector_responses (needs beam_pattern, time_delay, osc terms)
'''
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
compare("change_basis_detector_to_ec", og.hanford_detector_angles, op.hanford_detector_angles)
compare("change_basis_detector_to_ec", og.livingston_detector_angles, op.livingston_detector_angles)

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
