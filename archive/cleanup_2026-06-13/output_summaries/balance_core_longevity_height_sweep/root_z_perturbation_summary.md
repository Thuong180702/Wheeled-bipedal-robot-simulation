# Balance-Core Study Summary

- Cases: 9
- Passed: 3
- Failed: 6
- Invalid initial setup: 0

**Conclusion:** root_z_perturbation_robustness_narrow: pass_1000=[-0.01,0.00], fail_1000=[-0.03,-0.02,+0.01,+0.02,+0.03], pass_5000=[-0.01], fail_5000=[0.00]

## Cases

- **root_z_minus_030mm_1000** [FAIL] type=root_z_perturbation duration=1000 actual=1000 setup_valid=True failure_mode=F2.1
- **root_z_minus_020mm_1000** [FAIL] type=root_z_perturbation duration=1000 actual=1000 setup_valid=True failure_mode=F2.1
- **root_z_minus_010mm_1000** [PASS] type=root_z_perturbation duration=1000 actual=1000 setup_valid=True failure_mode=None
- **root_z_plus_000mm_1000** [PASS] type=root_z_perturbation duration=1000 actual=1000 setup_valid=True failure_mode=None
- **root_z_plus_010mm_1000** [FAIL] type=root_z_perturbation duration=1000 actual=1000 setup_valid=True failure_mode=F1.2
- **root_z_plus_020mm_1000** [FAIL] type=root_z_perturbation duration=1000 actual=1000 setup_valid=True failure_mode=F1.2
- **root_z_plus_030mm_1000** [FAIL] type=root_z_perturbation duration=1000 actual=1000 setup_valid=True failure_mode=F1.2
- **root_z_minus_010mm_5000** [PASS] type=root_z_perturbation duration=5000 actual=5000 setup_valid=True failure_mode=None
- **root_z_plus_000mm_5000** [FAIL] type=root_z_perturbation duration=5000 actual=5000 setup_valid=True failure_mode=failed_validation
