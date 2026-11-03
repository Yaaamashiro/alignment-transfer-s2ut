"""Synthetic coefficient contracts for tests of unrelated native IO/parity.

Real tau/gradient estimation is exercised in test_distillation_native. These
small values are explicit fixtures, never production defaults or fallbacks.
"""
from alignment_transfer_s2ut import benchmark as b
from alignment_transfer_s2ut.alignment.calibration import binding,runtime_identity
from alignment_transfer_s2ut.io import digest,write_json
from pathlib import Path


def fixture_plan(baseline,output,method="distillation",prior_type="t2tt",manifest=None,units=None,teacher=None,
                 anneal_steps=None,beta=None,lambda_align=None,smoothing=0.,all_layers=False):
    if method=="baseline":return b.make_plan(baseline,output,method)
    plan=b.make_plan(baseline,output,method,prior_type,manifest,units,teacher,all_layers=all_layers,for_calibration=True)
    value=dict(format="alignment-calibration-v2",tau_calibration_unit="microbatch",runtime=runtime_identity(),binding=binding(plan),implementation_hash=plan["implementation_hash"],
        lambda_kl=1.,lambda_mask=1.,tau_kl=.4,tau_mask=.4,selected_top_p=.8,synthetic_test_fixture=True)
    value["artifact_hash"]=digest(value)
    path=Path(output).with_suffix(".calibration.json");write_json(path,value,overwrite=True)
    plan=b.make_plan(baseline,output,method,prior_type,manifest,units,teacher,all_layers=all_layers,calibration=path)
    if beta==0:  # Explicit zero-supervision raw-forward parity test, not a calibration.
        plan["transfer"]["calibrated"]["lambda_kl"]=0.
        plan["transfer"]["calibrated"]["lambda_mask"]=0.
        plan["contract_hash"]=digest({k:v for k,v in plan.items() if k!="contract_hash"})
    return plan
