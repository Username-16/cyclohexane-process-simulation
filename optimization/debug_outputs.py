# debug_outputs.py  — run once, then delete
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from optimization.simulation_adapter import create_flowsheet_evaluator

evaluator = create_flowsheet_evaluator()
result = evaluator({
    "stage_volume_1": 3.0, "stage_volume_2": 3.0, "stage_volume_3": 3.0,
    "stage_volume_4": 3.0, "stage_volume_5": 3.0, "stage_volume_6": 3.0,
    "h2_recycle_fraction": 0.85,     "liquid_recycle_fraction": 0.35,
    "distillate_recycle_fraction": 0.95, "h2_benzene_feed_ratio": 3.5,
    "distillate_LK_mole_frac": 0.85, "distillate_HK_mole_frac": 0.15,
    "bottoms_LK_mole_frac": 0.05,    "bottoms_HK_mole_frac": 0.90,
    "reflux_ratio_factor": 3.0,
})

def flatten(d, prefix=""):
    """Recursively flatten nested dict to dot-notation keys."""
    rows = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            rows.update(flatten(v, key))
        elif isinstance(v, (int, float)):
            rows[key] = v
    return rows

flat = flatten(result)
print(f"\n{'Key':<55} {'Value'}")
print("─" * 75)
for k, v in sorted(flat.items()):
    print(f"  {k:<53} {v:.6g}")
