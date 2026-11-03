"""Load the canonical research YAML, also bundled verbatim into wheels."""
from pathlib import Path
import yaml
from .alignment.losses import METHODS, annealing_policy


def load_conditions(path=None):
    if path is None:
        # Editable/Colab checkout: editing the research YAML changes the plan.
        source = Path(__file__).resolve().parents[2] / "configs/experiments/conditions.yaml"
        path = source if source.is_file() else Path(__file__).with_name("conditions.yaml")
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or set(value) != set(METHODS):
        raise ValueError("conditions must declare the four research methods")
    for method, settings in value.items():
        if settings.get("method") != method:
            raise ValueError("condition method identifier mismatch")
        if method != "baseline" and settings.get("aggregation") != "worst_layer":
            raise ValueError("research conditions require worst_layer aggregation")
    if value["mask_distillation"].get("mask_type") != "top_p":
        raise ValueError("mask condition requires top_p")
    annealing_policy(value["annealed_distillation"])
    return value
