"""Rule-based educational advice from the CED estimate and its dominant SHAP factors.

Messages live in config/recommendations.yaml so a midwife or nutritionist can
edit them without touching code. A factor message is shown only when that
factor pushed this woman's CED estimate up, and the wording always describes
the model's estimate, never a cause. The disclaimer is always included.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Advice:
    """Messages for the advice panel."""

    status_message: str
    hb_message: str
    driver_messages: list[str] = field(default_factory=list)
    disclaimer: str = ""
    draft_banner: str | None = None

    def all_messages(self) -> list[str]:
        return [self.status_message, *self.driver_messages, self.hb_message, self.disclaimer]


def build_advice(ced_probability: float, threshold: float, contributions: Sequence[Mapping[str, Any]],
                 rec_cfg: Mapping[str, Any]) -> Advice:
    """Advice for one woman.

    ``contributions`` are the CED SHAP contributions ordered by absolute size,
    each with ``feature`` and ``shap``.
    """
    status = rec_cfg["ced_status"]["above" if ced_probability >= threshold else "below"]
    drivers = rec_cfg.get("drivers") or {}
    limit = int(rec_cfg.get("max_driver_messages", 3))
    positive = sorted((c for c in contributions if c["shap"] > 0 and drivers.get(c["feature"])),
                      key=lambda c: -c["shap"])
    review = rec_cfg.get("review", {})
    banner = None if review.get("status") == "reviewed" and review.get("reviewed_by") else review.get("banner")
    return Advice(
        status_message=" ".join(status.split()),
        hb_message=" ".join(rec_cfg["hb_always"].split()),
        driver_messages=[" ".join(drivers[c["feature"]].split()) for c in positive[:limit]],
        disclaimer=" ".join(rec_cfg["disclaimer"].split()),
        draft_banner=banner,
    )
