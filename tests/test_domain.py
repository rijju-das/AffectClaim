from __future__ import annotations

import pytest

from affectclaim.domain.models import AffectDistribution, BoundingBox


def test_bounding_box_rejects_invalid_geometry() -> None:
    with pytest.raises(ValueError, match="positive width"):
        BoundingBox(0.5, 0.1, 0.5, 0.9)


def test_affect_distribution_requires_normalisation() -> None:
    with pytest.raises(ValueError, match="sum to 1.0"):
        AffectDistribution({"positive": 0.4, "unclear": 0.4})


def test_affect_distribution_reports_top_label() -> None:
    distribution = AffectDistribution({"pleased": 0.7, "unclear": 0.3})
    assert distribution.top_label == "pleased"
    assert distribution.top_probability == pytest.approx(0.7)
