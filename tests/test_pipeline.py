from __future__ import annotations

from pathlib import Path

from affectclaim.adapters.manifest import ManifestAdapter
from affectclaim.config import AppConfig
from affectclaim.domain.models import ClaimAction
from affectclaim.factory import build_manifest_pipeline

ROOT = Path(__file__).resolve().parents[1]


def test_sample_manifest_exercises_all_reporting_actions() -> None:
    adapter = ManifestAdapter.from_json(ROOT / "examples/sample_manifest.json")
    pipeline = build_manifest_pipeline(adapter, AppConfig.from_json(ROOT / "configs/default.json"))

    records = [record for sample in adapter.samples() for record in pipeline.analyse(sample)]

    assert [record.action for record in records] == [
        ClaimAction.DIRECT,
        ClaimAction.QUALIFIED,
        ClaimAction.ABSTAIN,
    ]
    assert records[0].evidence.necessity > records[1].evidence.necessity
    assert records[2].realised_text == "A person wearing a hat is riding a horse."


def test_claim_record_is_json_serialisable() -> None:
    import json

    adapter = ManifestAdapter.from_json(ROOT / "examples/sample_manifest.json")
    pipeline = build_manifest_pipeline(adapter, AppConfig.from_json(ROOT / "configs/default.json"))
    record = pipeline.analyse(adapter.samples()[0])[0]

    encoded = json.dumps(record.to_dict())
    assert '"action": "direct"' in encoded
    assert '"family": "face"' in encoded
