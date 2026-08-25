from __future__ import annotations

import json
from pathlib import Path

from affectclaim.cli import main

ROOT = Path(__file__).resolve().parents[1]


def test_cli_writes_structured_records(tmp_path: Path, capsys: object) -> None:
    del capsys
    output = tmp_path / "records.json"
    result = main(
        [
            "run",
            "--manifest",
            str(ROOT / "examples/sample_manifest.json"),
            "--config",
            str(ROOT / "configs/default.json"),
            "--output",
            str(output),
        ]
    )

    records = json.loads(output.read_text(encoding="utf-8"))
    assert result == 0
    assert len(records) == 3
    assert {record["action"] for record in records} == {"direct", "qualified", "abstain"}
