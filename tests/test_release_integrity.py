from __future__ import annotations

import json
import py_compile
import re
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_PARTS = {
    ".git",
    ".idea",
    ".venv",
    "__pycache__",
    "outputs",
    "logs",
    "cache",
}


def test_all_numbered_scripts_are_present() -> None:
    scripts = sorted((ROOT / "scripts").glob("[0-9][0-9]_*.py"))
    numbers = [int(path.name[:2]) for path in scripts]
    assert numbers == list(range(1, 41)) + [42, 43]


def test_all_scripts_compile() -> None:
    for path in sorted((ROOT / "scripts").glob("*.py")):
        py_compile.compile(str(path), doraise=True)


def test_experiment_configuration_matches_release() -> None:
    config = yaml.safe_load(
        (ROOT / "configs" / "experiment.yaml").read_text(encoding="utf-8-sig")
    )
    assert config["project"]["version"] == "1.0.0"
    assert config["project"]["seeds"] == [42, 123, 2024, 3407, 7777]
    assert config["modalities"]["text"]["fields"] == [
        "description",
        "location",
    ]
    assert config["modalities"]["text"]["max_length"] == 64
    assert len(config["modalities"]["graph"]["feature_relations"]) == 14
    assert config["modalities"]["graph"]["feature_dimension"] == 28
    assert config["modalities"]["metadata"]["feature_dimension"] == 10
    assert config["evaluation"]["bootstrap_replicates"] == 10000
    assert config["routing"]["mapping"]["TGM"] == "text_metadata"


def test_input_manifest_is_portable() -> None:
    manifest = json.loads(
        (ROOT / "manifests" / "input_manifest.json").read_text(
            encoding="utf-8-sig"
        )
    )
    expected = {
        "user_file": (
            "user.json",
            "53faeeb165cea4aa65a19b0b3a37c880e5c33e42d64dca53b273cc3a62f98fcd",
        ),
        "label_file": (
            "label.csv",
            "02062e53d1d05128ca554d792a826aa8200c855e294b9520d1697c96b7632efc",
        ),
        "split_file": (
            "split.csv",
            "f68c890c824f6f63141ed8228eb412ccb18e127f645761caaf24ed56224c0fd9",
        ),
        "edge_file": (
            "edge.csv",
            "ab6ca64a4e99acabde3ccf6c1112cb36174c298f56bbf58508976ead69475c19",
        ),
    }
    for key, (filename, digest) in expected.items():
        assert manifest[key]["path"] == filename
        assert manifest[key]["sha256"].lower() == digest


def test_final_evidence_audit_passed() -> None:
    report = json.loads(
        (ROOT / "evidence" / "final_evidence_audit.json").read_text(
            encoding="utf-8-sig"
        )
    )
    assert report["status"] == "PASS"


def test_publishable_text_contains_no_absolute_windows_paths() -> None:
    suffixes = {".py", ".toml", ".yaml", ".yml", ".json", ".md", ".cff", ".txt"}
    absolute_windows_path = re.compile(r"[A-Za-z]:\\")
    violations: list[str] = []

    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(part in EXCLUDED_PARTS or part.endswith(".egg-info") for part in path.parts):
            continue
        if path.name == "config.local.yaml" or path.suffix.lower() not in suffixes:
            continue
        text = path.read_text(encoding="utf-8-sig", errors="replace")
        if absolute_windows_path.search(text):
            violations.append(str(path.relative_to(ROOT)))

    assert not violations, violations


def test_citation_and_license_exist() -> None:
    citation = yaml.safe_load(
        (ROOT / "CITATION.cff").read_text(encoding="utf-8-sig")
    )
    assert citation["version"] == "1.1.0"
    assert citation["doi"] == "10.5281/zenodo.24199691"
    assert citation["license"] == "MIT"
    assert (ROOT / "LICENSE").is_file()

def test_overlap_sensitivity_release_evidence() -> None:
    evidence_dir = ROOT / "evidence" / "twibot20_overlap_sensitivity"
    required = [
        "overlap_audit.json",
        "split_overlap_crosstab.csv",
        "disjoint_seed_results.csv",
        "disjoint_summary.csv",
        "disjoint_paired_inference.csv",
    ]
    for name in required:
        assert (evidence_dir / name).is_file(), name

    assert (
        ROOT
        / "evidence"
        / "figures"
        / "fig07_twibot20_external_validation.pdf"
    ).is_file()

    assert not (
        evidence_dir / "twibot20_test_overlap_records.csv"
    ).exists()
