"""Materialize the user-approved v24 profile from completed immutable research evidence."""
import argparse
import json
from pathlib import Path

from backend.core import ROOT, MODEL, sha256
from backend.frozen_encoder import digest


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n")


def preserve(path, value):
    if path.exists() and read(path) != value:
        raise ValueError(f"Refusing to replace evidence: {path}")
    write(path, value)


def promote(research):
    source = research / "OSNet-AIN-x1.0/variant_24_dual_role/runs/dual_role_v1"
    manifest, results, profile = (read(source/n) for n in ("manifest.json", "results.json", "dual_role_profile.json"))
    if (results["status"] != "complete" or results["signature"] != digest(manifest)
            or not results["protected_unchanged"] or results["optimizer_updates"] or results["bn_updates"]
            or results["threshold_fit"] or profile != manifest["dual_role_plan"]["profile"]):
        raise ValueError("Need the complete, unchanged and untrained v24")
    for receipt in source.glob("tasks/*/complete.json"):
        value = read(receipt)
        if value["signature"] != results["signature"]:
            raise ValueError("Task signature changed")
        for name, expected in value["artifacts"].items():
            path = (source/name).resolve()
            if not path.is_relative_to(source.resolve()) or sha256(path) != expected:
                raise ValueError("Source task artifact changed")
    for name, expected in manifest["source_sha256"].items():
        if sha256(research/name) != expected:
            raise ValueError(f"Research implementation changed: {name}")
    for name, expected in manifest["protected"].items():
        if sha256(name) != expected:
            raise ValueError(f"Protected input changed: {name}")
    current = results["evaluations"]["fresh_validation"]
    if (not current["fresh_image_inference"] or not current["parity"]["bit_exact_vectors"]
            or not current["parity"]["v22_csv_byte_parity"] or results["comparison"]["mAP_delta"] != 0
            or results["comparison"]["quality_points_delta"] <= 0):
        raise ValueError("Completed v24 does not meet its approved promotion basis")
    profiles = read(ROOT/"models/profiles.json")
    r1 = profiles["RC_R1_equal3_v18"]
    if sha256(ROOT/r1["bundle"]) != profile["r1_bundle"]["sha256"] or sha256(MODEL) != profile["mvp"]["sha256"]:
        raise ValueError("Application weights differ from verified v24")
    evidence_path = ROOT/"docs/V24_QUALITY_EVIDENCE.json"
    preserve(evidence_path, {"source_results_sha256": sha256(source/"results.json"), "results": results,
                             "source_signature": results["signature"], "profile": profile})
    report = {"model": "MVP_dual_role_v24", "threshold": current["threshold"],
              "validation": {"mAP_at_10": current["ranking"]["mAP@10"], "candidate_F1": current["candidates"]["F1"],
                             "TNR": current["candidates"]["TNR"], "candidate_C": current["candidates"]["C"],
                             "known_queries": current["ranking"]["n_scored"], "unknown_queries": current["ranking"]["n_openset_excluded"]},
              "provenance": {"evidence": str(evidence_path.relative_to(ROOT)), "evidence_sha256": sha256(evidence_path),
                             "kind": "observed development validation, not hidden test", "ranking": "MVP_legacy",
                             "candidate": "RC_R1_equal3_v18", "source_signature": results["signature"]}}
    report_path = ROOT/"models/MVP_dual_role_v24.metrics.json"
    preserve(report_path, report)
    profiles["MVP_dual_role_v24"] = {"kind": "dual_role", "size": [208, 256], "dimension": 2048,
        "bundle": r1["bundle"], "bundle_sha256": r1["bundle_sha256"], "ranking": "legacy", "candidate_policy": "raw_top1",
        "lambda": .5, "metrics": str(report_path.relative_to(ROOT)), "metrics_sha256": sha256(report_path)}
    write(ROOT/"models/profiles.json", profiles)
    previous = read(ROOT/"release_decision.json")
    if previous["active_profile"] != "MVP_dual_role_v24":
        preserve(ROOT/"docs/RELEASE_DECISION_BEFORE_V24.json", previous)
    write(ROOT/"release_decision.json", {
        "active_profile": "MVP_dual_role_v24", "rollback_profile": "MVP_legacy", "candidate_profile": "RC_R1_equal3_v18",
        "reserve_profile": "RC_R1_single_v18", "promoted": True, "authority": "explicit user approval after completed v24",
        "quality_evidence": str(evidence_path.relative_to(ROOT)), "quality_evidence_sha256": sha256(evidence_path),
        "official_submission_ready": False, "local_acceptance_passed": False,
        "verification_scope": "quality verified; new application runtime verification recorded separately",
        "previous_release_decision": "docs/RELEASE_DECISION_BEFORE_V24.json", "open_checks": previous["open_checks"]})
    print("MVP_dual_role_v24 promoted; rollback MVP_legacy; no weights or thresholds changed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research", required=True, type=Path)
    promote(parser.parse_args().research.resolve())
