"""Materialize the explicitly approved, calibration-selected v25 without new weights."""
import argparse
from pathlib import Path

from backend.core import ROOT, MODEL, sha256
from backend.frozen_encoder import digest
from tools.promote_dual_role import read, write, preserve

PROFILE = "MVP_fusion_v25"
SELECTED = {"name": "r1w50_k20_q3_l50", "k1": 20, "k2": 3, "lambda": .5, "r1_weight": .5}


def promote(research):
    source = research/"OSNet-AIN-x1.0/variant_25_map_search/runs/map_search_v1"
    manifest, results, selection = (read(source/n) for n in ("manifest.json", "results.json", "frozen_selection.json"))
    if (manifest["version"] != 25 or results["status"] != "complete"
            or results["signature"] != digest(manifest) or not results["protected_unchanged"]
            or results["optimizer_updates"] or results["encoder_forwards"] or results["threshold_fit"]
            or selection != results["selection"] or selection["selected"] != SELECTED
            or selection["selection_split"] != "calibration" or selection["signature"] != results["signature"]):
        raise ValueError("Need completed, unchanged v25 and the approved frozen selection")
    specs = manifest["map_search_plan"]["systems"]
    if len(specs) != 84 or set(results["calibration"]) != {s["name"] for s in specs}:
        raise ValueError("Incomplete calibration grid")
    if max(specs, key=lambda s: results["calibration"][s["name"]]["ranking"]["mAP@10"]) != SELECTED:
        raise ValueError("Winner differs from calibration selection")
    if selection["evaluations"] != [specs[0], SELECTED] or set(results["evaluations"]) != {s["name"] for s in selection["evaluations"]}:
        raise ValueError("Unexpected validation selection")
    receipts = list(source.glob("tasks/*/complete.json"))
    if len(receipts) != 86:
        raise ValueError("Incomplete task receipts")
    for receipt in receipts:
        value = read(receipt)
        if value["signature"] != results["signature"] or not value["artifacts"]:
            raise ValueError("Task signature changed")
        for name, expected in value["artifacts"].items():
            path = (source/name).resolve()
            if not path.is_relative_to(source.resolve()) or sha256(path) != expected:
                raise ValueError("Source artifact changed")
    for split, key in (("calibration", "calibration"), ("validation", "evaluations")):
        for name, result in results[key].items():
            if result != read(source/"tasks"/f"{split}_{name}"/"result.json"):
                raise ValueError("Aggregate differs from verified task")
    for name, expected in manifest["source_sha256"].items():
        if sha256(research/name) != expected:
            raise ValueError(f"Research code changed: {name}")
    for name, expected in manifest["protected"].items():
        if sha256(name) != expected:
            raise ValueError(f"Protected input changed: {name}")
    base, current = (results["evaluations"][s["name"]] for s in selection["evaluations"])
    if (current["ranking"]["mAP@10"] <= base["ranking"]["mAP@10"]
            or current["candidates"] != base["candidates"] or current["threshold"] != base["threshold"]
            or sha256(Path(current["export"])/"candidates.csv") != sha256(Path(base["export"])/"candidates.csv")):
        raise ValueError("Promotion must improve ranking while preserving candidates")
    profiles = read(ROOT/"models/profiles.json")
    profile24 = profiles["MVP_dual_role_v24"]
    source24 = Path(manifest["map_search_plan"]["source_directory"])
    weights = read(source24/"dual_role_profile.json")
    if (sha256(MODEL) != weights["mvp"]["sha256"]
            or sha256(ROOT/profile24["bundle"]) != weights["r1_bundle"]["sha256"]
            or current["threshold"] != weights["threshold"]):
        raise ValueError("Application weights/threshold differ from source")
    evidence_path = ROOT/"docs/V25_QUALITY_EVIDENCE.json"
    compact = lambda x: {k: x[k] for k in ("system", "threshold", "ranking", "candidates", "protocol_sha256")}
    preserve(evidence_path, {"source_signature": results["signature"], "source_results_sha256": sha256(source/"results.json"),
        "selection": selection, "baseline": compact(base), "winner": compact(current),
        "delta": results["selected_vs_v24"], "protected_files_verified": len(manifest["protected"]),
        "task_receipts_verified": len(receipts), "scope": "observed development validation, not hidden test"})
    report = {"model": PROFILE, "threshold": current["threshold"], "validation": {
        "mAP_at_10": current["ranking"]["mAP@10"], "candidate_F1": current["candidates"]["F1"],
        "TNR": current["candidates"]["TNR"], "candidate_C": current["candidates"]["C"],
        "known_queries": current["ranking"]["n_scored"], "unknown_queries": current["ranking"]["n_openset_excluded"]},
        "provenance": {"evidence": str(evidence_path.relative_to(ROOT)), "evidence_sha256": sha256(evidence_path),
                       "ranking": SELECTED, "candidate": "RC_R1_equal3_v18", "source_signature": results["signature"],
                       "kind": "observed development validation, not hidden test"}}
    report_path = ROOT/f"models/{PROFILE}.metrics.json"
    preserve(report_path, report)
    profiles[PROFILE] = {**profile24, "r1_weight": .5, "metrics": str(report_path.relative_to(ROOT)),
                         "metrics_sha256": sha256(report_path)}
    write(ROOT/"models/profiles.json", profiles)
    previous = read(ROOT/"release_decision.json")
    if previous["active_profile"] == PROFILE:
        print("v25 assets already staged; existing acceptance evidence preserved")
        return
    preserve(ROOT/"docs/RELEASE_DECISION_BEFORE_V25.json", previous)
    write(ROOT/"release_decision.json", {"active_profile": PROFILE, "rollback_profile": "MVP_dual_role_v24",
        "legacy_profile": "MVP_legacy", "promoted": True, "authority": "explicit user approval after v25 review",
        "quality_evidence": str(evidence_path.relative_to(ROOT)), "quality_evidence_sha256": sha256(evidence_path),
        "official_submission_ready": False, "local_acceptance_passed": False,
        "verification_scope": "application verification pending; completed research quality comparison verified",
        "previous_release_decision": "docs/RELEASE_DECISION_BEFORE_V25.json", "open_checks": previous["open_checks"]})
    print("v25 staged; weights unchanged; verify application before switching live demo")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research", required=True, type=Path)
    promote(parser.parse_args().research.resolve())
