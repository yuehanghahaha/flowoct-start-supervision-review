"""Paper-selected best and a resumable latest; no archive of intermediate best predictions."""
from experiment import ROOT, read, write, sha, checkpoint


def storage_budget(trained_bytes, generator_bytes, epochs=200, checkpoint_every=0):
    prediction_bytes = 256 * 304 * 304 * 4
    periodic = (epochs // checkpoint_every) * trained_bytes if checkpoint_every else 0
    # The two best slots make pointer commits/recovery safe; an inactive slot is reused.
    phases = {
        "latest_atomic_update": periodic + 6 * trained_bytes + 2 * generator_bytes,
        "best_atomic_update": periodic + 3 * trained_bytes + 3 * generator_bytes,
        "final_validation_and_test": periodic + 3 * trained_bytes + 2 * generator_bytes + 60 * prediction_bytes,
    }
    margin = 10 * 1024 ** 3
    return {"required_bytes": max(phases.values()) + margin, "phase_bytes": phases,
            "margin_bytes": margin, "periodic_snapshots": epochs // checkpoint_every if checkpoint_every else 0,
            "complete_prediction_cases": 60,
            "scope": "Future output budget; existing read-only data/cache excluded; no clinical arrays evaluated for this estimate."}


def store_best(payload):
    pointer = ROOT / "checkpoints/best.json"
    current = read(pointer) if pointer.exists() else None
    if current and current["protocol_id"] != payload["protocol_id"]:
        raise ValueError("The existing best belongs to a different protocol")
    slot = 1 - current["slot"] if current else 0
    target = ROOT / f"checkpoints/best_slot_{slot}.pt"
    checkpoint(target, payload)
    record = {"path": target.relative_to(ROOT).as_posix(), "sha256": sha(target),
              "slot": slot, **payload["best"], "protocol_id": payload["protocol_id"]}
    write(pointer, record)
    return record


def restore_best(record):
    if record is None:
        return
    if sha(ROOT / record["path"]) != record["sha256"]:
        raise ValueError("The completed-epoch best checkpoint is missing or changed")
    write(ROOT / "checkpoints/best.json", record)
