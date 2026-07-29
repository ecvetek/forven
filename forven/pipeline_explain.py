"""Explainable pipeline: why each strategy is where it is, and what moves it.

Read-only aggregation behind ``GET /api/lifecycle/pipeline/explain`` (fleet) and
``GET /api/lifecycle/strategies/{id}/explain`` (single). For every strategy in an
active stage it answers the four operator questions:

1. why is it stuck — the same gate evaluators the real promotion runs, plus the
   readiness checklists, classified through the gate-rejection taxonomy so an
   evidence wait is never presented as a quality failure (or vice versa);
2. which evidence is missing or stale, and how old what exists is;
3. what specific action unblocks it;
4. what transition is expected next, and what triggers it.

Read-only by contract: promotion gates run via ``evaluate_promotion(dry_run=True)``
(no rejection records, no dethrone approvals, no symbol auto-assign, no
graduation snapshots), and the gauntlet rollup already evaluates with
``record_rejection=False``. Polling this surface must never mutate pipeline state.
"""

import logging
from datetime import datetime, timezone
from typing import Any

from forven.db import get_db
from forven.util import normalize_stage

log = logging.getLogger("forven.pipeline_explain")

# Forward flow of the tradable pipeline (see forven.util.normalize_stage).
_NEXT_STAGE = {
    "quick_screen": "gauntlet",
    "gauntlet": "paper",
    "paper": "live_graduated",
    "live_graduated": None,
    "research_only": None,
}

ACTIVE_STAGES = ("quick_screen", "gauntlet", "paper", "live_graduated")

_STAGE_LABELS = {
    "quick_screen": "Quick screen",
    "gauntlet": "Gauntlet",
    "paper": "Paper trading",
    "live_graduated": "Live (graduated)",
    "research_only": "Research only",
}

# What fires the next transition, keyed by CURRENT stage.
_TRANSITION_TRIGGERS = {
    "quick_screen": "Automatic — promoted once the quick-screen gate passes",
    "gauntlet": (
        "Automatic — the gauntlet workflow promotes once every required "
        "validation test passes and the gauntlet→paper gate clears"
    ),
    "paper": (
        "Gated — the strict paper→live gate must pass on forward paper "
        "evidence; live graduation then runs through the approvals workflow"
    ),
}

_LIVE_STANDING_NOTE = (
    "Holds live allocation on the graduated schedule; the decay kill-switch "
    "or a dethrone recommendation can demote it"
)

# Raw stage/status spellings that normalize_stage maps to terminal states —
# pre-filtered in SQL so the fleet query never drags the archive in.
_TERMINAL_STAGE_ALIASES = (
    "archived",
    "retired",
    "trash",
    "killed",
    "deprecated",
    "rejected",
    "failed",
    "backtest_failed",
    "backtest-failed",
    "backtestfailed",
)

# Mirror of brain._SLOT_CONTENTION_MARKERS: slot contention is a capacity
# conflict awaiting a dethrone, not a merit failure — it needs its own badge.
_SLOT_CONTENTION_MARKERS = ("awaiting dethrone", "slot occupied", "duplicate with active strategy")

# reason_code -> (action_key, operator-facing unblock label). Keys reuse the
# readiness checklist's action vocabulary (policy._action_for_check) where one
# exists so the frontend can keep a single action registry.
_REASON_CODE_ACTIONS = {
    "no_metrics_error": ("run_backtest", "Run a backtest so the strategy has metrics to judge"),
    "artifacts_pending": ("run_validation_suite", "Run the gauntlet validation suite"),
    "validation_in_flight": ("wait", "Wait — a validation run is in flight; its verdict lands automatically"),
    "stale_validation": ("run_validation_suite", "Re-run validation — it predates the latest optimization"),
    "stale_engine_artifacts": ("wait", "Artifacts predate the current backtest engine — re-validation is queued automatically"),
    "stale_data_artifacts": ("wait", "Artifacts were scored on different data semantics — re-validation is queued automatically"),
    "missing_evidence": ("run_validation_suite", "Run or re-run the missing validation tests until their verdicts persist"),
    "wfa_window_insufficient": ("run_validation_suite", "Re-run walk-forward with a window sized to the strategy's trade cadence"),
    "insufficient_paper_evidence": ("wait", "Keep paper trading — forward evidence is still accumulating"),
    "source_reconciliation_pending": ("wait", "Source reconciliation has not measured this pair yet — it runs on schedule"),
    "dsr_unavailable": ("run_optimization", "Deflated-Sharpe inputs are missing — re-run optimization so trial counts are stamped"),
    "source_divergence_reject": ("fix_data", "Validation data diverges from the venue — backfill/reconcile data, then re-test"),
    "zero_trade": ("review_strategy", "Strategy produces no signals — revise the entry logic or archive it"),
    "duplicate_reject": ("review_slot", "Duplicates an active strategy on the same market — await the dethrone or retarget"),
    "not_found": ("review_strategy", "Strategy record is incomplete — investigate"),
}
_SLOT_ACTION = ("review_slot", "Slot occupied by an incumbent — resolve the pending dethrone or retarget the market")
_DEFAULT_MERIT_ACTION = ("review_strategy", "Failed a quality gate on merit — revise parameters, re-optimize, or archive")

# Labels for the readiness checklists' actionable keys (policy._action_for_check).
_STEP_ACTION_LABELS = {
    "run_timeframe_sweep": "Run the multi-timeframe backtest sweep",
    "run_validation_suite": "Run the gauntlet validation suite",
    "run_optimization": "Run parameter optimization",
    "apply_best_params": "Apply the best optimization parameters",
    "run_confirmation_backtest": "Run a confirmation backtest on the optimized params",
}


def _parse_ts(value: object) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _age_days(value: object, now: datetime) -> float | None:
    ts = _parse_ts(value)
    if ts is None:
        return None
    return round(max(0.0, (now - ts).total_seconds() / 86400.0), 1)


def _action_payload(key: str | None, label: str | None) -> dict | None:
    if not key:
        return None
    return {"key": key, "label": label or key}


def _blocker(reason: str, code: str, kind: str, source: str, action: tuple[str, str] | None, **extra) -> dict:
    payload = {
        "reason": str(reason),
        "code": code,
        "kind": kind,
        "source": source,
        "action": _action_payload(*action) if action else None,
    }
    payload.update({k: v for k, v in extra.items() if v is not None})
    return payload


def _classify_gate_reason(reason: str) -> tuple[str, str, tuple[str, str]]:
    """Map a gate rejection to (code, kind, action) for the explain payload."""
    from forven.policy import classify_rejection_reason

    lowered = str(reason or "").lower()
    if any(marker in lowered for marker in _SLOT_CONTENTION_MARKERS):
        return "slot_contention", "contention", _SLOT_ACTION
    code, kind = classify_rejection_reason(reason)
    action = _REASON_CODE_ACTIONS.get(code)
    if action is None:
        action = _DEFAULT_MERIT_ACTION if kind == "merit" else ("wait", "Waiting on evidence — no operator action required yet")
    return code, kind, action


def _latest_result_times(strategy_id: str) -> dict[str, dict]:
    """Newest non-deleted backtest and optimization rows (evidence recency)."""
    out: dict[str, dict] = {}
    with get_db() as conn:
        for result_type in ("backtest", "optimization"):
            row = conn.execute(
                """SELECT result_id, created_at FROM backtest_results
                   WHERE strategy_id = ?
                     AND LOWER(TRIM(COALESCE(result_type, 'backtest'))) = ?
                     AND (deleted_at IS NULL OR TRIM(COALESCE(deleted_at, '')) = '')
                   ORDER BY datetime(created_at) DESC LIMIT 1""",
                (strategy_id, result_type),
            ).fetchone()
            if row:
                out[result_type] = {"result_id": row["result_id"], "at": row["created_at"]}
    return out


def _last_rejection(strategy_id: str, stage_changed_at: str | None) -> tuple[dict | None, int]:
    """Latest gate rejection + how many landed during the current stage stay."""
    with get_db() as conn:
        row = conn.execute(
            """SELECT gate, reason_code, reason_text, created_at FROM gate_rejections
               WHERE strategy_id = ?
               ORDER BY datetime(created_at) DESC LIMIT 1""",
            (strategy_id,),
        ).fetchone()
        count = 0
        if stage_changed_at:
            count_row = conn.execute(
                """SELECT COUNT(*) AS c FROM gate_rejections
                   WHERE strategy_id = ? AND datetime(created_at) >= datetime(?)""",
                (strategy_id, stage_changed_at),
            ).fetchone()
            count = int(count_row["c"] or 0) if count_row else 0
    if not row:
        return None, count
    return (
        {
            "gate": row["gate"],
            "reason_code": row["reason_code"],
            "reason_text": row["reason_text"],
            "at": row["created_at"],
        },
        count,
    )


def _pending_approval(strategy_id: str) -> dict | None:
    with get_db() as conn:
        row = conn.execute(
            """SELECT id, approval_type, requested_status, reason, created_at FROM approvals
               WHERE target_type = 'strategy' AND target_id = ? AND status = 'pending_approval'
               ORDER BY datetime(created_at) DESC LIMIT 1""",
            (strategy_id,),
        ).fetchone()
    if not row:
        return None
    return {
        "id": row["id"],
        "approval_type": row["approval_type"],
        "requested_status": row["requested_status"],
        "reason": row["reason"],
        "at": row["created_at"],
    }


def _last_paper_trade_at(strategy_id: str, since: str | None) -> str | None:
    params: list[object] = [strategy_id]
    where_since = ""
    if since:
        where_since = " AND datetime(closed_at) >= datetime(?)"
        params.append(since)
    with get_db() as conn:
        row = conn.execute(
            "SELECT MAX(datetime(closed_at)) AS last_at FROM trades "
            "WHERE COALESCE(strategy_id, strategy) = ? "
            "AND status = 'CLOSED' "
            "AND LOWER(COALESCE(execution_type, '')) LIKE 'paper%'" + where_since,
            tuple(params),
        ).fetchone()
    return row["last_at"] if row and row["last_at"] else None


def _readiness_blockers(steps: list[dict], source: str, seen_reasons: set[str]) -> list[dict]:
    """Failed/warning readiness steps as blockers (dedup on reason text)."""
    blockers: list[dict] = []
    for step in steps or []:
        if step.get("status") not in ("failed", "warning"):
            continue
        detail = str(step.get("detail") or step.get("name") or "")
        if detail in seen_reasons:
            continue
        seen_reasons.add(detail)
        code, kind, action = _classify_gate_reason(detail)
        actionable = step.get("actionable")
        if actionable:
            action = (actionable, _STEP_ACTION_LABELS.get(actionable, actionable))
            # A checklist step with a run-action is evidence GATHERING; when the
            # taxonomy can't classify its prose ("Multi-TF sweep incomplete"),
            # absence-of-evidence is the honest read, not a merit failure.
            if code == "gate_reject":
                code, kind = "missing_evidence", "evidence"
        blockers.append(
            _blocker(
                detail,
                code,
                kind if step.get("status") == "failed" else "advisory",
                source,
                action,
                step=step.get("name"),
                extra=step.get("extra"),
            )
        )
    return blockers


def _validation_evidence(tests: dict, now: datetime) -> dict:
    """Compact per-test evidence view from the gauntlet status rollup."""
    out: dict[str, dict] = {}
    for key, payload in (tests or {}).items():
        if not isinstance(payload, dict):
            continue
        at = payload.get("completed_at") or payload.get("created_at")
        out[key] = {
            "status": payload.get("status"),
            "verdict": payload.get("verdict"),
            "at": at,
            "age_days": _age_days(at, now),
            "stale": payload.get("stale"),
            "stale_engine": payload.get("stale_engine"),
        }
    return out


def _classify_status(
    stage: str,
    promotable: bool | None,
    blockers: list[dict],
    pending_approval: dict | None,
    in_flight: bool,
) -> str:
    if stage == "live_graduated":
        return "live"
    if stage == "research_only":
        return "parked"
    if promotable:
        return "ready"
    if pending_approval:
        return "awaiting_operator"
    if any(b.get("kind") == "contention" for b in blockers):
        return "slot_contention"
    if in_flight or any(b.get("code") == "validation_in_flight" for b in blockers):
        return "in_flight"
    # The promotion gate is authoritative for the stuck-vs-failed verdict;
    # readiness-checklist steps are supporting detail and must not flip a
    # warm-up wait into "failed on merit" (their prose classifies loosely).
    gate_blockers = [
        b for b in blockers if str(b.get("source", "")).endswith(("_gate", "_workflow"))
    ]
    hard = [b for b in (gate_blockers or blockers) if b.get("kind") in ("evidence", "merit")]
    if hard and all(b.get("kind") == "evidence" for b in hard):
        return "waiting_evidence"
    if any(b.get("kind") == "merit" for b in hard):
        return "blocked_merit"
    return "waiting_evidence" if blockers else "unknown"


def _explain_row(row: dict, now: datetime) -> dict:
    from forven.policy import (
        check_paper_live_readiness,
        check_promotion_readiness,
        evaluate_promotion,
    )

    strategy_id = str(row.get("id") or "").strip()
    stage = normalize_stage(str(row.get("stage") or row.get("status") or ""))
    stage_changed_at = str(row.get("stage_changed_at") or "").strip() or None
    next_stage = _NEXT_STAGE.get(stage)

    promotable: bool | None = None
    gate_reason: str | None = None
    blockers: list[dict] = []
    seen_reasons: set[str] = set()
    readiness_steps: list[dict] = []
    gauntlet_summary: dict | None = None
    in_flight = False

    evidence: dict[str, Any] = {}
    latest = _latest_result_times(strategy_id)
    if "backtest" in latest:
        evidence["last_backtest_at"] = latest["backtest"]["at"]
        evidence["last_backtest_age_days"] = _age_days(latest["backtest"]["at"], now)
    if "optimization" in latest:
        evidence["last_optimization_at"] = latest["optimization"]["at"]
        evidence["last_optimization_age_days"] = _age_days(latest["optimization"]["at"], now)

    if stage == "quick_screen":
        promotable, reason = evaluate_promotion(
            strategy_id, stage, "gauntlet", record_rejection=False, dry_run=True
        )
        gate_reason = str(reason)
        if not promotable:
            code, kind, action = _classify_gate_reason(reason)
            seen_reasons.add(gate_reason)
            blockers.append(_blocker(gate_reason, code, kind, "quick_screen_gate", action))

    elif stage == "gauntlet":
        from forven.gauntlet.status import get_strategy_gauntlet_status

        status_payload = get_strategy_gauntlet_status(strategy_id)
        if status_payload.get("ok"):
            promotable = bool(status_payload.get("ready_for_paper"))
            gate_reason = status_payload.get("promotion_reason")
            workflow_status = str(status_payload.get("workflow_status") or "")
            in_flight = workflow_status == "running"
            gauntlet_summary = {
                "workflow_status": workflow_status or None,
                "current_step": status_payload.get("current_step"),
                "required_tests": status_payload.get("required_tests"),
                "missing_required": status_payload.get("missing_required"),
                "tests_passed": status_payload.get("tests_passed"),
                "tests_total": status_payload.get("tests_total"),
                "composite_robustness_score": status_payload.get("composite_robustness_score"),
                "min_robustness_score": status_payload.get("min_robustness_score"),
            }
            evidence["validation_tests"] = _validation_evidence(status_payload.get("tests") or {}, now)
            if not promotable and gate_reason:
                code, kind, action = _classify_gate_reason(gate_reason)
                seen_reasons.add(str(gate_reason))
                blockers.append(_blocker(str(gate_reason), code, kind, "gauntlet_gate", action))
            if in_flight and status_payload.get("current_step"):
                reason = f"Gauntlet workflow running — current step: {status_payload.get('current_step')}"
                if reason not in seen_reasons:
                    seen_reasons.add(reason)
                    blockers.append(
                        _blocker(
                            reason,
                            "validation_in_flight",
                            "evidence",
                            "gauntlet_workflow",
                            ("wait", "Wait — the gauntlet is processing this strategy"),
                        )
                    )
        readiness = check_promotion_readiness(strategy_id)
        readiness_steps = readiness.get("steps") or []
        blockers.extend(_readiness_blockers(readiness_steps, "promotion_readiness", seen_reasons))

    elif stage == "paper":
        promotable, reason = evaluate_promotion(
            strategy_id, stage, "live_graduated", record_rejection=False, dry_run=True
        )
        gate_reason = str(reason)
        if not promotable:
            code, kind, action = _classify_gate_reason(reason)
            seen_reasons.add(gate_reason)
            blockers.append(_blocker(gate_reason, code, kind, "paper_live_gate", action))
        readiness = check_paper_live_readiness(strategy_id)
        readiness_steps = readiness.get("steps") or []
        blockers.extend(_readiness_blockers(readiness_steps, "paper_live_readiness", seen_reasons))
        paper_evidence: dict[str, Any] = {}
        for step in readiness_steps:
            extra = step.get("extra")
            if isinstance(extra, dict) and step.get("name") in ("paper_duration", "paper_trades"):
                paper_evidence[step["name"]] = extra
        last_trade_at = _last_paper_trade_at(strategy_id, stage_changed_at)
        if last_trade_at:
            paper_evidence["last_trade_at"] = last_trade_at
            paper_evidence["last_trade_age_days"] = _age_days(last_trade_at, now)
        if paper_evidence:
            evidence["paper"] = paper_evidence

    pending_approval = _pending_approval(strategy_id)
    last_rejection, rejections_in_stage = _last_rejection(strategy_id, stage_changed_at)
    if last_rejection:
        last_rejection["age_days"] = _age_days(last_rejection.get("at"), now)

    status = _classify_status(stage, promotable, blockers, pending_approval, in_flight)

    next_action = None
    if pending_approval and stage != "live_graduated":
        next_action = _action_payload(
            "review_approval",
            f"Review pending approval #{pending_approval['id']} ({pending_approval.get('approval_type') or 'approval'})",
        )
    elif status == "ready" and next_stage:
        next_action = _action_payload("promote", f"Ready — promote to {_STAGE_LABELS.get(next_stage, next_stage)}")
    elif blockers:
        next_action = blockers[0].get("action")

    next_transition = None
    if next_stage:
        next_transition = {
            "to_stage": next_stage,
            "label": f"{_STAGE_LABELS.get(stage, stage)} → {_STAGE_LABELS.get(next_stage, next_stage)}",
            "trigger": _TRANSITION_TRIGGERS.get(stage),
        }
    elif stage == "live_graduated":
        next_transition = {"to_stage": None, "label": "Live", "trigger": _LIVE_STANDING_NOTE}

    return {
        "id": strategy_id,
        "display_id": row.get("display_id"),
        "name": row.get("display_name") or row.get("name") or strategy_id,
        "symbol": row.get("symbol"),
        "timeframe": row.get("timeframe"),
        "type": row.get("type"),
        "stage": stage,
        "stage_label": _STAGE_LABELS.get(stage, stage),
        "stage_changed_at": stage_changed_at,
        "days_in_stage": _age_days(stage_changed_at, now),
        "demotion_count": row.get("demotion_count"),
        "status": status,
        "promotable": promotable,
        "gate_reason": gate_reason,
        "blockers": blockers,
        "next_action": next_action,
        "next_transition": next_transition,
        "evidence": evidence,
        "readiness_steps": readiness_steps,
        "gauntlet": gauntlet_summary,
        "pending_approval": pending_approval,
        "last_rejection": last_rejection,
        "rejections_in_stage": rejections_in_stage,
    }


_STAGE_RANK = {"quick_screen": 1, "gauntlet": 2, "paper": 3, "live_graduated": 4, "research_only": 5}

_STRATEGY_COLUMNS = (
    "id, display_id, name, display_name, type, symbol, timeframe, stage, status, "
    "status_reason, demotion_count, stage_changed_at, created_at, updated_at"
)


def explain_strategy(strategy_id: str) -> dict:
    """Full pipeline explanation for one strategy."""
    from forven.gauntlet.store import sanitize_non_finite

    clean_id = str(strategy_id or "").strip()
    if not clean_id:
        return {"ok": False, "error": "strategy_id_required", "strategy_id": strategy_id}
    with get_db() as conn:
        row = conn.execute(
            f"SELECT {_STRATEGY_COLUMNS} FROM strategies WHERE id = ?", (clean_id,)
        ).fetchone()
    if not row:
        return {"ok": False, "error": "strategy_not_found", "strategy_id": clean_id}
    now = datetime.now(timezone.utc)
    payload = _explain_row(dict(row), now)
    return sanitize_non_finite({"ok": True, "generated_at": now.isoformat(), "strategy": payload})


def explain_pipeline(stage: str | None = None, limit: int = 200) -> dict:
    """Fleet-wide pipeline explanation for every strategy in an active stage.

    ``stage`` filters to one canonical stage (aliases accepted); default is the
    four tradable stages. Longest-stuck strategies sort first within a stage.
    """
    from forven.gauntlet.store import sanitize_non_finite

    requested: tuple[str, ...] = ACTIVE_STAGES
    if stage:
        requested = (normalize_stage(stage),)
    try:
        cap = max(1, min(int(limit), 1000))
    except (TypeError, ValueError):
        cap = 200

    placeholders = ",".join("?" for _ in _TERMINAL_STAGE_ALIASES)
    with get_db() as conn:
        rows = conn.execute(
            f"""SELECT {_STRATEGY_COLUMNS} FROM strategies
                WHERE LOWER(TRIM(COALESCE(stage, status, ''))) NOT IN ({placeholders})
                ORDER BY datetime(COALESCE(stage_changed_at, updated_at, created_at)) ASC""",
            _TERMINAL_STAGE_ALIASES,
        ).fetchall()

    now = datetime.now(timezone.utc)
    strategies: list[dict] = []
    errors: list[dict] = []
    truncated = False
    for raw in rows:
        row = dict(raw)
        row_stage = normalize_stage(str(row.get("stage") or row.get("status") or ""))
        if row_stage not in requested:
            continue
        if len(strategies) >= cap:
            truncated = True
            break
        try:
            strategies.append(_explain_row(row, now))
        except Exception as exc:  # one bad strategy must not blank the fleet view
            log.warning("pipeline explain failed for %s: %s", row.get("id"), exc)
            errors.append({"id": row.get("id"), "error": str(exc)})

    strategies.sort(
        key=lambda s: (
            _STAGE_RANK.get(str(s.get("stage")), 99),
            -(s.get("days_in_stage") or 0.0),
        )
    )

    by_stage: dict[str, int] = {}
    by_status: dict[str, int] = {}
    for s in strategies:
        by_stage[str(s.get("stage"))] = by_stage.get(str(s.get("stage")), 0) + 1
        by_status[str(s.get("status"))] = by_status.get(str(s.get("status")), 0) + 1

    from forven.policy import load_pipeline_config

    preset = str(load_pipeline_config().get("pipeline_preset") or "default")

    return sanitize_non_finite(
        {
            "ok": True,
            "generated_at": now.isoformat(),
            "pipeline_preset": preset,
            "stages": list(requested),
            "counts": {"by_stage": by_stage, "by_status": by_status},
            "truncated": truncated,
            "strategies": strategies,
            "errors": errors,
        }
    )
