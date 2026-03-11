from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from senate_model.model import SenateModel
from senate_model.agents import Senator
from senate_model.entities import CONSTITUENT_CATEGORIES

RYG = LinearSegmentedColormap.from_list("ryg", ["#e02020", "#ffe000", "#20c020"])


def compute_session_net_impact(model: SenateModel) -> dict[int, float]:
    last_session = model.current_session - 1
    bills = [b for b in model.resolved_bills if b.session_introduced == last_session]
    if not bills:
        return {}
    n_provisions = sum(len(b.provisions) for b in bills)
    result: dict[int, float] = {}
    for senator in model.agents_by_type[Senator]:
        profile = senator.constituent_profile
        total = 0.0
        for bill in bills:
            sign = 1.0 if bill.status.value == "passed" else -1.0
            for provision in bill.provisions:
                p_impact, p_weight = 0.0, 0.0
                for group, impact in provision.beneficiary_groups.items():
                    w = profile.composition.get(group, 0.0)
                    p_impact += impact * w
                    p_weight += w
                if p_weight > 0:
                    p_impact /= p_weight
                total += sign * p_impact
        result[senator.unique_id] = total / n_provisions if n_provisions > 0 else 0.0
    return result


def build_tooltip(senator: Senator, net_impact: float | None) -> str:
    profile = senator.constituent_profile
    lines = [
        f"Seat {senator.seat_id}",
        f"District  econ={profile.ideology.economic:+.2f}  soc={profile.ideology.social:+.2f}",
        f"Personal  econ={senator.personal_ideology.economic:+.2f}  soc={senator.personal_ideology.social:+.2f}",
        f"Approval: {profile.approval:.2f}   Reputation: {senator.reputation:.2f}",
        f"Net impact (last session): {net_impact:+.3f}" if net_impact is not None else "Net impact: n/a",
        f"Election in {senator.time_until_election} sessions",
        "",
    ]
    for category, groups in CONSTITUENT_CATEGORIES.items():
        lines.append(f"{category.capitalize()}:")
        for group in groups:
            lines.append(f"  {group.value:<22} {profile.composition.get(group, 0.0):.2f}")
    return "\n".join(lines)


def print_session_bills(model: SenateModel) -> None:
    senator_map = {s.unique_id: s for s in model.agents_by_type[Senator]}
    history_ids = {e["bill_id"] for e in model.session_history}
    bills = {b.id: b for b in model.resolved_bills if b.id in history_ids}
    group_totals: dict[str, float] = {}

    for entry in model.session_history:
        bill = bills.get(entry["bill_id"])
        status = entry["status"].upper()
        passed = entry["status"] == "passed"

        if bill is not None:
            author = senator_map.get(bill.authored_by)
            author_str = (f"senator {bill.authored_by} (seat {author.seat_id})" if author
                          else f"senator {bill.authored_by} (seat unknown, replaced)")
        else:
            author_str = "unknown"

        print(
            f"\nBill #{entry['bill_id']} [{status}]  {entry['yes']} yes / {entry['no']} no  —  authored by {author_str}")

        if bill is None:
            print("  (provisions not available)")
            continue

        for p in bill.provisions:
            benefit_str = "  ".join(
                f"{g.value}={w:+.2f}" for g, w in sorted(p.beneficiary_groups.items(), key=lambda x: -x[1]))
            print(
                f"  [{p.type.value}] econ={p.ideology_position.economic:+.2f}  soc={p.ideology_position.social:+.2f}  |  {benefit_str}")
            sign = 1.0 if passed else 0.0
            for g, w in p.beneficiary_groups.items():
                key = g.value if hasattr(g, "value") else g
                group_totals[key] = group_totals.get(key, 0.0) + sign * w

    print("\n--- Session Group Impact Summary ---")
    for group, total in sorted(group_totals.items(), key=lambda x: -x[1]):
        bar = ("+" if total >= 0 else "-") * int(abs(total) * 5)
        print(f"  {group:<22} {total:+.2f}  {bar}")


def run_viz_loop(model: SenateModel, draw_fn, map_seed: int = 0) -> None:
    """Main session loop. draw_fn(model, map_seed) is called after each session."""
    try:
        while True:
            model.run_session()
            print_session_bills(model)
            draw_fn(model, map_seed)
            plt.close("all")
            prompt = f"Session {model.current_session - 1} done. Press Enter for next session, q to quit: "
            if input(prompt).strip().lower() == "q":
                break
    except KeyboardInterrupt:
        pass
    print("Exiting.")
