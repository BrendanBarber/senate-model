from senate_model.model import SenateModel
from senate_model.agents import Senator
from senate_model.entities import CONSTITUENT_CATEGORIES

model = SenateModel(n_seats=10, seed=42)

for senator in model.agents_by_type[Senator]:
    profile = senator.constituent_profile
    print(
        f"Seat {senator.seat_id:3d}\n"
        f"  District ideology : econ={profile.ideology.economic:+.2f}  soc={profile.ideology.social:+.2f}\n"
        f"  Personal ideology : econ={senator.personal_ideology.economic:+.2f}  soc={senator.personal_ideology.social:+.2f}\n"
        f"  Approval          : {profile.approval:.2f}\n"
        f"  Election in       : {senator.time_until_election} sessions\n"
        f"  Deal tolerance    : {senator.deal_tolerance}\n"
    )

for senator in model.agents_by_type[Senator]:
    profile = senator.constituent_profile
    print(f"Seat {senator.seat_id} — Constituent Composition")
    for category, groups in CONSTITUENT_CATEGORIES.items():
        print(f"  {category.capitalize()}")
        for group in groups:
            weight = profile.composition.get(group, 0.0)
            print(f"    {group.value:<25} {weight:.3f}")
    print()
