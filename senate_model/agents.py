from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import mesa

from .entities import Ideology, ConstituentProfile, Deal, Bill, DealStatus

if TYPE_CHECKING:
    from .model import SenateModel


class VoteStance(str, Enum):
    FOR       = "for"
    AGAINST   = "against"
    UNDECIDED = "undecided"


class Vote(str, Enum):
    YES     = "yes"
    NO      = "no"
    ABSTAIN = "abstain"


class Senator(mesa.Agent):
    """
    A senator occupying a permanent seat. seat_id survives replacements; unique_id does not.

    Personal ideology is sampled with noise around the district ideology, creating tension
    between the senator's own views and constituent pressure.

    deal_tolerance is private — other senators only observe refusals.
    Relationships and inference data are seeded neutral and updated by the Resolution phase.
    """

    def __init__(
        self,
        model: SenateModel,
        seat_id: int,
        constituent_profile: ConstituentProfile,
        ideology_noise: float = 0.15,
        reputation: float = 0.5,
    ):
        super().__init__(model)

        self.seat_id = seat_id
        self.constituent_profile = constituent_profile

        self.personal_ideology = Ideology(
            constituent_profile.ideology.economic + model.rng.normal(0, ideology_noise),
            constituent_profile.ideology.social   + model.rng.normal(0, ideology_noise),
        )

        # Stagger three cohorts evenly across the election cycle
        cohort = seat_id % 3
        self.time_until_election: int = (cohort + 1) * (model.sessions_per_election_cycle // 3)

        self.deal_tolerance: int = int(model.rng.integers(1, 6))
        self.deals_owed_to_others: list[Deal] = []
        self.deals_owed_to_me:     list[Deal] = []

        self.relationships:     dict[int, float] = {}  # senator_id -> [-1, 1]
        self.inferred_models:   dict[int, list]  = {}  # senator_id -> [(ProvisionType, Ideology)]
        self.confidence_scores: dict[int, float] = {}
        self.reputation: float = reputation

        self.current_stance: VoteStance = VoteStance.UNDECIDED
        self.proposal_queue: Bill | None = None
        self.session_vote_alignment: list[float] = []

    @property
    def is_up_for_election(self) -> bool:
        return self.time_until_election <= 0

    @property
    def at_deal_capacity(self) -> bool:
        return len(self.deals_owed_to_others) >= self.deal_tolerance

    def init_relationships(self, all_senator_ids: list[int]) -> None:
        for sid in all_senator_ids:
            if sid == self.unique_id:
                continue
            self.relationships[sid]     = float(self.model.rng.normal(0, 0.05))
            self.inferred_models[sid]   = []
            self.confidence_scores[sid] = 0.0

    def reset_session_state(self) -> None:
        self.current_stance = VoteStance.UNDECIDED
        self.proposal_queue = None
        self.session_vote_alignment = []
        self.deals_owed_to_others = [d for d in self.deals_owed_to_others if d.expiry > 1]
        self.deals_owed_to_me     = [d for d in self.deals_owed_to_me     if d.expiry > 1]

    def _district_impact(self, bill: Bill) -> float:
        """Weighted average provision impact on this senator's district, in [-1, 1]."""
        profile = self.constituent_profile
        total = 0.0
        for provision in bill.provisions:
            p_impact, p_weight = 0.0, 0.0
            for group, impact in provision.beneficiary_groups.items():
                w = profile.composition.get(group, 0.0)
                p_impact += impact * w
                p_weight += w
            if p_weight > 0:
                p_impact /= p_weight
            total += p_impact
        return total / len(bill.provisions)

    def evaluate_bill(self, bill: Bill) -> VoteStance:
        if not bill.provisions:
            return VoteStance.UNDECIDED

        ideology_score, constituent_score, impact_score = 0.0, 0.0, 0.0
        max_dist = Ideology.MAX_DISTANCE

        for provision in bill.provisions:
            ideology_score    += 1.0 - self.personal_ideology.distance_to(provision.ideology_position) / max_dist
            constituent_score += 1.0 - self.constituent_profile.ideology.distance_to(provision.ideology_position) / max_dist
            net_impact, total_weight = 0.0, 0.0
            for group, impact in provision.beneficiary_groups.items():
                dw = self.constituent_profile.composition.get(group, 0.0)
                net_impact   += impact * dw
                total_weight += dw
            if total_weight > 0:
                net_impact /= total_weight
            impact_score += (net_impact + 1.0) / 2.0

        n = len(bill.provisions)
        ideology_score    /= n
        constituent_score /= n
        impact_score      /= n

        # Electoral pressure ramps up when approval drops below ~0.75
        pressure = max(0.0, 1.0 - (self.constituent_profile.approval - 0.5) * 4.0)

        score = (
            0.25                * ideology_score +
            (0.35 + 0.20 * pressure) * constituent_score +
            (0.40 - 0.20 * pressure) * impact_score
        )

        if score >= 0.55:   return VoteStance.FOR
        if score <= 0.45:   return VoteStance.AGAINST
        return VoteStance.UNDECIDED

    def vote_from_stance(self) -> Vote:
        for deal in self.deals_owed_to_others:
            if (deal.status == DealStatus.ACCEPTED
                    and deal.obligation.obligation_type == "vote"
                    and deal.obligation.target_bill_id == self.model.active_bill.id):
                return Vote.YES if deal.obligation.vote == "yes" else Vote.NO

        if self.current_stance == VoteStance.FOR:      return Vote.YES
        if self.current_stance == VoteStance.AGAINST:  return Vote.NO
        return Vote.ABSTAIN

    def _deal_acceptance_score(self, deal: Deal, bill: Bill) -> float:
        """Score in [0, 1] for willingness to accept a deal. >= 0.5 means accept."""
        rel       = self.relationships.get(deal.offered_by, 0.0)
        pressure  = max(0.0, 1.0 - (self.constituent_profile.approval - 0.5) * 4.0)
        impact    = self._district_impact(bill)
        return (
            0.40 * (rel + 1.0) / 2.0 +
            0.40 * (impact + 1.0) / 2.0 +
            0.20 * pressure
        )

    def step(self) -> None:
        pass
