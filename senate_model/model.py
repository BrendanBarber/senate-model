from __future__ import annotations

import logging
from enum import Enum

import numpy as np
import mesa

from senate_model import config
from .entities import (
    Ideology, ConstituentProfile, ConstituentGroup,
    Provision, ProvisionType, AmendmentSurface,
    Bill, BillStatus, Deal, DealStatus, DealObligation,
    CONSTITUENT_CATEGORIES,
)
from .agents import Senator, VoteStance, Vote

log = logging.getLogger(__name__)


class SessionPhase(str, Enum):
    ELECTION_CHECK = "Election Check"
    PROPOSAL       = "Proposal"
    DISCUSSION     = "Discussion"
    NEGOTIATION    = "Negotiation"
    REVISION       = "Revision"
    VOTING         = "Voting"
    RESOLUTION     = "Resolution"


_PHASE_ORDER = list(SessionPhase)


def _constituent_approval_delta(senator: Senator, bill: Bill, passed: bool) -> float:
    if not bill.provisions:
        return 0.0

    profile = senator.constituent_profile
    net_impact = 0.0
    for provision in bill.provisions:
        p_impact, p_weight = 0.0, 0.0
        for group, impact in provision.beneficiary_groups.items():
            w = profile.composition.get(group, 0.0)
            p_impact += impact * w
            p_weight += w
        if p_weight > 0:
            p_impact /= p_weight
        net_impact += p_impact
    net_impact /= len(bill.provisions)

    if not passed:
        net_impact = -net_impact

    my_vote = bill.vote_record.get(senator.unique_id)
    bill_helped = net_impact > 0
    voted_yes   = my_vote == Vote.YES.value
    voted_correctly = (bill_helped and voted_yes) or (not bill_helped and not voted_yes)

    return float(np.clip(net_impact * 0.3 + (0.03 if voted_correctly else -0.03), -0.15, 0.15))


class SenateModel(mesa.Model):
    def __init__(
        self,
        n_seats: int = 100,
        sessions_per_election_cycle: int = 6,
        ideology_noise: float = 0.15,
        approval_threshold: float = 0.5,
        vote_majority: float = 0.5,
        seed: int | None = 42,
    ):
        super().__init__(seed=seed)

        self.n_seats                     = n_seats
        self.sessions_per_election_cycle = sessions_per_election_cycle
        self.ideology_noise              = ideology_noise
        self.approval_threshold          = approval_threshold
        self.vote_majority               = vote_majority

        self.current_session: int          = 0
        self.current_phase: SessionPhase   = SessionPhase.ELECTION_CHECK
        self.bill_queue:     list[Bill]    = []
        self.active_bill:    Bill | None   = None
        self.passed_bills:   list[int]     = []
        self.failed_bills:   list[int]     = []
        self.resolved_bills: list[Bill]    = []
        self.session_history: list[dict]   = []
        self.coalition_tracker: dict       = {}

        self._next_bill_id      = 0
        self._next_provision_id = 0
        self._next_deal_id      = 0

        self.constituent_profiles: dict[int, ConstituentProfile] = {}
        self._init_seats()
        self._init_senators()

        self.datacollector = mesa.DataCollector(
            model_reporters={
                "Session":        "current_session",
                "Phase":          lambda m: m.current_phase.value,
                "Passed Bills":   lambda m: len(m.passed_bills),
                "Failed Bills":   lambda m: len(m.failed_bills),
                "Avg Approval":   lambda m: float(np.mean([
                    s.constituent_profile.approval for s in m.agents_by_type[Senator]
                ])) if m.agents_by_type[Senator] else 0.5,
                "Avg Reputation": lambda m: float(np.mean([
                    s.reputation for s in m.agents_by_type[Senator]
                ])) if m.agents_by_type[Senator] else 0.5,
            },
            agent_reporters={
                "Seat":                "seat_id",
                "Ideology_Economic":   lambda a: a.personal_ideology.economic,
                "Ideology_Social":     lambda a: a.personal_ideology.social,
                "Approval":            lambda a: a.constituent_profile.approval,
                "Reputation":          "reputation",
                "Time_Until_Election": "time_until_election",
            },
        )
        self.datacollector.collect(self)
        log.info("Model initialised: %d seats, seed=%s", n_seats, seed)

    # --- Initialisation ---

    def _init_seats(self) -> None:
        cfg      = config.get("demographics")
        min_w    = cfg["constraints"]["min_within_category"]
        max_w    = cfg["constraints"]["max_within_category"]
        geo_econ = cfg["geographic_economic"]
        geo_cult = cfg["geographic_cultural"]
        geo_gen  = cfg["geographic_generational"]
        eco_gen  = cfg["economic_generational"]
        eco_cult = cfg["economic_cultural"]
        geo_types = ["rural", "urban", "suburban"]

        def biased_weights(
            groups: list[ConstituentGroup],
            bias_sources: list[tuple[float, dict]],
        ) -> dict[ConstituentGroup, float]:
            # Dirichlet draw then apply additive biases scaled by source weight
            raw = dict(zip(groups, self.rng.dirichlet(np.ones(len(groups)))))
            for source_weight, bias_table in bias_sources:
                for group_name, delta in bias_table.items():
                    g = ConstituentGroup(group_name)
                    if g in raw:
                        raw[g] += delta * source_weight
            for g in groups:
                raw[g] = max(min_w, min(max_w, raw[g]))
            total = sum(raw.values())
            return {g: raw[g] / total for g in groups}

        for seat_id in range(self.n_seats):
            ideology    = Ideology(float(self.rng.uniform(-1, 1)), float(self.rng.uniform(-1, 1)))
            geo_weights = dict(zip(geo_types, self.rng.dirichlet(np.ones(3))))

            econ_groups = [
                ConstituentGroup.INDUSTRIAL_LABOR, ConstituentGroup.AGRICULTURAL_LABOR,
                ConstituentGroup.SMALL_BUSINESS, ConstituentGroup.LARGE_BUSINESS,
                ConstituentGroup.PUBLIC_SECTOR,
            ]
            econ_comp = biased_weights(econ_groups, [(geo_weights[g], geo_econ[g]) for g in geo_types])

            cult_comp = biased_weights(
                [ConstituentGroup.RELIGIOUS, ConstituentGroup.SECULAR],
                [(geo_weights[g], geo_cult[g]) for g in geo_types] +
                [(econ_comp[ConstituentGroup(k)], v) for k, v in eco_cult.items()],
            )

            gen_comp = biased_weights(
                [ConstituentGroup.YOUNG, ConstituentGroup.MIDDLE_AGE, ConstituentGroup.OLD],
                [(geo_weights[g], geo_gen[g]) for g in geo_types] +
                [(econ_comp[ConstituentGroup(k)], v) for k, v in eco_gen.items()],
            )

            geo_group_map = {
                "rural": ConstituentGroup.RURAL,
                "urban": ConstituentGroup.URBAN,
                "suburban": ConstituentGroup.SUBURBAN,
            }
            composition = {
                **econ_comp, **cult_comp, **gen_comp,
                **{geo_group_map[k]: v for k, v in geo_weights.items()},
            }

            self.constituent_profiles[seat_id] = ConstituentProfile(
                ideology=ideology,
                composition=composition,
                approval=float(np.clip(self.rng.normal(0.65, 0.05), 0.5, 0.8)),
                approval_decay_rate=float(self.rng.uniform(0.01, 0.1)),
            )

    def _init_senators(self) -> None:
        for seat_id in range(self.n_seats):
            Senator(
                model=self,
                seat_id=seat_id,
                constituent_profile=self.constituent_profiles[seat_id],
                ideology_noise=self.ideology_noise,
                reputation=float(np.clip(self.rng.normal(0.5, 0.1), 0.25, 0.75)),
            )
        all_ids = [s.unique_id for s in self.agents_by_type[Senator]]
        for senator in self.agents_by_type[Senator]:
            senator.init_relationships(all_ids)

    def _replace_senator(self, seat_id: int, old_senator: Senator) -> None:
        old_senator.remove()
        profile = self.constituent_profiles[seat_id]
        profile.approval = 0.5
        new_s = Senator(model=self, seat_id=seat_id, constituent_profile=profile, ideology_noise=self.ideology_noise)
        new_s.init_relationships([s.unique_id for s in self.agents_by_type[Senator]])
        log.debug("  Seat %d: new senator elected (id=%d)", seat_id, new_s.unique_id)

    # --- ID helpers ---

    def _alloc_bill_id(self) -> int:
        v = self._next_bill_id; self._next_bill_id += 1; return v

    def _alloc_provision_id(self) -> int:
        v = self._next_provision_id; self._next_provision_id += 1; return v

    def _alloc_deal_id(self) -> int:
        v = self._next_deal_id; self._next_deal_id += 1; return v

    # --- Bill factory ---

    def _create_bill(self, author: Senator) -> Bill:
        groups   = list(ConstituentGroup)
        ptypes   = list(ProvisionType)
        surfaces = list(AmendmentSurface)
        provisions = []

        for _ in range(int(self.rng.integers(1, 4))):
            lo, hi = 0.0, 20.0
            ideology_position = Ideology(
                float(np.clip(author.personal_ideology.economic + self.rng.normal(0, 0.2), -1, 1)),
                float(np.clip(author.personal_ideology.social   + self.rng.normal(0, 0.2), -1, 1)),
            )

            compositions = np.array([author.constituent_profile.composition.get(g, 0.0) for g in groups])
            compositions /= compositions.sum()

            n_g       = int(self.rng.integers(2, 5))
            n_winners = max(1, (n_g + 1) // 2)
            winner_indices = self.rng.choice(len(groups), size=n_winners, replace=False, p=compositions).tolist()

            loser_p   = 1.0 - compositions; loser_p /= loser_p.sum()
            available = [i for i in range(len(groups)) if i not in winner_indices]
            avail_p   = np.array([loser_p[i] for i in available]); avail_p /= avail_p.sum()
            loser_indices = self.rng.choice(available, size=min(n_g - n_winners, len(available)), replace=False, p=avail_p).tolist()

            bgroups, total_benefit = {}, 0.0
            for i in winner_indices:
                w = float(self.rng.uniform(0.3, 1.0))
                bgroups[groups[i]] = w
                total_benefit += w * author.constituent_profile.composition.get(groups[i], 0.0)

            total_loser_weight = sum(
                author.constituent_profile.composition.get(groups[i], 0.0) for i in loser_indices
            ) or 1.0
            cost_per_unit = total_benefit / total_loser_weight

            for i in loser_indices:
                bgroups[groups[i]] = -float(np.clip(cost_per_unit * self.rng.uniform(0.8, 1.2), 0.3, 1.0))

            provisions.append(Provision(
                id=self._alloc_provision_id(),
                type=ptypes[int(self.rng.integers(len(ptypes)))],
                parameter=float(self.rng.uniform(lo, hi)),
                parameter_range=(lo, hi),
                parameter_sensitivity=float(self.rng.uniform(0.1, 1.0)),
                ideology_position=ideology_position,
                beneficiary_groups=bgroups,
                amendment_surface=surfaces[int(self.rng.integers(len(surfaces)))],
                authored_by=author.unique_id,
            ))

        return Bill(
            id=self._alloc_bill_id(),
            authored_by=author.unique_id,
            provisions=provisions,
            session_introduced=self.current_session,
            status=BillStatus.DRAFT,
        )

    # --- Phase handlers ---

    def _phase_election_check(self) -> None:
        log.info("[Session %d] Phase: Election Check", self.current_session)
        reelected = replaced = 0
        for senator in list(self.agents_by_type[Senator]):
            senator.time_until_election -= 1
            if not senator.is_up_for_election:
                continue
            if senator.constituent_profile.approval >= self.approval_threshold:
                senator.constituent_profile.approval = 0.5
                senator.time_until_election = self.sessions_per_election_cycle
                reelected += 1
                log.debug("  Seat %d re-elected (approval=%.2f)", senator.seat_id, senator.constituent_profile.approval)
            else:
                log.debug("  Seat %d lost re-election (approval=%.2f)", senator.seat_id, senator.constituent_profile.approval)
                self._replace_senator(senator.seat_id, senator)
                replaced += 1
        log.info("  %d re-elected, %d replaced", reelected, replaced)

    def _phase_proposal(self) -> None:
        log.info("[Session %d] Phase: Proposal", self.current_session)
        senators = list(self.agents_by_type[Senator])
        authors  = self.rng.choice(senators, size=min(10, len(senators)), replace=False).tolist()

        self.bill_queue = []
        for senator in authors:
            bill = self._create_bill(senator)
            bill.status = BillStatus.PROPOSED
            senator.proposal_queue = bill
            self.bill_queue.append(bill)
        log.info("  %d bills drafted", len(self.bill_queue))

        if self.bill_queue:
            self.active_bill = self.bill_queue.pop(0)
            self.active_bill.status = BillStatus.NEGOTIATION
            log.info("  Active bill: id=%d authored_by=%d (%d provisions)",
                     self.active_bill.id, self.active_bill.authored_by, len(self.active_bill.provisions))

    def _phase_discussion(self) -> None:
        log.info("[Session %d] Phase: Discussion (bill %d)", self.current_session,
                 self.active_bill.id if self.active_bill else -1)
        if self.active_bill is None:
            return
        stance_counts: dict[str, int] = {}
        for senator in self.agents_by_type[Senator]:
            senator.current_stance = senator.evaluate_bill(self.active_bill)
            k = senator.current_stance.value
            stance_counts[k] = stance_counts.get(k, 0) + 1
        log.info("  Stances: %s", "  ".join(f"{k}={v}" for k, v in sorted(stance_counts.items())))

    def _phase_negotiation(self) -> None:
        log.info("[Session %d] Phase: Negotiation (bill %d)", self.current_session,
                 self.active_bill.id if self.active_bill else -1)
        if self.active_bill is None:
            return

        senators   = list(self.agents_by_type[Senator])
        for_s      = [s for s in senators if s.current_stance == VoteStance.FOR]
        against_s  = [s for s in senators if s.current_stance == VoteStance.AGAINST]
        undecided  = [s for s in senators if s.current_stance == VoteStance.UNDECIDED]

        if (not for_s and not against_s) or not undecided:
            log.info("  No deals possible (FOR=%d, AGAINST=%d, UNDECIDED=%d)", len(for_s), len(against_s), len(undecided))
            return

        # Build the best offer per undecided target from both sides, then resolve
        best_offer: dict[int, tuple[float, Deal, Senator]] = {}
        for vote_side, creditors in [("yes", for_s), ("no", against_s)]:
            for creditor in creditors:
                targets = sorted(undecided, key=lambda s: creditor.relationships.get(s.unique_id, 0.0), reverse=True)
                for target in targets:
                    if target.at_deal_capacity:
                        continue
                    deal = Deal(
                        id=self._alloc_deal_id(),
                        offered_by=creditor.unique_id,
                        offered_to=target.unique_id,
                        trigger_bill_id=self.active_bill.id,
                        obligation=DealObligation(
                            obligation_type="vote",
                            target_bill_id=self.active_bill.id,
                            vote=vote_side,
                        ),
                        expiry=1,
                        status=DealStatus.PROPOSED,
                    )
                    score    = target._deal_acceptance_score(deal, self.active_bill)
                    existing = best_offer.get(target.unique_id)
                    if existing is None or score > existing[0]:
                        best_offer[target.unique_id] = (score, deal, creditor)
                    break  # one offer attempt per creditor per bill

        deals_made = deals_rejected = 0
        for target in undecided:
            entry = best_offer.get(target.unique_id)
            if entry is None:
                continue
            score, deal, creditor = entry
            if score >= 0.5:
                deal.status = DealStatus.ACCEPTED
                target.deals_owed_to_others.append(deal)
                creditor.deals_owed_to_me.append(deal)
                creditor.relationships[target.unique_id]  = min(1.0, creditor.relationships.get(target.unique_id, 0.0) + 0.05)
                target.relationships[creditor.unique_id]  = min(1.0, target.relationships.get(creditor.unique_id, 0.0) + 0.05)
                deals_made += 1
                log.debug("  Deal accepted (%s): senator %d -> %d (score=%.2f)",
                          deal.obligation.vote, creditor.unique_id, target.unique_id, score)
            else:
                deal.status = DealStatus.REJECTED
                creditor.relationships[target.unique_id] = max(-1.0, creditor.relationships.get(target.unique_id, 0.0) - 0.02)
                deals_rejected += 1
                log.debug("  Deal rejected: senator %d -> %d (score=%.2f)", creditor.unique_id, target.unique_id, score)

        log.info("  Deals made=%d  rejected=%d  remaining undecided=%d", deals_made, deals_rejected, len(undecided))

    def _phase_revision(self) -> None:
        log.info("[Session %d] Phase: Revision (bill %d)", self.current_session,
                 self.active_bill.id if self.active_bill else -1)
        if self.active_bill:
            self.active_bill.status = BillStatus.VOTING

    def _phase_voting(self) -> None:
        log.info("[Session %d] Phase: Voting (bill %d)", self.current_session,
                 self.active_bill.id if self.active_bill else -1)
        if self.active_bill is None:
            return
        vote_record: dict[int, str] = {s.unique_id: s.vote_from_stance().value for s in self.agents_by_type[Senator]}
        self.active_bill.vote_record = vote_record
        tally: dict[str, int] = {}
        for v in vote_record.values():
            tally[v] = tally.get(v, 0) + 1
        log.info("  Tally: %s", "  ".join(f"{k}={v}" for k, v in sorted(tally.items())))

    def _phase_resolution(self) -> None:
        log.info("[Session %d] Phase: Resolution (bill %d)", self.current_session,
                 self.active_bill.id if self.active_bill else -1)
        if self.active_bill is None:
            return

        votes = self.active_bill.vote_record
        yes_n = sum(1 for v in votes.values() if v == Vote.YES.value)
        no_n  = sum(1 for v in votes.values() if v == Vote.NO.value)
        total = yes_n + no_n

        passed = total > 0 and yes_n / total > self.vote_majority
        if passed:
            self.active_bill.status = BillStatus.PASSED
            self.passed_bills.append(self.active_bill.id)
            log.info("  Bill %d PASSED (%d yes / %d no)", self.active_bill.id, yes_n, no_n)
        else:
            self.active_bill.status = BillStatus.FAILED
            self.failed_bills.append(self.active_bill.id)
            log.info("  Bill %d FAILED (%d yes / %d no)", self.active_bill.id, yes_n, no_n)
        self.resolved_bills.append(self.active_bill)

        approval_deltas: list[float] = []
        for senator in self.agents_by_type[Senator]:
            my_vote = votes.get(senator.unique_id, Vote.ABSTAIN.value)
            delta   = _constituent_approval_delta(senator, self.active_bill, passed)
            approval_deltas.append(delta)

            profile = senator.constituent_profile
            profile.approval = float(np.clip(profile.approval + delta, 0.0, 1.0))
            profile.approval += (0.5 - profile.approval) * profile.approval_decay_rate

            aligned = (
                (my_vote == Vote.YES.value and senator.current_stance == VoteStance.FOR) or
                (my_vote == Vote.NO.value  and senator.current_stance == VoteStance.AGAINST)
            )
            senator.session_vote_alignment.append(1.0 if aligned else 0.0)

        log.info("  Approval: avg_delta=%+.3f  avg_approval=%.3f",
                 float(np.mean(approval_deltas)),
                 float(np.mean([s.constituent_profile.approval for s in self.agents_by_type[Senator]])))

        senator_list = list(self.agents_by_type[Senator])
        for i, s1 in enumerate(senator_list):
            for s2 in senator_list[i + 1:]:
                v1, v2 = votes.get(s1.unique_id), votes.get(s2.unique_id)
                if v1 is None or v2 is None:
                    continue
                d = 0.01 if v1 == v2 else -0.01
                s1.relationships[s2.unique_id] = max(-1.0, min(1.0, s1.relationships.get(s2.unique_id, 0.0) + d))
                s2.relationships[s1.unique_id] = max(-1.0, min(1.0, s2.relationships.get(s1.unique_id, 0.0) + d))

        self.session_history.append({
            "session": self.current_session,
            "bill_id": self.active_bill.id,
            "status": self.active_bill.status.value,
            "yes": yes_n,
            "no": no_n,
            "votes": dict(votes),
        })

        if self.bill_queue:
            self.active_bill = self.bill_queue.pop(0)
            self.active_bill.status = BillStatus.NEGOTIATION
            log.info("  Next bill: id=%d (%d remaining in queue)", self.active_bill.id, len(self.bill_queue))
        else:
            log.info("  Session %d complete — passed=%d  failed=%d",
                     self.current_session, len(self.passed_bills), len(self.failed_bills))
            self.active_bill = None
            self.current_session += 1
            for senator in self.agents_by_type[Senator]:
                senator.reset_session_state()

    # --- Mesa step ---

    _PHASE_HANDLERS = {
        SessionPhase.ELECTION_CHECK: "_phase_election_check",
        SessionPhase.PROPOSAL:       "_phase_proposal",
        SessionPhase.DISCUSSION:     "_phase_discussion",
        SessionPhase.NEGOTIATION:    "_phase_negotiation",
        SessionPhase.REVISION:       "_phase_revision",
        SessionPhase.VOTING:         "_phase_voting",
        SessionPhase.RESOLUTION:     "_phase_resolution",
    }

    def step(self) -> None:
        getattr(self, self._PHASE_HANDLERS[self.current_phase])()
        if self.current_phase == SessionPhase.RESOLUTION and self.active_bill is not None:
            self.current_phase = SessionPhase.DISCUSSION
        else:
            idx = _PHASE_ORDER.index(self.current_phase)
            self.current_phase = _PHASE_ORDER[(idx + 1) % len(_PHASE_ORDER)]
        self.datacollector.collect(self)

    def run_session(self) -> None:
        log.info("=== Starting session %d ===", self.current_session)
        start = self.current_session
        while self.current_session == start:
            self.step()
        log.info("=== Session %d finished ===", self.current_session - 1)
