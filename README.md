# senate-model
A simulation of a unicameral legislature using Mesa agent-based modeling.

---

## Ideology

A 2D point in continuous space. Both axes are floats clamped to [-1, 1] on construction.

- **Economic axis**: -1 = interventionist, +1 = market-oriented
- **Social axis**: -1 = progressive, +1 = traditional

`MAX_DISTANCE` is the diagonal of the space: `sqrt(2) * 2 ≈ 2.828`.

Every senator and every provision has a position in this space.

---

## Constituent Profile

Represents the district a senator comes from. Persists across senator replacements; the seat retains its profile when a senator is voted out.

- **Ideology**: 2D point (the district's ideological center of mass)
- **Composition**: map of `ConstituentGroup` -> float, weights summing to 1. Groups span four categories:
  - *Economic*: `industrial_labor`, `agricultural_labor`, `small_business`, `large_business`, `public_sector`
  - *Geographic*: `urban`, `suburban`, `rural`
  - *Cultural*: `religious`, `secular`
  - *Generational*: `young`, `middle_age`, `old`
- **Approval**: float 0-1; initialized from `clip(normal(0.65, 0.05), 0.5, 0.8)` at seat creation. Resets to `0.5` when a senator is re-elected or replaced.
- **Approval decay rate**: float sampled from `uniform(0.01, 0.1)`. Applied each resolution as `approval += (0.5 - approval) * decay_rate`, pulling approval toward neutral after every vote regardless of bill outcome.

### Composition Generation

Composition is not drawn independently per category. Geographic weights (`rural`, `urban`, `suburban`) are drawn from a 3-group Dirichlet, then used as mixing weights to bias the economic, cultural, and generational draws via cross-correlation tables (`geographic_economic`, `geographic_cultural`, `geographic_generational`). Economic groups additionally bias cultural and generational groups via `economic_cultural` and `economic_generational` tables. All per-group values are clamped to `[min_within_category, max_within_category]` before renormalization. These correlation tables are loaded from `config`.

---

## Senator

### Identity
- **`unique_id`**: assigned by Mesa on construction; changes when a senator is replaced.
- **`seat_id`**: permanent seat identifier, survives replacements.
- **Personal ideology**: 2D point, sampled as `clip(constituent_ideology_axis + normal(0, noise), -1, 1)` on each axis independently.
- **Constituent profile**: reference to the seat's `ConstituentProfile`.

### Election State
- **`time_until_election`**: integer sessions. Initialized as `(seat_id % 3 + 1) * (sessions_per_cycle // 3)`, staggering three cohorts evenly. Decremented by 1 each Election Check phase.
- **`is_up_for_election`**: true when `time_until_election <= 0`.

### Deal State
- **`deal_tolerance`**: integer sampled from `randint(1, 6)` (1-5 inclusive), hidden from other senators.
- **`deals_owed_to_others`**: list of `Deal` objects this senator must honor.
- **`deals_owed_to_me`**: list of `Deal` objects others owe this senator.
- **`at_deal_capacity`**: true when `len(deals_owed_to_others) >= deal_tolerance`. Other senators observe only the refusal, not the reason.

Deals with `expiry <= 1` are pruned from both lists at the start of each session (`reset_session_state`). A deal with `expiry=1` survives exactly one more reset before being dropped.

### Relationships and Knowledge
- **`relationships`**: map of `senator_id (unique_id)` -> float in `[-1, 1]`. Initialized to `normal(0, 0.05)` for all peers.
- **`inferred_models`**: map of `senator_id` -> list of `(ProvisionType, Ideology)` pairs. Initialized empty; not yet populated by the Resolution phase in the current implementation.
- **`confidence_scores`**: map of `senator_id` -> float. Initialized to `0.0`.
- **`reputation`**: float 0-1, visible to all. Initialized from `clip(normal(0.5, 0.1), 0.25, 0.75)` at model start; replacement senators start at exactly `0.5`.

### Session State
- **`current_stance`**: `VoteStance` enum (`for` / `against` / `undecided`). Reset to `undecided` each session.
- **`proposal_queue`**: the `Bill` this senator will propose this session, or `None`.
- **`session_vote_alignment`**: running list of 0.0/1.0 values recording whether each vote matched the senator's stance.

---

## Senator Generation

Used at simulation start and when a senator is replaced after a failed election.

1. **Personal ideology**: sample each axis as `clip(constituent_ideology_axis + normal(0, ideology_noise), -1, 1)`.
2. **Deal tolerance**: sample from `randint(1, 6)` (1-5 inclusive), hidden.
3. **Reputation**: `clip(normal(0.5, 0.1), 0.25, 0.75)` at model start; `0.5` for replacement senators.
4. **Relationships**: `normal(0, 0.05)` for all current peers. Inferred models and confidence scores initialized empty/zero.
5. **Election offset**: `(seat_id % 3 + 1) * (sessions_per_cycle // 3)`.

When replacing a senator, the old senator is removed (`senator.remove()`), outstanding deals on both lists lapse, and relationship scores are reinitialized. The seat's `ConstituentProfile` is retained; its approval resets to `0.5`.

---

## Provision

- **`id`**: integer, allocated from a global counter.
- **`type`**: `ProvisionType` enum: `Tax`, `Spending`, `Regulatory`, `Social`.
- **`parameter`**: float sampled from `uniform(0, 20)`.
- **`parameter_range`**: `(0.0, 20.0)`.
- **`parameter_sensitivity`**: float from `uniform(0.1, 1.0)`.
- **`ideology_position`**: 2D point, sampled near the author's personal ideology with `normal(0, 0.2)` noise per axis.
- **`beneficiary_groups`**: map of `ConstituentGroup` -> impact weight. Positive = benefit, negative = burden. See Bill Factory for generation logic.
- **`amendment_surface`**: `AmendmentSurface` enum: `open`, `restricted`, `locked`. Sampled uniformly.
- **`authored_by`**: author senator's `unique_id`.

### Automatic Label

`Tax {param:.1f}% affecting {top groups}` / `Spending ${param:.1f}B affecting {top groups}` / `{Type} {param:.2f} affecting {top groups}`. Used for display only.

---

## Bill

- **`id`**: integer, allocated from a global counter.
- **`authored_by`**: author senator's `unique_id`.
- **`provisions`**: list of 1-3 `Provision` objects (sampled from `randint(1, 4)`).
- **`status`**: `BillStatus` enum: `draft`, `proposed`, `negotiation`, `voting`, `passed`, `failed`, `tabled`.
- **`vote_record`**: map of `senator unique_id` -> vote string (`yes` / `no` / `abstain`), populated during Voting.
- **`amendment_history`**: list of `AmendmentRecord(provision_id, old_value, new_value, session_timestamp)`. Not populated in the current implementation (no amendment offers are made during negotiation).
- **`session_introduced`**: session integer.

### Bill Label

`Bill #{id}: {most_extreme_provision.label}` where extremity is distance from the senate ideology centroid if provided, otherwise distance from origin.

### Bill Factory

Each provision is generated as follows:

1. Draw `n_g` from `randint(2, 5)`, the total number of affected groups.
2. `n_winners = max(1, (n_g + 1) // 2)`, a majority of `n_g`.
3. Winner groups are sampled without replacement from all `ConstituentGroup` values, weighted by the author's district composition.
4. Loser groups are sampled without replacement from the remaining groups, weighted inversely by composition.
5. Each winner gets a raw benefit weight from `uniform(0.3, 1.0)`.
6. Total benefit is the sum of `benefit_weight * district_composition[group]` for each winner.
7. `cost_per_unit = total_benefit / sum(district_composition[loser])`. Cost is calibrated to balance total benefit relative to district composition weights.
8. Each loser gets `−clip(cost_per_unit * uniform(0.8, 1.2), 0.3, 1.0)`.

> Note: this cost calibration is relative to the *author's* district composition, which can produce imbalanced raw weights for groups underrepresented in that district.

---

## Deal

- **`id`**: integer, allocated from a global counter.
- **`offered_by`**: creditor senator's `unique_id`.
- **`offered_to`**: debtor senator's `unique_id`.
- **`trigger_bill_id`**: bill ID that activates this deal.
- **`obligation`**: `DealObligation` with fields `obligation_type` (`"vote"` or `"provision_stance"`), `target_bill_id`, `vote` (`"yes"` / `"no"`), `provision_type`, `stance`. Only `"vote"` obligations are created by the current negotiation logic.
- **`expiry`**: sessions until the deal lapses. Deals with `expiry <= 1` are pruned at each `reset_session_state`. A deal with `expiry=1` survives one more reset.
- **`status`**: `DealStatus` enum: `proposed`, `accepted`, `rejected`, `honored`, `broken`.

Deal-breaking consequences (reputation/relationship penalties) are defined in the data model but not yet enforced in the simulation loop.

---

## Model Time

- **`current_session`**: integer, incremented when the bill queue empties after Resolution.
- **`sessions_per_election_cycle`**: constant (default 6), tunable.
- **Election cohorts**: three cohorts with offsets `1×(cycle//3)`, `2×(cycle//3)`, `3×(cycle//3)` sessions.

---

## Senator Election and Replacement

At the start of each session, `time_until_election` is decremented for every senator before checking elections. Senators with `time_until_election <= 0` after decrement are evaluated:

- If `approval >= approval_threshold` (default 0.5): re-elected. Approval resets to `0.5`, `time_until_election` resets to `sessions_per_election_cycle`.
- If `approval < approval_threshold`: removed and replaced. New senator generated from the same seat's `ConstituentProfile` (approval also resets to `0.5`).

---

## Model State Machine

Each call to `step()` executes one phase and advances `current_phase`. `run_session()` calls `step()` in a loop until `current_session` increments.

The session processes bills one at a time. The first bill becomes active during Proposal. Subsequent bills are dequeued in Resolution. The session ends (and `current_session` increments) when Resolution finds the bill queue empty.

### Phase Order

```
Election Check -> Proposal -> Discussion -> Negotiation -> Revision -> Voting -> Resolution
                                  ^                                                  |
                                  +-------------- (if queue non-empty) --------------+
```

When Resolution finds more bills in the queue, `current_phase` is set back to `Discussion` directly, not back to Proposal. The cycle fully restarts at Election Check only when the queue is exhausted and the session increments.

### 1. Election Check

Decrement `time_until_election` for all senators. Evaluate and replace/re-elect those at zero.

### 2. Proposal

Up to 10 senators are selected at random (without replacement) to author bills. Each generates one bill via the Bill Factory; the bill is set to `PROPOSED` status and added to `bill_queue`. The first bill is immediately dequeued, set to `NEGOTIATION`, and assigned as `active_bill`.

### 3. Discussion

Each senator calls `evaluate_bill(active_bill)` and sets their `current_stance`. The evaluation blends personal ideology alignment, constituent ideology alignment, and district impact, with electoral pressure (derived from current approval) shifting weight from impact toward constituent alignment when approval is low.

```
score = 0.25 * ideology_score
      + (0.35 + 0.20 * pressure) * constituent_score
      + (0.40 - 0.20 * pressure) * impact_score

pressure = max(0.0, 1.0 - (approval - 0.5) * 4.0)
```

`score >= 0.55` -> `FOR`; `score <= 0.45` -> `AGAINST`; otherwise `UNDECIDED`.

### 4. Negotiation

The model attempts to move undecided senators by brokering deals. No amendment offers are made in the current implementation.

For each side (`FOR` senators offering "yes" deals, `AGAINST` senators offering "no" deals), each creditor targets undecided senators ordered by descending relationship score, skipping any target already at deal capacity. Only one offer attempt is made per creditor per bill.

The best-scoring offer per undecided target (across all creditors from both sides) is resolved:

- **`_deal_acceptance_score`**:
  ```
  score = 0.40 * (relationship + 1) / 2
        + 0.40 * (district_impact + 1) / 2
        + 0.20 * electoral_pressure
  ```
- If `score >= 0.5`: deal accepted. Both parties append the deal to their respective lists. Relationship scores for both parties increase by `+0.05`.
- If `score < 0.5`: deal rejected. Creditor's relationship with the target decreases by `-0.02`.

### 5. Revision

`active_bill.status` is set to `VOTING`. No parameter amendments are applied in the current implementation.

### 6. Voting

Each senator calls `vote_from_stance()`:
1. If the senator has an accepted deal with `obligation_type == "vote"` targeting this bill, that obligation is honored.
2. Otherwise: `FOR` -> `YES`, `AGAINST` -> `NO`, `UNDECIDED` -> `ABSTAIN`.

Votes are recorded in `bill.vote_record` as strings.

### 7. Resolution

**Pass/fail**: `yes / (yes + no) > vote_majority` (default 0.5). Abstentions are excluded from the denominator. A bill with zero yes+no votes fails.

**Approval update** per senator:
```
delta = clip(net_impact * 0.3 + (±0.03 alignment bonus), -0.15, +0.15)
approval = clip(approval + delta, 0, 1)
approval += (0.5 - approval) * decay_rate   # decay toward neutral
```
`net_impact` is the weighted average provision impact on the senator's district, sign-flipped if the bill failed. The `±0.03` bonus is added if the senator's public vote matched their stance, subtracted otherwise.

**Relationship update**: for every senator pair, if their votes match, both relationships increase by `+0.01`; if they differ, both decrease by `-0.01`.

**Inferred model update**: not yet implemented.

**Session history**: one entry appended per bill: `{session, bill_id, status, yes, no}`. Relationship deltas, deal counts, and amendment records are not currently stored in session history.

**Queue management**: if `bill_queue` is non-empty, the next bill is dequeued and `current_phase` is set to `Discussion`. Otherwise `active_bill` is cleared, `current_session` increments, and all senators call `reset_session_state()`.

---

## Global Model Fields

- **`bill_queue`**: bills queued for the current session.
- **`active_bill`**: bill currently being processed.
- **`current_phase`**: `SessionPhase` enum.
- **`passed_bills`**: list of passed bill IDs.
- **`failed_bills`**: list of failed bill IDs.
- **`resolved_bills`**: list of fully-resolved `Bill` objects (both passed and failed), used for impact analysis.
- **`session_history`**: list of `{session, bill_id, status, yes, no}` dicts, one per resolved bill.
- **`coalition_tracker`**: dict field present but not populated by the simulation loop.
- **`constituent_profiles`**: map of `seat_id` -> `ConstituentProfile`, persists across replacements.
- **`datacollector`**: Mesa `DataCollector` recording model- and agent-level stats each step.
