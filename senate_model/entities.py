from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


@dataclass
class Ideology:
    """2-D ideological point, both axes in [-1, 1].
    economic: -1 = interventionist, +1 = market-oriented
    social:   -1 = progressive,     +1 = traditional
    """
    economic: float
    social: float
    MAX_DISTANCE: float = field(default=2.0 ** 0.5 * 2, init=False, repr=False, compare=False)

    def __post_init__(self):
        self.economic = max(-1.0, min(1.0, float(self.economic)))
        self.social   = max(-1.0, min(1.0, float(self.social)))

    def distance_to(self, other: Ideology) -> float:
        return ((self.economic - other.economic) ** 2 + (self.social - other.social) ** 2) ** 0.5


class ConstituentGroup(str, Enum):
    INDUSTRIAL_LABOR   = "industrial_labor"
    AGRICULTURAL_LABOR = "agricultural_labor"
    SMALL_BUSINESS     = "small_business"
    LARGE_BUSINESS     = "large_business"
    PUBLIC_SECTOR      = "public_sector"
    URBAN              = "urban"
    SUBURBAN           = "suburban"
    RURAL              = "rural"
    RELIGIOUS          = "religious"
    SECULAR            = "secular"
    YOUNG              = "young"
    MIDDLE_AGE         = "middle_age"
    OLD                = "old"


CONSTITUENT_CATEGORIES: dict[str, list[ConstituentGroup]] = {
    "economic": [
        ConstituentGroup.INDUSTRIAL_LABOR, ConstituentGroup.AGRICULTURAL_LABOR,
        ConstituentGroup.SMALL_BUSINESS, ConstituentGroup.LARGE_BUSINESS,
        ConstituentGroup.PUBLIC_SECTOR,
    ],
    "geographic":   [ConstituentGroup.URBAN, ConstituentGroup.SUBURBAN, ConstituentGroup.RURAL],
    "cultural":     [ConstituentGroup.RELIGIOUS, ConstituentGroup.SECULAR],
    "generational": [ConstituentGroup.YOUNG, ConstituentGroup.MIDDLE_AGE, ConstituentGroup.OLD],
}


@dataclass
class ConstituentProfile:
    """Ideological and demographic character of a district. Persists across senator replacements."""
    ideology: Ideology
    composition: dict[ConstituentGroup, float]  # weights summing to 1
    approval: float = 0.5
    approval_decay_rate: float = 0.05


class ProvisionType(str, Enum):
    TAX        = "Tax"
    SPENDING   = "Spending"
    REGULATORY = "Regulatory"
    SOCIAL     = "Social"


class AmendmentSurface(str, Enum):
    OPEN       = "open"        # freely negotiable
    RESTRICTED = "restricted"  # can shift slightly
    LOCKED     = "locked"      # accept-or-remove only


@dataclass
class Provision:
    id: int
    type: ProvisionType
    parameter: float
    parameter_range: tuple[float, float]
    parameter_sensitivity: float
    ideology_position: Ideology
    beneficiary_groups: dict[ConstituentGroup, float]  # group -> impact weight (+/-)
    amendment_surface: AmendmentSurface
    authored_by: int  # senator unique_id

    @property
    def label(self) -> str:
        top = sorted(
            [(g, w) for g, w in self.beneficiary_groups.items() if w > 0],
            key=lambda x: -x[1],
        )[:2]
        group_str = ", ".join(g.value for g, _ in top) if top else "general"
        if self.type == ProvisionType.TAX:
            return f"Tax {self.parameter:.1f}% affecting {group_str}"
        elif self.type == ProvisionType.SPENDING:
            return f"Spending ${self.parameter:.1f}B affecting {group_str}"
        else:
            return f"{self.type.value} {self.parameter:.2f} affecting {group_str}"


class BillStatus(str, Enum):
    DRAFT       = "draft"
    PROPOSED    = "proposed"
    NEGOTIATION = "negotiation"
    VOTING      = "voting"
    PASSED      = "passed"
    FAILED      = "failed"
    TABLED      = "tabled"


@dataclass
class AmendmentRecord:
    provision_id: int
    old_value: float
    new_value: float
    session_timestamp: int


@dataclass
class Bill:
    id: int
    authored_by: int  # senator unique_id
    provisions: list[Provision]
    session_introduced: int
    status: BillStatus = BillStatus.DRAFT
    vote_record: dict[int, str] = field(default_factory=dict)
    amendment_history: list[AmendmentRecord] = field(default_factory=list)

    def label(self, senate_centroid: Optional[Ideology] = None) -> str:
        if not self.provisions:
            return f"Bill #{self.id}"
        if senate_centroid is not None:
            key = lambda p: p.ideology_position.distance_to(senate_centroid)
        else:
            key = lambda p: (p.ideology_position.economic ** 2 + p.ideology_position.social ** 2) ** 0.5
        return f"Bill #{self.id}: {max(self.provisions, key=key).label}"


class DealStatus(str, Enum):
    PROPOSED = "proposed"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    HONORED  = "honored"
    BROKEN   = "broken"


@dataclass
class DealObligation:
    obligation_type: str                       # "vote" | "provision_stance"
    target_bill_id: Optional[int] = None
    vote: Optional[str] = None                 # "yes" | "no"
    provision_type: Optional[ProvisionType] = None
    stance: Optional[str] = None              # "support" | "oppose"


@dataclass
class Deal:
    id: int
    offered_by: int       # creditor senator unique_id
    offered_to: int       # debtor senator unique_id
    trigger_bill_id: int
    obligation: DealObligation
    expiry: int           # sessions until deal lapses if trigger never fires
    status: DealStatus = DealStatus.PROPOSED
