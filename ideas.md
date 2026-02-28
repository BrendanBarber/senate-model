### Create a chamber visualization

*Shows the bills, senators, and coalitions. This can be used to figure out why
bills are passing so easily.*

### Negotiation phase logic for SenateModel.

Deal-making flow:
  1. FOR senators identify UNDECIDED senators as targets (sorted by relationship score)
  2. Each FOR senator attempts to offer deals until they hit capacity or run out of targets
  3. UNDECIDED senators evaluate offers based on: relationship, electoral pressure, and
     whether the bill's demographic impact on their district is positive enough to risk it
  4. Accepted deals commit the debtor to vote YES; this is resolved in vote_from_stance

**Need to implement revision phase**
