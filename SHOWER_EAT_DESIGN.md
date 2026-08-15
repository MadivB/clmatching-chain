# Eager shower absorption — design record (UNVERSIONED until proven; then v0.9.0)

*(User scheme, 2026-08-15. No version name until measured more efficient or
more accurate than v0.8.8.)*

## The idea (user)

Showers must be treated differently: once a shower is recognized AND
placed, immediately let it absorb the surrounding hits that clearly belong
to it — **unless a candidate touches another non-track cluster** (the
two-showers-overlap protection).

Two trigger paths:

1. **Explicit**: the clustering stage already outputs shower-typed
   clusters; the moment one is placed → trigger absorption around it.
2. **Emergent**: every time a big (>30 MeV) cluster is placed, check
   whether multiple clusters at NEIGHBORING locations are placed at the
   SAME t0 — and they are not just a linear coincidence (not two distant
   collinear tracks). If so, a shower exists in this TPC even though the
   clustering never typed one; surrounding hits gain a tendency toward
   that t0 (unless overflow or the amplitude doesn't fit).

## The eating stage

Once a shower is recognized at t0_s: gradually absorb surrounding spatial
clusters **until 60 % of the shower's needed photon/amplitude at t0_s is
fulfilled** (light-closure target, measured on the shower's support
channels: explained fraction F = Σ min(predicted, actual) / Σ actual).

Priority is NOT spatial distance. Rank candidates by how much they WANT
this t0: the ratio between the candidate's own light loss at its best
alternative and at t0_s. A spatially-far cluster that strongly prefers
t0_s (large best/second-best ratio) beats a spatially-near cluster that is
ambivalent (its own preference is another t0 almost as good). Guards per
absorption: overflow check (support-restricted), amplitude fit, and the
non-track-contact veto.

## Chain integration (prototype)

Both hooks live in the backbone section (pre-namespace), operating on the
same structures as the pool (b_full, hit_t0, image_maps, info_b):

- **Hook A** — after Phase-1 places shower-typed + multi-TPC clusters
  into b_full: absorb around each placed shower label.
- **Hook B** — after the merged pool commits: detect emergent shower
  groups among placed big clusters (same t0 within 3 ticks, members within
  30 cm, and non-collinear OR containing a non-track member OR >= 3
  members) and absorb around each group.

Candidates: not-yet-placed imaged clusters below the P2 boundary (30 MeV);
big clusters stay with the pool. Absorbed labels are registered through the
existing `merged_placed` plumbing (t0-candidate registration,
assignment_info, worklist purges) — no new downstream paths. Everything
truth-audited per absorption.

Flag: `--shower-eat` (off by default). A/B vs v0.8.8 on file 0000001, both
headline metrics + the shower-TPC cohort; adoption bar unchanged.

## Relation to prior art in this chain

- NOT the shelved shower-TPC dedicated path (that REPLACED the whole-TPC
  flow: redo −15pp). This augments the standard flow.
- Generalizes `--early-expand` (stage 3.25) with: shower-specific
  triggering, the 60 % light-closure target, preference-margin ranking
  instead of geodesic distance, and the two-shower contact veto.
- The preference-margin idea is the user's best/second-best ratio — the
  same regret notion as v0.3.0 expansion, sharpened into the primary
  ranking key.
