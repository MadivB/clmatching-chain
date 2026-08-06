"""Build the v0.3.0 algorithm + failure-analysis deck in Billy's SlideKit style."""
import os, sys
KIT = "/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/skills/SlideKit"
sys.path.append(KIT)
from slidekit import new_deck, set_title_slide, add_content_slide, finish, fit_image

HERE = "/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/CLMatching_v0.1_backboneOnly/testing"
F = os.path.join(HERE, "deck_figs")
G = os.path.join(HERE, "bad_tpc_gallery")
MEET = "ND Prototype Analysis Meeting"
def fig(name): return os.path.join(F, name)
def gal(name): return os.path.join(G, name)
def R(path, max_w, max_h, top=1.85):
    """right-side image, left clamped so the right edge stays <= 12.85 in."""
    im = fit_image(path, left=0, top=top, max_w=max_w, max_h=max_h)
    im["left"] = round(12.85 - im["width"], 3)
    return im

prs = new_deck()
set_title_slide(prs,
    "Interaction-Level Charge-Light Matching\nv0.3.0: how the algorithm works, and where it fails",
    ["Yuxuan Wang (Billy)", "PI: Ryan Patterson", MEET])

def S(title, bullets=None, images=None, captions=None, body_width=None, body_height=None, notes=None):
    add_content_slide(prs, title=title, bullets=bullets, images=images, captions=captions,
                      body_width=body_width, body_height=body_height, meeting=MEET, notes=notes)

# ---- 2 · the task ----
img = R(fig("fig_lc_good.png"), 6.1, 5.1)
S("The task: give every charge hit a drift-time t0",
  ["ND-LAr spill: many neutrino / rock interactions overlap in each drift volume (TPC)",
   "Charge gives (y,z) and relative drift; the ABSOLUTE t0 (drift x) comes from matching to the scintillation-light flash",
   "Each TPC sees several flashes -> must decide which flash each charge cluster belongs to",
   "Success = |reco t0 - truth t0| <= 10 ticks (160 ns)",
   "v0.3.0 today: 94.9% of all hits correct (95.2% of non-shower energy), 100% coverage"],
  images=[img], body_width=6.2,
  captions=[dict(text="when placements are right, predicted (orange) tracks actual (blue) light",
                 left=img["left"], top=img["top"]+img["height"]+0.08, width=6.0)],
  notes="The charge tells us where in y-z and how far the electrons drifted, but not the absolute time. "
        "The light flash gives the absolute t0. In a busy TPC there are several flashes, so the whole game "
        "is deciding which flash each cluster of charge belongs to.")

# ---- 3 · pipeline overview ----
img = fit_image(fig("fig_pipeline.png"), left=0.4, top=2.7, max_w=11.9, max_h=2.9)
img["left"] = (13.33 - img["width"]) / 2
S("The pipeline at a glance",
  ["Strategy: place the confident, high-energy structures first, then peel off progressively "
   "smaller / more ambiguous clusters, and force-cover the tail"],
  images=[img], body_height=0.9,
  notes="High-level: we lock in the things we're sure about first - long tracks and showers - because "
        "they constrain the light budget. Then we work down to the small, dim clusters, and finally force "
        "everything to have an answer.")

# ---- 4 · clustering ----
S("Step 1 - Clustering: find the candidate interactions",
  ["Group charge hits into clusters per TPC: straight tracks, showers, blobs",
   "Intuition: one cluster ~ one interaction's charge, and it will receive ONE t0",
   "v0.2.5 vertex-merge: tracks emerging from a common vertex are merged into one multi-prong cluster",
   "- brighter and more constrained -> places as a single unit",
   "Why: fragmented prongs each pick a flash independently -> a classic source of swaps"],
  notes="Everything downstream is per-cluster, so clustering decides the units. A key recent addition is "
        "vertex-merge: if several prongs share a vertex we treat them as one object, because a bright merged "
        "object has a much less ambiguous t0 than a single faint prong.")

# ---- 5 · charge->light ----
img = R(fig("fig_lc_good.png"), 6.1, 5.1)
S("Step 2 - Predict the light each cluster would make",
  ["Charge2Light ML (perceiver): from a cluster's 3D charge, predict its per-channel light waveform",
   "This 'template' is the expected light image if the cluster flashed at t0 = 0",
   "Intuition: shift the template to any candidate t0 and compare to the actual light",
   "The whole matcher = place clusters at t0s so the SUM of templates best explains the actual light",
   "Prediction is ~80% of true amplitude (a systematic) but spatially faithful"],
  images=[img], body_width=6.2,
  captions=[dict(text="a correctly-placed TPC: sum of predicted templates (orange) explains the actual light (blue)",
                 left=img["left"], top=img["top"]+img["height"]+0.08, width=6.0)],
  notes="This ML step is the heart of it. For each cluster we predict what light it should have made, per "
        "channel and per time-tick. Then matching becomes: slide these templates in time and find the "
        "arrangement whose sum best reproduces the light we actually saw.")

# ---- 6 · backbone ----
S("Step 3 - Backbone: place the confident structures first",
  ["Order: showers -> >=3-TPC straight tracks -> energy-banded permutation batches for the 1-2 TPC pool",
   "Each placement scans t0; a soft-cosine matched filter picks the best flash (a hard gate hurts)",
   "Intuition: long tracks / showers are bright and span many channels -> their t0 is unambiguous",
   "- lock them first so they constrain everyone else's light budget",
   "Overflow trigger + rescue: unexplained EXCESS light (saturation, >=3 channels excess) means a cluster "
   "is mis-placed or a second one is hiding -> Stage A/B repair"],
  notes="We don't place everything at once. The bright, long objects have essentially one possible t0, so "
        "we commit those first. The overflow trigger is our safety net: if after placing something there's "
        "still a big pile of unexplained light, we know we got it wrong or missed a cluster, and we repair.")

# ---- 7 · P2 ----
S("Step 4 - P2: place the small-cluster pool jointly",
  ["The 1-2 TPC clusters left after the backbone are placed in energy order",
   "Energy-banded permutation (batch cap 5; commit N-1, last cluster returns to the pool)",
   "Intuition: within an energy band, jointly test which cluster->flash assignment fits best,",
   "- instead of a greedy one-at-a-time choice that locks in early mistakes",
   "P2-placed cohort accuracy: 92.5% -> 94.0% once the police (next step) is added"],
  notes="For the smaller clusters we test permutations within an energy band rather than committing greedily. "
        "The idea is that a few similar-energy clusters competing for a few flashes should be decided together.")

# ---- 8 · V2 + police ----
S("Step 5 - V2 + placement police: catch mistakes at commit",
  ["V2: after placement, re-examine each cluster - would moving it to another flash lower the light loss?",
   "Placement police: every P2 / V2 placement or move is PROBED at its t0 the instant it commits",
   "On alarm (overflow gate fires) resolve immediately: keep / rewind / rescan (rescan usually wins)",
   "Intuition: correct a bad placement BEFORE it corrupts the light budget that later clusters see",
   "Backbone integrity is asserted unchanged through the whole chain"],
  notes="V2 is a second-look move stage. The police is stricter: the moment any placement commits we probe "
        "it, and if it looks wrong we fix it on the spot, because one bad placement poisons the residual "
        "light that everyone downstream matches against.")

# ---- 9 · phase 3 ----
S("Steps 6-7 - Phase-3: absorb the tail, then force coverage",
  ["Sub-50 MeV clusters carry ~12% of energy but are too dim to place on their own",
   "Expansion: attach each to a placed interaction ('family') by geodesic distance + light 'regret' insertion",
   "- a family sitting at one t0 is bright and pins the attachment",
   "Matrix: an error-matrix sweeps whatever the expansion leaves",
   "Force-coverage: assign everything still unplaced by pure geometry -> 100% coverage",
   "Intuition: a small cluster inherits the t0 of the interaction it physically belongs to"],
  notes="The long tail of tiny clusters can't be matched on their own light. Instead we attach them to the "
        "bright interaction they geometrically belong to and let them inherit its t0. Force-coverage is the "
        "final mandate that nothing is left unassigned.")

# ---- 10 · results ----
img = R(fig("fig_results.png"), 6.1, 5.0, top=1.9)
S("Where we stand: v0.3.0 accuracy",
  ["94.9% of ALL hits get the correct t0 (94.6% energy-weighted)",
   "Non-shower TPCs (<=1.4 GeV): 95.2% of energy correct",
   "Shower TPCs (>1.4 GeV): 93.3% - a dedicated approach comes later",
   "100% coverage (force-coverage assigns everything)",
   "39 MiniProdN5p1 events; |reco-truth| <= 10 ticks"],
  images=[img], body_width=6.2,
  notes="These are energy-weighted, 100% coverage numbers. The shower walls above 1.4 GeV are the weakest "
        "and we've scoped a separate path for them.")

# ---- 11 · taxonomy ----
img = R(fig("fig_taxonomy.png"), 5.8, 4.9, top=1.95)
S("Where it fails: 91% of mistakes are t0 SWAPS",
  ["Non-shower wrong energy = 4.0% of truth energy - and it is highly structured",
   "SWAP (91%): two interactions in one TPC each get the OTHER's flash",
   "LANDED (5.5%): grabbed a real flash owned by a neighbouring TPC (shared light)",
   "OFF-flash (3.5%): placement drifts a few ticks past the 10-tick cut",
   "Clustering is correct - only the t0 LABELS move"],
  images=[img], body_width=6.4,
  notes="This is the most important slide. The residual error is not random dust - 91% of it is one failure "
        "mode: two real interactions swapping their flashes. That tells us it's a matching-policy problem, "
        "not a light-prediction problem.")

# ---- 12 · swap in 3D ----
im1 = fit_image(gal("b0_0000002_ev1_tpc5_reco3d.png"), left=0.6, top=2.7, max_w=6.1, max_h=4.2)
im2 = fit_image(gal("b0_0000002_ev1_tpc5_truth3d.png"), left=6.75, top=2.7, max_w=6.1, max_h=4.2)
S("The swap, seen in 3D (TPC 5)",
  ["Same charge, same clusters - only the t0 grouping differs between reco and truth",
   "The wrong hits form whole coherent tracks, not scattered noise"],
  images=[im1, im2], body_height=1.0,
  captions=[dict(text="RECO t0 (matcher)", left=2.4, top=6.95, width=3.0),
            dict(text="TRUTH t0", left=8.7, top=6.95, width=3.0)],
  notes="Left is what the matcher decided, right is truth. Notice the two big clusters have simply exchanged "
        "their colours - their t0 labels are swapped. The clustering is perfect; only the time assignment is wrong.")

# ---- 13 · swap in light ----
im = R(fig("fig_lc_tpc5.png"), 6.3, 5.3, top=1.75)
S("The swap in the light: two real flashes, crossed",
  ["L257 (295 MeV): truly at t0=33, but placed at 398",
   "L16 (185 MeV): truly at 398, but placed at 34",
   "BOTH t0 are real, bright flash peaks",
   "-> the summed light is near-degenerate under the swap",
   "Wrong hits (red rings) = the whole track that got the wrong flash"],
  images=[im], body_width=5.6,
  notes="Here's the same event in the light. Both 33 and 398 are genuine flashes, and both clusters emit "
        "light, so swapping which cluster sits at which flash barely changes the total predicted light. That "
        "degeneracy is exactly why the mistake survives.")

# ---- 14 · why swaps ----
im = R(fig("fig_lc_tpc38.png"), 6.1, 5.3, top=1.75)
S("Why swaps survive - three reinforcing gaps",
  ["1. Never tested: clusters are grouped by reco-t0, so the two swap partners land in different groups "
   "and the swap permutation is never co-scored",
   "2. Degenerate objective: both flashes bright -> summed light barely moves under the swap; a greedy "
   "'biggest cluster -> brightest flash' bias",
   "3. No anchor: the swapped cluster's flash trigger did not fire (or was overridden)",
   "All fixable: the correct flash IS present in the TPC"],
  images=[im], body_width=5.7,
  captions=[dict(text="TPC 38: two flashes only 37 ticks apart get crossed",
                 left=im["left"], top=im["top"]+im["height"]+0.06, width=6.1)],
  notes="Three things line up. We never even test the swap because the two clusters are in different time "
        "groups. Even if we did, the summed-light score can't tell them apart. And the flash trigger that "
        "could have anchored the right answer didn't fire confidently.")

# ---- 15 · minority modes ----
im = R(fig("fig_lc_tpc37.png"), 6.1, 5.3, top=1.75)
S("Minority failure modes",
  ["LANDED (5.5%): a long through-going muon grabs a real flash whose charge is in ANOTHER TPC "
   "(light is shared across TPCs)",
   "OFF-flash (3.5%): the flash trigger found the right t0, but the placement drifted ~12 ticks",
   "- just over the 10-tick cut -> a near-miss, largely benign",
   "Both involve long tracks whose light is spread / shared -> ambiguous association"],
  images=[im], body_width=5.7,
  captions=[dict(text="TPC 37: a 6-TPC muon placed 12 ticks late (135 vs true 123)",
                 left=im["left"], top=im["top"]+im["height"]+0.06, width=6.1)],
  notes="The rest is two small classes, both about long tracks whose light is spread across TPCs. The "
        "off-flash ones are near-misses - the trigger was right, the placement just drifted a couple ticks "
        "past our cut.")

# ---- 16 · summary ----
S("Summary and next steps",
  ["v0.3.0 full chain: 94.9% all-hits / 95.2% non-shower energy, 100% coverage",
   "Dominant residual = pairwise t0 swaps (91%) - a matching-policy problem, not light availability",
   "Discriminator validated: a saturation-masked per-channel SPATIAL cosine prefers the correct t0 (7/7 clean cases)",
   "- a swap-fix pass recovers the big clean swaps (e.g. +1.4pp on the worst TPC)",
   "- but net-neutral as a blanket pass (broad light -> small margins); need energy-weighted / joint assignment",
   "Next: dedicated handling for shower TPCs (>1.4 GeV)"],
  notes="To close: the model is solid and the pipeline is at ~95% on the interesting cohort. The remaining "
        "error is almost entirely swaps, which is encouraging because it's a policy problem we can attack. "
        "We have a discriminator that works on clean cases; making it a reliable blanket pass is the next step.")

out = os.path.join(HERE, "v030_presentation")
os.makedirs(out, exist_ok=True)
path = os.path.join(out, "CLMatching v0.3.0 algorithm and failures.pptx")
finish(prs, path)
print("wrote", path)
