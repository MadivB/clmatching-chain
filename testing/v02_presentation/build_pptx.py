"""Build the v0.2 charge-light matching talk in Billy's SlideKit format."""
import os
import sys

sys.path.append("/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/skills/SlideKit")
from slidekit import new_deck, set_title_slide, add_content_slide, finish, fit_image

HERE = os.path.dirname(os.path.abspath(__file__))
F = os.path.join(HERE, "figs")
MEET = "ND Prototypes analysis meeting"

prs = new_deck()
set_title_slide(prs, "Interaction Level Charge Light Matching\n"
                     "v0.2 placement, the overflow alarm, and where we stand",
                ["Yuxuan Wang (Billy)", "PI: Ryan Patterson", MEET])

# ---- outline ----
add_content_slide(prs, title="Outline",
    bullets=[
        "Quick recap: what charge-light matching does",
        "How the main placement unfolds (step by step)",
        "The overflow alarm and the saving moves",
        "Policing the later stages",
        "Performance: 95.1% -> 97.25%",
        "Where we still fail -> showers get their own algorithm next",
    ],
    meeting=MEET,
    notes="Today I want to walk you through the new v0.2 placement, which is mostly "
          "about placing clusters in a smarter order, plus a physics alarm that catches "
          "wrong placements, and then show where we stand and what's left.")

# ---- recap ----
img = fit_image(os.path.join(F, "ex0_type1_light_charge.png"), left=5.7, top=1.85, max_w=7.1, max_h=4.9)
add_content_slide(prs, title="Recap: every cluster needs a flash time",
    bullets=[
        "Each charge cluster gets an ML-predicted light image (what it would shine on each channel)",
        "Placement = slide that image in time so the sum over all clusters matches the recorded light",
        "- Poisson-weighted chi2, prediction clipped at ADC saturation",
        "A hit counts as correct if |reco t0 - true t0| <= 10 ticks",
        "Test set: 39 MiniProdN5p1 events, ~2 TeV of hits",
    ],
    body_width=4.9, images=[img],
    captions=[dict(text="One TPC: recorded light (blue), summed model light (orange), charge in between.",
                   left=5.7, top=img["top"] + img["height"] + 0.08, width=7.0)],
    meeting=MEET,
    notes="Quick recap. For every charge cluster we predict the light it would make. "
          "Placing a cluster means sliding that predicted pulse in time until the total "
          "predicted light matches what we recorded. We call a hit correct when its "
          "reconstructed time is within ten ticks of the truth.")

# ---- placement step 1 ----
add_content_slide(prs, title="How the placement unfolds (1): the easy ones go first",
    bullets=[
        "Showers are placed first, as before",
        "Then tracks that cross 3 or more TPCs",
        "- they light up several walls at once -> basically only one time fits them",
        "What is left: clusters living in 1-2 TPCs, >= 50 MeV",
        "- these are the ones that can be confused with each other -> the 'pool'",
        "Old v0.1 placed everything in an arbitrary order -> an early mistake poisons everything after it",
    ],
    meeting=MEET,
    notes="The idea is simple: place the things we are sure about first. Showers, then the long "
          "tracks that cross several TPCs - those can hardly go anywhere else. What remains is a pool "
          "of one-or-two-TPC clusters, and those are the tricky ones, because several of them can "
          "plausibly claim the same flash. The old code placed them in whatever order they came, "
          "and one early mistake would mess up everything placed afterwards.")

# ---- placement step 2 ----
add_content_slide(prs, title="How the placement unfolds (2): gather the rivals",
    bullets=[
        "Take the most energetic cluster still in the pool (energy E1)",
        "Gather the other pool clusters that",
        "- share a TPC with it, and",
        "- are almost as energetic (E > (1-s) * E1, s = 0.6 works best)",
        "That is the batch: the clusters that could realistically be mixed up with each other",
        "Cap the batch at 5 members",
    ],
    meeting=MEET,
    notes="One round of the loop starts like this. Take the biggest cluster left in the pool. Then "
          "gather the others that live in the same TPC and are almost as bright - those are exactly "
          "the ones the fit could confuse with it. A dim cluster can't steal a bright cluster's flash, "
          "so we don't worry about those. That group, at most five clusters, is our batch.")

# ---- placement step 3 ----
add_content_slide(prs, title="How the placement unfolds (3): try every order",
    bullets=[
        "Place the batch one at a time; each cluster lands where it fits best, given everything already placed",
        "Different orders end up in different arrangements",
        "- whoever goes first gets the best flash",
        "So: just try ALL the orders (5 clusters -> 120 tries)",
        "Keep the arrangement that explains the recorded light best",
        "Cheap: a few seconds per event (FFT scan + shared work between orders)",
    ],
    meeting=MEET,
    notes="Now the key trick. We place the batch members one at a time, and each one simply goes "
          "where it fits best given what's already there. But of course the result depends on who "
          "goes first. Instead of guessing, we try every order - at most 120 of them - and keep the "
          "ending where the light is explained best. This sounds expensive but it's seconds per event.")

# ---- placement step 4 ----
add_content_slide(prs, title="How the placement unfolds (4): keep most, hold one back",
    bullets=[
        "From the winning order, trust every placement except the LAST one placed",
        "- it had the fewest options left -> least reliable",
        "The last one goes back into the pool and competes again in the next round",
        "Repeat: new most-energetic cluster, new batch, ... until the pool is empty",
        "Then two cheap clean-up passes:",
        "- swap: two placed clusters exchange times if that fits better",
        "- rescan: suspicious ones get one more look against the finished picture",
    ],
    meeting=MEET,
    notes="From the winning order we commit everything except the very last cluster placed - it had "
          "the least freedom, so we let it go back to the pool and try again in the next round with "
          "the next energy band. Rinse and repeat until the pool is empty. At the end, two quick "
          "clean-ups: pairs of clusters can swap times if that fits better, and anything that looks "
          "suspicious gets one more scan against the now-complete picture.")

# ---- cosine ----
add_content_slide(prs, title="One more ingredient in every scan: does the leftover light look like me?",
    bullets=[
        "The chi2 alone can be fooled: the ML amplitude is ~20-30% off on bright walls",
        "But the SHAPE across channels is reliable",
        "At each candidate time, compare the cluster's channel pattern with the pattern of still-unexplained light",
        "- cosine similarity; used as a soft penalty: loss x (1 + (1 - cos))",
        "Note: a hard cosine cut makes things worse - penalise, never forbid",
    ],
    meeting=MEET,
    notes="One extra ingredient inside every scan. The chi-square alone can be fooled because our "
          "light amplitude prediction has sizable systematics. The shape across channels is much more "
          "trustworthy. So at every candidate time we ask: does the light that is still unexplained "
          "look like this cluster's pattern? That's just a cosine similarity, and we fold it in as a "
          "soft penalty. We tried a hard cut - it actually hurts. Soft is the way.")

# ---- overflow alarm ----
img = fit_image(os.path.join(F, "trigger_example.png"), left=6.2, top=1.85, max_w=6.6, max_h=5.0)
add_content_slide(prs, title="The overflow alarm: you cannot claim more light than there is",
    bullets=[
        "Light only adds up -> predicted > recorded is physically impossible",
        "z = (pred - actual) / sqrt(200 x actual)   (200 ADC ~ 1 photon)",
        "Alarm: >= 3 channels with excess >= 500 ADC and z > 2,",
        "or the leftover light doesn't look like the cluster (cos < 0.75)",
        "Catches ~half of wrong placements; false alarms only cost a re-check",
    ],
    body_width=5.3, images=[img],
    captions=[dict(text="A wrongly placed track: model (purple+gray) overshoots the recorded light (black) on many channels at once.",
                   left=6.2, top=img["top"] + img["height"] + 0.08, width=6.5)],
    meeting=MEET,
    notes="Now the alarm. The physics is simple: light from different clusters only adds up, so if "
          "our prediction sits above the recorded light, something is definitely wrong there. We "
          "measure that in units of photon statistics, and ring the alarm when three channels "
          "overshoot significantly, or when the leftover light just doesn't look like the cluster.")

# ---- saving moves ----
add_content_slide(prs, title="When the alarm rings: three saving moves",
    bullets=[
        "Several clusters share the flagged time slot ->",
        "- keep one at the slot, let the others re-place, try every order (like the main loop)",
        "Everybody insists on staying ->",
        "- maybe two TPC pieces were wrongly stitched into one cluster:",
        "- cut off the piece that wants to leave, give it a NEW cluster id, re-associate it downstream",
        "The flagged cluster is alone ->",
        "- the flash it claims may not exist at all: rescan the full time range, or detach a piece",
        "Every move is adopted only if the total light model actually improves",
    ],
    meeting=MEET,
    notes="When the alarm rings we have three moves, depending on the situation. If several clusters "
          "share the slot, we replay them in every order, like the main loop. If everyone insists on "
          "staying, we suspect the cluster itself is wrongly glued together across TPCs, so we cut "
          "off the piece that wants to leave and give it a fresh identity. And if the cluster is "
          "alone and screaming, the flash it claims may simply not exist - so it gets a full rescan. "
          "Nothing is adopted unless the light model genuinely improves.")

# ---- police ----
img = fit_image(os.path.join(F, "vfail3_channels.png"), left=6.2, top=1.85, max_w=6.6, max_h=5.0)
add_content_slide(prs, title="Policing the later stages: no overflow at the moment of placement",
    bullets=[
        "The later stages (large-cluster association, light rescue) used to place and move things unchecked",
        "Now every new placement or move is probed on the spot",
        "Alarm rings -> pick the best of: keep / undo the move / rescan",
        "- the fresh rescan won every single time over undoing",
        "Least-constrained cohort: 92.5% -> 94.0%",
    ],
    body_width=5.3, images=[img],
    captions=[dict(text="The motivating case: a late-stage move planted an 80k pulse on a 32k flash - 10 sigma overflow everywhere. Now caught immediately.",
                   left=6.2, top=img["top"] + img["height"] + 0.08, width=6.5)],
    meeting=MEET,
    notes="The same alarm now polices the later stages too. Before, those stages could plant a huge "
          "wrong pulse and nobody would object - here's the case that motivated this, ten sigma of "
          "overflow across the whole wall. Now every placement is checked the moment it happens, and "
          "if the alarm rings we keep, undo, or rescan - whichever explains the light best.")

# ---- pipeline at a glance ----
img = fit_image(os.path.join(F, "pipeline_flow.png"), left=1.0, top=2.9, max_w=11.4, max_h=3.6)
img["left"] = (13.33 - img["width"]) / 2
add_content_slide(prs, title="The whole chain at a glance",
    bullets=["Saturated channels (clipped > 4 ticks) are vetoed from every scan and probe"],
    body_height=1.0, images=[img],
    meeting=MEET,
    notes="Putting it together: predict, place the sure things, run the batch loop for the tricky "
          "pool, let the alarm and the saving moves clean up, and police the two later stages. One "
          "hygiene rule everywhere: saturated channels are excluded from every comparison.")

# ---- performance ----
img = fit_image(os.path.join(F, "progression.png"), left=5.6, top=1.85, max_w=7.3, max_h=5.0)
add_content_slide(prs, title="Performance: 95.1% -> 97.25%",
    bullets=[
        "Every step stacks: ordering +1.0, refinements +0.3,",
        "saving moves +1.3 (backbone), police +0.24",
        "Clusters >= 50 MeV: 97.4% correct at 100% coverage",
        "34 of 39 events improved end-to-end",
        "Note: numbers are energy-weighted, +-10 ticks",
    ],
    body_width=4.8, images=[img],
    meeting=MEET,
    notes="The bottom line. Each layer stacks: the ordering buys a percent, the saving moves another "
          "1.3 on the backbone, and the police another quarter. We end at 97.25 percent of assigned "
          "energy within ten ticks, and for clusters above 50 MeV it's 97.4 with full coverage.")

img = fit_image(os.path.join(F, "ex4_type4_wrong3d.png"), left=3.0, top=1.85, max_w=8.0, max_h=5.1)
img["left"] = (13.33 - img["width"]) / 2
add_content_slide(prs, title="Performance: a busy TPC done right",
    bullets=[],
    images=[img],
    captions=[dict(text="4.3 GeV of hits in one TPC after the full chain: red would mean wrong - there is no red.",
                   left=2.5, top=img["top"] + img["height"] + 0.08, width=8.5)],
    meeting=MEET,
    notes="Here's what success looks like: a TPC with over four GeV of activity, every single hit "
          "within ten ticks. Red points would be mistakes - there aren't any.")

# ---- where we fail ----
img = fit_image(os.path.join(F, "tpc_energy_vs_wrong.png"), left=1.0, top=2.75, max_w=11.6, max_h=4.0)
img["left"] = (13.33 - img["width"]) / 2
add_content_slide(prs, title="Where the remaining 2.7% lives: only in busy TPCs",
    bullets=["Quiet TPCs (< 300 MeV) are essentially perfect (99.7%); accuracy falls steadily with TPC activity"],
    body_height=0.9, images=[img],
    captions=[dict(text="Left: wrong energy vs TPC activity (one point per event x TPC). Middle: accuracy vs activity. Right: mistake-TPCs are the busy tail.",
                   left=1.2, top=6.85, width=11.0)],
    meeting=MEET,
    notes="And where do we still fail? Only in busy TPCs. Below a few hundred MeV of activity we are "
          "essentially perfect. The mistakes live where a big shower lights up the whole wall - then "
          "every candidate flash looks plausible and the template matching degenerates.")

# ---- conclusion ----
img = fit_image(os.path.join(F, "ex0_type4_wrong3d.png"), left=6.4, top=1.85, max_w=6.4, max_h=4.9)
add_content_slide(prs, title="Conclusion: showers are the last frontier",
    bullets=[
        "Order + try-every-order + physics alarm + policing: 95.1% -> 97.25%",
        "Quiet TPCs > 99.4% - the algorithm can be trusted there as-is",
        "All remaining mistakes sit in shower-dominated / multi-GeV TPCs",
        "- even 'clean straight track' failures live inside such TPCs",
        "Next: a dedicated path that RECOGNISES a shower-dominated TPC",
        "and hands it to a separate algorithm instead of the standard scan",
    ],
    body_width=5.5, images=[img],
    captions=[dict(text="The target of the next algorithm: a 2.8 GeV TPC where dense activity misleads whole clusters (red = wrong).",
                   left=6.4, top=img["top"] + img["height"] + 0.08, width=6.3)],
    meeting=MEET,
    notes="To conclude: smarter ordering, trying every order among rivals, a physics overflow alarm, "
          "and policing every stage bring us from 95.1 to 97.25 percent. Quiet TPCs are basically "
          "solved. Everything that's left sits in shower-dominated TPCs, so the next development is "
          "a separate path: recognise that a TPC is shower-dominated, and treat it with a dedicated "
          "algorithm instead of the standard scan.")

out = os.path.join(HERE, "CLMatching v0.2 Update.pptx")
finish(prs, out)
print("saved", out)
