"""Assemble the bad-TPC diagnosis Artifact (self-contained HTML, embedded JPGs)."""
import base64, glob, json, os

HERE = os.path.abspath(os.path.dirname(__file__))
G = os.path.join(HERE, "bad_tpc_gallery")
TH = os.path.join(G, "thumbs")

def b64(path):
    with open(path, "rb") as f:
        return "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()

# tag stem per case index -> find its thumbs
ctx = json.load(open(os.path.join(G, "_all_context.json")))
stem = {c["idx"]: f"b{c['idx']}_{c['file']}_ev{c['event']}_tpc{c['tpc']}" for c in ctx}

# distilled, adversarially-verified diagnoses (all 10 CONFIRMED, correct answer available)
CASES = {
 0: dict(tpc="ev1 · TPC 5", E=870, eff=.42, wrong=506, cls="SWAP",
    what="A 295 MeV multi-prong vertex (true t0≈33) and a 184 MeV crossing track (true t0≈398) were each given the OTHER's flash; their satellite clusters followed.",
    why="Both t0 are strong real flashes and the smaller track grabbed the largest flash. All four clusters had group=None → the swap permutation was never co-grouped or tested; trigger fired=false, cmin=0.955."),
 1: dict(tpc="ev3 · TPC 69", E=1062, eff=.53, wrong=504, cls="SWAP",
    what="A 394 MeV shower (true t0=508) was placed on t0=160, while the fragments that truly fire t0=160 were pushed to 508.",
    why="Both flashes are large (h≈367k vs 326k) and the shower's diffuse light is near-degenerate between them. group=null and trig=null on both partners → swap permutation never evaluated."),
 2: dict(tpc="ev12 · TPC 38", E=1318, eff=.64, wrong=478, cls="SWAP",
    what="A 340 MeV cluster (true t0=386) landed on the single brightest flash t0=349; its partner track (true 349) was pushed to 386. Two flashes only 37 ticks apart.",
    why="group=null + trig=null → untested permutation, so the matcher defaulted to 'biggest cluster → biggest flash'. Compounded by a loss-driven fragmentation: a 490-track piece moved to 387 because loss_best(85579) < loss_stay(89628)."),
 3: dict(tpc="ev3 · TPC 24", E=1137, eff=.62, wrong=435, cls="MIXED",
    what="5+ overlapping interactions with ~20 comparable flashes; the clusterer fragmented them into 35 sub-clusters that snapped onto neighbouring flashes — a cyclic many-way scramble, plus a shower bleeding across TPCs 24/25/27.",
    why="group=null on every dominant fragment; track triggers fired=false (cmin 0.907/0.986). Extreme crowding makes the light constraint near-degenerate and the clustering itself contributes."),
 4: dict(tpc="ev5 · TPC 28", E=1088, eff=.62, wrong=414, cls="SWAP",
    what="A 3-cycle rotation of flash ownership: a 240 MeV cluster (true 59) was pulled onto the tallest flash t0=78 (h=752k); the vacated 59 was taken by a 130 MeV cluster; its true t0=9 taken by another.",
    why="Strong bias toward the tallest flash for the largest cluster + group=None on all three + a true trigger at t0=9 that fired=false (cmin=0.952). Placed and true t0 sit exactly on real flashes → genuinely degenerate light."),
 5: dict(tpc="ev6 · TPC 1", E=906, eff=.56, wrong=386, cls="SWAP",
    what="A ~400 MeV shower (true 252) and a 187 MeV track (true 131) overlap in z–y; ~322 MeV of the shower inherited the track's t0=131 group and vice-versa. Wrong t0 reprojects charge to the wrong drift-x → phantom parallel copies in the display.",
    why="group_hits_by_time drew the group boundary through the overlapping bundle in the wrong place; group=None, trig=null → the swap that would fix it was never tested."),
 6: dict(tpc="ev12 · TPC 48", E=824, eff=.79, wrong=174, cls="SWAP",
    what="A long through-going track (true t0=182) was placed on the real flash at t0=70; the whole track flips to wrong.",
    why="A same-TPC partner interaction owns t0=70; the track and its partner exchanged flashes. Both t0 land on strong real flashes; the correct assignment was available but never jointly tested."),
 7: dict(tpc="ev12 · TPC 2", E=1228, eff=.77, wrong=276, cls="SWAP",
    what="A long through-going track spanning TPCs 0/2/4/6/8 (true t0=591) was placed on t0=348, which is genuinely owned by a different interaction.",
    why="A greedy per-cluster light template matched the 348 flash almost perfectly while undershooting at 591. group=null on the dominant cluster → the joint swap was never enumerated; trigger only weakly confident (cmin=0.569)."),
 8: dict(tpc="ev12 · TPC 37", E=818, eff=.76, wrong=198, cls="OFF-FLASH",
    what="A long muon (true t0=123, spanning 6 TPCs) was placed 12 ticks late at t0=135 — off any flash. The whole 174 MeV track flips because dt=12 exceeds the 10-tick cut by only 2.",
    why="NOT a swap: the flash trigger FOUND the right t0 (fired=true, cmin=0.263, 0.2 ticks from the real flash) but the downstream light-placement drifted. A marginal placement-drift on a long multi-TPC track (group=None)."),
 9: dict(tpc="ev12 · TPC 35", E=644, eff=.90, wrong=64, cls="OFF-FLASH",
    what="The same L86 muon as TPC 37 (spans 6 TPCs), same 12-tick placement drift (135 vs true 123); plus a few tiny sub-MeV fragments snapped onto distant strong flashes.",
    why="Placement drift 2.2 ticks over the cut — a near-miss, largely benign. Trigger fired=true and found t0=123; only the placement stage moved it."),
}

CLS_COLOR = {"SWAP": "swap", "MIXED": "mixed", "OFF-FLASH": "off", "LANDED": "landed"}

cards = []
for i in range(len(ctx)):
    c = CASES[i]; s = stem[i]
    w3 = os.path.join(TH, f"{s}_wrong3d.jpg")
    lc = os.path.join(TH, f"{s}_light_charge.jpg")
    cards.append(f"""
    <article class="card">
      <header class="card-h">
        <span class="chip {CLS_COLOR[c['cls']]}">{c['cls']}</span>
        <h3>{c['tpc']}</h3>
        <div class="stats">
          <span><b>{c['eff']:.2f}</b><i>eff</i></span>
          <span><b>{c['wrong']}</b><i>MeV wrong</i></span>
          <span><b>{c['E']}</b><i>MeV TPC</i></span>
        </div>
      </header>
      <div class="imgs">
        <figure><img loading="lazy" src="{b64(w3)}" alt="wrong-hits 3D"><figcaption>Wrong hits (red) vs correct (gray)</figcaption></figure>
        <figure><img loading="lazy" src="{b64(lc)}" alt="light and charge"><figcaption>Actual vs predicted light · charge z–y</figcaption></figure>
      </div>
      <div class="diag">
        <p><span class="lab what">WHAT</span>{c['what']}</p>
        <p><span class="lab why">WHY</span>{c['why']}</p>
      </div>
    </article>""")

CARDS = "\n".join(cards)

HTML = f"""<title>ND-LAr charge–light matcher: where t0 goes wrong</title>
<style>
:root{{
  --bg:#eef1f5; --panel:#ffffff; --ink:#141a21; --muted:#5b6774; --line:#d9e0e8;
  --accent:#0e8fa8; --swap:#c76a00; --mixed:#8a5cf6; --off:#4a90c2; --wrong:#d83a3f; --ok:#8b98a5;
  --shadow:0 1px 3px rgba(20,30,45,.08),0 6px 20px rgba(20,30,45,.06);
}}
@media (prefers-color-scheme:dark){{
  :root{{ --bg:#0c1015; --panel:#141b23; --ink:#e6edf3; --muted:#8b98a5; --line:#232e3a;
    --accent:#38c0d8; --swap:#f5a524; --mixed:#a884ff; --off:#5aa9de; --wrong:#ff5a5f; --ok:#6c7a88;
    --shadow:0 1px 3px rgba(0,0,0,.4),0 8px 28px rgba(0,0,0,.35);}}
}}
:root[data-theme="dark"]{{ --bg:#0c1015; --panel:#141b23; --ink:#e6edf3; --muted:#8b98a5; --line:#232e3a;
  --accent:#38c0d8; --swap:#f5a524; --mixed:#a884ff; --off:#5aa9de; --wrong:#ff5a5f; --ok:#6c7a88;
  --shadow:0 1px 3px rgba(0,0,0,.4),0 8px 28px rgba(0,0,0,.35);}}
:root[data-theme="light"]{{ --bg:#eef1f5; --panel:#ffffff; --ink:#141a21; --muted:#5b6774; --line:#d9e0e8;
  --accent:#0e8fa8; --swap:#c76a00; --mixed:#8a5cf6; --off:#4a90c2; --wrong:#d83a3f; --ok:#8b98a5;
  --shadow:0 1px 3px rgba(20,30,45,.08),0 6px 20px rgba(20,30,45,.06);}}
*{{box-sizing:border-box}}
body,.wrap{{margin:0}}
.wrap{{background:var(--bg);color:var(--ink);min-height:100vh;
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  line-height:1.55;padding:clamp(18px,4vw,44px);}}
.mono{{font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace;font-variant-numeric:tabular-nums}}
.inner{{max-width:1180px;margin:0 auto}}
.eyebrow{{font-size:.74rem;letter-spacing:.14em;text-transform:uppercase;color:var(--accent);font-weight:700}}
h1{{font-size:clamp(1.6rem,3.4vw,2.5rem);margin:.15em 0 .1em;line-height:1.08;text-wrap:balance;letter-spacing:-.01em}}
.sub{{color:var(--muted);max-width:64ch;font-size:1.02rem}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:14px;margin:26px 0 8px}}
.kpi{{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:16px 18px;box-shadow:var(--shadow)}}
.kpi .n{{font-size:1.9rem;font-weight:750;letter-spacing:-.02em}}
.kpi .l{{color:var(--muted);font-size:.82rem;margin-top:2px}}
.kpi .n small{{font-size:.9rem;color:var(--muted);font-weight:500}}
h2{{font-size:1.28rem;margin:38px 0 6px;letter-spacing:-.01em}}
.note{{color:var(--muted);font-size:.9rem;margin:0 0 14px}}
.tax{{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:14px}}
.tax .t{{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:16px 18px;box-shadow:var(--shadow);border-top:4px solid var(--c)}}
.tax .t .pct{{font-size:1.7rem;font-weight:750}}
.tax .t h4{{margin:.2em 0 .3em;font-size:.98rem}}
.tax .t p{{margin:0;color:var(--muted);font-size:.87rem}}
.bar{{height:14px;border-radius:8px;overflow:hidden;display:flex;margin:14px 0 4px;border:1px solid var(--line)}}
.bar i{{display:block;height:100%}}
.mech{{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:6px 22px;box-shadow:var(--shadow);margin-top:8px}}
.mech ol{{padding-left:20px}} .mech li{{margin:12px 0}} .mech b{{color:var(--ink)}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(520px,1fr));gap:18px;margin-top:10px}}
@media(max-width:560px){{.grid{{grid-template-columns:1fr}}}}
.card{{background:var(--panel);border:1px solid var(--line);border-radius:16px;overflow:hidden;box-shadow:var(--shadow);display:flex;flex-direction:column}}
.card-h{{padding:14px 18px 8px;display:flex;flex-wrap:wrap;align-items:center;gap:10px;border-bottom:1px solid var(--line)}}
.card-h h3{{margin:0;font-size:1.05rem;flex:1;font-family:ui-monospace,monospace}}
.chip{{font-size:.7rem;font-weight:800;letter-spacing:.06em;padding:4px 9px;border-radius:20px;color:#fff}}
.chip.swap{{background:var(--swap)}}.chip.mixed{{background:var(--mixed)}}.chip.off{{background:var(--off)}}.chip.landed{{background:var(--off)}}
.stats{{display:flex;gap:14px;width:100%;margin-top:6px}}
.stats span{{display:flex;flex-direction:column;font-family:ui-monospace,monospace}}
.stats b{{font-size:1.05rem}} .stats i{{font-style:normal;color:var(--muted);font-size:.72rem}}
.imgs{{display:grid;grid-template-columns:1fr 1fr;gap:2px;background:var(--line)}}
@media(max-width:640px){{.imgs{{grid-template-columns:1fr}}}}
.imgs figure{{margin:0;background:var(--panel)}}
.imgs img{{width:100%;height:auto;display:block;background:#fff}}
.imgs figcaption{{font-size:.72rem;color:var(--muted);padding:5px 10px;text-align:center}}
.diag{{padding:12px 18px 16px}}
.diag p{{margin:.35em 0;font-size:.9rem}}
.lab{{display:inline-block;font-size:.66rem;font-weight:800;letter-spacing:.08em;padding:2px 7px;border-radius:5px;margin-right:8px;vertical-align:1px}}
.lab.what{{background:color-mix(in srgb,var(--accent) 18%,transparent);color:var(--accent)}}
.lab.why{{background:color-mix(in srgb,var(--wrong) 16%,transparent);color:var(--wrong)}}
.recs{{counter-reset:r;display:grid;gap:10px;margin-top:8px}}
.rec{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px 18px 14px 54px;position:relative;box-shadow:var(--shadow)}}
.rec::before{{counter-increment:r;content:counter(r);position:absolute;left:16px;top:14px;width:26px;height:26px;border-radius:50%;background:var(--accent);color:#fff;font-weight:750;display:grid;place-items:center;font-size:.85rem}}
.rec b{{display:block;margin-bottom:2px}} .rec span{{color:var(--muted);font-size:.9rem}}
footer{{color:var(--muted);font-size:.8rem;margin-top:34px;border-top:1px solid var(--line);padding-top:14px}}
code{{font-family:ui-monospace,monospace;background:color-mix(in srgb,var(--muted) 14%,transparent);padding:1px 5px;border-radius:4px;font-size:.86em}}
</style>
<div class="wrap"><div class="inner">
  <div class="eyebrow">ND-LAr charge–light matching · v0.3.0 diagnostic</div>
  <h1>Where the matcher gets t0 wrong — and why</h1>
  <p class="sub">39 spills, full v0.3.0 chain (<span class="mono">phase3 both + force-coverage</span>, D50 run). Ten of the worst charge-TPCs, each visually diagnosed and adversarially verified. A hit is <b>correct</b> when |reco − truth| ≤ 10 ticks (160 ns).</p>

  <div class="kpis">
    <div class="kpi"><div class="n">94.87<small>%</small></div><div class="l">of ALL hits get the correct t0 (every TPC)</div></div>
    <div class="kpi"><div class="n">95.24<small>%</small></div><div class="l">energy correct, non-shower (≤1.4 GeV)</div></div>
    <div class="kpi"><div class="n">93.33<small>%</small></div><div class="l">energy correct, shower TPCs (&gt;1.4 GeV)</div></div>
    <div class="kpi"><div class="n" style="color:var(--swap)">91<small>%</small></div><div class="l">of wrong energy is pairwise <b>t0 swaps</b></div></div>
  </div>

  <h2>The mistake, in one sentence</h2>
  <p class="note">When two interactions share a charge-TPC, the matcher assigns each one the <i>other's</i> scintillation flash. It is almost never inventing light or parking charge on empty time — it is <b>mis-routing real light to the wrong charge</b>.</p>

  <div class="bar">
    <i style="background:var(--swap);width:91%" title="SWAP 91%"></i>
    <i style="background:var(--mixed);width:5.5%" title="LANDED 5.5%"></i>
    <i style="background:var(--off);width:3.5%" title="OFF-FLASH 3.5%"></i>
  </div>
  <div class="tax">
    <div class="t" style="--c:var(--swap)"><div class="pct" style="color:var(--swap)">91.0%</div><h4>SWAP</h4><p>Two real interactions exchange flashes. Both the right and the wrong flash are real, large, and close — the summed-light χ² is near-degenerate under the swap.</p></div>
    <div class="t" style="--c:var(--mixed)"><div class="pct" style="color:var(--mixed)">5.5%</div><h4>LANDED ON FLASH</h4><p>Placed on a real flash that fired but is owned by a different interaction — often a long track grabbing a neighbouring TPC's flash (cross-TPC light sharing).</p></div>
    <div class="t" style="--c:var(--off)"><div class="pct" style="color:var(--off)">3.5%</div><h4>OFF-FLASH</h4><p>Placed off any flash. Mostly near-misses: the trigger found the right t0 but the placement stage drifted ~12 ticks, just over the 10-tick cut.</p></div>
  </div>

  <h2>Why swaps survive — three reinforcing defects</h2>
  <div class="mech"><ol>
    <li><b>The swap permutation is never tested.</b> <code>group_hits_by_time</code> buckets clusters by reco-t0, so two interactions at different flashes land in <i>different</i> groups (<code>group=None</code> on the swap partners in every case). The joint "A→flash-X, B→flash-Y vs. the swap" comparison is never enumerated.</li>
    <li><b>The light objective is degenerate — and sometimes wrong.</b> Both flashes are real and large, so summed predicted-vs-actual light barely changes under the swap. The matcher then defaults to a greedy <i>biggest-cluster→brightest-flash</i> pairing. In TPC 38 the group-move loss actively <i>rewarded</i> the wrong move (<code>loss_best &lt; loss_stay</code>).</li>
    <li><b>No confident trigger anchor.</b> The swapped clusters have <code>trig fired=false</code> with high <code>cmin</code> (0.85–0.97) or <code>trig=null</code> — the flash trigger never locked the correct t0, so nothing corrected the greedy default.</li>
  </ol></div>
  <p class="note">All 10 cases were verified <b>fixable</b> — the true flash is present in the TPC, so the correct answer was achievable. The two OFF-FLASH cases are a distinct, smaller defect: a correct trigger overridden by a drifting <i>placement</i> stage.</p>

  <h2>The ten worst TPCs</h2>
  <div class="grid">{CARDS}</div>

  <h2>What would move the needle</h2>
  <div class="recs">
    <div class="rec"><b>Co-group co-located clusters and test the swap permutation.</b><span>Group by spatial overlap / shared flash-candidate set, not just reco-t0 proximity. For every pair of clusters whose candidate flashes overlap, score the swap and keep the lower joint cost. Directly attacks the 91% + 5.5% ≈ 96% of the error.</span></div>
    <div class="rec"><b>Replace greedy per-cluster argmax with joint one-flash-one-owner assignment.</b><span>A Hungarian-style match over the cluster×flash cost matrix removes double-booking as a mechanism rather than patching it. This is the highest-leverage single change.</span></div>
    <div class="rec"><b>Score spatial per-channel light, and add an energy↔amplitude term.</b><span>The summed-light term is flat under the swap; compare per-channel predicted-vs-actual light (not total amplitude — in TPC 69 the <i>wrong</i> flash is actually brighter than the true one), and penalize "small cluster on the brightest flash" (TPC 5, 28, 38; the bright-cluster-on-dim-flash inversion in TPC 48).</span></div>
    <div class="rec"><b>Promote a confident trigger to an authoritative anchor; snap off-flash placements.</b><span>When the trigger fired with low <code>cmin</code>, lock that t0 against the drifting placement stage, and snap off-flash placements to the nearest <i>triggered</i> flash peak. Recovers the OFF-FLASH near-misses (TPC 37/35, correct t0=123 drifted to 135) and the trigger-overridden swap in TPC 48.</span></div>
    <div class="rec"><b>Address crowded/fragmented TPCs and shower bleed separately.</b><span>TPC 24-type many-way scrambles combine clustering fragmentation with degenerate light; joint assignment helps but shower diffuse-light bleed across TPCs needs its own handling (already scoped for &gt;1.4 GeV TPCs).</span></div>
  </div>

  <footer>
    Source: <span class="mono">v030D50_out/dumps</span> (39 events) · displays in <span class="mono">bad_tpc_gallery/</span> (interactive 3D HTML per case: wrong3d / reco3d / truth3d) · taxonomy from <span class="mono">badtpc_swap.py</span>, diagnoses adversarially verified (10/10 CONFIRMED). Non-shower efficiency 95.24% cross-checks the D50 NS-total-eff 95.19%. Force-coverage sets coverage to 100%, so every wrong hit is a visible mistake.
  </footer>
</div></div>"""

out = os.path.join(G, "bad_tpc_report.html")
open(out, "w").write(HTML)
print("wrote", out, f"({len(HTML)/1e6:.2f} MB)")
