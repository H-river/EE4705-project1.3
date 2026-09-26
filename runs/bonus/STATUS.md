# BONUS STATUS (updated 2026-09-26 10:23 +08)
Stage: 8 — 8.1–8.4 done; 8.6 extra (±20 cm demos) training (~70 min)
Grasp /30 (C1 C2 C3 C4): scripted 29 30 29 30 | ACT 29 26 0 30 | ACT+bottle C3 26
8.1 data size: flat (500 demos saturate C1) | 8.2 bottle: C3 0 → 26/30
8.3 learned PLACE: C1 29/30 = script, closer to centre (0.33 vs 0.45 cm); C2 26/30 (far layouts)
8.4 final50: scripted 48/50, ACT+bottle 47/50; 0 false claims everywhere
Incident 10:15: OOM (too many eval workers) — c2_20 re-run, trainer resumed
Next: ACT on 2k + bottle + ±20 cm → C2 / final50 f20; then refresh docs
Blocked: nothing
