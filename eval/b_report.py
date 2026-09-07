"""Build a local results page for a Qwen B refinement run (no external services)."""
from pathlib import Path
import argparse

PAGE = r'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Qwen B — plans and episodes</title>
<style>
:root{color-scheme:dark;font-family:system-ui,sans-serif;background:#0c1422;color:#e8eef7}body{max-width:1250px;margin:0 auto;padding:36px 24px}h1{font-size:36px;margin:12px 0}h2{font-size:23px}p{line-height:1.65;color:#b4c2d5}.eyebrow{color:#7de3c0;letter-spacing:.12em;font-size:13px}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}.card{background:#142136;border:1px solid #2b405b;border-radius:14px;padding:22px}.number{font-size:38px;font-weight:700;color:#90e8c7}.label{font-size:13px;color:#aabbd0}a{color:#86d8ff}table{width:100%;border-collapse:collapse}td,th{text-align:left;vertical-align:top;padding:12px;border-bottom:1px solid #2b405b}th{color:#9fb7d1}.good{color:#90e8c7}.bad{color:#ffafad}.muted{color:#9eb1c9;font-size:14px}.tag{display:inline-block;border:1px solid #476078;padding:4px 10px;border-radius:20px;font-size:12px}.episode img{width:100%;border-radius:9px;aspect-ratio:1.6;object-fit:cover}.episode h3{margin-bottom:4px}button,select{background:#1d344f;color:#e8eef7;border:1px solid #476078;padding:8px 12px;border-radius:7px;cursor:pointer}.row{display:flex;align-items:center;justify-content:space-between;gap:16px}section{margin-top:32px}code{font-size:13px}@media(max-width:850px){.grid{grid-template-columns:1fr}.table{overflow:auto}.row{display:block}h1{font-size:28px}}
</style>
<div class="eyebrow">EE4705 · STUDENT B · LIVE QWEN</div>
<h1>From instructions to checked plans</h1>
<p>Fixed scene inputs and independent expected answers. These scores measure B's planning, including search, clarification and replanning. Physical episodes below use demo RGB-D A and demo motion C in MuJoCo.</p>
<div class="grid" id="scores"></div>
<section class="card"><h2>What changed</h2><p>The adapter now records and removes consistent duplicate IDs, while rejecting conflicting fields. Explicit candidate groups and decision rules prevent guessing between multiple stones and distinguish a held-object conflict from an unsupported task. The model prompt was frozen before the 32-case evaluation.</p><p>The old failed API response, both development rounds and failed physical episodes remain available. A demo C release fix opens the gripper before detaching the weld, preventing closed fingers from ejecting the object.</p><a href="historical_failure/original_audit.json">Original failed Qwen request + repair</a> · <a href="pytest.txt">Offline regression output</a></section>
<section><div class="row"><h2>Recorded episodes</h2><span class="tag">Raw model records are inside each B.model event</span></div><div class="grid" id="episodes"></div></section>
<section><div class="row"><h2>Independent evaluation cases</h2><select id="filter"><option value="all">All cases</option><option value="pass">Passed cases</option><option value="fail">Failed cases</option></select></div><p class="muted">Each case includes the instruction, scene, expected status/IDs, returned plan and failure details. Open the audit folder from the case result for raw Qwen responses.</p><div class="table"><table><thead><tr><th>Case</th><th>Instruction</th><th>Result</th><th>Ground truth</th></tr></thead><tbody id="cases"></tbody></table></div></section>
<section class="card"><h2>Interpret the result</h2><p>Each case is one sample. API/format errors count as failures. A few replan cases contain two model calls. Development results are tuning results; the evaluation set was run once after freezing B. These are curated tests of a small skill vocabulary, not a guarantee of performance on arbitrary instructions or scenes.</p><p>Videos are annotated workflow replays with display pauses. They are not the assignment's uncut final assessment recording. The missing-target episode cannot complete because the requested dark-red stone is absent.</p></section>
<script>
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
async function read(p){const r=await fetch(p);if(!r.ok)throw Error(p);return r.json()}
let evaluation;
async function main(){
 const groups=[['Development · first run','development_r1'],['Development · refined','development_r2'],['Independent evaluation','evaluation']];
 for(const [label,path] of groups){try{const d=await read(path+'/summary.json');const card=document.createElement('div');card.className='card';card.innerHTML=`<div class="label">${label}</div><div class="number">${d.n_correct}/${d.n_cases}</div><p>${(100*d.accuracy).toFixed(1)}% correct${d.complete===false?' · running':''}</p><a href="${path}/index.html">Cases and raw records →</a>`;document.querySelector('#scores').append(card);if(path==='evaluation')evaluation=d}catch(e){}}
 const episodes=[['Standard move · before release fix','live_success'],['Failed placement after B replan · preserved','live_replan'],['Standard move · final code','live_success_fixed'],['B replan after missed grasp · final code','live_replan_fixed'],['Move the blue cube','cube_transfer'],['Missing dark-red stone · incomplete task','missing_target']];
 for(const [label,name] of episodes){try{const d=await read('episodes/'+name+'/episode.json');const card=document.createElement('div');card.className='card episode';card.innerHTML=`<a href="episodes/${name}/"><img src="episodes/${name}/final.png" alt="Final frame: ${esc(label)}"></a><h3>${esc(label)}</h3><p class="${d.actual_success?'good':'bad'}">${esc(d.outcome)} · actual ${d.actual_success?'PASS':'FAIL'}</p><p class="muted">${esc(d.instruction)}</p><a href="episodes/${name}/">Play video + inspect events →</a>`;document.querySelector('#episodes').append(card)}catch(e){}}
 render();document.querySelector('#filter').onchange=render;
}
function render(){if(!evaluation)return;const mode=document.querySelector('#filter').value;document.querySelector('#cases').innerHTML=evaluation.cases.filter(r=>mode==='all'||(mode==='pass')===r.correct).map(r=>`<tr><td><a href="evaluation/${esc(r.id)}/result.json">${esc(r.id)}</a><div class="muted">${esc(r.category)}</div></td><td>${esc(r.instruction)}<div class="muted">${r.latency_s.toFixed(1)} s</div></td><td class="${r.correct?'good':'bad'}">${r.correct?'PASS':'FAIL'}<div>${esc(r.errors.join('; '))}</div></td><td><a href="evaluation/${esc(r.id)}/case.json">Input + expected</a><br><a href="evaluation/${esc(r.id)}/audit/">Raw model records</a></td></tr>`).join('')}
main();
</script></html>'''


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args(argv)
    if not (args.root / "evaluation/summary.json").is_file():
        parser.error("Expected evaluation/summary.json under the run root")
    (args.root / "index.html").write_text(PAGE)
    print(args.root.resolve() / "index.html")


if __name__ == "__main__":
    main()
