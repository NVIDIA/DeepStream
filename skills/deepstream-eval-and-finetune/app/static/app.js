// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved. Apache-2.0.
const $ = id => document.getElementById(id);
const GT_C = '#46a302', ON_C = '#46a302', OFF_C = '#e08a00';
// user-adjustable detection box + label colors (so boxes pop on any background, e.g. green PCB);
// remembered across sessions. box='' => keep the default per-target colors.
let UICOLORS = {box:'', text:'#ffffff'};
try{ UICOLORS = Object.assign(UICOLORS, JSON.parse(localStorage.getItem('dsBoxColors')||'{}')); }catch(e){}
const PALETTE = ['#46a302','#ff00ff','#00e5ff','#ffe600','#ff3b30','#ffffff','#000000'];
function saveColors(){ try{ localStorage.setItem('dsBoxColors', JSON.stringify(UICOLORS)); }catch(e){} }
let SHOW_LABELS = true;   // toggle drawn box labels (class/conf) on/off — boxes always shown
try{ SHOW_LABELS = JSON.parse(localStorage.getItem('dsShowLabels') ?? 'true'); }catch(e){}
function setShowLabels(on){
  SHOW_LABELS = !!on;
  try{ localStorage.setItem('dsShowLabels', JSON.stringify(SHOW_LABELS)); }catch(e){}
  document.querySelectorAll('.labelsChk').forEach(c=> c.checked = SHOW_LABELS);
  if(typeof redrawFinalFrames==='function') redrawFinalFrames();
  if(typeof redrawBaselineFrames==='function') redrawBaselineFrames();
  if(LAST_CONF) renderHardest(LAST_CONF.r, LAST_CONF.which, LAST_CONF.conf);   // Analyze hardest gallery
  if(PG_RESULT) renderPg();                                                    // Playground result
}
function setColor(kind, c){
  if(kind==='box') UICOLORS.box=c; else UICOLORS.text=c;
  document.querySelectorAll(`input[type=color][data-ck="${kind}"]`).forEach(i=>i.value=c);
  saveColors();
  if(typeof redrawFinalFrames==='function') redrawFinalFrames();
  if(typeof redrawBaselineFrames==='function') redrawBaselineFrames();
  if(typeof renderHardest==='function' && LAST_CONF) renderHardest(LAST_CONF.r, LAST_CONF.which, LAST_CONF.conf);
  if(typeof renderPg==='function' && PG_RESULT) renderPg();
}
function setupColorControls(){
  document.querySelectorAll('.swatches').forEach(sw=>{
    sw.innerHTML = PALETTE.map(c=>`<button class="sw" data-ck="${sw.dataset.ck}" data-c="${c}" style="background:${c}" title="${c}"></button>`).join('')
      + `<button class="sw swclear" data-ck="${sw.dataset.ck}" data-c="" title="default">×</button>`;
  });
  document.querySelectorAll('input[type=color][data-ck]').forEach(inp=>{
    inp.value = inp.dataset.ck==='box' ? (UICOLORS.box||'#46a302') : (UICOLORS.text||'#ffffff');
    inp.oninput = ()=> setColor(inp.dataset.ck, inp.value);
  });
  document.querySelectorAll('.sw').forEach(b=> b.onclick = ()=> setColor(b.dataset.ck, b.dataset.c));
  document.querySelectorAll('.labelsChk').forEach(c=>{ c.checked = SHOW_LABELS; c.onchange = ()=> setShowLabels(c.checked); });
  document.querySelectorAll('.recolorbtn').forEach(btn=> btn.onclick = async ()=>{
    if(!sel) return;
    btn.disabled=true; const o=btn.textContent; btn.textContent='Rendering…';
    try{ const r=await jpost('/recolor_report',{preset:sel, box_color:UICOLORS.box||'', text_color:UICOLORS.text||'', conf:cmpConf});
         btn.textContent = r&&r.ok ? 'Report updated ✓' : 'Failed'; }
    catch(e){ btn.textContent='Failed'; }
    finally{ setTimeout(()=>{btn.disabled=false; btn.textContent=o;}, 2200); }
  });
}

let PRESETS = [], DATA_CACHE = {}, sel = null, curStep = 1, polling = false, lastPhase = null, LAST = null, chartTick = 0, _confT = null, LAST_CONF = null, PG_FILE = null, PG_RESULT = null, ftSub = 'automl';
let _curveT = null;   // auto-refresh timer for the Fine-tune curve of a CLI/container-launched run
let cmpConf = 0.2, FINAL_SAMPLES = [], FRAME_DATA = {};
let blConf = 0.2, BL_IDS = [], BL_DATA = {};
// Advanced training-settings helpers removed — custom runs use the server's auto-by-class-count
// defaults (or AutoML searches the hyperparameters). Epochs + eval split are the only Setup knobs.

async function jget(u){ const r = await fetch(u); if(!r.ok) throw new Error((await r.json()).detail||r.status); return r.json(); }
async function jpost(u,b){ const r = await fetch(u,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b||{})}); return r.json(); }
const fmt = (v,d=3) => (v==null||v<0) ? '—' : (+v).toFixed(d);
function fmtBytes(n){
  if(!n) return '0 B';
  const u=['B','KiB','MiB','GiB','TiB']; let i=0, v=+n;
  while(v>=1024 && i<u.length-1){v/=1024;i++;}
  return `${v.toFixed(i?1:0)} ${u[i]}`;
}
function fmtRunDate(ts){
  if(!ts) return '';
  try{
    return new Date(ts * 1000).toLocaleString(undefined, {
      year: 'numeric', month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit'
    });
  }catch(e){ return ''; }
}
function runMeta(p){
  const bits = [];
  bits.push(`${p.model_id} → ${p.dataset_id || p.dataset_label || 'dataset'}`);
  bits.push(`split ${p.eval_split || 'valid'}`);
  bits.push(`${p.epochs || 30} ep`);
  if(p.hparams && p.hparams.lr) bits.push(`lr ${p.hparams.lr}`);
  if(p.run_note) bits.push(p.run_note);
  return bits.join(' · ');
}
const icons = () => { if(window.lucide) lucide.createIcons(); };
const RUN = () => PRESETS.find(p=>p.key===sel);

// ----------------------------------------------------------------- init
async function init(){
  try{
    const sy = await jget('/system');
    const bits = [];
    if(sy.gpu) bits.push(`<span class="pill gpu">${sy.gpu.name} · ${sy.gpu.mem_used_gb}/${sy.gpu.mem_total_gb} GB</span>`);
    bits.push(`<span class="pill ${sy.in_container?'':'busy'}">${sy.in_container?'in container':'host (viewer)'}</span>`);
    bits.push(`<span class="muted">${sy.precision.toUpperCase()}</span>`);
    $('sysinfo').innerHTML = bits.join('');
  }catch(e){}
  await refreshRuns();
  setStep(1);
  $('startRun').onclick = startRun;
  $('goFinetune').onclick = startFinetune;
  $('backToSetup').onclick = () => { refreshRuns(); setStep(1); };
  const discard = () => { if(confirm('Discard this run? (no evaluation)')) jpost('/run/cancel',{}); };
  $('cancelRun').onclick = discard;
  $('cancelEval').onclick = discard;
  $('stopRun').onclick = async () => {
    const r = await jpost('/run/stop',{});
    if(r && r.ok){ $('stopHint').textContent = 'Stopping training at the latest completed epoch…'; }
    else { $('stopHint').textContent = (r && r.detail) || 'could not stop'; }
  };
  $('evaluateRun').onclick = () => startEvaluate();
  $('continueBtn').onclick = async () => {
    const n = parseInt($('continueEpochs').value) || 20;
    if(!confirm(`Continue training "${sel}" for +${n} epochs?\n\nWarm-starts from the current weights (fresh LR schedule), then re-deploys + re-evaluates + re-reports. Overwrites this run (a weights backup is kept as checkpoints/final_prev).`)) return;
    const r = await jpost('/run/continue', {preset: sel, extra_epochs: n});
    if(!(r && r.ok)){ alert((r && r.detail) || 'could not start continue training'); return; }
    lastPhase = null;                              // let the poller drive the live fine-tune view
  };
  $('restart').onclick = () => { setStep(1); refreshRuns(); };
  $('ctxNew').onclick = () => setStep(1);
  $('lightbox').onclick = () => $('lightbox').classList.remove('open');
  $('sysBtn').onclick = openSysModal;
  $('sysClose').onclick = () => $('sysModal').classList.remove('open');
  $('sysModal').onclick = (ev) => { if(ev.target === $('sysModal')) $('sysModal').classList.remove('open'); };
  $('qcBtn').onclick = openQcModal;
  $('qcClose').onclick = () => $('qcModal').classList.remove('open');
  $('qcModal').onclick = (ev) => { if(ev.target === $('qcModal')) $('qcModal').classList.remove('open'); };
  document.addEventListener('keydown', e => { if(e.key === 'Escape'){ $('lightbox').classList.remove('open'); $('sysModal').classList.remove('open'); $('qcModal').classList.remove('open'); } });
  $('chartLoss').onclick = () => { if($('chartLoss').getAttribute('src')) zoom($('chartLoss').src, 'Training curves'); };
  $('chartAcc').onclick = () => { if($('chartAcc').getAttribute('src')) zoom($('chartAcc').src, 'Deployed accuracy & per-class AP'); };
  document.querySelectorAll('#ftSubtabs .subtab').forEach(b => b.onclick = () => setFtSub(b.dataset.sub));
  $('cmpConf').oninput = () => {
    cmpConf = parseFloat($('cmpConf').value); $('cmpConfVal').textContent = cmpConf.toFixed(3);
    redrawFinalFrames();   // real-time: re-render from cached boxes, no fetch
  };
  // Analyze tab (confusion matrix) has its own conf/IoU/model controls
  $('confWhich').onchange = () => { refreshModelContext('conf'); loadConfusion(); };
  $('confConf').oninput = () => { $('confConfVal').textContent = parseFloat($('confConf').value).toFixed(2);
    clearTimeout(_confT); _confT = setTimeout(loadConfusion, 300); };
  $('confIou').oninput = () => { $('confIouVal').textContent = parseFloat($('confIou').value).toFixed(2);
    clearTimeout(_confT); _confT = setTimeout(loadConfusion, 250); };
  // Playground controls
  $('pgPick').onclick = () => $('pgFile').click();
  $('pgFile').onchange = () => { if($('pgFile').files[0]) uploadAndInfer($('pgFile').files[0]); };
  $('pgConf').oninput = () => { $('pgConfVal').textContent = parseFloat($('pgConf').value).toFixed(2); renderPg(); };
  $('pgWhich').onchange = () => { refreshModelContext('pg'); if(PG_FILE) uploadAndInfer(PG_FILE); };
  const dz = $('pgDrop');
  dz.ondragover = e => { e.preventDefault(); dz.classList.add('drag'); };
  dz.ondragleave = () => dz.classList.remove('drag');
  dz.ondrop = e => { e.preventDefault(); dz.classList.remove('drag'); const f=(e.dataTransfer.files||[])[0]; if(f) uploadAndInfer(f); };
  $('blConf').oninput = () => {
    blConf = parseFloat($('blConf').value); $('blConfVal').textContent = blConf.toFixed(3);
    redrawBaselineFrames();
  };
  setupColorControls(); // box/text color palette + pickers (Step 2 + Step 4)
  pollLoop();
  icons();
}

async function refreshRuns(){
  try{ const d = await jget('/presets'); PRESETS = d.presets; DATA_CACHE = d.data_cache||{}; }catch(e){ return; }
  if(DATA_CACHE.enabled){
    $('cacheBox').style.display='';
    const staged = DATA_CACHE.staged_sources
      ? ` · ${DATA_CACHE.staged_sources} staged source(s), ${fmtBytes(DATA_CACHE.staged_bytes)}`
      : '';
    $('cacheStatus').innerHTML=`<b>Fast Windows/WSL dataset cache:</b> <code>${DATA_CACHE.volume||DATA_CACHE.root}</code>${staged}. Reused across UI restarts.`;
  } else {
    $('cacheBox').style.display='none';
    $('f_refresh_cache').checked=false;
  }
  renderQuickfill(); renderRuns(); renderStepper(); updateCtx();
}

function renderQuickfill(){
  const demos = PRESETS.filter(p=>p.kind==='preset');
  $('quickfill').innerHTML = `<span class="ql">demos</span>` +
    demos.map(p=>`<span class="chip" data-fill="${p.key}">${p.label}</span>`).join('');
  document.querySelectorAll('.chip[data-fill]').forEach(c=>c.onclick=()=>{
    const p = PRESETS.find(x=>x.key===c.dataset.fill);
    $('f_model').value=p.model_id; $('f_dataset').value=p.dataset_id;
    $('f_split').value=p.eval_split||'valid'; $('f_epochs').value=p.epochs||30;
    $('recipeBanner').innerHTML='';
  });
}

function stateChip(p){
  if(p.has_finetuned) return `<span class="rstate cmp">compared</span>`;
  if(p.has_baseline) return `<span class="rstate base">baseline ready</span>`;
  if(p.is_automl) return `<span class="rstate none">blocked: baseline missing</span>`;
  if(p.has_checkpoint) return `<span class="rstate base">trained</span>`;
  return `<span class="rstate none">no results</span>`;
}

function renderRuns(){
  // Newest run first (built-in demos have created_at 0 -> fall to the bottom). The server already
  // sorts, but sort here too so the list order + date/time are guaranteed regardless of source.
  const rows = PRESETS.slice().sort((a,b)=>(b.created_at||0)-(a.created_at||0));
  $('runlist').innerHTML = rows.map(p=>`
    <div class="runrow ${p.key===sel?'sel':''}" data-key="${p.key}">
      <div class="rinfo">
        <div class="rlabel">${p.label} <span class="badge ${p.kind==='custom'?'custom':''}">${p.kind==='custom'?'custom':'demo'}</span>${p.created_at?` <span class="rdate" style="color:#9aa0a6;font-weight:400;font-size:0.82em;white-space:nowrap">· 🕒 ${fmtRunDate(p.created_at)}</span>`:''}</div>
        <div class="rmeta">${runMeta(p)}</div>
      </div>
      ${stateChip(p)}
      <div class="ractions">
        ${p.has_baseline?`<a data-act="vbase" data-key="${p.key}">view baseline</a>`:''}
        ${p.has_finetuned?`<a data-act="vcmp" data-key="${p.key}">view comparison</a>`:''}
        <button data-act="run" data-key="${p.key}">${p.has_baseline?'re-run':'run baseline'}</button>
        ${(p.kind==='custom'||p.has_baseline||p.has_finetuned)?`<a data-act="del" data-key="${p.key}" class="del" title="delete this run's data to reclaim disk">${p.kind==='custom'?'delete':'clear'}</a>`:''}
      </div>
    </div>`).join('');
  document.querySelectorAll('#runlist .runrow').forEach(row => row.onclick = e => {
    if(e.target.closest('[data-act]')) return;
    selectRun(row.dataset.key);
  });
  document.querySelectorAll('#runlist [data-act]').forEach(el => el.onclick = e => {
    e.preventDefault(); e.stopPropagation();
    const k = el.dataset.key, act = el.dataset.act; sel = k;
    if(act==='vbase'){ gotoStep(2); }
    else if(act==='vcmp'){ gotoStep(4); }
    else if(act==='run'){ runExisting(k); }
    else if(act==='del'){ deleteRun(k); }
  });
}

// Delete a run's data (reclaim disk). Custom runs leave the history; demos keep their row.
async function deleteRun(k){
  const p = PRESETS.find(x => x.key === k) || {};
  const msg = p.kind === 'custom'
    ? `Delete run "${p.label}"?\n\nRemoves its TRT engines, checkpoints, dataset, eval and report, and removes it from history. This cannot be undone. (Your source dataset is NOT touched.)`
    : `Clear results for "${p.label}"?\n\nDeletes its engines, checkpoints, eval and report. The demo stays in the list (re-runnable). This cannot be undone.`;
  if(!confirm(msg)) return;
  let r; try{ r = await jpost('/run/delete', {preset: k}); }catch(e){ alert('Delete failed: ' + e); return; }
  if(!r || !r.ok){ alert((r && r.detail) || 'Delete failed'); return; }
  if(sel === k) sel = null;
  await refreshRuns();
  setStep(1);
}

// Open a run at the most advanced step it has results for.
function selectRun(k){
  sel = k; const r = RUN();
  if(r && r.has_finetuned) gotoStep(4);
  else if(r && r.has_checkpoint) gotoStep(3);  // trained checkpoint, curves + Evaluate gate
  else if(r && (r.has_training || r.is_automl)) gotoStep(3);     // training / AutoML sweep in progress -> Fine-tune (3a)
  else if(r && r.has_baseline) gotoStep(2);
  else { setStep(1); }
  renderRuns();
}

// ----------------------------------------------------------------- step navigation
function avail(step){
  if(step===1) return true;
  const r = RUN(); if(!r) return false;
  const live = LAST||{}; const here = live.preset===sel;
  if(step===2) return !!r.has_baseline || (here && ['baseline','awaiting_gate_a','finetune','evaluating','awaiting_eval','done'].includes(live.phase));
  if(step===3) return !!r.has_baseline && (
    !!r.has_finetuned || !!r.has_checkpoint || !!r.has_training || !!r.is_automl
    || (here && ['finetune','evaluating','awaiting_eval','done'].includes(live.phase))
  );
  if(step===4 || step===5) return !!r.has_finetuned || (here && live.phase==='done');
  if(step===6) return !!r.has_baseline || (here && ['finetune','evaluating','awaiting_eval','done'].includes(live.phase));
  return false;
}

function renderStepper(){
  document.querySelectorAll('.step').forEach(s=>{
    const k=+s.dataset.step, ok=avail(k);
    s.classList.toggle('active', k===curStep);
    s.classList.toggle('done', k<curStep && ok);
    s.classList.toggle('clickable', ok && k!==curStep);
    s.classList.toggle('disabled', !ok);
    s.onclick = ok ? () => gotoStep(k) : null;
  });
}

function updateCtx(){
  const bar=$('ctxbar'), r=RUN();
  if(curStep===1 || !r){ bar.classList.add('hide'); return; }
  bar.classList.remove('hide');
  $('ctxLabel').textContent = r.label;
  $('ctxMeta').textContent = `${r.model_id} → ${r.dataset_id}`;
}

function setStep(n){
  if(_curveT && n!==3){ clearInterval(_curveT); _curveT=null; }   // stop live-curve poll when leaving Fine-tune tab
  curStep = n;
  for(let i=1;i<=6;i++) $('panel'+i).classList.toggle('active', i===n);
  renderStepper(); updateCtx(); icons();
}

function gotoStep(n){
  if(!avail(n)) return;
  setStep(n);
  if(n===2) showStep2();
  else if(n===3) enterFinetuneView();
  else if(n===4) loadFinal(true);
  else if(n===5) loadConfusion();   // Analyze tab: confusion matrix + hardest images
  else if(n===6) enterPlayground(); // Playground tab: upload an image -> live deployed detections
}

// ----------------------------------------------------------------- run control
function enterBaselineRunning(key){
  sel = key; lastPhase = null;
  $('baselineResults').style.display='none'; $('baselineRunning').style.display='';
  $('b_stage').textContent='starting…'; $('b_count').textContent=''; $('b_bar').style.width='0%'; $('b_log').textContent='';
  setStep(2); renderRuns();
}

async function startRun(){
  $('recipeBanner').innerHTML='';
  const model = $('f_model').value.trim(), dataset = $('f_dataset').value.trim();
  const split = $('f_split').value.trim()||'valid', epochs = parseInt($('f_epochs').value)||30;
  const refresh_dataset_cache = !!$('f_refresh_cache').checked;
  if(!model||!dataset){ $('recipeBanner').innerHTML=`<div class="banner warn">Enter both a model and a dataset.</div>`; return; }
  const match = PRESETS.find(p=>p.kind==='preset' && p.model_id===model && p.dataset_id===dataset);
  let res;
  if(match){ res = await jpost('/run/baseline',{preset:match.key, refresh_dataset_cache}); res.key = match.key; }
  else {
    // No manual hyperparameter overrides — the server auto-tunes lr/warmup/scheduler/batch to the
    // detected class count (or AutoML searches them). Epochs + eval split are the only knobs.
    res = await jpost('/run/custom', {model_id:model, dataset_id:dataset, eval_split:split, epochs, refresh_dataset_cache});
  }
  if(res.needs_recipe){ $('recipeBanner').innerHTML=`<div class="banner warn"><b>Needs a deploy recipe.</b> ${res.message}</div>`; return; }
  if(!res.ok){ $('recipeBanner').innerHTML=`<div class="banner warn">${res.detail||'could not start run'}</div>`; return; }
  await refreshRuns();
  enterBaselineRunning(res.key);
}

async function runExisting(key){
  const res = await jpost('/run/baseline',{preset:key, refresh_dataset_cache:!!$('f_refresh_cache').checked});
  if(!res.ok){ alert(res.detail||'could not start'); return; }
  enterBaselineRunning(key);
}

async function startFinetune(){
  const res = await jpost('/run/finetune', {preset: sel});
  if(!res.ok){ alert(res.detail||'could not start fine-tune'); return; }
  lastPhase = null; setStep(3); enterFinetuneView();
}

// ----------------------------------------------------------------- step 2 (baseline)
function showStep2(){
  const running = LAST && LAST.preset===sel && LAST.phase==='baseline';
  $('baselineRunning').style.display = running ? '' : 'none';
  $('baselineResults').style.display = running ? 'none' : '';
  if(!running) loadBaseline(true);
}

async function loadBaseline(force){
  let r; try{ r = await jget('/results/baseline?preset='+sel); }catch(e){ return; }
  $('b_tiles').innerHTML = [
    tile('mAP', fmt(r.map)), tile('mAP@50', fmt(r.map_50)),
    tile('FPS', r.fps!=null?(+r.fps).toFixed(1):'—', r.fps_source),
    tile('Detection rate', r.detection_rate!=null?(100*r.detection_rate).toFixed(0)+'%':'—'),
    tile('Images', r.n_eval),
  ].join('');
  $('b_pertable').innerHTML = `<tr><th>Class</th><th class="num">AP</th><th class="num">Detections</th></tr>` +
    (r.per_class.length ? r.per_class.map(c=>`<tr><td>${c.name}</td><td class="num">${fmt(c.ap)}</td><td class="num">${c.detections}</td></tr>`).join('')
     : `<tr><td colspan="3" class="muted">No target class detected by the stock model (honest mAP 0).</td></tr>`);
  BL_IDS = r.images.map(im=>im.image_id); BL_DATA = {};
  $('b_frames').innerHTML = r.images.map(im=>frameDuo(im.image_id, im.objects)).join('');
  icons();
  BL_IDS.forEach(loadBaselineFrame);
}

function frameDuo(id, nobj){
  return `<div class="frame"><div class="cap">Frame #${id} · ${nobj} ground-truth objects</div>
    <div class="duo">
      <div class="col"><span class="t" id="bgtt_${id}">Ground truth</span>${imgCell('bgt_'+id)}</div>
      <div class="col"><span class="t" id="prt_${id}">Baseline (stock)</span>${imgCell('pr_'+id)}</div>
    </div><div class="zoomhint">click an image to enlarge</div>
    <div class="boxlist" id="bbx_${id}"></div></div>`;
}

async function loadBaselineFrame(id){
  let bx; try{ bx = await jget(`/eval_boxes?preset=${sel}&leg=orig&image_id=${id}&conf=0.005`); }catch(e){ bx={gt:[],pred:[]}; }
  BL_DATA[id] = {url:`/eval_image?preset=${sel}&leg=orig&image_id=${id}`, w:bx.w||640, h:bx.h||640,
                 gt: bx.gt||[], pred: bx.pred||[]};
  renderBaselineFrame(id);
}
function redrawBaselineFrames(){ BL_IDS.forEach(renderBaselineFrame); }
function renderBaselineFrame(id){
  const d = BL_DATA[id]; if(!d) return;
  const pred = d.pred.filter(b=>b.score==null || b.score>=blConf);
  $('bgtt_'+id).textContent = `Ground truth (${d.gt.length})`;
  $('prt_'+id).textContent = `Baseline (${pred.length})`;
  drawBoxes('bgt_'+id, d.url, d.w, d.h, d.gt.map(b=>({xyxy:b.xyxy, name:b.name, color:GT_C})), `Frame #${id} ground truth`);
  drawBoxes('pr_'+id, d.url, d.w, d.h, pred.map(b=>({xyxy:b.xyxy, name:b.name, conf:b.score, color:b.on_target?ON_C:OFF_C})), `Frame #${id} baseline`);
  const gtChips = d.gt.map(b=>`<span class="bchip gt">${b.name}</span>`).join('') || '<span class="muted">none</span>';
  const plChips = pred.length ? pred.slice().sort((a,b)=>b.score-a.score)
      .map(b=>`<span class="bchip ${b.on_target?'on':'off'}">${b.name} ${(+b.score).toFixed(3)}</span>`).join('')
      : '<span class="muted">none above threshold</span>';
  $('bbx_'+id).innerHTML =
    `<div class="bxrow"><span class="bxlab">Ground truth (${d.gt.length}):</span> ${gtChips}</div>` +
    `<div class="bxrow"><span class="bxlab">Baseline (${pred.length}):</span> ${plChips}</div>`;
}

// HTML for one image cell: a NATIVE <img> (the browser renders the photo reliably — never a
// black canvas) + a transparent <canvas> overlay on top that draws ONLY the boxes.
function imgCell(canvasId){
  return `<span class="imgwrap"><img class="frameimg" id="im_${canvasId}" alt=""><canvas class="boxlayer" id="${canvasId}"></canvas></span>`;
}
// Draw ONLY the boxes on the transparent overlay (no drawImage — the <img> shows the photo). The
// canvas coord space = the eval canvas (w×h, box coords are in those pixels); CSS scales both the
// <img> and the overlay to the same display size so boxes align. Decoupled from image load, so
// there is no decode race and no black box.
function drawBoxes(canvasId, url, w, h, boxes, cap){
  const cv = $(canvasId); if(!cv) return;
  const im = $('im_'+canvasId);
  if(im && im.getAttribute('src') !== url){ im.src = url; }   // native image, set once (cached on revisit)
  cv.width = w||640; cv.height = h||640;
  const ctx = cv.getContext('2d'); ctx.clearRect(0,0,cv.width,cv.height);
  const fs = Math.max(11, cv.width/64); ctx.font = `${fs}px Segoe UI`;
  const lw = Math.max(2, cv.width/300);
  boxes.forEach(b=>{
    // user-picked box color applies to ALL boxes (GT + baseline + fine-tuned) — separate panels
    const oc = UICOLORS.box ? UICOLORS.box : b.color;
    const [x1,y1,x2,y2]=b.xyxy;
    ctx.lineWidth=lw; ctx.strokeStyle=oc; ctx.strokeRect(x1,y1,x2-x1,y2-y1);
    if(SHOW_LABELS){   // label = class (+ confidence); toggled off to show boxes only
      const t = b.name + (b.conf!=null?' '+(+b.conf).toFixed(3):'');
      const tw = ctx.measureText(t).width+6;
      ctx.fillStyle=oc; ctx.fillRect(x1,y1-fs-2,tw,fs+2);
      ctx.lineWidth=Math.max(2,fs/6); ctx.strokeStyle='rgba(0,0,0,0.85)'; ctx.lineJoin='round';
      ctx.strokeText(t,x1+3,y1-4);
      ctx.fillStyle=UICOLORS.text||'#fff'; ctx.fillText(t,x1+3,y1-4);
    }
  });
  cv.onclick = () => zoomComposite(im, cv, cap);
}
// Zoom: composite the (already displayed) <img> + the box overlay into one image.
function zoomComposite(im, cv, cap){
  const t = document.createElement('canvas'); t.width = cv.width; t.height = cv.height;
  const c = t.getContext('2d');
  try{ if(im && im.naturalWidth) c.drawImage(im, 0, 0, t.width, t.height); }catch(e){}
  c.drawImage(cv, 0, 0);
  zoom(t.toDataURL(), cap);
}

function metricLabel(name){
  // Keep mixed-case mAP labels (avoid .ml { text-transform: uppercase } → "MAP@50").
  const known = {mAP:'mAP','mAP@50':'mAP@50','mAP@75':'mAP@75'};
  return `<span class="met-lbl">${known[name] || name}</span>`;
}
function tile(label,val,delta,cls){
  const lbl = (typeof label === 'string' && label.startsWith('<')) ? label : metricLabel(label);
  return `<div class="tile ${cls||''}"><div class="ml">${lbl}</div><div class="mv">${val}</div>${delta?`<div class="md muted">${delta}</div>`:''}</div>`;
}

// Deploy + DeepStream eval for the selected/best checkpoint (Fine-tune or Compare tab).
async function startEvaluate(){
  const r = await jpost('/run/evaluate',{preset: sel});
  if(!(r && r.ok)){ alert((r && r.detail) || 'could not start evaluation'); return; }
  gotoStep(3);
}

// ----------------------------------------------------------------- step 3 (fine-tune)
function enterFinetuneView(){
  const here = LAST && LAST.preset===sel;
  const ph = here ? LAST.phase : null;
  const training = ph==='finetune';
  const evaluating = ph==='evaluating';
  const paused = ph==='awaiting_eval';
  $('finetuneRunning').style.display = (training||evaluating) ? '' : 'none';
  $('f_log').style.display = (training||evaluating) ? '' : 'none';
  $('finetuneActions').style.display = training ? '' : 'none';   // "Stop training" only while training
  const r = RUN();
  const idle = !training && !evaluating && !paused;
  refreshDeployEvalGate(paused, r, idle);
  // AutoML runs get sub-tabs: 3a AutoML sweep · 3b Fine-tuning. Other runs go straight to training.
  const isAuto = !!(r && r.is_automl);
  $('ftSubtabs').style.display = isAuto ? '' : 'none';
  if(isAuto){ applyFtSub(); loadAutoml(); }
  else { $('automlView').style.display='none'; $('ftTrainWrap').style.display=''; }
  $('curveLive').textContent = training ? '· updating live' : '';
  // live canvas curve: from the active poll snapshot if training, else from the run's train.log
  if(training && LAST && LAST.curve){ drawCurve(LAST.curve); }
  else { jget('/results/curve?preset='+encodeURIComponent(sel)+'&t='+Date.now()).then(drawCurve).catch(()=>drawCurve({})); }
  // Reset report charts ONLY when switching to a different run, so we never show another run's
  // image — but a continue/refresh of the SAME run keeps its prior curve on screen until the
  // updated chart loads (refreshCharts swaps in place on load), avoiding a blank-graph flash.
  const curSrc = $('chartLoss').getAttribute('src') || '';
  const sameRun = curSrc.indexOf('preset='+encodeURIComponent(sel)+'&') >= 0;
  if(!sameRun){
    ['chartLoss','chartAcc'].forEach(id=>{ $(id).removeAttribute('src'); $(id).style.display='none'; });
    $('lossHdr').style.display='none'; $('accHdr').style.display='none'; $('chartNote').style.display='';
  }
  refreshCharts();
  loadSummary();
  // A CLI/container-launched run isn't driven by the browser poller, so its curve would only
  // update on manual tab re-open. Auto-refresh curve + charts + flags every 20s while the run is
  // still training (has a checkpoint but no final comparison yet). Cleared on tab-leave (setStep)
  // or once the run is compared. The live UI-launched run keeps using the poller (LAST.curve).
  if(_curveT){ clearInterval(_curveT); _curveT=null; }
  // Auto-refresh while a run is training but the browser isn't driving it (CLI/container AutoML or
  // finetune): as soon as it has training logs (has_training) OR a checkpoint, and isn't finished
  // (no final comparison yet). has_training covers the trial/early phase before any checkpoint exists.
  const liveCli = !training && !evaluating && r && (r.has_training || r.has_checkpoint || r.is_automl) && !r.has_finetuned;
  if(liveCli){
    const mine = sel;
    _curveT = setInterval(async ()=>{
      if(curStep!==3 || sel!==mine){ clearInterval(_curveT); _curveT=null; return; }
      await refreshRuns();
      try{ drawCurve(await jget('/results/curve?preset='+encodeURIComponent(mine)+'&t='+Date.now())); }catch(e){}
      refreshCharts();
      const rr = RUN();
      if(rr && rr.is_automl) loadAutoml();   // keep the 3a sweep table live during trials/final
      if(rr && rr.has_finetuned){ clearInterval(_curveT); _curveT=null; renderStepper(); refreshDeployEvalGate(false, rr, true); }
    }, 20000);
    $('curveLive').textContent = '· updating live';
  }
}

// ---- AutoML sweep sub-view (3a) ------------------------------------------------------------
function applyFtSub(){
  const a = ftSub === 'automl';
  $('automlView').style.display = a ? '' : 'none';
  $('ftTrainWrap').style.display = a ? 'none' : '';
  document.querySelectorAll('#ftSubtabs .subtab').forEach(b => b.classList.toggle('active', b.dataset.sub === ftSub));
}
function setFtSub(which){ ftSub = which; applyFtSub(); if(which === 'automl') loadAutoml(); }

async function loadAutoml(){
  let d; try{ d = await jget('/results/automl?preset='+encodeURIComponent(sel)+'&t='+Date.now()); }catch(e){ return; }
  if(!d || !d.is_automl){ $('automlView').innerHTML=''; $('automlPhaseChip').textContent=''; return; }
  const phaseTxt = d.phase==='trials' ? `searching — ${d.n_scored}/${d.n_trials} trials scored`
                 : d.phase==='final' ? `final training (${d.final_epochs} ep) at best config`
                 : 'complete';
  $('automlPhaseChip').textContent = '· '+phaseTxt;
  // 3b ("Fine-tuning") shows ONLY the final full-length training — make that explicit while the
  // sweep is still searching (its curve is empty until a winner is chosen; trial curves live in 3a).
  if(d.phase==='trials'){
    const cn=$('chartNote'); if(cn){ cn.style.display='';
      cn.textContent = `AutoML sweep still searching — this “3b Fine-tuning” curve is the FINAL ${d.final_epochs||''}-epoch training and stays empty until a winner is chosen. Per-trial scores are on “3a AutoML sweep”.`; }
  }
  const fmtMap = v => v==null ? '<span class="muted">—</span>' : `<b>${(100*v).toFixed(1)}%</b>`;
  const fmtSci = v => v==null ? '—' : Number(v).toExponential(1);
  const rows = (d.trials||[]).map(t=>{
    const win = (t.i===d.winner);
    const st = t.status==='scored' ? (win?'<span class="pill win">★ winner</span>':'<span class="pill ok">scored</span>')
             : t.status==='running' ? '<span class="pill run">running…</span>' : '<span class="pill">pending</span>';
    return `<tr class="${win?'winrow':''}">
      <td><b>#${t.i}</b></td><td>${fmtSci(t.lr)}</td><td>${t.warmup}</td><td>${t.sched}</td>
      <td>${fmtSci(t.wd)}</td><td>${t.epochs}</td><td>${fmtMap(t.best_map)}</td><td>${st}</td></tr>`;
  }).join('');
  const fc = d.final_config;
  const finalHtml = fc
    ? `<div class="automlfinal"><b>Winner (trial #${d.winner}, ${fmtMap(d.winner_map)}) → final training:</b>
        lr ${fmtSci(fc.lr)} · warmup ${fc.warmup} · ${fc.sched} · wd ${fmtSci(fc.wd)} · <b>${fc.epochs} epochs</b> on the full dataset.
        Watch it on the <a class="lnk" data-sub="train">3b Fine-tuning</a> tab; deployed before/after lands in <b>Compare</b>.</div>`
    : `<div class="automlfinal muted">Once all ${d.n_trials||''} trials score, the winning config trains a full ${d.final_epochs||''}-epoch model — its config appears here and training shows on <b>3b Fine-tuning</b>.</div>`;
  const legend = (d.trials||[]).map(t=>`<span><i style="background:${AUTOML_COLORS[t.i%AUTOML_COLORS.length]}"></i> #${t.i}${t.i===d.winner?' ★':''}</span>`).join('');
  // Extra context: the searched grid + the best/worst spread (explains how much the search mattered).
  const uniq=(k,f)=> [...new Set((d.trials||[]).map(t=>t[k]))].filter(v=>v!=null).sort((a,b)=>(a>b?1:-1)).map(f||(x=>x)).join(', ');
  const sci=v=>Number(v).toExponential(1);
  const scored=(d.trials||[]).filter(t=>t.best_map!=null);
  const best=scored.length?Math.max(...scored.map(t=>t.best_map)):null;
  const worst=scored.length?Math.min(...scored.map(t=>t.best_map)):null;
  const bestT=scored.find(t=>t.best_map===best), worstT=scored.find(t=>t.best_map===worst);
  const spread=(best&&worst&&worst>0)?(best/worst).toFixed(1)+'×':'—';
  const summary = `<div class="automlsummary">
    <div><span class="k">Search space</span> lr {${uniq('lr',sci)}} · warmup {${uniq('warmup')}} · scheduler {${uniq('sched')}} · weight&nbsp;decay {${uniq('wd',sci)}}</div>
    <div><span class="k">Budget</span> ${d.n_trials||'?'} trials × ${d.trial_epochs||'?'} ep on a data subset → the winner trains ${d.final_epochs||'?'} ep on the full set · scores are PyTorch eval mAP on the 120-image eval subset (a trend signal, not the deployed DeepStream mAP)</div>
    ${scored.length>1?`<div><span class="k">Spread</span> best <b>${(100*best).toFixed(1)}%</b> (trial&nbsp;#${bestT.i}) vs worst <b>${(100*worst).toFixed(1)}%</b> (trial&nbsp;#${worstT.i}) — <b>${spread}</b> · a wide spread means the searched knobs matter here (LR dominates)</div>`:''}
  </div>`;
  $('automlView').innerHTML = `
    <p class="sub">Hyperparameter search: each trial trains <b>${d.trial_epochs}</b> epochs on a data subset; the best per-epoch <b>eval mAP</b> wins, then a full <b>${d.final_epochs}</b>-epoch model trains at that config. Curves + scores are PyTorch eval mAP — the deployed DeepStream mAP is measured later in <b>Compare</b>.</p>
    ${summary}
    <table class="tbl automltbl">
      <thead><tr><th>Trial</th><th>LR</th><th>Warmup</th><th>Scheduler</th><th>Weight&nbsp;decay</th><th>Epochs</th><th>Best eval mAP</th><th>Status</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
    <h3 class="sec">Per-trial eval mAP curves <span class="muted" style="font-weight:400">— each line is one trial's search over its ${d.trial_epochs||''} epochs (winner in bold); the AutoML sweep, separate from the final fine-tuning on 3b</span></h3>
    <div class="curvewrap"><canvas id="automlChart" width="1000" height="260"></canvas>
      <div class="clegend" id="automlLegend">${legend}</div></div>
    ${finalHtml}`;
  drawAutomlChart(d.trials, d.winner);
  $('automlView').querySelectorAll('a.lnk[data-sub]').forEach(a => a.onclick = () => setFtSub(a.dataset.sub));
}

const AUTOML_COLORS = ['#76B900','#1f77b4','#e08a00','#9467bd','#d62728','#17becf','#8c564b','#555555'];
function drawAutomlChart(trials, winner){
  const cv = $('automlChart'); if(!cv) return;
  const ctx = cv.getContext('2d'); const W = cv.width, H = cv.height;
  ctx.clearRect(0,0,W,H);
  const series = (trials||[]).filter(t=>t.curve && (t.curve.epoch||[]).length);
  if(!series.length){ ctx.fillStyle='#999'; ctx.font='13px Segoe UI'; ctx.fillText('waiting for the first trial eval…',16,26); return; }
  const pad={l:46,r:12,t:12,b:26};
  let maxEp=1, maxV=0.05;
  series.forEach(t=>t.curve.epoch.forEach((e,i)=>{ maxEp=Math.max(maxEp,e); maxV=Math.max(maxV,t.curve.eval_map[i]||0); }));
  const X=e=>pad.l+(W-pad.l-pad.r)*(e/maxEp), Y=v=>H-pad.b-(H-pad.t-pad.b)*(v/maxV);
  ctx.fillStyle='#888'; ctx.font='10px Segoe UI';
  for(let g=0;g<=4;g++){ const v=maxV*g/4, y=Y(v);
    ctx.strokeStyle='#f2f2f2'; ctx.beginPath(); ctx.moveTo(pad.l,y); ctx.lineTo(W-pad.r,y); ctx.stroke();
    ctx.fillText((100*v).toFixed(0)+'%',6,y+3); }
  ctx.strokeStyle='#e0e0e0'; ctx.lineWidth=1; ctx.beginPath(); ctx.moveTo(pad.l,pad.t); ctx.lineTo(pad.l,H-pad.b); ctx.lineTo(W-pad.r,H-pad.b); ctx.stroke();
  series.forEach(t=>{
    const col=AUTOML_COLORS[t.i%AUTOML_COLORS.length], win=(t.i===winner);
    ctx.strokeStyle=col; ctx.lineWidth=win?3:1.5; ctx.globalAlpha=win?1:0.8; ctx.beginPath();
    t.curve.epoch.forEach((e,i)=>{ const x=X(e), y=Y(t.curve.eval_map[i]||0); i?ctx.lineTo(x,y):ctx.moveTo(x,y); });
    ctx.stroke();
    const li=t.curve.epoch.length-1; if(li>=0){ ctx.globalAlpha=1; ctx.fillStyle=col; ctx.beginPath(); ctx.arc(X(t.curve.epoch[li]),Y(t.curve.eval_map[li]||0),win?3.5:2.5,0,7); ctx.fill(); }
  });
  ctx.globalAlpha=1; ctx.fillStyle='#666'; ctx.font='10px Segoe UI'; ctx.fillText('epoch →',W-pad.r-46,H-9);
}

// Show Evaluate / Re-evaluate when: first deploy needed, soft-stop pause, OR DeepStream is stale.
async function refreshDeployEvalGate(paused, r, idle){
  let stale = false;
  if(r && r.has_checkpoint){
    try{ const c = await jget('/results/checkpoint?preset='+encodeURIComponent(sel)); stale = !!c.deploy_stale; }
    catch(e){}
  }
  const canEval = paused || (idle && r && r.has_checkpoint && (!r.has_finetuned || stale));
  $('evalGate').style.display = canEval ? '' : 'none';
  if(canEval) updateEvalButton(paused, stale);
  const trained = idle && r && r.has_checkpoint;
  $('finetuneDone').style.display = trained ? '' : 'none';
  $('continueAction').style.display = (idle && r && r.has_finetuned && !stale) ? '' : 'none';
}

// Evaluate button: show best vs latest epoch from /results/checkpoint (matches load_best config).
async function updateEvalButton(stoppedEarly, stale){
  const lbl = $('evaluateRunLbl'), title = $('evalGateTitle'), msg = $('evalGateMsg');
  const cancel = $('cancelEval');
  if(!lbl) return;
  let c; try{ c = await jget('/results/checkpoint?preset='+encodeURIComponent(sel)); }catch(e){
    lbl.textContent = stale ? 'Re-evaluate checkpoint' : 'Evaluate checkpoint';
    if(title) title.textContent = stoppedEarly ? 'Training stopped.' : 'Training complete.';
    if(msg) msg.textContent = 'Deploy → DeepStream eval → report.';
    if(cancel) cancel.style.display = stoppedEarly ? '' : 'none';
    icons(); return;
  }
  const ep = c.epoch, kind = c.selection === 'best' ? 'best' : 'latest';
  const met = (c.metric||'').replace(/^eval_/,'');
  const mv = c.metric_value!=null ? ` · ${met}=${(+c.metric_value).toFixed(3)}` : '';
  const tr = (c.map!=null && c.map_50!=null)
    ? ` Training at that epoch: mAP ${(+c.map).toFixed(3)}, mAP@50 ${(+c.map_50).toFixed(3)}.`
    : '';
  const dep = c.deployed_checkpoint_epoch;
  lbl.textContent = stale ? `Re-evaluate ${kind} epoch (epoch ${ep})` : `Evaluate ${kind} epoch (epoch ${ep})`;
  if(title){
    title.textContent = stale
      ? `DeepStream deploy is out of date (deployed epoch ${dep??'?'})`
      : (stoppedEarly ? 'Training stopped.' : 'Checkpoint ready.');
  }
  if(msg){
    const why = stale
      ? `Section ② in <b>Compare</b> still shows DeepStream results from <b>epoch ${dep??'?'}</b>. `
        + `Re-deploy the <b>${kind}</b> checkpoint (epoch <b>${ep}</b>${mv}) through ONNX → TensorRT → DeepStream.${tr}`
      : (c.load_best_model_at_end
        ? `Deploys the <b>best</b> epoch by <code>${c.metric||'eval_map'}</code> (epoch <b>${ep}</b>${mv}), not the last trained epoch.${tr}`
        : `Deploys the <b>latest</b> completed epoch (<b>${ep}</b>).${tr}`);
    msg.innerHTML = why + ' Updates section ② in Compare.';
  }
  if(cancel) cancel.style.display = (stoppedEarly && !stale) ? '' : 'none';
  icons();
}

// Plain-language fine-tune summary for non-ML reviewers.
async function loadSummary(){
  const els = [$('ftSummary'), $('cmpSummary')].filter(Boolean);
  if(!els.length) return;
  const hide = ()=>els.forEach(el=>el.style.display='none');
  let s; try{ s = await jget('/results/summary?preset='+sel); }catch(e){ hide(); return; }
  if(!s.has_results && !s.has_training){ hide(); return; }
  let am={}; try{ am = await jget('/results/automl?preset='+sel); }catch(e){}
  const isAuto = !!(am && am.is_automl); const win = am.winner; const wmap = am.winner_map;
  const pct = (s.map50_before!=null && s.map50_after!=null)
    ? `<b>${(100*s.map50_after).toFixed(1)}%</b> mAP@50 (was ${(100*s.map50_before).toFixed(1)}%`
      + (s.x?`, ${s.x}× better`:'') + `)`
    : (s.map50_training!=null
      ? `<b>${(100*s.map50_training).toFixed(1)}%</b> mAP@50 on the training eval set (PyTorch — not deployed yet)`
      : '—');
  const lr = (s.lr!=null)? (+s.lr).toExponential(1).replace('e','e') : '—';
  const ec = s.eval_checkpoint || {};
  const ckpt = (s.deployed_epoch!=null)
    ? `<li><b>Deployed checkpoint (the “after” / fine-tuned model):</b> epoch <b>${s.deployed_epoch}</b> of ${s.total_epochs??s.epochs??'—'} `
      + `(most-trained).${(s.min_evalloss_epoch!=null && s.min_evalloss_epoch!==s.deployed_epoch)
          ? ` Lowest eval-loss was epoch ${s.min_evalloss_epoch}, but for DETR detectors eval-loss is a poor accuracy proxy — the most-trained checkpoint is deployed and validated by the deployed mAP above.` : ''}</li>`
    : (ec.epoch!=null
      ? `<li><b>Checkpoint to evaluate:</b> <b>${ec.selection==='best'?'best':'latest'}</b> epoch `
        + `<b>${ec.epoch}</b> of ${ec.total_epochs??s.epochs??'—'}`
        + (ec.map!=null ? ` — training mAP <b>${(+ec.map).toFixed(3)}</b>` : '')
        + (ec.map_50!=null ? `, mAP@50 <b>${(+ec.map_50).toFixed(3)}</b>` : '')
        + (ec.metric_value!=null && ec.map==null ? ` (${(ec.metric||'eval_map').replace(/^eval_/,'')}=${(+ec.metric_value).toFixed(3)})` : '')
        + `${ec.selection==='best' ? ' — selected by load_best_model_at_end, not the last trained epoch.' : '.'}</li>` : '');
  // stored-model location (#3)
  const store = s.model_onnx ? `<li><b>Model stored:</b> <code>${s.model_name||''}</code><br>
      <span class="muted">checkpoint:</span> <code>${s.model_checkpoint||'—'}</code><br>
      <span class="muted">ONNX:</span> <code>${s.model_onnx}</code><br>
      <span class="muted">TRT engine:</span> <code>${s.model_engine||'—'}</code><br>
      <span class="muted">nvinfer config:</span> <code>${s.nvinfer_config||'—'}</code></li>` : '';
  // dataset class distribution (train + eval per-class counts) — explains imbalance + eval coverage
  const dd = s.dataset_distribution;
  const N = n => (+n||0).toLocaleString();
  const distTable = (dd && dd.classes && dd.classes.length) ? `
    <div class="sc-h" style="margin-top:12px">Dataset class distribution</div>
    <table class="disttbl"><tr><th>Class</th><th class="num">Train (inst / imgs)</th><th class="num">Eval (inst / imgs)</th></tr>
    ${dd.classes.map(c=>`<tr class="${c.eval_instances?'':'zero'}"><td>${c.name}</td>
        <td class="num">${N(c.train_instances)} / ${N(c.train_images)}</td>
        <td class="num">${N(c.eval_instances)} / ${N(c.eval_images)}</td></tr>`).join('')}
    <tr class="tot"><td>Total</td>
        <td class="num">${N(dd.train_total_instances)} / ${N(dd.n_train_images)}</td>
        <td class="num">${N(dd.eval_total_instances)} / ${N(dd.n_eval_images)}</td></tr></table>
    <p class="sc-foot">inst = annotated objects, imgs = images with the class. Greyed rows have <b>0 eval instances</b> (not in the KPI eval slice → mAP averages only over the classes that are).</p>` : '';
  // training configuration table — the actual fine-tune hyperparameters + how each was set.
  // For an AutoML run the tuned settings come from the winning trial, NOT the Setup page.
  const howTuned = isAuto ? `AutoML-selected (winner trial #${win})` : 'auto by class count';
  const tcRows = [
    ['Learning rate', lr, isAuto ? howTuned : 'auto by class count · Advanced override'],
    ['LR scheduler', s.sched||'—', howTuned],
    ['Warmup ratio', s.warmup??'—', howTuned],
    ['Epochs', s.epochs??'—', isAuto ? 'AutoML final-run epochs (you set the count)' : 'you set it on Setup'],
    ['Batch size', s.batch??'—', 'default 8 · Advanced override'],
    ['Grad accumulation', s.grad_accum??1, 'fixed'],
    ['Weight decay', s.weight_decay??'—', isAuto ? howTuned : 'fixed'],
    ['Precision', s.precision||'—', 'fixed (mixed precision)'],
    ['Eval / best by', `${s.eval_strategy||'epoch'} · ${s.metric_for_best_model||'eval_loss'}`, 'fixed'],
  ];
  const autoBanner = isAuto
    ? `<p class="sc-foot" style="border-left:3px solid #2ea043;padding:6px 10px;background:rgba(46,160,67,.08);border-radius:4px">
        <b>AutoML run:</b> the tuned parameters below were selected by the sweep — winner <b>trial #${win}</b>${wmap!=null?` (subset eval mAP ${(+wmap).toFixed(3)})`:''} —
        <b>not</b> the Setup-page values. The full 16-trial leaderboard is on <b>Fine-tune → 3a AutoML sweep</b>.</p>` : '';
  const cfgFoot = isAuto
    ? `<p class="sc-foot">The four tuned settings (learning rate, LR scheduler, warmup, weight decay) were chosen by the
        <b>AutoML sweep</b> (winner trial #${win}), then trained for <b>${s.epochs??'?'} epochs</b> on the full dataset.
        Batch size / precision / eval strategy are fixed defaults. Setup-page manual values were <b>not</b> used for the tuned four.</p>`
    : `<p class="sc-foot">Chosen from this dataset's <b>${s.n_classes??'?'} classes</b> (not copied from a prior run):
      <b>≥3 classes</b> use lr 2.5e-5 / cosine / warmup 0.1 — RT-DETR's classification head diverges at higher LR
      with several classes; <b>≤2 classes</b> use lr 1e-4 / linear. <i>Advanced settings</i> override the auto
      values; the rest are fixed.</p>`;
  const trainCfg = `
    <div class="sc-h" style="margin-top:12px">Training configuration${isAuto?' — AutoML-selected':''}</div>
    ${autoBanner}
    <table class="disttbl"><tr><th>Parameter</th><th>Value</th><th>How it was set</th></tr>
    ${tcRows.map(r=>`<tr><td>${r[0]}</td><td>${r[1]}</td><td class="muted">${r[2]}</td></tr>`).join('')}</table>
    ${cfgFoot}`;
  const html = `<div class="sc-h">Summary</div>
    <p>Fine-tuned <b>${s.model_id}</b> on <b>${s.dataset_id}</b> (${s.n_classes} classes${s.classes&&s.classes.length?': '+s.classes.join(', '):''}).</p>
    <ul class="sc-list">
      <li><b>Trained</b> on ${s.n_train??'—'} images · <b>evaluated</b> on ${s.eval_images??'—'} held-out images.</li>
      <li><b>${s.epochs??'—'} epochs</b> at learning rate ${lr} (${s.sched||'—'} schedule).</li>
      <li><b>Batch size</b> ${s.batch??'—'}${(s.grad_accum&&s.grad_accum>1)?` × ${s.grad_accum} grad-accum (effective ${s.batch*s.grad_accum})`:' (single GPU)'}.</li>
      <li><b>Detection accuracy:</b> ${pct} — higher is better; measured on the real DeepStream pipeline.</li>
      ${s.best_class?`<li><b>Best class:</b> ${s.best_class} (AP ${(s.best_ap).toFixed(2)}).</li>`:''}
      ${ckpt}
      ${store}
    </ul>
    ${distTable}
    ${trainCfg}
    <p class="sc-foot">mAP@50 = how well boxes match ground truth at 50% overlap, averaged over classes.</p>`;
  // Fine-tune tab gets the FULL card. Compare tab gets ONLY the fine-tune parameters + epochs +
  // which model — no duplicating the summary/dataset/accuracy already shown elsewhere on Compare.
  const modelLine = (s.deployed_epoch!=null || s.model_engine)
    ? `<p class="sc-foot"><b>“After” (fine-tuned) model:</b> checkpoint epoch <b>${s.deployed_epoch??'—'}</b>${s.model_engine?` → <code>${s.model_engine}</code>`:''}</p>` : '';
  const cmpHtml = `${trainCfg}${modelLine}`;
  const ft=$('ftSummary'); if(ft){ ft.style.display=''; ft.innerHTML=html; }
  const cmp=$('cmpSummary'); if(cmp){ cmp.style.display=''; cmp.innerHTML=cmpHtml; }
}

// Report-style charts (matplotlib PNGs from the server). A probe Image swaps src only on
// success, so a not-yet-available chart leaves the previous one in place (no flicker).
function refreshCharts(){
  if(!sel) return;
  const t = Date.now();
  const loss=$('chartLoss'), lh=$('lossHdr'), note=$('chartNote');
  const lp = new Image();
  lp.onload = () => { loss.src = lp.src; loss.style.display=''; lh.style.display=''; note.style.display='none'; };
  lp.onerror = () => { if(!loss.src){ loss.style.display='none'; lh.style.display='none'; note.style.display=''; } };
  lp.src = `/chart?preset=${encodeURIComponent(sel)}&kind=loss&t=${t}`;
  const acc=$('chartAcc'), ah=$('accHdr');
  const ap = new Image();
  ap.onload = () => { acc.src = ap.src; acc.style.display=''; ah.style.display=''; };
  ap.onerror = () => { acc.style.display='none'; ah.style.display='none'; };
  ap.src = `/chart?preset=${encodeURIComponent(sel)}&kind=acc&t=${t}`;
  const mp=$('chartMap'), mh=$('mapHdr');
  if(mp){ const mi = new Image();
    mi.onload = () => { mp.src = mi.src; mp.style.display=''; mh.style.display=''; };
    mi.onerror = () => { if(!mp.src){ mp.style.display='none'; mh.style.display='none'; } };
    mi.src = `/chart?preset=${encodeURIComponent(sel)}&kind=map&t=${t}`;
  }
}

// ----------------------------------------------------------------- step 4 (compare)
async function loadFinal(force){
  loadSummary();   // render the fine-tune config / which-model card on the Compare page too (AutoML-aware)
  let r; try{ r = await jget('/results/final?preset='+sel); }catch(e){ return; }
  const a = r.accuracy||{};
  const ck = r.checkpoint_evaluated || {};
  const depEp = r.deployed_checkpoint_epoch;
  const kind = ck.selection==='best' ? 'Best' : 'Latest';
  const ckEp = ck.epoch;

  // Stale deploy warning: DeepStream results are from a different epoch than selected checkpoint.
  const warn = $('cmpStaleWarn');
  if(warn){
    if(r.deploy_stale && ckEp!=null && depEp!=null){
      const kind = ck.selection==='best' ? 'best' : 'latest';
      warn.innerHTML = `<b>DeepStream results are out of date.</b> Section ② is from deployed <b>epoch ${depEp}</b>, `
        + `but the selected checkpoint is <b>epoch ${ckEp}</b>. `
        + `<div class="actions" style="margin-top:10px">`
        + `<button class="btn primary" id="cmpReEvaluate"><i data-lucide="gauge"></i> Re-evaluate ${kind} epoch (${ckEp}) in DeepStream</button>`
        + ` <button class="btn" id="cmpGoFinetune"><i data-lucide="sparkles"></i> Open Fine-tune tab</button>`
        + `</div>`;
      $('cmpReEvaluate').onclick = () => startEvaluate();
      $('cmpGoFinetune').onclick = () => gotoStep(3);
      warn.style.display = '';
      icons();
    } else { warn.style.display='none'; warn.innerHTML=''; }
  }
  const epBadge = (el, ep, suffix) => {
    if(!el) return;
    el.textContent = ep!=null ? `(epoch ${ep}${suffix||''})` : '';
  };
  epBadge($('ckTrainEp'), ckEp, ck.selection==='best' ? ' · best by '+((ck.metric||'eval_map').replace(/^eval_/,'')) : '');
  epBadge($('ckDeployEp'), depEp, r.deploy_stale ? ' · stale' : '');

  const trainTile = (lbl, val) => (val==null ? '' :
    `<div class="tile train"><div class="ml">${metricLabel(lbl)}</div><div class="mv">${fmt(val,3)}</div>`
    + `<div class="md">PyTorch training eval</div></div>`);
  // Primary metric first (matches "best by map" badge and section ② below).
  $('c_tiles_train').innerHTML =
    (ckEp!=null ? trainTile('mAP', ck.map) : '')
    + (ckEp!=null ? trainTile('mAP@50', ck.map_50) : '')
    + (ckEp!=null ? trainTile('mAP@75', ck.map_75) : '')
    || '<p class="muted">No training metrics for the selected checkpoint yet — finish training or re-run Evaluate.</p>';

  const accTile = (lbl,key)=>{
    const m = a[key]; if(!m) return '';
    const up = (m.delta||0) >= 0;
    const xs = m.x!=null ? ` · ${m.x}×` : '';
    return `<div class="tile ${up?'up':'down'}"><div class="ml">${metricLabel(lbl)}</div>
      <div class="mv">${fmt(m.original,3)} <span class="ar">→</span> ${fmt(m.finetuned,3)}</div>
      <div class="md">${up?'▲':'▼'} ${(m.delta>=0?'+':'')}${fmt(m.delta,3)}${xs} · DeepStream</div></div>`;
  };
  const perf = r.perf||{};
  const qo = perf.original?.trtexec_qps, qf = perf.finetuned?.trtexec_qps;
  const hasDs = a.map_50 || a.map;
  $('c_tiles_ds').innerHTML = hasDs
    ? accTile('mAP','map') + accTile('mAP@50','map_50') + accTile('mAP@75','map_75')
      + (qf!=null?tile('Engine FPS (TRT FP16)', (+qf).toFixed(0), (qo!=null&&Math.abs(qo-qf)<1?'unchanged (same arch)':'')+' · DeepStream'):'')
    : '<p class="muted">No DeepStream deploy eval yet — use <b>Evaluate '+kind.toLowerCase()+' epoch</b> on the Fine-tune tab.</p>';

  const trainPc = ck.per_class_ap || {};
  $('c_pertable').innerHTML = `<tr><th>Class</th><th class="num">Eval GT</th>`
    + (ckEp!=null ? `<th class="num">Training AP<br><span class="muted">epoch ${ckEp}</span></th>` : '')
    + `<th class="num">Baseline AP<br><span class="muted">DeepStream</span></th>`
    + `<th class="num">Fine-tuned AP<br><span class="muted">DeepStream</span></th></tr>`
    + r.per_class.map(c=>{
      const tr = trainPc[c.name];
      return `<tr><td>${c.name}</td><td class="num">${c.det_gt==null?'—':c.det_gt}</td>`
        + (ckEp!=null ? `<td class="num">${tr==null?'—':fmt(tr)}</td>` : '')
        + `<td class="num">${c.original==null?'—':fmt(c.original)}</td>`
        + `<td class="num">${fmt(c.finetuned)}</td></tr>`;
    }).join('');
  renderVerdict(r);
  FINAL_SAMPLES = (r.frames && r.frames.length) ? r.frames : (r.samples||[]).map(s=>s.id);
  FRAME_DATA = {};
  $('c_frames').innerHTML = FINAL_SAMPLES.map(id=>`
    <div class="frame"><div class="cap">Frame #${id}</div>
      <div class="trio">
        <div class="col"><span class="t" id="gtt_${id}">Ground truth</span>${imgCell('gt_'+id)}</div>
        <div class="col"><span class="t" id="blt_${id}">Baseline (stock)</span>${imgCell('bl_'+id)}</div>
        <div class="col"><span class="t" id="ftt_${id}">Fine-tuned</span>${imgCell('ft_'+id)}</div>
      </div>
      <div class="zoomhint">click an image to enlarge</div>
      <div class="boxlist" id="bx_${id}"></div>
      <div class="framecounts" id="fc_${id}"></div></div>`).join('');
  icons();
  FINAL_SAMPLES.forEach(loadFrameData);
  const dl = $('dlReport');
  if(r.has_report){ dl.style.display=''; dl.href = '/report?preset='+sel; } else { dl.style.display='none'; }
  icons();
}

// per-frame, per-class counts under each comparison image
// Compare-tab sample frames. Boxes are fetched ONCE per frame (down to conf 0.05) and cached;
// the slider then filters + redraws client-side, so it's truly real-time (no per-drag fetch).
async function loadFrameData(id){
  let bo, bf;
  try{ bo = await jget(`/eval_boxes?preset=${sel}&leg=orig&image_id=${id}&conf=0.005`); }catch(e){ bo={gt:[],pred:[]}; }
  try{ bf = await jget(`/eval_boxes?preset=${sel}&leg=ft&image_id=${id}&conf=0.005`); }catch(e){ bf={gt:[],pred:[]}; }
  FRAME_DATA[id] = {url:`/eval_image?preset=${sel}&leg=orig&image_id=${id}`, w:bo.w||640, h:bo.h||640,
                    gt: bo.gt||[], baseline: bo.pred||[], ft: bf.pred||[]};
  renderFrame(id);
}

// ---- Deployed model context (Analyze + Playground) ----
let MODEL_CTX = null;

function renderModelBanner(el, m, ctx){
  if(!el) return;
  if(!m || !m.available){
    el.style.display = 'none'; el.innerHTML = ''; return;
  }
  const stale = m.role === 'finetuned_deploy' && m.deploy_stale;
  el.className = 'modelbanner' + (el.id === 'confModelBanner' ? ' modelbanner-main' : '')
    + (stale ? ' warn' : (m.role === 'finetuned_deploy' ? ' train' : ''));
  const mainWide = el.id === 'confModelBanner';
  const titleBlock = mainWide && ctx && ctx.run_label
    ? `<div class="mb-title">Run: <b>${ctx.run_label}</b> · ${m.title||'Deployed model'}</div>`
    : (ctx && ctx.run_label
      ? `<div class="mb-run"><span class="muted">Run</span> ${ctx.run_label}</div>`
        + `<div class="mb-title">${m.title||'Deployed model'}</div>`
      : `<div class="mb-title">${m.title||'Deployed model'}</div>`);
  const ds = (m.map!=null || m.map_50!=null)
    ? `<div class="mb-meta">DeepStream on ${m.n_eval||'?'} eval images: `
      + `mAP <b>${fmt(m.map)}</b>, mAP@50 <b>${fmt(m.map_50)}</b></div>` : '';
  let train = '';
  if(m.role === 'finetuned_deploy' && (m.training_map!=null || m.training_map_50!=null)){
    train = `<div class="mb-meta">Training eval at same checkpoint (PyTorch, 120-image val): `
      + `mAP <b>${fmt(m.training_map)}</b>, mAP@50 <b>${fmt(m.training_map_50)}</b></div>`;
  }
  const dsName = (m.dataset_id||'').split('/').filter(Boolean).pop() || 'dataset';
  const ck = m.checkpoint ? `<div class="mb-meta">Weights: <code>${m.checkpoint}</code></div>` : '';
  const staleNote = stale
    ? `<div class="mb-stale">⚠ Deployed epoch <b>${m.deployed_epoch??'?'}</b> ≠ selected checkpoint epoch `
      + `<b>${m.checkpoint_epoch??'?'}</b> — re-evaluate on the Fine-tune tab for fresh analysis.</div>` : '';
  el.innerHTML = titleBlock
    + `<div class="mb-meta">${m.subtitle||''}${m.model_id ? ` · ${m.model_id} on ${dsName}` : ''}</div>`
    + ds + train + ck + staleNote;
  el.style.display = '';
}

function updateModelSelectLabels(ctx){
  if(!ctx) return;
  const label = (m, fallback) => {
    if(!m || !m.available) return fallback + ' (not deployed)';
    if(m.role === 'stock_baseline') return m.title || fallback;
    const ep = m.deployed_epoch ?? m.checkpoint_epoch;
    const kind = m.checkpoint_selection === 'best' ? 'best' : 'latest';
    let s = `Fine-tuned · ep ${ep??'?'} (${kind})`;
    if(ctx.run_label) s = `${ctx.run_label} — ${s}`;
    return s;
  };
  const confSel = $('confWhich'), pgSel = $('pgWhich');
  if(confSel){
    const o = confSel.querySelector('option[value=ft]');
    const b = confSel.querySelector('option[value=orig]');
    if(o) o.textContent = label(ctx.ft, 'Fine-tuned');
    if(b) b.textContent = ctx.orig && ctx.orig.available ? (ctx.orig.title || 'Baseline') : 'Baseline (not run)';
    if(!ctx.ft || !ctx.ft.available){
      if(confSel.value === 'ft' && ctx.orig && ctx.orig.available) confSel.value = 'orig';
      if(o) o.disabled = true;
    } else if(o) o.disabled = false;
  }
  if(pgSel){
    const o = pgSel.querySelector('option[value=ft]');
    const b = pgSel.querySelector('option[value=orig]');
    if(o) o.textContent = label(ctx.ft, 'Fine-tuned');
    if(b) b.textContent = ctx.orig && ctx.orig.available ? (ctx.orig.title || 'Baseline') : 'Baseline (not run)';
    if(!ctx.ft || !ctx.ft.available){
      if(pgSel.value === 'ft' && ctx.orig && ctx.orig.available) pgSel.value = 'orig';
      if(o) o.disabled = true;
    } else if(o) o.disabled = false;
  }
}

async function refreshModelContext(whichBanner){
  if(!sel) return null;
  try{ MODEL_CTX = await jget('/results/model_context?preset='+encodeURIComponent(sel)); }
  catch(e){ MODEL_CTX = null; return null; }
  updateModelSelectLabels(MODEL_CTX);
  const which = whichBanner === 'pg'
    ? (($('pgWhich')||{}).value || 'ft')
    : (($('confWhich')||{}).value || 'ft');
  const m = which === 'orig' ? MODEL_CTX.orig : MODEL_CTX.ft;
  renderModelBanner(whichBanner === 'pg' ? $('pgModelBanner') : $('confModelBanner'), m, MODEL_CTX);
  return MODEL_CTX;
}

// ---- Error analysis: confusion matrix + hardest-images gallery ----
async function loadConfusion(){
  if(!sel) return;
  await refreshModelContext('conf');
  const which = ($('confWhich')||{}).value || 'ft';
  const conf = parseFloat(($('confConf')||{}).value || '0.2');
  const iou = parseFloat(($('confIou')||{}).value || '0.5');
  $('confWrap').innerHTML = '<p class="muted">Analyzing…</p>';
  let r;
  try{ r = await jget(`/results/confusion?preset=${encodeURIComponent(sel)}&which=${which}&conf=${conf}&iou=${iou}`); }
  catch(e){
    $('confSummary').innerHTML='';
    const leg = which === 'orig' ? 'baseline' : 'fine-tuned';
    $('confWrap').innerHTML = `<p class="muted">No ${leg} eval results yet — finish deploy + evaluate first.</p>`;
    $('hardFrames').innerHTML=''; return;
  }
  if(r.model) renderModelBanner($('confModelBanner'), r.model, MODEL_CTX);
  LAST_CONF = {r, which, conf};
  renderConfSummary(r.summary); renderConfMatrix(r); renderHardest(r, which, conf);
}

function renderConfSummary(s){
  const el = $('confSummary'); if(!el) return;
  if(!s){ el.innerHTML=''; return; }
  el.innerHTML = '<div class="cshow"><b>How to read:</b> each cell counts ground-truth boxes (row) by what the model predicted at that location (column). '
    + 'Green diagonal = correct; red = class confusion; amber = misses (<i>(missed)</i> column) or spurious detections (<i>(background)</i> row).</div>'
    + '<ul class="csins">' + (s.insights||[]).map(x=>`<li>${x}</li>`).join('') + '</ul>';
}

function renderConfMatrix(r){
  const rows=r.rows||[], cols=r.cols||[], C=r.confusion||{};
  let mx=1; rows.forEach(rw=>cols.forEach(c=>{ const v=(C[rw]||{})[c]||0; if(v>mx)mx=v; }));
  const cell=(rw,c)=>{ const v=(C[rw]||{})[c]||0; if(!v) return '<td class="cm0"></td>';
    const a=Math.min(1,0.18+0.82*v/mx);
    const bg = rw===c ? `rgba(46,125,50,${a})`                          // correct (diagonal)
             : (c==='(missed)'||rw==='(background)') ? `rgba(184,134,11,${a})`  // miss / spurious
             : `rgba(192,57,43,${a})`;                                  // class confusion
    return `<td class="cmv" style="background:${bg}" title="GT ${rw} → pred ${c}: ${v}">${v}</td>`; };
  let h='<div class="cmwrap"><table class="cmtbl"><tr><th class="cmcorner">GT \\ Pred</th>'+
    cols.map(c=>`<th>${c}</th>`).join('')+'</tr>'+
    rows.map(rw=>`<tr><th>${rw}</th>`+cols.map(c=>cell(rw,c)).join('')+'</tr>').join('')+'</table></div>';
  const f=x=>x==null?'—':(+x).toFixed(3);
  h+='<table class="systbl pr"><tr><th>class</th><th>precision</th><th>recall</th><th>TP</th><th>FP</th><th>FN</th></tr>'+
    (r.classes||[]).map(n=>{ const v=r.per_class[n]||{};
      return `<tr><td>${n}</td><td>${f(v.precision)}</td><td>${f(v.recall)}</td><td>${v.tp||0}</td><td>${v.fp||0}</td><td>${v.fn||0}</td></tr>`; }).join('')+'</table>';
  $('confWrap').innerHTML=h;
}

function renderHardest(r, which, conf){
  const hard=(r.hardest||[]).filter(h=>h.errors>0).slice(0,6);
  const m = r.model || (MODEL_CTX && (which === 'orig' ? MODEL_CTX.orig : MODEL_CTX.ft));
  if(!hard.length){ $('hardFrames').innerHTML='<p class="muted">No error frames at this threshold — clean result.</p>'; return; }
  $('hardFrames').innerHTML = hard.map(h=>`
    <div class="frame"><div class="cap">Frame #${h.image_id} — ${h.fp} FP · ${h.fn} miss</div>
      <div class="duo">
        <div class="col"><span class="t">Ground truth</span>${imgCell('hgt_'+h.image_id)}</div>
        <div class="col"><span class="t">Predicted (${m && m.title ? m.title : which})</span>${imgCell('hpr_'+h.image_id)}</div>
      </div></div>`).join('');
  icons();
  hard.forEach(async h=>{
    let b; try{ b=await jget(`/eval_boxes?preset=${sel}&leg=${which}&image_id=${h.image_id}&conf=${conf}`); }catch(e){ return; }
    const url=`/eval_image?preset=${sel}&leg=${which}&image_id=${h.image_id}`;
    drawBoxes('hgt_'+h.image_id, url, b.w||640, b.h||640, (b.gt||[]).map(x=>({xyxy:x.xyxy,name:x.name,color:GT_C})), `#${h.image_id} ground truth`);
    drawBoxes('hpr_'+h.image_id, url, b.w||640, b.h||640, (b.pred||[]).map(x=>({xyxy:x.xyxy,name:x.name,conf:x.score,color:x.on_target?ON_C:OFF_C})), `#${h.image_id} predicted`);
  });
}

function redrawFinalFrames(){ FINAL_SAMPLES.forEach(renderFrame); }

function renderFrame(id){
  const d = FRAME_DATA[id]; if(!d) return;
  const keep = b => (b.score==null) || b.score >= cmpConf;
  const bl = d.baseline.filter(keep), ft = d.ft.filter(keep);
  $('gtt_'+id).textContent = `Ground truth (${d.gt.length})`;
  $('blt_'+id).textContent = `Baseline (${bl.length})`;
  $('ftt_'+id).textContent = `Fine-tuned (${ft.length})`;
  drawBoxes('gt_'+id, d.url, d.w, d.h, d.gt.map(b=>({xyxy:b.xyxy, name:b.name, color:GT_C})), `Frame #${id} — ground truth`);
  drawBoxes('bl_'+id, d.url, d.w, d.h, bl.map(b=>({xyxy:b.xyxy, name:b.name, conf:b.score, color:b.on_target?ON_C:OFF_C})), `Frame #${id} — baseline`);
  drawBoxes('ft_'+id, d.url, d.w, d.h, ft.map(b=>({xyxy:b.xyxy, name:b.name, conf:b.score, color:b.on_target?ON_C:OFF_C})), `Frame #${id} — fine-tuned (≥${cmpConf.toFixed(3)})`);
  // reviewable box list: GT classes, and fine-tuned class + confidence
  const gtChips = d.gt.map(b=>`<span class="bchip gt">${b.name}</span>`).join('') || '<span class="muted">none</span>';
  const ftChips = ft.length ? ft.sort((a,b)=>b.score-a.score)
      .map(b=>`<span class="bchip ${b.on_target?'on':'off'}">${b.name} ${(+b.score).toFixed(3)}</span>`).join('')
      : '<span class="muted">none above threshold</span>';
  $('bx_'+id).innerHTML =
    `<div class="bxrow"><span class="bxlab">Ground truth (${d.gt.length}):</span> ${gtChips}</div>` +
    `<div class="bxrow"><span class="bxlab">Fine-tuned ≥${cmpConf.toFixed(3)} (${ft.length}):</span> ${ftChips}</div>`;
  // per-class count table (client-side, real-time)
  const tally = arr => arr.reduce((m,b)=>(m[b.name]=(m[b.name]||0)+1,m),{});
  const g=tally(d.gt), b0=tally(bl), f0=tally(ft);
  const classes=[...new Set([...Object.keys(g),...Object.keys(b0),...Object.keys(f0)])].sort((x,y)=>(g[y]||0)-(g[x]||0));
  $('fc_'+id).innerHTML = classes.length ? (
    `<table class="tbl fctbl"><tr><th>Class</th><th class="num">GT</th><th class="num">Baseline</th><th class="num">Fine-tuned</th></tr>` +
    classes.map(c=>`<tr><td>${c}</td><td class="num">${g[c]||0}</td><td class="num">${b0[c]||0}</td><td class="num">${f0[c]||0}</td></tr>`).join('') + `</table>`) : '';
}

// ----------------------------------------------------------------- polling
async function pollLoop(){
  if(polling) return; polling = true;
  const tick = async () => {
    let s;
    try{ s = await jget('/run/status'); }catch(e){ setTimeout(tick,1500); return; }
    renderStatus(s);
    setTimeout(tick, 1500);
  };
  tick();
}

function renderStatus(s){
  LAST = s;
  const pill = $('statePill');
  pill.className = 'pill' + (s.status==='running'?' busy':s.status==='error'?' err':'');
  pill.textContent = s.phase;
  $('statusMsg').textContent = s.stage_label ? s.stage_label : (s.error||'Ready.');
  // A run training via an external CLI/container harness isn't in the UI's STATE (phase='idle'),
  // so reflect it here instead of showing "idle": the server flags it via s.external.
  if(s.external){
    const isSweep = s.external.kind==='automl' && s.external.phase==='trials';
    pill.className = 'pill busy';
    pill.textContent = isSweep ? 'automl sweep' : 'fine-tuning';
    $('statusMsg').textContent = (isSweep ? 'AutoML sweep running'
        : (s.external.kind==='automl' ? 'Fine-tuning — final training' : 'Fine-tuning'))
      + ` · ${s.external.label} (launched via CLI/container)`;
  }
  $('elapsed').textContent = s.elapsed ? s.elapsed+'s' : '';

  // bottom status bar: overall stage % + within-stage progress (images / epochs)
  const running = s.status==='running';
  const sp = s.stage_progress;
  const fmtN = n => (n==null?'':(+n).toLocaleString());
  let txt = '';
  if(running){
    if(s.pct!=null && s.total) txt = `Stage ${s.done}/${s.total} · ${s.pct}% overall`;
    if(sp && sp.total) txt += `${txt?'  ·  ':''}${fmtN(sp.done)}/${fmtN(sp.total)} ${sp.unit||'items'}${sp.pct!=null?` (${sp.pct}%)`:''}`;
  }
  $('sbProg').textContent = txt;
  const sb = $('sbBar'); const wrap = sb.parentElement;
  const barpct = (sp && sp.pct!=null) ? sp.pct : (s.pct||0);
  if(running){ wrap.classList.add('on'); sb.style.width = barpct+'%'; }
  else { wrap.classList.remove('on'); sb.style.width='0%'; }

  const pct = s.total ? Math.round(100*s.done/s.total) : 0;
  const cnt = s.total ? `${s.done}/${s.total}` : '';
  // keep the running widgets fresh regardless of which step is visible
  if(s.phase==='baseline'){
    $('b_stage').textContent = s.stage_label||'…'; $('b_count').textContent = cnt;
    $('b_bar').style.width = pct+'%'; $('b_log').textContent = (s.log_tail||[]).join('\n'); $('b_log').scrollTop=1e9;
  }
  if(s.phase==='finetune' || s.phase==='evaluating'){
    $('f_stage').textContent = s.stage_label||'…'; $('f_count').textContent = cnt;
    $('f_bar').style.width = pct+'%'; $('f_log').textContent = (s.log_tail||[]).join('\n'); $('f_log').scrollTop=1e9;
    if(curStep===3 && ++chartTick % 3 === 0) refreshCharts();   // report PNG charts — ~4.5s cadence
  }
  // Live canvases: redraw whenever the Fine-tune tab is open and we have curve data — covers finetune,
  // evaluating AND done (the poll runs continuously), so the loss/mAP graphs stay populated after
  // training finishes instead of going blank. (s.curve is the merged history+live curve.)
  if(curStep===3 && s.curve && (s.curve.train_epoch||[]).length){
    drawCurve(s.curve);
  }

  // auto-advance only on phase change AND only if we're following the active run
  const following = (sel === s.preset) || sel === null;
  if(s.phase !== lastPhase){
    if(following && s.preset){
      sel = s.preset;
      if(s.phase==='baseline') gotoStep(2);
      else if(s.phase==='finetune' || s.phase==='evaluating' || s.phase==='awaiting_eval') gotoStep(3);
      else if(s.phase==='awaiting_gate_a'){ refreshRuns().then(()=>gotoStep(2)); }
      else if(s.phase==='done'){ refreshRuns().then(()=>gotoStep(4)); }
    }
    if(['error','done','awaiting_gate_a','awaiting_eval'].includes(s.phase)) refreshRuns();
    lastPhase = s.phase;
  }
  renderStepper(); updateCtx();
}

// ----------------------------------------------------------------- live training curve (canvas)
// Detailed "is fine-tuning helping?" assessment on the Compare tab.
function renderVerdict(r){
  const el = $('cmpVerdict'); if(!el) return;
  const m = (r.accuracy||{}).map_50;
  if(!m){ el.style.display='none'; return; }
  const before = m.original||0, after = m.finetuned||0, d = after-before;
  let verdict, cls;
  if(after>=0.5 && d>=0.2){ verdict='Fine-tuning clearly helped'; cls='good'; }
  else if(d>=0.1){ verdict='Fine-tuning helped'; cls='good'; }
  else if(d>0.02){ verdict='Fine-tuning helped modestly'; cls='ok'; }
  else if(d>-0.02){ verdict='Fine-tuning made little difference'; cls='ok'; }
  else { verdict='Fine-tuning did not help (regressed)'; cls='bad'; }
  const pc = r.per_class||[];
  const gained = pc.filter(c=> (c.finetuned||0) > (c.original||0)+0.02)
                   .sort((a,b)=>(b.finetuned-(b.original||0))-(a.finetuned-(a.original||0)));
  const lost   = pc.filter(c=> c.original!=null && (c.finetuned||0) < c.original-0.02);
  const weak   = pc.filter(c=> (c.finetuned||0) < 0.05);
  const chip = (c,showFrom)=>`<span class="bchip ${(c.finetuned||0)>=(c.original||0)?'on':'off'}">${c.name} `
      + `${showFrom&&c.original!=null?fmt(c.original,2)+'->':''}${fmt(c.finetuned,2)}</span>`;
  const xtxt = m.x!=null ? ` (${m.x}x better)` : '';
  const ck = r.checkpoint_evaluated || {};
  const depEp = r.deployed_checkpoint_epoch;
  let html = `<div class="sc-h verdict ${cls}">${verdict}</div>`;
  if(ck.epoch!=null && (ck.map!=null || ck.map_50!=null)){
    html += `<p><b>① Training (epoch ${ck.epoch}):</b> mAP `
      + `${ck.map!=null?(100*ck.map).toFixed(1)+'%':'—'}`
      + (ck.map_50!=null ? `, mAP@50 ${(100*ck.map_50).toFixed(1)}%` : '') + ` <span class="muted">(PyTorch)</span></p>`;
  }
  if(depEp!=null){
    html += `<p><b>② DeepStream deployed${r.deploy_stale?' (stale — epoch '+depEp+')':''}:</b> mAP@50 `
      + `${(100*before).toFixed(1)}% → ${(100*after).toFixed(1)}%`
      + `${d>=0?' (+':' ('}${(100*d).toFixed(1)} pts${xtxt}).</p>`;
  } else {
    html += `<p>DeepStream mAP@50 <b>${(100*before).toFixed(1)}% → ${(100*after).toFixed(1)}%</b>`
      + `${d>=0?' (+':' ('}${(100*d).toFixed(1)} pts${xtxt}).</p>`;
  }
  if(gained.length) html += `<p><b>Improved (${gained.length}):</b> ${gained.map(c=>chip(c,true)).join(' ')}</p>`;
  if(lost.length)   html += `<p><b>Regressed (${lost.length}):</b> ${lost.map(c=>chip(c,true)).join(' ')}</p>`;
  if(weak.length)   html += `<p><b>Still weak / not learned (${weak.length}):</b> ${weak.map(c=>chip(c,false)).join(' ')}`
      + ` <span class="muted">- may need more epochs or more examples of these classes.</span></p>`;
  html += `<p class="sc-foot">Baseline classes at 0 mean the stock COCO model has no such class - fine-tuning is what creates that capability. Confidence scores are often modest even when mAP is high; use the slider to reveal detections.</p>`;
  el.innerHTML = html; el.style.display='';
}

function drawCurve(c){
  c = c || {};
  const cv = $('curve'); if(!cv) return;
  const W = cv.width = cv.clientWidth||1000, H = cv.height;
  const ctx = cv.getContext('2d'); ctx.clearRect(0,0,W,H);
  const all = [...(c.train_loss||[]), ...(c.eval_loss||[])];
  const eps = [...(c.train_epoch||[]), ...(c.eval_epoch||[])];
  if(!all.length){ ctx.fillStyle='#999'; ctx.font='13px Segoe UI'; ctx.fillText('waiting for the first training log…',16,28); return; }
  const pad = {l:58,r:16,t:16,b:40};
  const xMax = Math.max(...eps, 1), yMax = Math.max(...all), yMin = Math.min(...all, 0);
  const X = e => pad.l + (e/xMax)*(W-pad.l-pad.r);
  const Y = v => pad.t + (1-(v-yMin)/((yMax-yMin)||1))*(H-pad.t-pad.b);
  // axes
  ctx.strokeStyle='#e4e4e4'; ctx.lineWidth=1; ctx.beginPath();
  ctx.moveTo(pad.l,pad.t); ctx.lineTo(pad.l,H-pad.b); ctx.lineTo(W-pad.r,H-pad.b); ctx.stroke();
  ctx.fillStyle='#999'; ctx.font='11px Segoe UI';
  // y ticks (min / mid / max) + light gridlines
  ctx.textAlign='right'; ctx.textBaseline='middle';
  for(let k=0;k<=2;k++){ const v=yMin+(yMax-yMin)*k/2, y=Y(v);
    ctx.fillText(v.toFixed(1), pad.l-8, y);
    ctx.strokeStyle='#f0f0f0'; ctx.beginPath(); ctx.moveTo(pad.l,y); ctx.lineTo(W-pad.r,y); ctx.stroke(); }
  // x ticks (0 / mid / max)
  ctx.textAlign='center'; ctx.textBaseline='top';
  for(let k=0;k<=2;k++){ const e=xMax*k/2; ctx.fillText(e.toFixed(e<10?1:0), X(e), H-pad.b+6); }
  // axis titles
  ctx.fillStyle='#555'; ctx.font='12px Segoe UI';
  ctx.fillText('epoch', (pad.l+W-pad.r)/2, H-16);
  ctx.save(); ctx.translate(14,(pad.t+H-pad.b)/2); ctx.rotate(-Math.PI/2);
  ctx.textAlign='center'; ctx.textBaseline='middle'; ctx.fillText('loss', 0, 0); ctx.restore();
  ctx.textAlign='left'; ctx.textBaseline='alphabetic';
  const line = (epArr,vArr,color)=>{
    if(!vArr||!vArr.length) return;
    ctx.strokeStyle=color; ctx.lineWidth=2; ctx.beginPath();
    epArr.forEach((e,i)=>{ const x=X(e),y=Y(vArr[i]); i?ctx.lineTo(x,y):ctx.moveTo(x,y); });
    ctx.stroke();
    ctx.fillStyle=color; epArr.forEach((e,i)=>{ ctx.beginPath(); ctx.arc(X(e),Y(vArr[i]),2.5,0,7); ctx.fill(); });
  };
  line(c.train_epoch, c.train_loss, '#76B900');
  line(c.eval_epoch, c.eval_loss, '#1f77b4');
}

// (per-epoch mAP is shown as the server-rendered report chart `/chart?kind=map` via refreshCharts —
//  the live canvas was removed; the PNG also includes the per-class AP panel.)

// ----------------------------------------------------------------- system & environment panel
// Dataset QC modal — health check over the ingested annotations (no training needed).
async function openQcModal(){
  const body = $('qcBody'); $('qcModal').classList.add('open');
  if(!sel){ body.innerHTML = '<p class="muted">Select a run first (Setup tab), then open Dataset QC.</p>'; return; }
  body.innerHTML = '<p class="muted">Analyzing dataset…</p>';
  let r; try{ r = await jget('/results/dataset_qc?preset='+encodeURIComponent(sel)); }
  catch(e){ body.innerHTML = '<p class="muted">Dataset not ingested yet — run/select a run that has finished ingest.</p>'; return; }
  const VC = {green:'#2e7d32', amber:'#b8860b', red:'#c0392b'};
  const verdict = r.verdict||'green';
  let h = `<div class="qcverdict" style="border-color:${VC[verdict]};color:${VC[verdict]}">`
        + `Dataset verdict: <b>${verdict.toUpperCase()}</b>${verdict==='green'?' — no major issues found':''}</div>`;
  if((r.flags||[]).length){
    h += '<div class="qcflags">' + r.flags.map(f=>
      `<div class="qcflag qc-${f.level}"><span class="qcdot" style="background:${VC[f.level]}"></span>${f.msg}</div>`).join('') + '</div>';
  }
  // per-class balance table
  h += '<h4 class="qch">Class balance</h4><table class="systbl"><tr><th>class</th><th>train inst</th><th>train img</th><th>valid inst</th><th>valid img</th></tr>';
  (r.classes||[]).forEach(c=>{ const v=r.combined_per_class[c]||{};
    h += `<tr><td>${c}</td><td>${v.train_instances??0}</td><td>${v.train_images??0}</td><td>${v.valid_instances??0}</td><td>${v.valid_images??0}</td></tr>`; });
  h += `</table><div class="muted" style="margin-top:4px">imbalance ratio (largest:smallest class) = ${r.imbalance_ratio}:1</div>`;
  // per-split image + geometry cards
  h += '<h4 class="qch">Per-split</h4><div class="qcsplits">';
  Object.entries(r.splits||{}).forEach(([sp,s])=>{
    const g=s.geometry||{};
    h += `<div class="qccard"><div class="qccardh">${sp}</div>`
      + `<div>${s.n_images} images · ${s.n_annotations} boxes · ${s.boxes_per_image_avg}/img avg</div>`
      + `<div>no-label images: <b>${s.no_label_images}</b></div>`
      + `<div>tiny: ${g.tiny} · huge: ${g.huge} · out-of-bounds: ${g.out_of_bounds} · degenerate: ${g.degenerate} · extreme-AR: ${g.extreme_aspect}</div></div>`;
  });
  h += '</div>';
  const lk = r.leakage||{count:0};
  h += `<h4 class="qch">Train ↔ valid leakage</h4><div>${lk.count? `<b style="color:${VC.red}">${lk.count}</b> shared image filename(s) — leakage inflates eval mAP. e.g. ${(lk.examples||[]).slice(0,5).join(', ')}` : 'none detected (by filename) ✓'}</div>`;
  h += '<div class="muted" style="margin-top:10px">Computed from the ingested COCO annotations (no image decode).</div>';
  body.innerHTML = h;
}

// ---- Inference playground (Step 6): upload an image -> run the DEPLOYED engine -> draw detections ----
async function enterPlayground(){
  await refreshModelContext('pg');
  if(!PG_RESULT) $('pgStatus').textContent = 'Drop or choose an image to run the deployed model on it.';
}

async function uploadAndInfer(file){
  if(!sel || !file) return;
  PG_FILE = file;
  const which = $('pgWhich').value;
  const m = MODEL_CTX && (which === 'orig' ? MODEL_CTX.orig : MODEL_CTX.ft);
  const leg = m && m.title ? m.title : (which === 'orig' ? 'baseline' : 'fine-tuned');
  $('pgStatus').textContent = `Running ${leg} on “${file.name}”… (deployed FP16 engine)`;
  const fd = new FormData(); fd.append('file', file);
  let r;
  try{
    const resp = await fetch(`/infer?preset=${encodeURIComponent(sel)}&which=${which}&conf=0.01`, {method:'POST', body:fd});
    if(!resp.ok){ const j = await resp.json().catch(()=>({})); throw (j.detail || ('HTTP '+resp.status)); }
    r = await resp.json();
  }catch(e){ $('pgStatus').textContent = 'Inference failed: ' + (typeof e==='string'?e:JSON.stringify(e)); $('pgResult').style.display='none'; return; }
  PG_RESULT = r;
  $('pgStatus').textContent = `Done — ${r.detections.length} raw detection(s) from the ${r.which} model. Drag the confidence slider to filter.`;
  renderPg();
}

function renderPg(){
  if(!PG_RESULT) return;
  const conf = parseFloat($('pgConf').value);
  const dets = (PG_RESULT.detections||[]).filter(d=>d.score>=conf);
  $('pgResult').style.display='';
  $('pgImgHost').innerHTML = imgCell('pgImg'); icons();
  $('pgCap').textContent = `${PG_RESULT.which} model — ${dets.length} detection(s) ≥ ${conf.toFixed(2)}`;
  drawBoxes('pgImg', PG_RESULT.image, PG_RESULT.w, PG_RESULT.h,
            dets.map(d=>({xyxy:d.xyxy, name:d.name, conf:d.score, color:'#ff3b30'})), 'Playground prediction');
  const chips = dets.length
    ? dets.map(d=>`<span class="bchip on">${d.name} ${(+d.score).toFixed(3)}</span>`).join('')
    : '<span class="muted">no detections above this confidence</span>';
  $('pgList').innerHTML = `<div class="bxrow"><span class="bxlab">Detections (${dets.length}):</span> ${chips}</div>`;
}

async function openSysModal(){
  const body = $('sysBody'); body.innerHTML = '<p class="muted">Loading…</p>';
  $('sysModal').classList.add('open');
  let sy; try{ sy = await jget('/system'); }catch(e){ body.innerHTML = '<p class="muted">Unavailable.</p>'; return; }
  const e = sy.environment || {};
  const row = (k, v) => v ? `<tr><td>${k}</td><td>${v}</td></tr>` : '';
  let html = '<table class="systbl">';
  if(sy.gpu) html += row('GPU (live)', `${sy.gpu.name} · ${sy.gpu.mem_used_gb} / ${sy.gpu.mem_total_gb} GB used`);
  html += row('Container image', e.container_image || sy.image);
  html += row('Running in', sy.in_container ? 'DeepStream container' : 'host (view-only)');
  html += row('OS', e.os) + row('CPU', e.cpu) + row('RAM', e.ram);
  html += row('GPU', e.gpu) + row('GPU driver', e.gpu_driver) + row('CUDA (driver)', e.cuda_driver);
  html += row('DeepStream', e.deepstream) + row('TensorRT', e.tensorrt) + row('Python', e.python);
  html += '</table>';
  const pk = e.packages || {};
  if(Object.keys(pk).length){
    html += '<div class="modalh2">Key package versions</div><table class="systbl pkgtbl">';
    for(const [k, v] of Object.entries(pk)) html += `<tr><td>${k}</td><td>${v}</td></tr>`;
    html += '</table>';
  }
  if(!sy.environment) html += '<p class="muted" style="margin-top:10px">Full hardware/stack details appear after a run generates <code>build/env_info.json</code>.</p>';
  body.innerHTML = html;
}

// ----------------------------------------------------------------- lightbox
function zoom(src, cap){
  $('lbImg').src = src; $('lbCap').textContent = cap||'';
  $('lightbox').classList.add('open');
}

init();
