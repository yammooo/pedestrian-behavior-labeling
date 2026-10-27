'use strict';
const $ = id => document.getElementById(id);
const letters = ['A', 'B', 'C', 'D'];
const colors = ['#73c5a6', '#7eaff0', '#efbf67', '#ce9be8'];
const shortClass = name => ({MOVING:'Moving',STOPPED:'Stopped',WAITING_TO_CROSS:'Waiting',CROSSING:'Crossing'}[name]||name);
const state = {dataset:'loki', split:'test', runs:'', metric:'accuracy', focus:'', axis:'', value:'0', scope:'matching', sort:'0', descending:'false', search:'', minimum:'0', page:0, locator:'', frame:0, active:0};
let catalog = [], result = null, track = null, selectedRow = null, context = null;
let generation = 0, queryGeneration = 0, mediaGeneration = 0, timer = null;
const selectedRuns = () => state.runs.split(',').filter(Boolean);
const error = message => {$('error').textContent = message || '';};
const node = (tag, text, cls) => {const n=document.createElement(tag); if(text!==undefined)n.textContent=text; if(cls)n.className=cls; return n;};
async function api(path, params={}, body=null) {
  const response = await fetch(path+(Object.keys(params).length?'?'+new URLSearchParams(params):''), body===null?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  const value = await response.json();
  if(!response.ok)throw new Error(value.error || response.statusText);
  return value;
}
function label(run) {const arms={g:'Geometry', 'g-rgb':'Geometry + RGB', 'g-dditr':'Geometry + LiDAR'};const config=arms[run.configuration]?`${arms[run.configuration]} (${run.configuration})`:run.configuration;const design=run.variant_label?`${run.variant_label} (${run.model_variant})`:run.architecture;const architecture=run.variant_label?` · ${run.architecture}`:'';const selected=run.selected_epoch?` · selected e${run.selected_epoch}`:'';const evaluation=run.evaluation_label?` · ${run.evaluation_label}`:'';return `${run.experiment} · ${design} · ${config} · ${run.source}${architecture} · max${run.epochs}/pat${run.patience} · seed${run.seed}${selected}${evaluation} · ${run.attempt}`;}
function options(select, values, selected) {
  select.replaceChildren();
  values.forEach(([value,text])=>{const option=node('option',text); option.value=value;select.append(option);});
  if([...select.options].some(o=>o.value===String(selected)))select.value=selected;
}
function modelSelectors() {
  const terms=$('run-search').value.toLowerCase().split(/\s+/).filter(Boolean), runs=selectedRuns();
  $('models').replaceChildren();
  letters.forEach((letter,i)=>{
    const wrap=node('label',`Model ${letter}${i?' (optional)':''}`), select=node('select');select.id='model-'+i;select.setAttribute('aria-label','Model '+letter);
    const matching=catalog.filter(r=>r.split===state.split && r.datasets.includes(state.dataset) && (r.id===runs[i] || terms.every(t=>label(r).toLowerCase().includes(t))));
    options(select,[['','Select a completed run'],...matching.map(r=>[r.id,label(r)])],runs[i]||'');
    [...select.options].forEach(o=>o.disabled=!!o.value && runs.includes(o.value) && o.value!==runs[i]);
    select.onchange=()=>{const values=letters.map((_,j)=>$('model-'+j).value).filter(Boolean); state.runs=values.join(',');state.locator='';state.sort='0';state.axis='';state.value='0';state.page=0;modelSelectors();sortOptions();query();};
    wrap.append(select);$('models').append(wrap);
  });
}
function sortOptions() {
  const choices=selectedRuns().map((_,i)=>[String(i),letters[i]+' score']);
  selectedRuns().slice(1).forEach((_,i)=>choices.push([`${i+1}-0`,`${letters[i+1]} − A (signed difference)`]));
  options($('sort'),choices.length?choices:[['0','A score']],state.sort);
  state.sort=$('sort').value;
}
function syncControls() {
  ['dataset','split','metric','focus','scope','sort','descending','search','minimum'].forEach(id=>$(id).value=state[id]);
  modelSelectors();sortOptions();
}
function conditionOptions(axes) {
  options($('axis'),[['','All observations'],...Object.keys(axes).map(a=>[a,a])],state.axis);
  if($('axis').value!==state.axis)throw new Error('Saved condition is unavailable for these runs. Clear the condition or choose compatible runs.');
  options($('value'),state.axis?axes[state.axis].map((v,i)=>[String(i),v]):[['0','All']],state.value);
  state.value=$('value').value;
  $('value').disabled=!state.axis;$('scope').disabled=!state.axis;
}
function focusOptions(classes) {
  const values=[['','Overall errors']];
  classes.forEach((a,i)=>classes.forEach((b,j)=>{if(i!==j)values.push([`${i},${j}`,`${shortClass(a)} → ${shortClass(b)}`]);}));
  options($('focus'),values,state.focus);
  $('metric').querySelector('[value=confusion_rate]').disabled=!state.focus;
}
function format(value, metric=state.metric) {return value===null||value===undefined?'—':metric==='ce'?value.toFixed(3):(100*value).toFixed(1)+'%';}
function signed(value) {return value===null?'—':`${value>=0?'+':''}${(value*(state.metric==='ce'?1:100)).toFixed(state.metric==='ce'?3:1)}${state.metric==='ce'?'':' pp'}`;}
function pause() {if(timer!==null)clearTimeout(timer);timer=null;$('play').textContent='Play';}
function clearTrack() {
  generation++;mediaGeneration++;pause();track=null;context=null;selectedRow=null;
  ['rgb','bev','crop'].forEach(id=>{$(id).removeAttribute('src');$(id+'-placeholder').hidden=false;});
  ['play','step-back','step-next','previous-error','next-error','save','frame'].forEach(id=>$(id).disabled=true);
  $('track-title').textContent='Select a track';$('media-status').textContent='';$('media-frame-status').textContent='';
  ['score-table','confusion','probabilities','slot-info','model-labels'].forEach(id=>$(id).replaceChildren());
  $('timeline').getContext('2d').clearRect(0,0,$('timeline').width,$('timeline').height);
}
async function query(restore=null) {
  const version=++queryGeneration;error();pause();
  if(!selectedRuns().length){clearTrack();$('track-table').tBodies[0].replaceChildren();$('scope-info').textContent='Select model A to begin.';return;}
  $('track-count').textContent='Loading prediction index…';
  try {
    const data=await api('/api/tracks',state);if(version!==queryGeneration)return;
    if(restore && JSON.stringify(restore.hashes)!==JSON.stringify(data.hashes))throw new Error('Saved predictions changed. This case was not reopened.');
    result=data;conditionOptions(data.axes);focusOptions(data.classes);
    $('scope-info').textContent=`${state.split} split · Scores: ${state.axis && state.scope==='matching'?'accepted GT in matching frames':'accepted GT over the whole track'}. ${state.axis?'Filter: '+state.axis+' = '+data.axes[state.axis][+state.value]+'. ':''}${state.metric==='macro_f1'?'F1 averages GT-supported classes per track; not the four-class experiment score. ':''}Shared tracks: ${data.common}; missing from the intersection A–D: ${data.missing.join(' / ')}.`;
    renderPicker();
    if(restore){await openTrack(restore.state.locator,+restore.state.frame||0);}
    else if(data.rows.length){const row=data.rows.find(r=>r.locator===state.locator);if(row && track){selectedRow=row;renderScores();renderFrame();}else await openTrack((row||data.rows[0]).locator,row?state.frame:0);}
    else clearTrack();
    updateLink();
  } catch(e){if(version===queryGeneration){clearTrack();error(e.message);$('track-count').textContent='Could not load tracks';}}
}
function renderPicker() {
  const table=$('track-table'),head=node('tr');head.append(node('th','Track'),node('th','GT counts'));
  selectedRuns().forEach((_,i)=>{const th=node('th'),button=node('button',letters[i]);button.title='Sort by '+letters[i];button.onclick=()=>{state.sort=String(i);state.descending=state.metric==='ce'||state.metric==='confusion_rate'?'true':'false';state.page=0;syncControls();query();};th.append(button);head.append(th);});
  if(state.sort.includes('-'))head.append(node('th','Δ '+$('sort').selectedOptions[0].textContent.split(' (')[0]));
  table.tHead.replaceChildren(head);const body=table.tBodies[0];body.replaceChildren();
  result.rows.forEach(row=>{
    const tr=node('tr');tr.dataset.locator=row.locator;tr.tabIndex=0;if(row.locator===state.locator)tr.className='selected';
    const identity=JSON.parse(row.locator), name=node('td',identity.track_id||identity.tube_uid);name.append(node('small',identity.scenario||identity.clip));name.title=row.locator;
    const bar=node('div',undefined,'classbar'),support=row.scores[0].support,total=support.reduce((a,b)=>a+b,0);support.forEach((n,i)=>{if(n){const span=node('span');span.style.width=(100*n/total)+'%';span.style.background=colors[i%colors.length];span.title=result.classes[i]+': '+n;bar.append(span);}});name.append(bar);tr.append(name);
    const count=node('td',`${row.scores[0].gt_count} scored`);count.append(node('small',`${row.matching_gt} matching / ${row.total_gt} total`));tr.append(count);
    row.values.forEach((value,i)=>{const cell=node('td',format(value));if(state.metric==='confusion_rate')cell.append(node('small',`${row.scores[i].confusion_errors}/${row.scores[i].confusion_support}`));tr.append(cell);});
    if(state.sort.includes('-'))tr.append(node('td',signed(row.rank)));
    tr.onclick=()=>openTrack(row.locator);tr.onkeydown=e=>{if(e.key==='Enter')openTrack(row.locator);};body.append(tr);
  });
  $('track-count').textContent=`${result.total} matching tracks · ${state.metric==='ce'?'CE':state.metric==='macro_f1'?'Supported-class F1':state.metric==='confusion_rate'?'Confusion rate':'Accuracy'}`;
  $('page').textContent=`Page ${result.page+1} / ${Math.max(1,Math.ceil(result.total/result.page_size))}`;
  $('previous-page').disabled=result.page===0;$('next-page').disabled=(result.page+1)*result.page_size>=result.total;
}
async function openTrack(locator, frame=0) {
  clearTrack();const version=generation;error();state.locator=locator;state.frame=frame;
  selectedRow=result.rows.find(r=>r.locator===locator)||null;
  $('track-title').textContent=locator;$('media-status').textContent='Loading track predictions…';
  try {
    const data=await api('/api/track',{runs:state.runs,dataset:state.dataset,split:state.split,locator});if(version!==generation)return;
    track=data;state.active=Math.min(state.active,data.models.length-1);
    $('model-labels').replaceChildren();selectedRuns().forEach((id,i)=>{$('model-labels').append(node('div',letters[i]+': '+label(catalog.find(r=>r.id===id))));});
    $('frame').max=track.targets.length-1;state.frame=Math.max(0,Math.min(state.frame,track.targets.length-1));
    ['play','step-back','step-next','previous-error','next-error','save','frame'].forEach(id=>$(id).disabled=false);
    $('legend').replaceChildren();track.classes.forEach((c,i)=>{const span=node('span'),swatch=node('i',undefined,'swatch');swatch.style.background=colors[i%colors.length];span.append(swatch,document.createTextNode(c));$('legend').append(span);});
    $('road-overlay').disabled=state.dataset!=='road-waymo';
    renderScores();renderFrame();renderPicker();updateLink();
    $('media-status').textContent='Preparing same-frame context… Metrics and timelines are ready.';
    const job=await api('/api/media',{run:selectedRuns()[0],dataset:state.dataset,split:state.split,locator});
    if(version===generation)poll(job.job,version);
  } catch(e){if(version===generation)error(e.message);}
}
async function poll(id, version) {
  try {
    const job=await api('/api/job',{id});if(version!==generation)return;
    if(job.status==='ready'){context=job.result;$('media-status').textContent=`${context.cache_hit?'Cached context':'Context prepared in '+context.seconds.toFixed(1)+' s'} · yellow native box; cyan ROAD overlay is optional.`;renderFrame();}
    else if(job.status==='unavailable'||job.status==='cancelled'){$('media-status').textContent='Media unavailable: '+(job.error||job.status)+'. Scores and timelines remain usable.';}
    else {$('media-status').textContent=`${job.status}: ${job.done}/${job.total} frames. Scores and timelines remain usable.`;setTimeout(()=>poll(id,version),700);}
  } catch(e){if(version===generation)$('media-status').textContent='Media unavailable: '+e.message;}
}
function match(i) {return !state.axis || track.conditions[state.axis]?.[i]===+state.value;}
function isError(i) {
  if(!track.gt_valid[i] || (state.scope==='matching'&&!match(i)))return false;
  const pred=track.models[state.active].predicted[i];
  if(state.focus){const [a,b]=state.focus.split(',').map(Number);return track.targets[i]===a&&pred===b;}
  return track.targets[i]!==pred;
}
function navigateError(direction) {
  if(!track)return;pause();
  for(let i=state.frame+direction;i>=0&&i<track.targets.length;i+=direction)if(isError(i)){setFrame(i);return;}
  $('saved-status').textContent='No further matching error for '+letters[state.active];
}
function setFrame(i) {state.frame=Math.max(0,Math.min(i,track.targets.length-1));renderFrame();updateLink();}
async function showMedia(i) {
  const version=++mediaGeneration,key=context.key;
  $('media-frame-status').textContent=`Loading synchronized context for frame ${i+1}…`;
  try {
    const loaded=await Promise.all(['rgb','bev','crop'].map(name=>new Promise((resolve,reject)=>{
      const variant=name==='rgb'&&$('road-overlay').checked?'rgb-road':name==='crop'&&!$('tight-crop').checked?'crop-3x':name,image=new Image();
      image.onload=()=>resolve([name,image.src]);image.onerror=()=>reject(new Error('Could not load '+name));
      image.src='/media?'+new URLSearchParams({key,image:String(i).padStart(4,'0')+'-'+variant+'.jpg'});
    })));
    if(version!==mediaGeneration)return;
    loaded.forEach(([name,src])=>{$(name).src=src;$(name+'-placeholder').hidden=true;});
    const centerNote=context.frames[i].pedestrian_centered_bev_radius_m?` · ${context.frames[i].pedestrian_centered_bev_radius_m} m context radius centered on native 3D target`:' · no native 3D target center; radius outline omitted';
    $('media-frame-status').textContent=`${state.split} · context frame ${i+1}/${track.targets.length} · selected source ${track.source_times_us[i]??'missing'} µs${centerNote}`;
  } catch(e){if(version===mediaGeneration)$('media-frame-status').textContent='Context unavailable: '+e.message;}
}
function renderFrame() {
  if(!track)return;const i=state.frame;$('frame').value=i;
  const requested=(track.grid_times_us[i]-track.grid_times_us[0])/1e6;
  $('slot-info').textContent=`Frame ${i+1}/${track.targets.length} · nominal grid +${requested.toFixed(1)} s · source ${track.source_times_us[i]??'missing'} µs · GT ${track.gt_valid[i]?track.classes[track.targets[i]]:'unlabeled (unscored)'}${state.axis?' · '+(match(i)?'matches condition':'outside condition'):''}`;
  $('active-model').textContent='Inspecting '+letters[state.active];
  if(context){showMedia(i);
    const info=context.frames[i];$('availability').textContent=JSON.stringify({split:state.split,native:info,source_flags:Object.fromEntries(Object.entries(track.source_flags||{}).map(([k,v])=>[k,v[i]])),conditions:Object.fromEntries(Object.entries(track.conditions).map(([k,v])=>[k,result.axes[k]?.[v[i]]??v[i]]))},null,2);drawTrajectory();}
  drawTimeline();renderConfusion();
  const scores=track.models[state.active].probabilities[i];$('probabilities').replaceChildren();scores.forEach((p,j)=>{const item=node('div',`${track.classes[j]}: ${(100*p).toFixed(1)}%`);item.style.borderLeft=`8px solid ${colors[j%colors.length]}`;item.style.padding='3px 8px';$('probabilities').append(item);});
}
function drawTimeline() {
  const canvas=$('timeline'),ctx=canvas.getContext('2d'),n=track.targets.length,left=170,width=canvas.width-left-10,height=34;
  canvas.height=30+(track.models.length+2)*height;ctx.font='13px system-ui';
  const rows=[{name:'Accepted GT',values:track.targets,gt:true},...track.models.map((m,j)=>({name:letters[j]+(j===state.active?' · active':''),values:m.predicted,index:j}))];
  rows.forEach((row,r)=>{const top=24+r*height;ctx.fillStyle='#e5edf6';ctx.fillText(row.name,4,top+19);row.values.forEach((v,i)=>{const x=left+i*width/n,w=width/n;ctx.fillStyle=row.gt&&!track.gt_valid[i]?'#424c58':colors[v%colors.length];ctx.fillRect(x,top,Math.max(w,.5),23);if(!row.gt&&track.gt_valid[i]&&v!==track.targets[i]){ctx.fillStyle='#ff6262';ctx.fillRect(x,top+25,Math.max(w,.5),3);}});});
  const top=24+rows.length*height;ctx.fillStyle='#afbdd0';ctx.fillText(state.axis?'Condition match':'All context',4,top+16);
  for(let i=0;i<n;i++){ctx.fillStyle=match(i)?'#eac77e':'#303a49';ctx.fillRect(left+i*width/n,top,width/n,17);}
  const x=left+(state.frame+.5)*width/n;ctx.strokeStyle='#fff';ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(x,18);ctx.lineTo(x,canvas.height);ctx.stroke();ctx.fillStyle='#afbdd0';ctx.fillText('0 s',left,13);ctx.fillText(((n-1)*.2).toFixed(1)+' s',canvas.width-55,13);
}
function renderScores() {
  if(!track)return;$('detail-scope').textContent=state.axis&&state.scope==='matching'?'Matching frames':'Whole track';
  const table=node('table'),head=node('tr');['Model','Accuracy','CE','Supported F1','GT / class support'].forEach(t=>head.append(node('th',t)));table.append(head);
  track.models.forEach((model,i)=>{const score=selectedRow?.scores[i]||model.scores,row=node('tr');row.append(node('td',letters[i]),node('td',format(score.accuracy,'accuracy')),node('td',format(score.ce,'ce')),node('td',format(score.macro_f1,'macro_f1')),node('td',`${score.gt_count} · [${score.support.join(', ')}]`));row.onclick=()=>{state.active=i;renderFrame();};table.append(row);});
  $('score-table').replaceChildren(table);
}
function renderConfusion() {
  const model=track.models[state.active],score=selectedRow?.scores[state.active]||model.scores,table=node('table');
  const head=node('tr');head.append(node('th','GT ↓ / pred →'));track.classes.forEach(c=>{const th=node('th',shortClass(c));th.title=c;head.append(th);});table.append(head);
  score.confusion.forEach((values,i)=>{const row=node('tr');row.append(node('th',shortClass(track.classes[i])));values.forEach((v,j)=>{const cell=node('td',String(v));if(v&&i!==j)cell.style.color='#ffb5a9';row.append(cell);});table.append(row);});
  $('confusion-title').textContent=letters[state.active]+' confusion · GT rows / prediction columns';$('confusion').replaceChildren(table);
}
function drawTrajectory() {
  const canvas=$('trajectory'),ctx=canvas.getContext('2d');ctx.clearRect(0,0,400,400);
  const all=Object.values(context.trajectory).flat().filter(p=>p.every(v=>v!==null));if(!all.length)return;
  const xs=all.map(p=>p[0]),ys=all.map(p=>p[1]),cx=(Math.max(...xs)+Math.min(...xs))/2,cy=(Math.max(...ys)+Math.min(...ys))/2,span=Math.max(Math.max(...xs)-Math.min(...xs),Math.max(...ys)-Math.min(...ys),1);
  Object.entries(context.trajectory).forEach(([name,points])=>{ctx.strokeStyle=name==='ped_position'?'#ffe36d':'#6de1ff';ctx.beginPath();let previous=false;points.forEach(p=>{if(p.some(v=>v===null)){previous=false;return;}const x=200-(p[1]-cy)/span*340,y=200-(p[0]-cx)/span*340;if(previous)ctx.lineTo(x,y);else ctx.moveTo(x,y);previous=true;});ctx.stroke();const p=points[state.frame];if(p?.every(v=>v!==null)){ctx.fillStyle=ctx.strokeStyle;ctx.beginPath();ctx.arc(200-(p[1]-cy)/span*340,200-(p[0]-cx)/span*340,5,0,2*Math.PI);ctx.fill();}});
  ctx.fillStyle='#e5edf6';ctx.fillText('Fixed frame: +x up, +y left · yellow pedestrian / cyan ego',8,390);
}
function updateLink() {history.replaceState(null,'','#'+encodeURIComponent(JSON.stringify(state)));}
async function reviews() {
  try{const cases=await api('/api/reviews');$('reviews').replaceChildren();cases.slice().reverse().forEach(record=>{const item=node('div',undefined,'review'),button=node('button',record.reason||record.state.locator);button.onclick=async()=>{pause();queryGeneration++;clearTrack();Object.assign(state,record.state);state.split=record.state.split||record.split||'test';syncControls();if(!await refresh())return;state.axis=record.state.axis||'';await query(record);$('note').value=record.note;$('reason').value=record.reason;};item.append(button,node('small',` ${record.split||'test'} · ${record.saved_utc}`),node('p',record.note));$('reviews').append(item);});}catch(e){error(e.message);}
}
async function refresh() {const requestedSplit=state.split;try{const data=await api('/api/runs',{split:requestedSplit});if(state.split!==requestedSplit)return false;catalog=data.runs;modelSelectors();if(data.warnings.length)error(data.warnings.join('\n'));if(!catalog.length)$('scope-info').textContent=`No completed ${requestedSplit} predictions found under the configured outputs root.`;return true;}catch(e){if(state.split===requestedSplit)error(e.message);return false;}}
$('split').onchange=async()=>{queryGeneration++;clearTrack();$('track-table').tBodies[0].replaceChildren();$('track-count').textContent='';$('page').textContent='';state.split=$('split').value;state.runs='';state.axis='';state.value='0';state.locator='';state.page=0;modelSelectors();sortOptions();const requestedSplit=state.split;if(await refresh()&&state.split===requestedSplit)await query();};
['dataset','metric','focus','axis','value','scope','sort','descending','minimum'].forEach(id=>$(id).onchange=async()=>{
  pause();state[id]=$(id).value;state.page=0;
  if(id==='dataset'){state.runs='';state.axis='';state.value='0';state.locator='';modelSelectors();sortOptions();}
  if(id==='metric'){state.descending=state.metric==='ce'||state.metric==='confusion_rate'?'true':'false';$('descending').value=state.descending;}
  if(id==='focus'){state.metric=state.focus?'confusion_rate':'accuracy';state.descending=state.focus?'true':'false';$('metric').value=state.metric;$('descending').value=state.descending;}
  if(id==='axis'){state.value='0';options($('value'),state.axis?result.axes[state.axis].map((v,i)=>[String(i),v]):[['0','All']],'0');state.scope='matching';$('scope').value=state.scope;}
  if(id==='value')state.value=$('value').value;
  await query();
});
$('run-search').oninput=modelSelectors;
$('apply').onclick=()=>{state.search=$('search').value;state.minimum=$('minimum').value;state.page=0;query();};
$('search').onkeydown=e=>{if(e.key==='Enter')$('apply').click();};
$('refresh').onclick=async()=>{const requestedSplit=state.split;if(await refresh()&&state.split===requestedSplit&&selectedRuns().length)await query();};
$('previous-page').onclick=()=>{state.page--;query();};$('next-page').onclick=()=>{state.page++;query();};
$('frame').oninput=()=>{pause();setFrame(+$('frame').value);};
$('step-back').onclick=()=>{pause();setFrame(state.frame-1);};$('step-next').onclick=()=>{pause();setFrame(state.frame+1);};
$('previous-error').onclick=()=>navigateError(-1);$('next-error').onclick=()=>navigateError(1);
$('road-overlay').onchange=renderFrame;
$('tight-crop').onchange=renderFrame;
$('timeline').onclick=e=>{if(!track)return;pause();const rect=e.target.getBoundingClientRect(),x=(e.clientX-rect.left)*e.target.width/rect.width,y=(e.clientY-rect.top)*e.target.height/rect.height,row=Math.floor((y-24)/34);if(row>=1&&row<=track.models.length)state.active=row-1;setFrame(Math.floor((x-170)/(e.target.width-180)*track.targets.length));};
function tick(){if(!track)return;if(state.frame>=track.targets.length-1){pause();return;}setFrame(state.frame+1);timer=setTimeout(tick,200/+$('speed').value);}
$('play').onclick=()=>{if(timer!==null){pause();return;}if(state.frame===track.targets.length-1)setFrame(0);$('play').textContent='Pause';timer=setTimeout(tick,200/+$('speed').value);};
$('save').onclick=async()=>{try{const record=await api('/api/reviews',{}, {state:{...state},hashes:track.models.map(m=>m.sha256),reason:$('reason').value,note:$('note').value});$('saved-status').textContent='Saved '+record.saved_utc;await reviews();}catch(e){error(e.message);}};
$('reload-reviews').onclick=reviews;
window.addEventListener('hashchange',async()=>{try{queryGeneration++;clearTrack();const restored=JSON.parse(decodeURIComponent(location.hash.slice(1)));Object.assign(state,restored);state.split=restored.split||'test';syncControls();if(!await refresh())return;await query();}catch(e){error('Could not restore viewer link: '+e.message);}});
(async()=>{try{if(location.hash){const restored=JSON.parse(decodeURIComponent(location.hash.slice(1)));Object.assign(state,restored);state.split=restored.split||'test';}syncControls();if(!await refresh())return;await reviews();if(selectedRuns().length)await query();}catch(e){error('Could not restore viewer state: '+e.message);}})();
