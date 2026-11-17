"""Prepare self-contained HTML views from the retained observation and fit contracts."""

import math

HTML_REVISION = 1
PAGE_PATHS = ("overview/overview.html", "retraining-timeline.html", "retraining-explorer.html")
AXES = {
    "sequence": "Evento (sequência do fluxo)",
    "epoch": "Época",
    "candidate": "Candidato (ordem do ajuste)",
    "time": "Timestamp (ms desde Unix epoch)",
}
FRACTIONS = {
    "accuracy", "macro_f1", "positive_precision", "positive_recall", "average_precision",
    "recall_at_fpr_0_01", "precision", "recall", "f1", "absolute_retention", "gem_bwt", "post_learning_fm",
}
INITIAL = {"initial", "initial_fit", "warmup"}


def _coordinate(value):
    return type(value) in (int, float) and math.isfinite(value)


def _unit(metric, scope):
    if metric in FRACTIONS:
        return "proporção"
    if metric in {"mae", "rmse"}:
        return scope.get("target_unit") or "unidade do alvo"
    if metric.endswith("_seconds"):
        return "s"
    if metric.endswith("_bytes"):
        return "bytes"
    if metric.endswith("per_second"):
        return "predições/s"
    if metric == "release_delay_mean":
        return scope.get("release_delay_unit") or "unidade do relógio da fonte"
    if metric == "cost_per_1000_evaluated":
        return f"{scope.get('cost_currency') or 'unidade de custo'} / 1.000 exemplos"
    if metric.startswith("cost_"):
        return scope.get("cost_currency") or "unidade de custo do protocolo"
    if metric.endswith("_examples") or metric.endswith("_samples"):
        return "exemplos"
    if metric.endswith("_presentations"):
        return "apresentações"
    if metric in {"updates", "steps", "epochs", "alarm_count", "promoted_versions", "fit_count"}:
        return "contagem"
    return "unidade do protocolo"


def _metrics(scopes, position=None):
    for split, values in scopes.items():
        for key, value in (values or {}).items():
            reason = (values.get("availability") or {}).get(key) or {}
            if _coordinate(value) or value is None:
                # Scalar availability belongs in the table. Array/metadata keys belong in the sources.
                if key in {"confusion_matrix", "classes", "class_support", "class_recall", "pr_curve",
                           "roc_curve", "residuals", "ranking_method", "target_unit", "positive_class"}:
                    continue
                yield [split, key, value, _unit(key, values), position, reason.get("reason")]


def page_data(report, fits, chart_data, window_panels, fit_series):
    observations = report["tracking_observations"]
    windows = observations.get("windows") or []
    lineage = observations.get("lineage") or {}
    parent_scopes = {**(observations.get("scopes") or {}),
                     **{key: observations.get(key) for key in ("effort", "resources", "drift", "retention")}}
    base = {key: lineage.get(key) or report.get(key) for key in ("dataset", "policy", "protocol")}
    overview = list(_metrics(parent_scopes))
    for window in windows:
        coordinate = AXES.get(window.get("coordinate") or "sequence", "Coordenada do protocolo")
        overview.extend(_metrics({window["scope"]: window["quality"]}, f"{coordinate} · {window['step']}"))

    candidates = {row.get("fit_id"): row["step"] for row in windows if row.get("coordinate") == "candidate"}
    records = []
    for index, fit in enumerate(fits, 1):
        fit_id = fit["fit_id"]
        coordinate = fit.get("promotion_step")
        if not _coordinate(coordinate):
            coordinate = fit.get("end_seq")
        axis = "sequence"
        if fit_id in candidates:
            axis, coordinate = "candidate", candidates[fit_id]
        elif not _coordinate(coordinate) and _coordinate(fit.get("end_time_ms")):
            axis, coordinate = "time", fit["end_time_ms"]
        elif not _coordinate(coordinate) and fit.get("observation_epochs") and len(fits) == 1:
            axis, coordinate = "epoch", fit["observation_epochs"][-1]["epoch"]
        if not _coordinate(coordinate):
            axis, coordinate = None, None
        effort = fit.get("observation_effort") or {}
        resources = fit.get("observation_resources") or {}
        scopes = {**(fit.get("observation_scopes") or {}), "esforço": effort, "recursos": resources}
        records.append({
            "id": fit_id, "index": index,
            "initial": fit.get("action") in INITIAL or fit.get("deployment_action") in INITIAL,
            "action": fit.get("action"), "status": fit.get("status"),
            "promotion": fit.get("promotion_status"), "axis": axis, "x": coordinate,
            "start": fit.get("start_seq"), "end": fit.get("end_seq"), "time": fit.get("end_time_ms"),
            "checkpoint": fit.get("checkpoint"), "version": fit.get("deployed_version"),
            "duration": resources.get("duration_seconds", fit.get("train_seconds")),
            "volume": effort.get("training_presentations", fit.get("training_presentations")),
            "metrics": list(_metrics(scopes)),
        })

    timeline = {}
    for axis in AXES:
        compatible = [row for row in windows if (row.get("coordinate") or "sequence") == axis]
        timeline[axis] = chart_data(window_panels({**observations, "windows": compatible}, fits))
    drift = observations.get("drift") or {}
    events = []
    for key, label, color in (("alarms", "Alarme", "#aa2335"), ("changes", "Mudança de conceito", "#5f40a1")):
        for event in drift.get(key) or []:
            x = event.get("seq") if isinstance(event, dict) else event
            if _coordinate(x):
                events.append({"x": x, "label": label, "color": color})

    return {
        "overview": {"kind": "overview", "meta": base, "metrics": overview},
        "retraining-timeline": {
            "kind": "timeline", "meta": base, "axes": AXES, "events": events, "timeline": timeline,
            "fits": [{key: value for key, value in row.items() if key != "metrics"} for row in records],
        },
        "retraining-explorer": {
            "kind": "explorer", "meta": base, "axes": AXES,
            "fits": [{**record, "series": {
                axis: chart_data([panel for panel in fit_series(fit, axis) if panel.series])
                for axis in ("step", "epoch")
            }} for record, fit in zip(records, fits, strict=True)],
        },
    }


PAGE_STYLE = """
header, #controls {margin-bottom:1rem}
#controls {display:flex;flex-wrap:wrap;align-items:end;gap:.65rem}
#controls label {display:grid;gap:.25rem;margin:0}
#controls input, #controls select, button {
  font:inherit;padding:.35rem;border:1px solid #9eafc3;border-radius:4px
}
button {background:#eef5fd;color:#123054;cursor:pointer}
button:focus-visible, input:focus-visible, select:focus-visible, [tabindex]:focus-visible {
  outline:3px solid #0098b6;outline-offset:2px
}
dl {display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,12rem),1fr));gap:.6rem;margin:.4rem 0}
dl div {background:#eef5fd;padding:.45rem;overflow-wrap:anywhere}
dt {font-weight:600} dd {margin:0}
.wide {grid-column:1/-1;min-width:0}
.scroll {overflow:auto;max-height:30rem}
table {width:100%;border-collapse:collapse;text-align:left;font-variant-numeric:tabular-nums}
th, td {border-bottom:1px solid #dce4ef;padding:.42rem;vertical-align:top;overflow-wrap:anywhere}
th {background:#eef5fd;position:sticky;top:0}
[data-fit-marker] {cursor:pointer}
"""

PAGE_SCRIPT = """
const page = JSON.parse(document.querySelector('#page-data').textContent);
const overviewPanels = JSON.parse(document.querySelector('#chart-data').textContent);
const lineage = JSON.parse(document.querySelector('#run-lineage').textContent);
const main = document.querySelector('main');
const controls = document.querySelector('#controls');
const names = {dataset:'Dataset', policy:'Política', protocol:'Protocolo', complete:'Concluído',
  interrupted:'Interrompido', partial:'Parcial', promoted:'Promovido', rejected:'Rejeitado',
  candidate:'Candidato', pending:'Pendente', initial:'Treino inicial', initial_fit:'Treino inicial',
  reset_retrain:'Retreinamento', continue:'Continuação do treino', learn_one:'Aprendizado online',
  online:'Atualização online', failed:'Falhou'};
const display = v => v === null || v === undefined || v === '' ? 'Indisponível' :
  typeof v === 'number' ? v.toLocaleString('pt-BR', {maximumSignificantDigits:12}) :
  typeof v === 'object' ? (v.name || v.family || JSON.stringify(v)) : (names[v] || String(v));
function facts(parent, values) {
  const list = node('dl',parent);
  values.forEach(([title,value]) => {
    const item = node('div',list); node('dt',item,title); node('dd',item,display(value));
  });
}
facts(document.querySelector('header'), Object.entries(page.meta || {}).map(([k,v]) => [names[k],v]));
function section(title, target=main) {
  const box = node('section',target); box.className = 'wide'; node('h2',box,title); return box;
}
function field(parent,label,tag,attrs={}) {
  const wrap=node('label',parent,label); const input=node(tag,wrap);
  Object.entries(attrs).forEach(([k,v]) => input.setAttribute(k,v)); return input;
}
function action(parent,label,tap) {
  const button=node('button',parent,label); button.type='button'; button.addEventListener('click',tap);
  return button;
}
function table(parent, headers, rows, searchable=false) {
  const search = searchable ? field(parent,'Buscar resultados','input',{type:'search'}) : null;
  const scroll=node('div',parent); scroll.className='scroll'; scroll.tabIndex=0;
  const grid=node('table',scroll), head=node('tr',node('thead',grid)), body=node('tbody',grid);
  const collator=new Intl.Collator('pt-BR',{numeric:true,sensitivity:'base'});
  const buttons=[];
  let sort=-1, direction=1, offset=0;
  const pageSize=100;
  let previous, next, pagination;
  function draw() {
    let items=rows.filter(row => !search || row.some(value =>
      display(value).toLocaleLowerCase('pt-BR').includes(search.value.trim().toLocaleLowerCase('pt-BR'))));
    if (sort>=0) items=items.sort((a,b) => {
      const x=a[sort], y=b[sort];
      if (x==null || y==null) return x==null && y==null ? 0 : x==null ? 1 : -1;
      return direction*(typeof x==='number' && typeof y==='number' ? x-y :
        collator.compare(display(x),display(y)));
    });
    body.replaceChildren();
    if (!items.length) {
      const message=rows.length ? 'Nenhum resultado para a busca.' : 'Nenhum registro.';
      const cell=node('td',node('tr',body),message);
      cell.colSpan=headers.length;
    }
    if (searchable) {
      previous.disabled=offset===0;
      next.disabled=offset+pageSize>=items.length;
      pagination.textContent=items.length ?
        (offset+1)+'–'+Math.min(items.length,offset+pageSize)+' de '+items.length : '0 resultados';
      items=items.slice(offset,offset+pageSize);
    }
    items.forEach(row=>{const tr=node('tr',body); row.forEach(value=>node('td',tr,display(value)));});
  }
  headers.forEach((name,index) => {
    const th=node('th',head); th.scope='col';
    if (!searchable) { th.textContent=name; return; }
    buttons.push(th);
    action(th,name,() => {
      direction=sort===index ? -direction : 1; sort=index; offset=0;
      buttons.forEach(cell=>cell.removeAttribute('aria-sort'));
      th.setAttribute('aria-sort',direction===1 ? 'ascending' : 'descending'); draw();
    });
  });
  if (searchable) {
    const footer=node('div',parent);
    previous=action(footer,'Página anterior',()=>{offset-=pageSize;draw();});
    next=action(footer,'Próxima página',()=>{offset+=pageSize;draw();});
    pagination=node('output',footer);
    pagination.setAttribute('aria-live','polite');
    search.addEventListener('input',()=>{offset=0;draw();});
  }
  draw();
}
const metricHeaders=['Split ou grupo','Métrica','Valor','Unidade','Posição','Motivo de indisponibilidade'];
function describe(box,fit) {
  const type=fit.initial?'Treino inicial':fit.action;
  facts(box,[['Ajuste',fit.id],['Posição no histórico',fit.index],['Tipo',type],
    ['Status',fit.status],['Promoção',fit.promotion],['Posição no fluxo',fit.axis ?
      (page.axes[fit.axis]+' · '+display(fit.x)) : null],
    ['Intervalo de treino',fit.start == null && fit.end == null ? null :
      display(fit.start)+' → '+display(fit.end)],
    ['Fim registrado (UTC)',fit.time==null ? null : new Date(fit.time).toISOString()],
    ['Versão promovida',fit.version],['Run filha (ID)',lineage.children?.[fit.id]],
    ['Checkpoint',fit.checkpoint?.file],['SHA-256 do checkpoint',fit.checkpoint?.sha256],
    ['Duração (s)',fit.duration],['Apresentações ao modelo',fit.volume]]);
}
function explore() {
  const fits=page.fits, details=section('Ajuste selecionado'), plots=node('div',main);
  plots.className='wide';
  const bounds={min:1,max:Math.max(1,fits.length),step:1};
  const slider=field(controls,'Posição no histórico','input',{type:'range',...bounds});
  const direct=field(controls,'Ir para a posição','input',{type:'number',...bounds});
  const state=node('output',controls); state.setAttribute('aria-live','polite');
  slider.disabled=direct.disabled=fits.length<2;
  function show(index) {
    details.replaceChildren(); plots.replaceChildren();
    if (!fits.length) { state.textContent='Nenhum ajuste registrado.'; return; }
    const fit=fits[index-1]; slider.value=direct.value=index;
    state.textContent=index+' de '+fits.length+' · '+fit.id;
    describe(details,fit);
    table(details,metricHeaders,fit.metrics);
    Object.entries(fit.series).forEach(([axis,panels]) => {
      if (panels.length) renderPanels(panels,plots);
      else node('p',plots,'Série por '+(axis==='step'?'step':'época')+' indisponível neste ajuste.');
    });
  }
  slider.addEventListener('input',()=>show(Number(slider.value)));
  const select=()=>{if(direct.value && direct.validity.valid) show(Number(direct.value));};
  direct.addEventListener('input',select);
  direct.addEventListener('change',()=>show(Math.max(1,Math.min(fits.length,Math.round(Number(direct.value)||1)))));
  show(1);
}
function timeline() {
  const fits=page.fits;
  const status=field(controls,'Status ou promoção','select');
  const statuses=[...new Set(fits.flatMap(f=>[f.status,f.promotion]).filter(Boolean))];
  [['','Todos'],...statuses.map(v=>[v,display(v)])]
    .forEach(([value,label])=>{const option=node('option',status,label);option.value=value;});
  const axis=field(controls,'Eixo do intervalo','select');
  [['','Todos os eixos'],...Object.entries(page.axes)].forEach(([value,label])=>{
    const option=node('option',axis,label);option.value=value;
  });
  const lo=field(controls,'De','input',{type:'number',step:'any'});
  const hi=field(controls,'Até','input',{type:'number',step:'any'});
  const selection=section('Detalhes do ajuste');
  node('p',selection,'Selecione um marcador ou um ajuste na lista.');
  const charts=node('div',main); charts.className='wide';
  const records=section('Ajustes no recorte');
  const select=index=>{
    selection.replaceChildren();
    node('h2',selection,'Detalhes do ajuste');
    describe(selection,fits[index-1]);
  };
  function limits(values) {
    return values.map(input=>input.value==='' ? null : Number(input.value));
  }
  function draw() {
    charts.replaceChildren(); records.replaceChildren();
    selection.replaceChildren(); node('h2',selection,'Detalhes do ajuste');
    node('p',selection,'Selecione um marcador ou um ajuste na lista.');
    const [a,b]=limits([lo,hi]);
    const between=x=>Number.isFinite(x)&&(a==null||x>=a)&&(b==null||x<=b);
    const rows=fits.filter(f=>(!status.value||[f.status,f.promotion].includes(status.value)) &&
      (!axis.value||f.axis===axis.value) && (a==null&&b==null||between(f.x)));
    node('h2',records,'Ajustes no recorte · '+rows.length+' de '+fits.length);
    if (!rows.length) {
      node('p',records,fits.length?'Nenhum ajuste neste recorte.':'Nenhum ajuste registrado.');
    }
    const list=node('ol',records);
    rows.forEach(f=>{
      const item=node('li',list);
      action(item, (f.initial?'Treino inicial · ':'')+f.id,()=>select(f.index));
      node('span',item,' · '+display(f.status)+' · '+display(f.promotion)+' · '+
        (f.axis?page.axes[f.axis]+' · '+display(f.x):'Posição indisponível'));
    });
    table(records,['Ordem','Ajuste','Tipo','Status','Promoção','Eixo','Posição','Run filha'],
      rows.map(f=>[f.index,f.id,f.initial?'Treino inicial':f.action,f.status,f.promotion,
        page.axes[f.axis],f.x,lineage.children?.[f.id]]));
    Object.entries(page.axes).forEach(([kind,label])=>{
      if (axis.value && kind!==axis.value) return;
      const markers=rows.filter(f=>f.axis===kind).map(f=>({
        x:f.x,fitIndex:f.index,label:(f.initial?'Treino inicial · ':'')+f.id+' · '+display(f.promotion),
        color:f.promotion==='rejected'?'#a33224':f.initial?'#00568e':'#007a62'
      })).concat(kind==='sequence'?page.events.filter(e=>between(e.x)):[]);
      const panels=page.timeline[kind];
      if (!panels.length && !markers.length) return;
      const metricPositions=panels.flatMap(p=>p.series.flatMap(s=>s.points.map(pt=>pt.x)));
      const xValues=markers.map(m=>m.x).concat(metricPositions);
      const [left,right]=extent(xValues);
      const min=a??left, max=b??right;
      if (!(min<=max) || ![min,max].every(Number.isFinite)) return;
      const xRange=min===max ? [min-1,max+1] : [min,max];
      renderPanels((panels.length?panels:[{title:'Ajustes e eventos',series:[],yLim:[0,1],yLabel:'Eventos'}])
        .map(p=>({...p,xLabel:label,integerX:true,xRange,markers,selectFit:select,
          series:p.series.map(s=>({...s,points:s.points.filter(pt=>between(pt.x))}))})),charts);
    });
  }
  function axisChange() {
    lo.disabled=hi.disabled=!axis.value;
    lo.value=hi.value=''; draw();
  }
  axis.addEventListener('change',axisChange); status.addEventListener('change',draw);
  [lo,hi].forEach(input=>input.addEventListener('change',draw));
  action(controls,'Ampliar ×2',()=>{
    if (!axis.value) {
      const preferred=fits.find(f=>f.axis)?.axis;
      if(!preferred) return;
      axis.value=preferred; axisChange();
    }
    const values=fits.filter(f=>f.axis===axis.value).map(f=>f.x);
    if (!values.length) return;
    const [a,b]=extent(values), [low, high]=limits([lo,hi]);
    const start=low??a,end=high??b, quarter=(end-start)/4;
    lo.value=start+quarter; hi.value=end-quarter; draw();
  });
  action(controls,'Restaurar intervalo',()=>{lo.value=hi.value='';draw();});
  axisChange();
}
if(page.kind==='explorer') explore();
else if(page.kind==='timeline') timeline();
else {
  if(page.metrics) table(section('Resultados da execução'),metricHeaders,page.metrics,true);
  renderPanels(overviewPanels,main);
}
"""
