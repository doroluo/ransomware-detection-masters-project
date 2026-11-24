import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import {fileURLToPath} from 'node:url';
import {gunzipSync} from 'node:zlib';
import {FileBlob,SpreadsheetFile} from '@oai/artifact-tool';
const here=path.dirname(fileURLToPath(import.meta.url));
const args=process.argv.slice(2);
function option(name,fallback){const i=args.indexOf(name);if(i<0)return fallback;assert(args[i+1]&&!args[i+1].startsWith('--'),'Missing value for '+name);return path.resolve(args[i+1]);}
if(args.includes('--help')){console.log('node export.mjs --output FINAL.xlsx [--data records.json[.gz]] [--template baseline.xlsx] [--layout baseline_layout.json] [--preview-dir DIR]');process.exit(0);}
const output=option('--output',path.join(process.env.TEMP||process.cwd(),'Final_Paper_Results_With_Hyperparameters_2026-11-24.xlsx'));
const temp=path.dirname(output);
const previewDir=option('--preview-dir',path.join(temp,'previews'));
await fs.mkdir(previewDir,{recursive:true});
const dataPath=option('--data',path.join(here,'assets/attention_records.json.gz'));
const bytes=await fs.readFile(dataPath);
const data=JSON.parse((dataPath.endsWith('.gz')?gunzipSync(bytes).toString('utf8'):bytes.toString('utf8')).replace(/^\uFEFF/,''));
const base={fold_summary:{columns:['series','representation','ablation','model','pooling','fold','completed_seeds','expected_seeds','auc_mean','auc_seed_sd','pr_auc_mean','pr_auc_seed_sd','accuracy_mean','accuracy_seed_sd','detection_rate_mean','detection_rate_seed_sd','fpr_mean','fpr_seed_sd','macro_f1_mean','macro_f1_seed_sd']},oof:{columns:['series','representation','ablation','model','pooling','seed','n','accuracy','detection_rate','macro_f1','auc','pr_auc','fpr','tp','tn','fp','fn','threshold_policy']},jobs:{columns:['job_id','series','assignment','state','attempts','adopted','started_utc','finished_utc','elapsed_job_hours','note','skip_reason']}};
const layouts=JSON.parse(await fs.readFile(option('--layout',path.join(here,'assets/baseline_layout.json')),'utf8'));
const wb=await SpreadsheetFile.importXlsx(await FileBlob.load(option('--template',path.join(here,'assets/baseline.xlsx'))));
const get=n=>wb.worksheets.getItem(n);
function col(i){let s='';for(let n=i+1;n;n=Math.floor((n-1)/26))s=String.fromCharCode(65+(n-1)%26)+s;return s;}
const clean=v=>v==null?null:typeof v==='object'?JSON.stringify(v):v;
const metrics=['accuracy','detection_rate','precision','specificity','macro_f1','auc','pr_auc','fpr'];
const initial={};for(const n of ['Held-out runs','Validation grid','Frozen recipes','OOF by seed','Per-fold summary','Five-fold summary','Job status','Failures'])initial[n]=get(n).getUsedRange().values.length-1;
async function render(name,range,file){const b=await wb.render({sheetName:name,range,scale:1.1,format:'png'});await fs.writeFile(path.join(previewDir,file+'.png'),new Uint8Array(await b.arrayBuffer()));}
await render('Summary','A1:I22','before');

function append(name,keys,entries,extras=[]){
  console.log('Appending '+name+': '+entries.length);
  const sheet=get(name),table=sheet.tables.items[0],oldCount=initial[name];
  const all=[...keys,...extras.filter(k=>!keys.includes(k))];
  table.rows.add(null,entries.map(r=>keys.map(k=>clean(r[k]??r.spec?.[k]))));
  for(let i=0;i<entries.length;i++){
    const range=sheet.getRangeByIndexes(oldCount+i+1,0,1,keys.length);
    range.copyFrom(sheet.getRangeByIndexes(oldCount,0,1,keys.length),'all');
    range.values=[keys.map(k=>clean(entries[i][k]??entries[i].spec?.[k]))];
    if(keys.includes('run_configuration'))range.format.rowHeight=84;
  }
  if(all.length>keys.length){
    const name0=table.name,style=table.style;
    for(let c=keys.length;c<all.length;c++){
      const k=all[c];sheet.getRangeByIndexes(0,c,1,1).copyFrom(sheet.getRange('A1'),'all');
      sheet.getRangeByIndexes(0,c,1,1).values=[[k.replaceAll('_',' ')]];
      sheet.getRangeByIndexes(0,c,oldCount+entries.length+1,1).format.columnWidth=23;
      sheet.getRangeByIndexes(oldCount+1,c,entries.length,1).values=entries.map(r=>[clean(r[k]??r.spec?.[k])]);
    }
    table.delete();sheet.tables.add(`A1:${col(all.length-1)}${oldCount+entries.length+1}`,true,name0).style=style;
  }
  for(const [key,numerator] of [['precision','tp'],['specificity','tn']]){
    if(!all.includes(key)||!all.includes('tp'))continue;
    const num=col(all.indexOf(numerator)),fp=col(all.indexOf('fp'));
    sheet.getRangeByIndexes(oldCount+1,all.indexOf(key),entries.length,1).formulas=entries.map((_,i)=>{
      const r=oldCount+i+2,den=`${num}${r}+${fp}${r}`;return [`=IF(${den}=0,"n.a.",${num}${r}/(${den}))`];});
  }
  layouts[name]={keys:all,rows:oldCount+entries.length};
  return sheet;
}
const extra=['attention_dropout','attention_width','gat_head_policy','study'];
append('Held-out runs',layouts['Held-out runs'].keys,data.held_out,extra);
append('Validation grid',layouts['Validation grid'].keys,data.validation,extra);
const fkeys=layouts['Frozen recipes'].keys;
const frozen=append('Frozen recipes',fkeys,data.frozen,[...extra,'effective_gat_heads','selection_family']);
const fk=layouts['Frozen recipes'].keys;
for(let i=0;i<40;i++){
  const attention=i>=20;
  frozen.getRangeByIndexes(i+1,fk.indexOf('selection_family'),1,1).values=[[attention?'Attention follow-up':'Original staged grid']];
  if(attention){const r=data.frozen[i-20];frozen.getRangeByIndexes(i+1,fk.indexOf('effective_gat_heads'),1,1).values=[[r.model===1?'Unused':JSON.stringify(Array(r.gat_layers).fill(r.gat_heads))]];}
  else{
    const v=frozen.getUsedRange().values[i+1],m=v[fk.indexOf('model')],n=v[fk.indexOf('gat_layers')],h=v[fk.indexOf('gat_heads')];
    frozen.getRangeByIndexes(i+1,fk.indexOf('effective_gat_heads'),1,1).values=[[m===1?'Unused':JSON.stringify([...Array(Math.max(0,n-1)).fill(h),1])]];
  }
}
const pkeys=[...base.fold_summary.columns,'hardware_scope','selected_recipe'];
append('Per-fold summary',pkeys,data.per_fold);
const okeys=[...base.oof.columns.filter(k=>k!=='threshold_policy'),'precision','specificity','threshold_policy','hardware_scope'];
append('OOF by seed',okeys,data.oof);
const skeys=['series','representation','ablation','model','pooling','seeds','n',...metrics.flatMap(k=>[k+'_mean',k+'_seed_sd']),'all_main_metrics_at_90',...['accuracy','detection_rate','macro_f1','auc'].flatMap(k=>[k+'_ci_lower',k+'_ci_upper']),'hardware_scope','fold_specific_recipe_ids'];
const ss=data.summaries.map(g=>({series:'attention_followup',representation:'signal500_common_api',ablation:'none',model:g.model,pooling:g.pooling,seeds:3,n:5618,hardware_scope:'RTX 5080, attention study',fold_specific_recipe_ids:[0,1,2,3,4].map(f=>`A-M${g.model}-F${f}-${g.pooling.replace('graph_mean_max','GAT').toUpperCase()}`).join('; ')}));
const five=append('Five-fold summary',skeys,ss);
for(let i=0;i<4;i++){
  const row=initial['Five-fold summary']+2+i,start=initial['OOF by seed']+2+i*3,end=start+2;
  for(const m of metrics)for(const [suffix,fn] of [['mean','AVERAGE'],['seed_sd','STDEV']]){
    const c=col(okeys.indexOf(m));five.getRange(`${col(skeys.indexOf(m+'_'+suffix))}${row}`).formulas=[[`=${fn}('OOF by seed'!${c}${start}:${c}${end})`]];
  }
  five.getRange(`${col(skeys.indexOf('all_main_metrics_at_90'))}${row}`).formulas=[[`=IF(AND(${metrics.filter(k=>k!=='fpr').map(k=>`${col(skeys.indexOf(k+'_mean'))}${row}>=0.9`).join(',')}),"Yes","No")`]];
}
append('Failures',['source_record','kind','paper_included','model','seed','pooling','error','message','spec_json'],data.failures);
const jobs=[];for(let f=0;f<5;f++)for(let m=1;m<=3;m++)jobs.push({job_id:`attention-fold${f}-model${m}`,series:'attention_followup',assignment:'RTX 5080 attention study',state:'complete',finished_utc:Date.parse(data.completed_utc)/86400000+25569,note:m===1&&[2,3].includes(f)?'Selected confirmations completed; three screen candidates failed with CUDA OOM.':'Validation selection and all selected seed confirmations complete.'});
append('Job status',base.jobs.columns,jobs);
get('Job status').getRangeByIndexes(initial['Job status']+1,base.jobs.columns.indexOf('finished_utc'),15,1).setNumberFormat('yyyy-mm-dd hh:mm:ss');

const summary=get('Summary');
const baselineValues=summary.getRange('A11:H15').values,baselineFormulas=summary.getRange('A11:H15').formulas;
summary.getRange('A48:H52').copyFrom(summary.getRange('A11:H15'),'all');
summary.getRange('A48:H52').values=baselineValues;
for(let r=0;r<5;r++)for(let c=0;c<8;c++)if(baselineFormulas[r]?.[c])summary.getRangeByIndexes(47+r,c,1,1).formulas=[[baselineFormulas[r][c]]];
summary.getRange('A46').values=[['Original primary study — retained for comparison']];
summary.getRange('A46').format.font={name:'Arial',size:12,bold:true,color:'#24364B'};
summary.getRange('A2').values=[['Final campaign and attention-study results']];
summary.getRange('A4').values=[['Original local jobs complete']];
summary.getRange('D4').values=[['Attention completed 2026-11-23 23:05 Pacific.']];
summary.getRange('D6').values=[['678 original RTX + 60 attention RTX + 6 Radeon test runs.']];
summary.getRange('A6').values=[['Validation stage entries']];
const flag=col(layouts['Held-out runs'].keys.indexOf('paper_included'));
summary.getRange('B5').formulas=[[`=COUNTIFS('Held-out runs'!${flag}2:${flag}748,"Yes")`]];
summary.getRange('B6').formulas=[["=COUNTA('Validation grid'!A2:A3140)"]];
summary.getRange('A9').values=[['Completed attention study: signal500, five-fold pooled test metrics, mean across 3 seeds']];
const show=['accuracy','detection_rate','precision','specificity','macro_f1','auc','pr_auc'];
for(let i=0;i<4;i++){
  const row=initial['Five-fold summary']+2+i;
  summary.getRange(`A${12+i}`).values=[[`Model ${ss[i].model} / ${ss[i].pooling.replace('graph_mean_max','mean+max')}`]];
  summary.getRange(`B${12+i}:H${12+i}`).formulas=[show.map(k=>`='Five-fold summary'!${col(skeys.indexOf(k+'_mean'))}${row}`)];
}
summary.getRange('A17').values=[['Target: every displayed metric ≥90%; no attention-study model meets all seven metrics.']];
summary.getRange('A18').values=[['Attention combinations meeting all seven ≥90%']];
summary.getRange('G18').formulas=[[`=COUNTIFS('Five-fold summary'!A2:A33,"attention_followup",'Five-fold summary'!${col(skeys.indexOf('all_main_metrics_at_90'))}2:${col(skeys.indexOf('all_main_metrics_at_90'))}33,"Yes")`]];
summary.getRange('A20').values=[['594 successful attention screens + 6 OOM failures; all 60 selected held-out confirmations completed.']];
summary.getRange('A21').values=[['Original primary results are retained below the chart; series and hardware labels distinguish every study.']];
summary.charts.items[0].title='Attention follow-up: five-fold held-out metrics';

const guide=get('Hyperparameter guide');
const lines={
  5:'Held-out runs shows each exact combination. Frozen recipes lists 20 original + 20 attention-study recipes.',
  7:'Final export: original 5080 campaign + completed attention study + six validated Radeon returns; each remains labeled.',
  38:'Validation screens: up to 12 epochs. Original patience 13; attention-study patience 8. These are tuning scores.',
  42:'Original recipe IDs start M; attention-study IDs start A-M. Each identifies model, fold and pooling; seed is separate.',
  44:'Original GAT output layer uses one head. Attention-study GAT uses the chosen head count in every layer.',
  51:'Completed attention follow-up — successful screens and held-out confirmations are included',
  53:'Heads tested: 2, 4, 8, 12. Independent attention dropout: 0.0, 0.1, 0.2. Transformer width fixed at 96 (Models 1/3).',
  54:'Model 3 crosses Transformer heads × GAT heads × attention dropout × mean/CLS pooling; Model 1/2 use their active heads.',
  55:'All GAT layers use the selected head count in this follow-up, including the output layer. Actual heads are recorded.',
  56:'Feature dropout stays at each original fold’s frozen value. Attention dropout separately drops attention probabilities.',
  57:'Width normalization and GAT head policy also changed; differences from the original study cannot be attributed only to heads.',
  58:'Coverage: 594 successful screens, 6 CUDA-OOM screen failures, 60 successful final confirmations; all five folds complete.',
  59:'The 90% goal is not met across all seven pooled test metrics. Report held-out scores; do not choose settings from test results.',
  61:'Units: dropout is a probability (0.1 = 10%); LR and weight decay are scalar optimizer settings; metrics display percentages.',
  63:'Learning rate controls update size; weight decay penalizes large weights; batch size is samples per optimizer update.',
  64:'Width / GAT hidden size controls feature capacity. Layers control depth. Heads learn distinct attention patterns in parallel.',
  65:'Instruction limits truncate input: Model 1 uses the first sequence_length instructions; Model 3 uses block_length per block.',
  66:'Block count controls how many CFG blocks are retained (250/500/750/1000); signal and callwalk select blocks differently.',
  67:'Pooling combines features: mean averages; CLS learns a summary token; Model 2 graph pooling concatenates mean and max.',
  68:'Seed controls randomness; fold controls the held-out family group. These are repeat/evaluation controls, not model capacity.',
  69:'Epoch is one training-data pass. Patience stops after enough non-improving validation epochs; best_epoch identifies the saved model.',
  71:'Detailed metric and threshold definitions, timing scope, source hashes and hardware provenance remain in Methods.'
};
for(const [r,s] of Object.entries(lines))guide.getRange('A'+r).values=[[s]];
guide.getRange('A61:F71').format.font={name:'Arial',size:10,color:'#182B3C'};
guide.getRange('A61:F71').format.rowHeight=24;
// Long explanatory lines span an intentionally empty row; retain readable text and existing table widths.
const methods=get('Methods'),methodRows=methods.getUsedRange().values;
for(let i=1;i<methodRows.length;i++){
  if(methodRows[i][0]==='Scope')methods.getRange(`B${i+1}`).values=[['Original RTX 5080 campaign, completed attention follow-up, and six validated Radeon signal250 Model 3 fold-3 trials. Study series and execution profiles remain separate.']];
  if(methodRows[i][0]==='Guide coverage')methods.getRange(`B${i+1}`).values=[['Original grid: 2,545 stage entries / 2,465 distinct contracts. Attention grid: 594 successful screens plus six OOM failures. Full settings and source hashes remain in raw tables; original unused defaults remain preserved.']];
}
const notes=[
 ['Attention completion',data.completed_utc+'; all 15 fold/model selections complete, all 60 held-out confirmations successful.'],
 ['Attention source',data.source],['Attention records SHA-256',data.source_sha256],
 ['Attention evaluation','12 five-fold OOF seed results recomputed from saved test prediction CSVs. Each seed covers the same 5,618 sample IDs exactly once with verified labels. Each fold uses its validation-fitted threshold.'],
 ['Attention selection','Select using validation ROC-AUC then macro-F1; Model 3 mean/CLS selected as a pair. Test scores do not select recipes. Original feature dropout and other fold-specific scientific parameters stay fixed.'],
 ['Attention comparability','Transformer width fixed at 96 for Models 1/3; chosen GAT heads applied to all GAT layers. These architectural controls differ from the original grid, so cross-study score changes are not an isolated heads/dropout effect.'],
 ['Attention uncertainty','Five-fold summary reports mean and sample standard deviation across three pooled seed scores. New attention confidence interval cells are blank because bootstrap confidence intervals were not computed.'],
 ['Final export coverage','744 paper-included held-out trials: 678 original RTX 5080, 60 attention RTX 5080, six RX 9070 XT. Three historical uncapped Model 1 trials retained as excluded. Nine failures contain no fabricated scores.'],
 ['Final export snapshot','2026-11-24 Pacific; attention study completed November 23. Experiment sources unchanged. Radeon timing remains specific to RX 9070 XT / Ryzen 7 7800X3D GPU-resident execution.']
];
methods.tables.items[0].rows.add(null,notes);
methods.getRangeByIndexes(methodRows.length,0,notes.length,2).format.rowHeight=65;
methods.getRangeByIndexes(methodRows.length,1,notes.length,1).format.wrapText=true;
wb.recalculate();
assert.equal(summary.getRange('B5').values[0][0],744);assert.equal(summary.getRange('B6').values[0][0],3139);
for(let i=0;i<4;i++)for(const [m,v] of Object.entries(data.summaries[i].metrics)){
 const row=initial['Five-fold summary']+2+i;assert(Math.abs(five.getRange(`${col(skeys.indexOf(m+'_mean'))}${row}`).values[0][0]-v.mean)<1e-10,m);
}
console.log((await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:20},maxChars:1200})).ndjson);
for(const [name,range,file] of [['Summary','A1:I22','summary'],['Hyperparameter guide','A51:F71','guide_attention'],['Held-out runs','A688:K692','attention_runs'],['Frozen recipes','A21:H24','attention_recipes']])await render(name,range,file);
const filename=output;
await (await SpreadsheetFile.exportXlsx(wb)).save(filename);
await fs.writeFile(path.join(temp,'layout.json'),JSON.stringify({layouts,initial,skeys,okeys,pkeys},null,2));
console.log(JSON.stringify({filename,paperResults:744,attentionResults:60,validationEntries:3139,recipes:40}));
