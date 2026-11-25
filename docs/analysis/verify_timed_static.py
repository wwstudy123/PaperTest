"""独立核验私有归档、真实Sysbench日志及试验前GP快照；不连接数据库或模型接口。"""
from dataclasses import asdict
import csv,hashlib,json,math,re,statistics,tarfile
from pathlib import Path
import argparse,sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from admtune.contracts import fingerprint
from admtune.space import Knob,Space
from admtune.surrogate import GaussianProcess
root=ROOT
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--archive',type=Path,default=ROOT/'runs/archives/timed-static-100min-002.tar.gz')
parser.add_argument('--output',type=Path,default=ROOT/'runs/reverified-timed-static-100min-002')
args=parser.parse_args()
archive=args.archive;inventory=json.loads(archive.with_suffix('.inventory.json').read_text())
assert hashlib.sha256(archive.read_bytes()).hexdigest()==inventory['archive_sha256']
dest=args.output;dest.mkdir(parents=True,exist_ok=False)
with tarfile.open(archive) as t:
 members=t.getmembers();assert {m.name for m in members}==set(inventory['files']) and len(members)==len(inventory['files'])
 assert sum(m.size for m in members)<200_000_000
 for m in members:
  p=Path(m.name);assert m.isfile() and not p.is_absolute() and '..' not in p.parts and p.parts[0]=='runs' and m.size<50_000_000
  b=t.extractfile(m).read();assert hashlib.sha256(b).hexdigest()==inventory['files'][m.name]
  out=dest/p;out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(b);out.chmod(0o600)
p=dest/'runs/timed-static-100min-002';audit=dest/'runs/timed-static-100min-002-audit'
def load(f):return json.loads(f.read_text())
def lines(f):return [json.loads(x) for x in f.read_text().splitlines()]
m=load(p/'manifest.json');summary=load(p/'summary.json');post=load(audit/'post_run.json');runtime=load(audit/'runtime_checkpoint.json');n=summary['completed_until']
assert not m['simulated'] and not m['git_dirty'] and m['benchmark']['workload']['rand_type']=='uniform'
assert m['benchmark']['workload']['threads']==32 and m['benchmark']['workload']['duration']==60 and m['benchmark']['workload']['warmup_time']==15
assert summary['status']=='time_budget_checkpoint' and 2<=n<=100 and summary['next_trial']==n+1 and summary['initial_configuration_restored']
assert summary['wall_budget_seconds']==6000 and summary['budget']['session_elapsed_seconds']<=6000
assert post['counts']==[[2000000,1,2000000]]*4 and post['code_sha256']==m['frozen']['code_sha256']
for name,digest in load(p/'checksums.json').items():assert hashlib.sha256((p/name).read_bytes()).hexdigest()==digest
records=lines(p/'trials.jsonl');assert len(records)==n and len({r['trial_id'] for r in records})==n
assert [r['metadata']['seq'] for r in records]==list(range(1,n+1))
attempts=lines(audit/'attempts.jsonl');assert len(attempts)==n
assert all(a['result']==r and a['trial_id']==r['trial_id'] and a['seq']==r['metadata']['seq'] for a,r in zip(attempts,records))
space=Space([Knob(**spec) for spec in m['space']]);byid={r['trial_id']:r for r in records};models={}
for f in sorted((p/'gp_snapshots').glob('*.json')):
 s=load(f)
 assert len(s['training_ids'])==len(s['x'])==len(s['target'])
 for ident,x,y in zip(s['training_ids'],s['x'],s['target']):
  r=byid[ident];assert r['status']=='measured' and r['complete'] and r['effective'] and r['constraints_ok']
  assert all(abs(a-b)<1e-12 for a,b in zip(x,space.encode(r['config'])+r['context']))
  assert math.isclose(y,math.log(r['value']),abs_tol=1e-12)
 models[str(f.relative_to(p))]=(s,GaussianProcess.from_snapshot(s))
forecasts=[];best=None;expected=[];kinds={};lastelapsed=0
for a,r in zip(attempts,records):
 meta=r['metadata'];seq=meta['seq'];kind=meta['kind'];kinds[kind]=kinds.get(kind,0)+1
 assert a['submitted']['config']==r['config'] and a['submitted'].get('prediction')==r['prediction'] and meta['simulated'] is False
 assert lastelapsed<meta['active_elapsed_seconds']<=6000;lastelapsed=meta['active_elapsed_seconds']
 valid=r['status']=='measured' and r['complete'] and r['effective'] and r['constraints_ok'] and r['value'] is not None
 if valid:
  text=(p/meta['benchmark_log']).read_text()
  tps=float(re.search(r'transactions:\s*\d+\s*\(([\d.]+) per sec',text).group(1));p95=float(re.search(r'95th percentile:\s*([\d.]+)',text).group(1))
  duration=float(re.search(r'total time:\s*([\d.]+)s',text).group(1));errors=int(re.search(r'ignored errors:\s*(\d+)',text).group(1))
  assert tps==r['value'] and p95==meta['p95_latency_ms'] and duration>=59.9 and errors==0
  checks=meta['resize_checks'];assert len(checks)>=3 and all(c['status_code']==0 and c['configured_bytes']==c['expected_bytes'] for c in checks)
  assert checks[0]['expected_bytes']==r['config']['innodb_buffer_pool_size'] and checks[-1]['expected_bytes']==1073741824
 if valid and kind in {'baseline','startup','candidate','audit'} and (best is None or r['value']>best['value']):best=r
 expected.append({'trial':seq,'kind':kind,'valid':str(bool(valid)),'tps':r['value'], 'p95_ms':meta.get('p95_latency_ms'),'best_found_tps':None if best is None else best['value']})
 if r['prediction']:
  saved,model=models[meta['model_snapshot']]
  assert all(byid[i]['metadata']['seq']<seq for i in saved['training_ids'])
  pred=asdict(model.predict([space.encode(r['config'])+r['context']])[0])
  for field in ('mean','latent_std','observation_std'):assert math.isclose(pred[field],r['prediction'][field],rel_tol=1e-6,abs_tol=1e-7)
  if valid:
   mean=math.exp(pred['mean']);error=abs(mean/r['value']-1)*100
   forecasts.append({'trial':seq,'kind':kind,'actual_tps':r['value'],'predicted_tps':mean,'absolute_percent_error':error,
                     'covered_2_observation_std':abs(math.log(r['value'])-pred['mean'])<=2*pred['observation_std']})
with (p/'curve.csv').open() as f:rows=list(csv.DictReader(f))
assert len(rows)==n
for row,e in zip(rows,expected):
 for key,value in e.items():assert row[key]==('' if value is None else str(value)),(key,row[key],value)
refs=load(p/'reflections.json');covered=set()
for ref in refs:
 assert not set(ref['evidence_ids'])-set(byid)
 for fact in ref['facts']:
  r=byid[fact['trial_id']];assert fact['status']==r['status'] and fact['value']==r['value']
 covered.update(ref['evidence_ids'])
traces=[load(f) for f in sorted((p/'llm_calls').glob('*.json'))]
ledger=runtime['ledger'];assert len(traces)==ledger['llm_calls']<=40 and ledger['llm_tokens_or_reserved']<=160000 and ledger['trial_slots_used']==n
sessions=lines(p/'llm_usage_sessions.jsonl');known_tokens=sum(s['total_tokens'] for s in sessions);trace_tokens=sum((r.get('usage') or {}).get('total_tokens',0) for r in traces)
assert known_tokens==trace_tokens and known_tokens<=ledger['llm_tokens_or_reserved']
assert summary['budget']['llm_calls']==ledger['llm_calls']
assert records[-1]['metadata']['kind']=='verification' and records[-1]['config']==runtime['state']['incumbent'] and records[-1]['value']==summary['final_verification_value']
initial=records[0]['value'];last=records[-1]['value'];reference=[r['value'] for r in records if r['metadata']['kind']=='reference' and r['value']]
groups={}
for kind in kinds:
 f=[r for r in forecasts if r['kind']==kind]
 if f:groups[kind]={'n':len(f),'mape_pct':statistics.mean(r['absolute_percent_error'] for r in f),'coverage':sum(r['covered_2_observation_std'] for r in f)/len(f)}
result={'verified':True,'archive_sha256':inventory['archive_sha256'],'attempts':n,'valid':sum(r['status']=='measured' and r['complete'] and r['effective'] and r['constraints_ok'] for r in records),
 'counts_by_kind':kinds,'unique_configs':len({fingerprint(r['config']) for r in records}),'baseline_tps':initial,'verification_tps':last,
 'best_observed_search_tps':best['value'],'accepted_search_value':summary['search_value'],'verification_vs_own_baseline_pct':(last/initial-1)*100,
 'reference_median_tps':statistics.median(reference) if reference else None,'verification_vs_interleaved_reference_pct':(last/statistics.median(reference)-1)*100 if reference else None,
 'baseline_p95_ms':records[0]['metadata'].get('p95_latency_ms'),'verification_p95_ms':records[-1]['metadata'].get('p95_latency_ms'),
 'llm_calls':ledger['llm_calls'],'known_tokens':known_tokens,'charged_tokens_or_reserved':ledger['llm_tokens_or_reserved'],
 'reflection_count':len(refs),'reflected_records':len(covered),'final_gp_training_count':len(runtime['state']['gp']['training_ids']),
 'prospective_forecast_groups':groups,'active_seconds':ledger['elapsed_seconds'],'session_seconds':summary['budget']['session_elapsed_seconds'],
 'last_trial_elapsed_seconds':lastelapsed,'incumbent':runtime['state']['incumbent'],'git_revision':m['git_revision'],'can_resume':summary['can_resume']}
analysis=dest/'analysis';analysis.mkdir();(analysis/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
with (analysis/'predictions.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(forecasts[0]) if forecasts else ['trial'],lineterminator='\n');w.writeheader();w.writerows(forecasts)
print(json.dumps(result,ensure_ascii=False,indent=2))
