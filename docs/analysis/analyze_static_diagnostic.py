"""离线验证九次诊断私有归档并导出不含凭证的绘图数据；不运行数据库或LLM。"""
import csv,hashlib,json,math,re,statistics,tarfile
from pathlib import Path
import argparse,sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from admtune.static_diagnostic import plan,counter_delta,summarize
root=ROOT
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--archive',type=Path,default=ROOT/'runs/archives/static-diagnostic-001.tar.gz')
parser.add_argument('--output',type=Path,default=ROOT/'runs/collected-static-diagnostic-001')
args=parser.parse_args()
archive=args.archive;inv=json.loads(archive.with_suffix('.inventory.json').read_text());assert hashlib.sha256(archive.read_bytes()).hexdigest()==inv['archive_sha256']
dest=args.output;dest.mkdir(mode=0o700,parents=True,exist_ok=False)
with tarfile.open(archive) as t:
 members=t.getmembers();assert {m.name for m in members}==set(inv['files']) and len(members)==len(inv['files']) and sum(m.size for m in members)<100_000_000
 for m in members:
  p=Path(m.name);assert m.isfile() and not p.is_absolute() and '..' not in p.parts and p.parts[0]=='runs' and m.size<20_000_000
  data=t.extractfile(m).read();assert hashlib.sha256(data).hexdigest()==inv['files'][m.name];f=dest/p;f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(data);f.chmod(0o600)
p=dest/'runs/static-diagnostic-001';post=json.loads((dest/'runs/static-diagnostic-001-audit/post_run.json').read_text());manifest=json.loads((p/'manifest.json').read_text());summary=json.loads((p/'summary.json').read_text())
assert summary['completed'] and summary['completed_trials']==9 and summary['initial_configuration_restored'] and summary['llm_calls']==0
assert manifest['mode']=='zero_llm_static_diagnostic' and manifest['same_cgroup'] and not manifest['git_dirty'] and not manifest['preflight']['simulated']
assert post['counts']==[[2000000,1,2000000]]*4 and post['config_sha256']==manifest['preflight']['config_sha256']
assert manifest['benchmark']['workload']['duration']==60 and manifest['benchmark']['workload']['warmup_time']==15 and manifest['benchmark']['workload']['threads']==32 and manifest['benchmark']['workload']['rand_type']=='uniform'
for name,h in json.loads((p/'checksums.json').read_text()).items():assert hashlib.sha256((p/name).read_bytes()).hexdigest()==h
records=[json.loads(l) for l in (p/'trials.jsonl').read_text().splitlines()];assert len(records)==9 and manifest['plan']==plan()
rows=[]
for r,spec in zip(records,plan()):
 assert all(r[k]==v for k,v in spec.items()) and r['valid'] and r['config']==manifest['configs'][r['name']]
 m=r['measurement'];assert m['status']=='measured' and m['complete'] and m['effective'] and m['constraints_ok'] and m['metadata']['simulated'] is False
 log=(p/r['benchmark_log']).read_text();tps=float(re.search(r'transactions:\s*(\d+)\s*\(([\d.]+) per sec',log).group(2));tx=int(re.search(r'transactions:\s*(\d+)',log).group(1));lat=float(re.search(r'95th percentile:\s*([\d.]+)',log).group(1));duration=float(re.search(r'total time:\s*([\d.]+)s',log).group(1));errors=int(re.search(r'ignored errors:\s*(\d+)',log).group(1));avg=float(re.search(r'avg:\s*([\d.]+)',log).group(1))
 assert tps==m['value'] and lat==m['metadata']['p95_latency_ms'] and duration>=59.9 and errors==0
 d=counter_delta(r['telemetry']['before'],r['telemetry']['after']);assert d==r['telemetry']['delta'] and 59.9<=d['boundary_elapsed_seconds']<=70
 assert all(c['status_code']==0 and c['configured_bytes']==c['expected_bytes'] for c in r['resize_checks'])
 assert r['resize_checks'][-1]['expected_bytes']==1024**3
 before=r['telemetry']['before']['cgroup'];assert before['quota_us']/before['period_us']==12
 rows.append({**spec,'tps':tps,'p95_ms':lat,'mean_latency_ms':avg,'tps_per_p95':tps/lat,'transactions':tx,'duration_seconds':duration,'boundary_seconds':d['boundary_elapsed_seconds'],'mysqld_cpu_cores':d['mysqld_cpu_cores'],'cgroup_cpu_cores':d['cgroup_cpu_cores'],'throttled_period_pct':100*d['throttled_period_fraction'],'throttled_seconds':d['throttled_seconds'],'mysqld_read_mib_per_second':d['mysqld_read_bytes']/1024**2/d['boundary_elapsed_seconds'],'innodb_physical_reads':d['mysql']['Innodb_buffer_pool_reads'],'innodb_physical_reads_per_1000_transactions':d['mysql']['Innodb_buffer_pool_reads']/tx*1000,'innodb_logical_reads':d['mysql']['Innodb_buffer_pool_read_requests']})
assert summarize(records)==summary['aggregates']
aggregates=[]
fields=['tps','p95_ms','mean_latency_ms','tps_per_p95','mysqld_cpu_cores','cgroup_cpu_cores','throttled_period_pct','mysqld_read_mib_per_second','innodb_physical_reads_per_1000_transactions']
for name in ('Default','BufferOnly','ADMTuneConfig'):
 data=[r for r in rows if r['name']==name]
 aggregates.append({'name':name,'n':len(data),**{f'median_{k}':statistics.median(r[k] for r in data) for k in fields},'min_tps':min(r['tps'] for r in data),'max_tps':max(r['tps'] for r in data)})
paired=[]
for block in range(1,4):
 d={r['name']:r for r in rows if r['block']==block};a=d['ADMTuneConfig'];b=d['BufferOnly'];base=d['Default']
 paired.append({'block':block,'full_vs_buffer_tps_pct':100*(a['tps']/b['tps']-1),'full_vs_buffer_p95_pct':100*(a['p95_ms']/b['p95_ms']-1),'buffer_vs_default_tps_pct':100*(b['tps']/base['tps']-1),'full_vs_default_tps_pct':100*(a['tps']/base['tps']-1)})
out=root/'docs/experiment_data/static_diagnostic_001';out.mkdir(parents=True,exist_ok=True)
for name,data in [('observations.csv',rows),('aggregates.csv',aggregates),('paired_blocks.csv',paired)]:
 with (out/name).open('w') as f:w=csv.DictWriter(f,fieldnames=list(data[0]),lineterminator='\n');w.writeheader();w.writerows(data)
verification={'verified':True,'valid':9,'llm_calls':0,'archive_sha256':inv['archive_sha256'],'config_restored':True,'counts_restored':post['counts'],'git_revision':manifest['git_revision'],'elapsed_seconds':summary['elapsed_seconds'],'aggregates':aggregates,'paired_blocks':paired,'telemetry_scope':manifest['telemetry_scope'],'cache_protocol':manifest['cache_protocol']}
(out/'verification.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2)+'\n');(out/'checksums.json').write_text(json.dumps({f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in out.iterdir() if f.is_file() and f.name!='checksums.json'},indent=2)+'\n')
print(json.dumps(verification,ensure_ascii=False,indent=2))
