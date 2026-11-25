"""绘制综合目标下的固定数据库并发配置探针，不冒充ADMTune算法搜索。"""
from pathlib import Path
import csv,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[2];data=root/'docs/experiment_data/concurrency_probe_001'
v=json.loads((data/'verification.json').read_text());assert v['verified'] and v['valid']==9 and v['llm_calls']==0 and v['evaluation_objective']=='tps_per_p95:maximize'
rows=list(csv.DictReader((data/'observations.csv').open()));agg=list(csv.DictReader((data/'aggregates.csv').open()))
names=['Default','BufferOnly','Concurrency8'];labels=['Default','Buffer only','Concurrency 8'];colors=['#7f7f7f','#ff7f0e','#1f77b4']
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':10,'pdf.fonttype':42})
fig,axes=plt.subplots(1,4,figsize=(12.5,3.7))
for ax,field,title,ylabel in zip(axes,['tps','p95_ms','tps_per_p95','throttled_period_pct'],['(a) Throughput','(b) Tail latency','(c) Joint objective','(d) CPU throttling'],['Throughput (TPS)','Latency (ms)','TPS / P95 (ms)','Throttled periods (%)']):
 for i,name in enumerate(names):
  values=[float(r[field]) for r in rows if r['name']==name];a=next(x for x in agg if x['name']==name);med=float(a['median_'+field]);ax.bar(i,med,color=colors[i],alpha=.20,width=.6);ax.hlines(med,i-.31,i+.31,color=colors[i],lw=2);ax.scatter([i-.12,i,i+.12],values,color=colors[i],s=22,zorder=4,clip_on=False)
 ax.set_xticks([0,1,2],labels,rotation=18);ax.set_title(title);ax.set_ylabel(ylabel);ax.set_ylim(bottom=0);ax.grid(axis='y',alpha=.25,lw=.6);ax.set_axisbelow(True);ax.tick_params(direction='in',top=True,right=True)
 if field=='throttled_period_pct':ax.set_ylim(0,105)
fig.subplots_adjust(bottom=.27,top=.84,wspace=.45)
fig.text(.5,.02,'Unlimited request rate; 32 clients. Fixed database configurations, not algorithm search. All three observations shown; zero LLM calls.',ha='center',fontsize=8)
p=root/'docs/figures/concurrency_probe_001';fig.savefig(p.with_suffix('.png'),dpi=280,bbox_inches='tight');fig.savefig(p.with_suffix('.pdf'),bbox_inches='tight');plt.close(fig)
print('Saved fixed-configuration concurrency probe figures')
