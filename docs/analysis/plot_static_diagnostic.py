"""绘制已核验的九次手工配置诊断：中位数横线与全部观测，不冒充算法对比。"""
from pathlib import Path
import csv,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[2];data=root/'docs/experiment_data/static_diagnostic_001'
v=json.loads((data/'verification.json').read_text());assert v['verified'] and v['valid']==9 and v['llm_calls']==0
rows=list(csv.DictReader((data/'observations.csv').open()));agg=list(csv.DictReader((data/'aggregates.csv').open()));assert len(rows)==9
names=['Default','BufferOnly','ADMTuneConfig'];labels=['Default','Buffer only','Full config'];colors=['#7f7f7f','#ff7f0e','#1f77b4']
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':10,'pdf.fonttype':42})
fig,axes=plt.subplots(1,3,figsize=(10,3.6))
for ax,field,title,ylabel in zip(axes,['tps','p95_ms','throttled_period_pct'],['(a) Throughput','(b) Latency','(c) CPU throttling'],['Throughput (TPS)','Latency (ms)','Throttled periods (%)']):
 for i,name in enumerate(names):
  values=[float(r[field]) for r in rows if r['name']==name];a=next(x for x in agg if x['name']==name);med=float(a['median_'+field]);ax.bar(i,med,color=colors[i],alpha=.20,width=.6);ax.hlines(med,i-.31,i+.31,color=colors[i],lw=2);ax.scatter([i-.12,i,i+.12],values,color=colors[i],s=23,zorder=4)
 ax.set_xticks([0,1,2],labels,rotation=15);ax.set_title(title);ax.set_ylabel(ylabel);ax.set_ylim(bottom=0);ax.grid(axis='y',alpha=.25,lw=.6);ax.set_axisbelow(True);ax.tick_params(direction='in',top=True,right=True)
 if field=='throttled_period_pct':ax.set_ylim(0,105)
fig.subplots_adjust(bottom=.26,top=.85,wspace=.4)
fig.text(.5,.025,'Full config: fixed ADMTune result. Points: three measurements; horizontal lines: medians. Zero LLM calls.',ha='center',fontsize=8)
out=root/'docs/figures/static_diagnostic_001';fig.savefig(out.with_suffix('.png'),dpi=280,bbox_inches='tight');fig.savefig(out.with_suffix('.pdf'),bbox_inches='tight');plt.close(fig)
print('Saved measured configuration diagnostic PNG/PDF')
