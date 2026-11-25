"""用经过原始日志核验的100分钟预算实验绘图，禁止模拟填值。"""
import csv,json,math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'docs/experiment_data/timed_static_100min_001'
OUT=ROOT/'docs/figures'
verification=json.loads((DATA/'verification.json').read_text())
assert verification['verified'] and 2<=verification['attempts']<=100
assert verification['session_seconds']<=6000
own=list(csv.DictReader((DATA/'admtune_trials.csv').open()))
literature=list(csv.DictReader((DATA/'literature_curves.csv').open()))
assert len(own)==verification['attempts']
assert all(0<float(r['active_elapsed_seconds'])<=6000 for r in own)
assert own[-1]['kind']=='verification'
colors={'ADMTune':'#1f77b4','AgenticDB':'#d62728','AgentTune':'#ff7f0e','GPTuner':'#9467bd','CDBTune':'#2ca02c','Default':'#7f7f7f'}
markers={'ADMTune':'v','AgenticDB':'o','AgentTune':'D','GPTuner':'P','CDBTune':'*','Default':None}
methods=list(colors)
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':11,'axes.labelsize':12,'axes.titlesize':12,'legend.fontsize':9,'pdf.fonttype':42,'ps.fonttype':42})

def numeric(rows,field):return [float(r[field]) for r in rows]
def points(metric,method):
 if method=='ADMTune':
  field='best_found_tps' if metric=='throughput' else 'best_found_p95_ms'
  return numeric(own,'active_elapsed_seconds'),numeric(own,field)
 rows=[r for r in literature if r['method']==method and r['metric']==metric]
 return [60*float(r['time_min']) for r in rows],numeric(rows,'value')

def style(ax,ylabel):
 ax.set_xlabel('Tuning Time (min)');ax.set_ylabel(ylabel)
 ax.set_xlim(0,100);ax.set_xticks([0,20,40,60,80,100])
 ax.grid(True,color='#dedede',lw=.55,alpha=.75);ax.set_axisbelow(True)
 ax.tick_params(direction='in',top=True,right=True)
 for spine in ax.spines.values():spine.set_linewidth(.7)

def save(fig,name):
 OUT.mkdir(exist_ok=True)
 fig.savefig(OUT/f'{name}.png',dpi=300,bbox_inches='tight')
 fig.savefig(OUT/f'{name}.pdf',bbox_inches='tight')
 plt.close(fig)

# 默认配置是本次实测初始配置，其水平线不是新增试验数据。
baseline=float(own[0]['tps']);base_lat=float(own[0]['p95_ms'])
fig,ax=plt.subplots(figsize=(7.2,4.35))
combined=[]
for method in methods:
 if method=='Default':
  x=[0,100];y=[baseline,baseline]
  ax.plot(x,y,color=colors[method],ls='--',lw=1.1,label=method)
  for t,v in zip(x,y):combined.append({'method':method,'time_min':t,'throughput_tps':v,'series_kind':'reference_span','source':'current_run_initial_config','trial':''})
 else:
  sec,y=points('throughput',method);x=[s/60 for s in sec]
  ax.plot(x,y,label=method,color=colors[method],marker=markers[method],ms=4.5 if method=='ADMTune' else 3.4,lw=1.6 if method=='ADMTune' else 1.05,markevery=max(1,len(x)//9),zorder=5 if method=='ADMTune' else 3)
  for i,(t,v) in enumerate(zip(x,y)):
   combined.append({'method':method,'time_min':t,'throughput_tps':v,'series_kind':'best_found_observed' if method=='ADMTune' else 'published_vector_vertex','source':'timed-static-100min-001' if method=='ADMTune' else 'AgenticDB_Fig5_MS_Sysbench_Read','trial':own[i]['trial'] if method=='ADMTune' else ''})
style(ax,'Throughput (TPS)');ax.set_ylim(0,max(r['throughput_tps'] for r in combined)*1.1)
ax.set_title('Sysbench Read-only')
fig.legend(*ax.get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,1.015),ncol=3,frameon=False,columnspacing=1.25,handlelength=2)
fig.subplots_adjust(top=.79,bottom=.15,left=.12,right=.98)
save(fig,'admtune_100min_comparison')
with (DATA/'comparison_curve.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(combined[0]));w.writeheader();w.writerows(combined)

# 独立展示同一容器、同一协议的真实调优，末端星号为独立复测。
fig,axes=plt.subplots(1,2,figsize=(10,3.9))
for ax,metric,field,ylabel,base in zip(axes,['throughput','latency'],['tps','p95_ms'],['Throughput (TPS)','Latency (ms)'],[baseline,base_lat]):
 sec,best=points(metric,'ADMTune');x=[s/60 for s in sec]
 ax.plot(x,best,color=colors['ADMTune'],lw=1.5,marker='v',ms=3.8,markevery=max(1,len(x)//10),label='ADMTune')
 ax.axhline(base,color=colors['Default'],ls='--',lw=1,label='Default')
 refs=[r for r in own if r['kind']=='reference' and r['valid']=='True']
 ax.scatter([float(r['active_elapsed_seconds'])/60 for r in refs],numeric(refs,field),color='#999999',marker='x',s=20,label='Default recheck',zorder=6)
 last=own[-1];ax.scatter([float(last['active_elapsed_seconds'])/60],[float(last[field])],color='#d62728',marker='*',s=90,zorder=7,label='Final verification')
 style(ax,ylabel);ax.set_ylim(0,max(best+[float(last[field]),base]+numeric(refs,field))*1.15)
axes[0].set_title('(a) Throughput');axes[1].set_title('(b) Latency')
fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,1.03),ncol=4,frameon=False)
fig.subplots_adjust(top=.79,bottom=.18,wspace=.28)
save(fig,'admtune_100min_measured')
print(json.dumps({'figures':['admtune_100min_comparison','admtune_100min_measured'],'attempts':len(own),'baseline_tps':baseline,'final_verification_tps':float(own[-1]['tps'])},indent=2))
