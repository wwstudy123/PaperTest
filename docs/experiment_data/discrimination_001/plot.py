import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
rows=list(csv.DictReader(Path('docs/experiment_data/discrimination_001/metrics.csv').open()))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42})
fig,axes=plt.subplots(1,2,figsize=(10,3.9))
for dist,color,marker in [('special','#ff7f0e','D'),('uniform','#1f77b4','v')]:
 data=[r for r in rows if r['distribution']==dist]
 for ax,metric,title in zip(axes,['tps','p95_ms'],['Throughput (TPS)','P95 latency (ms)']):
  ax.plot(range(1,len(data)+1),[float(r[metric]) for r in data],color=color,marker=marker,lw=1.1,ms=4,label=dist)
  ax.set_title(title);ax.set_xlabel('Position in predeclared B-C-B[-C-B] block');ax.set_ylim(0,max(float(r[metric]) for r in rows)*1.12);ax.grid(alpha=.25);ax.set_xticks([1,2,3,4,5])
axes[0].set_ylabel('TPS');axes[1].set_ylabel('P95 (ms)')
fig.suptitle('Manual workload discrimination calibration (not ADMTune tuning)',y=.99,fontsize=13)
fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.91),ncol=2,frameon=False)
fig.text(.5,.025,'B: 1 GiB, C: 2 GiB buffer pool. 32 threads; 15 s warm-up + 60 s measurement. Zero LLM calls.',ha='center',fontsize=8)
fig.subplots_adjust(top=.73,bottom=.2,wspace=.25)
p=Path('docs/figures/discrimination_001');fig.savefig(p.with_suffix('.png'),dpi=240);fig.savefig(p.with_suffix('.pdf'))
