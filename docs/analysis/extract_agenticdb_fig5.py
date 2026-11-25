"""从用户提供的 AgenticDB 原文图5抽取矢量折线；不推造原始试验次数。"""
import argparse,csv,hashlib,json
from pathlib import Path
import pdfplumber
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--pdf',type=Path,required=True)
args=parser.parse_args()
root=Path(__file__).resolve().parents[2]
out=root/'docs/experiment_data/timed_static_100min_001';out.mkdir(exist_ok=True,parents=True)
pdf=args.pdf
sha=hashlib.sha256(pdf.read_bytes()).hexdigest()
assert sha=='e305e62a06d3af011945e25d8d5bfea9aca4f601b09d5a276957511aaf110696', '论文PDF版本不同，须重新校准坐标' 
with pdfplumber.open(pdf) as p:curves=p.pages[8].curves
specs={
 'throughput':dict(indices={'GPTuner':42,'AgentTune':44,'CDBTune':45,'AgenticDB':46},x0=159.26268390625003,x100=214.49356840625,y0=249.85327065455,value0=5000,dy=249.85327065455-241.13596322770002,dvalue=250,unit='TPS'),
 'latency':dict(indices={'GPTuner':47,'AgentTune':49,'CDBTune':50,'AgenticDB':51},x0=232.43320206250002,x100=291.7956475625,y0=249.32769226599999,value0=15,dy=249.32769226599999-239.40652393195,dvalue=1,unit='ms')}
meta={'source':'AgenticDB Figure 5, MS Sysbench Read, page 9','source_pdf_sha256':sha,'extraction':'original PDF vector vertices and axis ticks; no inferred evaluation counts','specifications':specs,'notes_zh':['原文基线保留原始横坐标及终止时间；不补造到100分钟的观测。','原文优化目标为TPS/P95，当前ADMTune为TPS最大化；原文AgenticDB含OS调优，当前ADMTune仅调DB参数。','原文与本次数据集、调参空间及执行容器并非同一实测协议，曲线用于来源明确的跨实验参照，不能据此宣称同条件优劣。','PDF折线坐标有路径简化及坐标舍入，不作为原始逐次试验记录。']}
rows=[]
for metric,s in specs.items():
 for method,i in s['indices'].items():
  pts=curves[i]['pts'];assert pts[0][0]==s['x0']
  for j,(x,top) in enumerate(pts):
   t=(x-s['x0'])/(s['x100']-s['x0'])*100
   value=s['value0']+(s['y0']-top)/s['dy']*s['dvalue']
   assert -1e-6<=t<=100+1e-6 and value>0
   rows.append(dict(method=method,metric=metric,vertex=j,time_min=round(t,9),value=round(value,9),unit=s['unit'],source='AgenticDB_Fig5_MS_Sysbench_Read',source_pdf_sha256=sha,page=9,curve_index=i,pdf_x=x,pdf_top=top))
with (out/'literature_curves.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(out/'literature_provenance.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:{m:next(r for r in reversed(rows) if r['method']==m and r['metric']==k)['value'] for m in s['indices']} for k,s in specs.items()},indent=2))
