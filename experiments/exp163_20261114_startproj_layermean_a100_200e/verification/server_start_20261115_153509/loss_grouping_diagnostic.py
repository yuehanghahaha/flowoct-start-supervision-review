import json
from pathlib import Path
import numpy as np
import torch
import common as c
import checks
import method
torch.set_num_threads(1)
cfg=c.read(c.ROOT/'experiment.json')
torch.manual_seed(11)
net=method.make_model(dict(base_channels=4))
sample=dict(source=torch.rand(1,1,8,8,8),target=torch.rand(1,1,8,8,8),layers=checks.layers_for(8,8),time=.37)
method.loss_terms(net,sample,cfg)
sample['time']=0.
total,terms,_=method.loss_terms(net,sample,cfg)
expected=terms['fm']+sum(cfg['loss_weights'][key]*terms[key] for key in method.PROJECTION_KEYS)
weighted=[cfg['loss_weights'][key]*value for key,value in terms.items()]
oracle=sum(float(value.detach()) for value in weighted)
epsilon=torch.finfo(torch.float32).eps
gamma=(len(weighted)-1)*epsilon/(1-(len(weighted)-1)*epsilon)
bound=gamma*sum(abs(float(value.detach())) for value in weighted)
params=tuple(net.parameters())
left=torch.autograd.grad(total,params,retain_graph=True)
right=torch.autograd.grad(expected,params,retain_graph=True)
diff=torch.cat([(a-b).reshape(-1) for a,b in zip(left,right)])
reference=torch.cat([b.reshape(-1) for b in right])
result=dict(total=float(total.detach()),regrouped=float(expected.detach()),absolute_difference=abs(float(total.detach())-float(expected.detach())),ulp_difference=abs(int(np.float32(float(total.detach())).view(np.int32))-int(np.float32(float(expected.detach())).view(np.int32))),weighted_terms={key:float((cfg['loss_weights'][key]*value).detach()) for key,value in terms.items()},float64_sum_of_float32_weighted_terms=oracle,sum_rounding_bound=bound,total_oracle_error=abs(float(total.detach())-oracle),regrouped_oracle_error=abs(float(expected.detach())-oracle),gradient_max_abs_difference=float(diff.abs().max()),gradient_relative_l2=float(torch.linalg.vector_norm(diff)/torch.linalg.vector_norm(reference)),gradients_bitwise_equal=all(torch.equal(a,b) for a,b in zip(left,right)),source_contract=c.source_contract(),scientific_source_modified=False)
assert result['total_oracle_error']<=bound and result['regrouped_oracle_error']<=bound
assert result['gradient_relative_l2']<=1e-6
c.write(Path(__file__).with_suffix('.json'),result)
print(json.dumps({key:value for key,value in result.items() if key!='source_contract'},indent=2))
