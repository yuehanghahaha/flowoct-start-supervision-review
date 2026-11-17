"""Depth-identifiability counterexamples, geometry, gradients and delivery checks."""
from pathlib import Path
import copy
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from microflow.geometry import slab
from microflow.subslab import relative_subslabs,subslab_l1
from microflow.losses import patient_objective
from microflow.data import structure_maps
from microflow.metrics import depth_diagnostics


def layers_for(depth,batch=1,height=1,width=1):
    return torch.linspace(1,depth,6)[None,:,None,None].expand(batch,6,height,width).clone()


class SubslabTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(4)

    def test_cross_bin_error_hidden_by_full_mip(self):
        truth=torch.zeros(1,1,16,1,1);truth[:,:,2]=1
        pred=torch.zeros_like(truth);pred[:,:,6]=1;pred.requires_grad_(True)
        layers=layers_for(16);layers[:,2]=16
        self.assertEqual(float((slab(pred,layers,native_depth=16)[0]-slab(truth,layers,native_depth=16)[0]).abs().sum()),0)
        p,v=relative_subslabs(pred,layers,native_depth=16)
        y,_=relative_subslabs(truth,layers,native_depth=16)
        loss=subslab_l1(p,y,v)
        self.assertAlmostEqual(float(loss),.5)
        loss.backward();self.assertGreater(float(pred.grad.abs().sum()),0)
        self.assertGreater(float(pred.grad[:,:,6].item()),0)

    def test_within_bin_ambiguity_is_not_claimed_solved(self):
        a=torch.zeros(1,1,16,1,1);a[:,:,1]=1
        b=torch.zeros_like(a);b[:,:,2]=1
        layers=layers_for(16);layers[:,2]=16
        p,v=relative_subslabs(a,layers,native_depth=16)
        y,_=relative_subslabs(b,layers,native_depth=16)
        self.assertEqual(float(subslab_l1(p,y,v)),0)
        self.assertGreater(float((a-b).abs().mean()),0)

    def test_warped_layers_and_unique_boundary_ownership(self):
        x=torch.arange(1,9,dtype=torch.float32)[None,None,:,None,None].expand(1,1,8,1,2)
        layers=layers_for(8,width=2);layers[:,0,0]=torch.tensor([1.,3.]);layers[:,2,0]=torch.tensor([5.,7.])
        projections,valid=relative_subslabs(x,layers,bins=2,native_depth=8)
        torch.testing.assert_close(projections,torch.tensor([[[[2.,4.]],[[5.,7.]]]]))
        self.assertTrue(valid.all())
        # Each basis volume marks one voxel, including exact internal boundaries.
        basis=torch.eye(9).reshape(9,1,9,1,1)
        layers=layers_for(9,batch=9);layers[:,2]=9
        p,_=relative_subslabs(basis,layers,bins=4,native_depth=9)
        torch.testing.assert_close(p.sum(1),torch.ones(9,1,1))
        self.assertEqual(int(p[2,:,0,0].argmax()),1)
        self.assertEqual(int(p[8,:,0,0].argmax()),3)

    def test_empty_invalid_and_thin_subslabs_have_finite_gradients(self):
        x=torch.randn(1,1,4,1,3,requires_grad=True)
        layers=layers_for(8,width=3);layers[:,0,0]=torch.tensor([2.,4.,float('nan')]);layers[:,2,0]=torch.tensor([2.,3.,8.])
        p,v=relative_subslabs(x,layers,native_depth=8)
        self.assertFalse(v.any());self.assertTrue(torch.isfinite(p).all())
        loss=subslab_l1(p,torch.zeros_like(p),v);self.assertEqual(float(loss),0)
        loss.backward();self.assertTrue(torch.isfinite(x.grad).all());self.assertEqual(float(x.grad.abs().sum()),0)
        x=torch.ones(1,1,4,1,1,requires_grad=True);layers=layers_for(8);layers[:,2]=3
        p,v=relative_subslabs(x,layers,bins=8,native_depth=8)
        self.assertEqual(int(v.sum()),1)
        # Seven empty bins do not dilute the only observed bin's error.
        loss=subslab_l1(p,torch.zeros_like(p),v)
        self.assertEqual(float(loss),1);loss.backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_masked_bin_and_patient_aggregation(self):
        p=torch.tensor([[[[2.,99.]],[[4.,8.]]],[[[8.,99.]],[[99.,99.]]]],requires_grad=True)
        v=torch.tensor([[[[1,0]],[[1,1]]],[[[1,0]],[[0,0]]]],dtype=torch.bool)
        loss=subslab_l1(p,torch.zeros_like(p),v)
        # patient0=(2+mean(4,8))/2=4; patient1=8; macro=6.
        self.assertEqual(float(loss),6);loss.backward()
        self.assertEqual(float(p.grad[~v].abs().sum()),0)
        for bins in (0,-1,True,2.5):
            with self.assertRaises(ValueError):relative_subslabs(torch.zeros(1,1,8,1,1),layers_for(8),bins=bins)

    def test_objective_component_coefficient_ramp_and_time_gradient(self):
        cfg=json.loads((ROOT/'configs/main.json').read_text())
        cfg['data']['native_layer_depth']=16;cfg['patch'].update(core=[16,4,4],halo=[0,0,0])
        class EndpointModel(torch.nn.Module):
            def __init__(self):
                super().__init__();self.field=torch.nn.Parameter(torch.randn(1,1,16,4,4)*.1)
            def encode_context(self,*args):return self.field.new_zeros(1,16,1,1,1)
            def forward_patch(self,*args):return self.field
        class Teacher(torch.nn.Module):
            def forward(self,x):return torch.cat((x,-x,x*.5),1)
        source=torch.ones(1,1,16,4,4)*.2;truth=source*.3;noise=torch.randn_like(source)
        layers=layers_for(16,height=4,width=4);layers[:,2]=16
        maps=structure_maps(np.zeros((4,4),bool),np.zeros((4,4),bool),np.zeros((4,4),bool),[3.,7.])
        cfg['loss']={k:(v if k not in ('fm_uniform','endpoint','weighted_mip','vessel_ce','cldice','gradient_match') else 0.) for k,v in cfg['loss'].items()}
        stats={'volume_rms':.1,'mip_rms':.2}
        for t in (0.,.5,.99,1.):
            model=EndpointModel()
            loss,parts=patient_objective(model,Teacher(),source,truth,layers,maps,noise,torch.tensor([t]),[(0,0,0)],cfg,stats,700)
            torch.testing.assert_close(loss,parts['subslab']*.15*.5)
            loss.backward();self.assertTrue(torch.isfinite(model.field.grad).all())
            if t<1:self.assertGreater(float(model.field.grad.abs().sum()),0)
            else:self.assertEqual(float(model.field.grad.abs().sum()),0)

    def test_evaluation_delivery_and_resume_include_depth_evidence(self):
        from microflow.evaluation import evaluate_split
        from microflow.runtime import read
        cfg=json.loads((ROOT/'configs/main.json').read_text());cfg['data']['native_layer_depth']=16
        truth=np.zeros((16,8,8),np.float32);truth[2]=1
        pred=np.zeros_like(truth);pred[6]=1
        layers=layers_for(16,height=8,width=8);layers[:,2]=16
        diagnostics,p,g=depth_diagnostics(pred,truth,layers[0].numpy(),cfg)
        self.assertEqual(diagnostics['subslab_mae_raw'],.5)
        class Dataset:
            def __init__(self,path):self.run=path;self.identities={}
            def split(self,phase):return [{'case_id':'synthetic'}]
            def identify(self,row,key):self.identities[row['case_id']+'/'+key]={'sha256':'synthetic'}
            def case(self,row):return torch.zeros(1,1,16,8,8),torch.as_tensor(truth)[None,None],layers
            def maps(self,row):return structure_maps(np.zeros((8,8),bool),np.zeros((8,8),bool),np.zeros((8,8),bool),[3.,7.])
        class Teacher(torch.nn.Module):
            def forward(self,x):return torch.cat((x,-x,x*.5),1)
        generated=torch.as_tensor(pred)[None,None]
        with tempfile.TemporaryDirectory(dir=ROOT/'verification') as tmp:
            folder=Path(tmp)/'validation';data=Dataset(Path(tmp))
            with patch('microflow.evaluation.generate',return_value=(generated,0.)) as generate:
                result=evaluate_split(None,Teacher(),data,'validation',cfg,{'mip_rms':.2},torch.device('cpu'),folder,'model','teacher')
                self.assertEqual(generate.call_count,1)
                self.assertEqual(result['depth_diagnostics']['subslab_mae_raw']['mean'],.5)
                self.assertTrue((folder/'depth_diagnostics.csv').is_file())
                self.assertTrue((folder/'synthetic/subslabs.png').is_file())
                self.assertFalse((folder/'synthetic/prediction.npy').exists())
                record=read(folder/'synthetic/record.json');self.assertIn('subslabs.png',record['files'])
                self.assertEqual(record['depth_diagnostics']['subslab_mae_raw'],.5)
                result2=evaluate_split(None,Teacher(),data,'validation',cfg,{'mip_rms':.2},torch.device('cpu'),folder,'model','teacher')
                self.assertEqual(generate.call_count,1);self.assertEqual(result,result2)
                altered=copy.deepcopy(cfg);altered['subslab']['bins']=8
                with self.assertRaises(ValueError):evaluate_split(None,Teacher(),data,'validation',altered,{'mip_rms':.2},torch.device('cpu'),folder,'model','teacher')

    def test_actual_engine_validates_only_every_ten_epochs(self):
        from microflow.engine import fit_flow
        cfg=json.loads((ROOT/'configs/main.json').read_text())
        self.assertEqual(cfg['train']['validation_every'],10)
        self.assertEqual(cfg['teacher']['validation_every'],250)
        cfg['train'].update(epochs=20,patient_groups_per_epoch=1)
        class Dataset:
            def __init__(self,path):self.run=path
            def split(self,phase):return [{'case_id':str(i),'split':'train'} for i in range(2)]
        calls=[]
        def evaluate(*args,**kwargs):
            calls.append(Path(args[7]).name)
            return {'selection_score':.1}
        def update(model,teacher,data,group,cfg,stats,device,opt,step):
            opt.zero_grad();loss=model(torch.ones(1,2)).square().mean();loss.backward();opt.step()
            return {'update':step,'losses':{'total':float(loss)},'seconds':0.}
        with tempfile.TemporaryDirectory(dir=ROOT/'verification') as tmp:
            with patch('microflow.engine.evaluate_split',side_effect=evaluate),patch('microflow.engine.optimizer_update',side_effect=update):
                receipt=fit_flow(torch.nn.Linear(2,1),None,Dataset(Path(tmp)),cfg,{},torch.device('cpu'),{'test':True},'teacher')
            self.assertEqual(calls,['e010','e020'])
            self.assertEqual(receipt['best']['epoch'],10)


if __name__=='__main__':unittest.main(verbosity=2)
