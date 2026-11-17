"""Meaningful CPU engineering checks using the full declared network widths/depths."""
from pathlib import Path
import copy
import csv
import itertools
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from microflow.runtime import read,save,load,seed_for
from microflow.model import MicroVesselFlow,VesselTeacher
from microflow.flow import reference
from microflow.geometry import uniform_tile,extract,region,starts
from microflow.data import structure_maps,patient_schedule,Dataset
from microflow.losses import patient_objective,soft_skeleton
from tests.reference_ssm import ReferenceMamba


class ProjectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(4);torch.manual_seed(176)

    def test_path_identity_all_times(self):
        a=torch.randn(2,1,8,8,8);b=torch.randn_like(a)
        for t in (0.,.125,.5,.875,.99999,1.):
            z=(1-t)*a+t*b;ref,q=reference(z,torch.full((2,),t))
            residual=b-a-ref
            torch.testing.assert_close(z+(1-t)*(ref+residual),b,atol=1e-6,rtol=1e-5)
            self.assertGreaterEqual(float(q.min()),.5)

    def test_uniform_partition_unbiased(self):
        a=np.random.default_rng(1).random((8,11,13));core=(8,5,5)
        total=0.
        for pos in itertools.product(*(range(0,n,c) for n,c in zip(a.shape,core))):
            roi=region(pos,core,a.shape);v=a[roi]
            total+=v.mean()*v.size/a.size
        self.assertAlmostEqual(total,float(a.mean()),places=12)

    def test_pixel_geometry_and_target_schedule(self):
        cap=np.zeros((24,24),bool);cap[12,3:21]=1
        large=np.zeros_like(cap);large[3:8,3:21]=1;faz=np.zeros_like(cap);faz[14:18,14:18]=1
        maps=structure_maps(cap,large,faz,[3.,7.])
        self.assertTrue(maps['small_centerline'].any());self.assertFalse((maps['capillary']&maps['large']).any())
        cfg=read(ROOT/'configs/main.json');rows=[{'case_id':str(i),'split':'train'} for i in range(140)]
        groups=list(patient_schedule(cfg,1,rows))
        self.assertEqual(len(groups),70);self.assertEqual(sum(t==0 for group in groups for row,t in group),35)
        self.assertEqual(len({r['case_id'] for g in groups for r,t in g}),140)
        self.assertEqual(groups,list(patient_schedule(cfg,1,rows)))

    def test_inference_coverage(self):
        shape=(8,11,13);core=(8,6,6);overlap=(0,2,2)
        coverage=np.zeros(shape)
        for pos in itertools.product(*(starts(n,c,o) for n,c,o in zip(shape,core,overlap))):coverage[region(pos,core,shape)]+=1
        self.assertTrue((coverage>0).all())

    def test_full_model_and_frozen_teacher_gradient(self):
        cfg=read(ROOT/'configs/main.json');cfg['model']['context_grid']=[2,2,2]
        cfg['model']['checkpoint_local']=True;cfg['data']['shape']=[8,12,12]
        cfg['patch']['core']=[8,8,8];cfg['patch']['halo']=[0,2,2]
        model=MicroVesselFlow(cfg['model'],ReferenceMamba).train()
        report=model.parameter_report();self.assertEqual(report['total'],27421329)
        self.assertLess(report['global_fraction'],.03)
        teacher=VesselTeacher().eval().requires_grad_(False)
        source=torch.rand(1,1,8,12,12);target=torch.rand_like(source)*.2;noise=torch.randn_like(source)
        layers=torch.tensor([1.,90.,220.,350.,450.,640.])[None,:,None,None].expand(1,6,12,12)
        cap=np.zeros((12,12),bool);cap[5,2:10]=1;large=np.zeros_like(cap);large[2:4,2:10]=1;faz=np.zeros_like(cap)
        maps=structure_maps(cap,large,faz,[3.,7.]);stats={'volume_rms':.1,'mip_rms':.2}
        loss,parts=patient_objective(model,teacher,source,target,layers,maps,noise,torch.tensor([0.]),[(0,0,0),(0,4,4)],cfg,stats,1400)
        self.assertTrue(torch.isfinite(loss));loss.backward()
        self.assertGreater(float(model.local_output.weight.grad.abs().sum()),0)
        self.assertTrue(all(p.grad is None for p in teacher.parameters()))
        # Direct input derivatives through the frozen teacher must remain enabled.
        image=torch.randn(1,1,12,12,requires_grad=True);teacher(image).square().mean().backward()
        self.assertGreater(float(image.grad.abs().sum()),0)
        # Enable nonzero output for a checkpointed vs uncheckpointed gradient comparison.
        with torch.no_grad():model.local_output.weight.normal_(std=.001)
        model.zero_grad(set_to_none=True)
        time=torch.tensor([.75]);args=(model,teacher,source,target,layers,maps,noise,time,[(0,0,0),(0,4,4)],cfg,stats,1400)
        loss,_=patient_objective(*args);loss.backward();grad=model.local_stem.weight.grad.clone()
        model.zero_grad(set_to_none=True);cfg['model']['checkpoint_local']=False
        loss,_=patient_objective(*args);loss.backward()
        torch.testing.assert_close(model.local_stem.weight.grad,grad,atol=2e-6,rtol=3e-3)

    def test_checkpoint_and_no_test_access(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'verification') as folder:
            p=Path(folder);linear=torch.nn.Linear(3,2);optimizer=torch.optim.AdamW(linear.parameters(),lr=.01)
            x=torch.randn(4,3)
            def step():optimizer.zero_grad();linear(x).square().sum().backward();optimizer.step()
            step();save(p/'state.pt',{'model':linear.state_dict(),'optimizer':optimizer.state_dict(),'rng':torch.get_rng_state()})
            step();expected={k:v.clone() for k,v in linear.state_dict().items()}
            state=load(p/'state.pt');linear.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);torch.set_rng_state(state['rng']);step()
            for k,v in linear.state_dict().items():torch.testing.assert_close(v,expected[k],atol=0,rtol=0)
            ds=object.__new__(Dataset);ds.allowed={'train'}
            with self.assertRaises(PermissionError):ds.split('test')

    def test_synchronous_whole_state_solver(self):
        from microflow.inference import generate
        from microflow.flow import noise_for
        class ConstantVelocity(torch.nn.Module):
            def encode_context(self,source,state,t):return state.new_zeros(1,16,2,2,2)
            def forward_patch(self,source,normalized,t,context):
                frac=t[:,None,None,None,None];q=(1-frac).square()+frac.square()
                actual=normalized*q.sqrt();ref=(2*frac-1)/q*actual
                return torch.full_like(actual,.25)-ref
        cfg=read(ROOT/'configs/main.json');cfg['inference'].update(core=[8,6,6],halo=[0,2,2],overlap=[0,2,2],steps=3)
        source=torch.zeros(1,1,8,11,13);stats={'volume_rms':.1};seed=13
        result,_=generate(ConstantVelocity(),source,cfg,stats,seed)
        expected=((noise_for(source.shape,torch.device('cpu'),seed)+.25)*.1).clamp(0,1)
        torch.testing.assert_close(result,expected,atol=1e-6,rtol=1e-6)

    def test_actual_engine_partial_epoch_resume(self):
        from microflow.engine import fit_flow
        from microflow.runtime import Pause
        cfg=read(ROOT/'configs/main.json');cfg['train'].update(epochs=2,patient_groups_per_epoch=2,validation_every=1,checkpoint_every=1)
        class SyntheticDataset:
            def __init__(self,path):self.run=path
            def split(self,phase):return [{'case_id':str(i),'split':'train'} for i in range(4)]
        def evaluation(*args,**kwargs):return {'selection_score':.1}
        pause_once=[False]
        def update(model,teacher,data,group,cfg,stats,device,opt,step):
            opt.zero_grad();loss=model(torch.randn(3,3)).square().mean();loss.backward();opt.step()
            if pause_once[0] and step==1:(data.run/'pause.request').touch();pause_once[0]=False
            return {'update':step,'losses':{'total':float(loss)},'seconds':.001}
        with tempfile.TemporaryDirectory(dir=ROOT/'verification') as folder:
            base=Path(folder);a=base/'continuous';b=base/'resumed';a.mkdir();b.mkdir()
            with patch('microflow.engine.optimizer_update',side_effect=update),patch('microflow.engine.evaluate_split',side_effect=evaluation):
                torch.manual_seed(12);model=torch.nn.Linear(3,1)
                fit_flow(model,None,SyntheticDataset(a),cfg,{},torch.device('cpu'),{'test':True},'teacher')
                expected=load(a/'checkpoints/latest.pt')
                torch.manual_seed(12);model=torch.nn.Linear(3,1);pause_once[0]=True
                with self.assertRaises(Pause):fit_flow(model,None,SyntheticDataset(b),cfg,{},torch.device('cpu'),{'test':True},'teacher')
                self.assertEqual(load(b/'checkpoints/latest.pt')['update'],1)
                (b/'pause.request').unlink();torch.randn(19)
                fit_flow(model,None,SyntheticDataset(b),cfg,{},torch.device('cpu'),{'test':True},'teacher')
                actual=load(b/'checkpoints/latest.pt')
                for key in ('model','ema'):
                    for name,value in expected[key].items():torch.testing.assert_close(value,actual[key][name],atol=0,rtol=0)
                self.assertEqual([r['update'] for e in (1,2) for r in read(b/f'train/epoch_{e:03d}.json')['steps']],[1,2,3,4])
                c=base/'pending_validation';c.mkdir();first=[True]
                def pending(*args,**kwargs):
                    if first[0]:first[0]=False;raise Pause('synthetic validation patient boundary')
                    return {'selection_score':.1}
                with patch('microflow.engine.evaluate_split',side_effect=pending):
                    torch.manual_seed(12);model=torch.nn.Linear(3,1)
                    with self.assertRaises(Pause):fit_flow(model,None,SyntheticDataset(c),cfg,{},torch.device('cpu'),{'test':True},'teacher')
                    self.assertTrue(load(c/'checkpoints/latest.pt')['awaiting_validation'])
                    fit_flow(model,None,SyntheticDataset(c),cfg,{},torch.device('cpu'),{'test':True},'teacher')
                    after=load(c/'checkpoints/latest.pt')
                    for name,value in expected['ema'].items():torch.testing.assert_close(value,after['ema'][name],atol=0,rtol=0)

    def test_real_dataset_adapter_without_test_reads(self):
        from microflow.data import statistics
        cfg=read(ROOT/'configs/main.json');cfg['data'].update(shape=[8,12,12],expected_counts={'train':1,'validation':1,'test':1})
        with tempfile.TemporaryDirectory(dir=ROOT/'verification') as folder:
            root=Path(folder);fields=['case_id','patient_id','split','npz','layers','capillary','large','faz','annotation_orientation','axis_order']
            rows=[]
            for i,phase in enumerate(('train','validation','test')):
                row={k:f'{phase}_{k}.npy' for k in ('npz','layers','capillary','large','faz')}
                row['npz']=phase+'.npz';row.update(case_id='C'+str(i),patient_id='P'+str(i),split=phase,annotation_orientation='YX',axis_order='ZYX');rows.append(row)
            with (root/'manifest.csv').open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
            a=np.random.default_rng(1).integers(0,100,(8,12,12),dtype=np.uint8)
            np.savez(root/'train.npz',oct=a,octa=a)
            layer=np.array([1.,90.,220.,350.,450.,640.],np.float32)[:,None,None]*np.ones((6,12,12),np.float32)
            np.save(root/'train_layers.npy',layer)
            for key in ('capillary','large','faz'):
                mask=np.zeros((12,12),np.uint8)
                if key=='capillary':mask[5,2:10]=1
                if key=='large':mask[2:4,2:10]=1
                np.save(root/('train_'+key+'.npy'),mask)
            dataset=Dataset(root/'manifest.csv',cfg,root/'run',{'train'})
            result=statistics(dataset,cfg);self.assertEqual(result['counts'][0],8*12*12)
            self.assertTrue(result['training_only'])
            with self.assertRaises(PermissionError):dataset.case(rows[2])
            self.assertFalse((root/'test.npz').exists())

    def test_complete_metric_groups(self):
        from microflow.metrics import evaluate,GROUPS
        cfg=read(ROOT/'configs/main.json');rng=np.random.default_rng(7)
        y=rng.random((8,12,12),dtype=np.float32)*.2
        layer=np.array([1.,90.,220.,350.,450.,640.],np.float32)[:,None,None]*np.ones((6,12,12),np.float32)
        cap=np.zeros((12,12),bool);cap[5,2:10]=1;large=np.zeros_like(cap);large[2:4,2:10]=1;faz=np.zeros_like(cap)
        maps=structure_maps(cap,large,faz,[3.,7.]);teacher=VesselTeacher().eval().requires_grad_(False)
        values,diag,slices,_,_=evaluate(y,y,layer,maps,teacher,{'mip_rms':.2},cfg,torch.device('cpu'))
        self.assertEqual(set(values),set(GROUPS));self.assertEqual(len(slices),12)
        for group in values.values():self.assertEqual(group['mae'],0);self.assertAlmostEqual(group['ssim'],1.,places=5)
        self.assertEqual(diag['selection_score'],0)

    def test_teacher_stage_freeze_and_reload(self):
        from microflow.engine import fit_teacher
        cfg=read(ROOT/'configs/main.json');cfg['teacher'].update(updates=2,batch_size=2,crop=8,validation_every=1)
        cap=np.zeros((12,12),bool);cap[5,2:10]=1;large=np.zeros_like(cap);large[2:4,2:10]=1;faz=np.zeros_like(cap)
        maps=structure_maps(cap,large,faz,[3.,7.])
        target=torch.rand(1,1,8,12,12)*.2
        layers=torch.tensor([1.,90.,220.,350.,450.,640.])[None,:,None,None].expand(1,6,12,12)
        class SyntheticTeacherDataset:
            def __init__(self,folder):self.run=folder
            def split(self,phase):return [{'case_id':phase+str(i)} for i in range(2)]
            def case(self,row):return target,target,layers
            def maps(self,row):return maps
        with tempfile.TemporaryDirectory(dir=ROOT/'verification') as folder:
            data=SyntheticTeacherDataset(Path(folder))
            teacher,receipt=fit_teacher(data,cfg,{'mip_rms':.2},torch.device('cpu'),{'test':True})
            self.assertTrue(all(not p.requires_grad for p in teacher.parameters()))
            self.assertEqual(receipt['updates'],2)
            again,receipt2=fit_teacher(data,cfg,{'mip_rms':.2},torch.device('cpu'),{'test':True})
            self.assertEqual(receipt,receipt2)
            for a,b in zip(teacher.parameters(),again.parameters()):torch.testing.assert_close(a,b,atol=0,rtol=0)


if __name__=='__main__':unittest.main(verbosity=2)
