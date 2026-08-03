from __future__ import annotations
import importlib.util, pathlib, sys, unittest
import torch
ROOT=pathlib.Path(__file__).resolve().parents[1]
PATH=ROOT/'ai_hypothesis/population_compute/gate9d_lbfgs_router_population_execution_v7.py'
spec=importlib.util.spec_from_file_location('gate9d_v7_test',PATH); m=importlib.util.module_from_spec(spec); sys.modules[spec.name]=m; spec.loader.exec_module(m)
class Contract(unittest.TestCase):
 def test_contract(self):
  self.assertEqual(m.BASE_HEAD,'ae1dc4c916dbc361748c2205cb177038cf4e47ce')
  self.assertEqual(m.POPULATION_SIZES,(9,16,64,256)); self.assertEqual(m.OPERATOR_COUNT,128)
  self.assertGreaterEqual(m.COUNTER_START,(1<<57)+0x5000)
 def test_learned_execute_matches_fixed(self):
  device=torch.device('cpu'); model=m.v6.analytic_model(device); c=m.v2.calibrate_thresholds(model,device)
  thresholds={g:c['gates'][g]['threshold'] for g in ('bias','contribution')}
  si=torch.tensor([[0,1,2,4,8,16,32,64,128]],dtype=torch.long); so=torch.tensor([[7,6,5,4,3,2,1,0,255]],dtype=torch.long); q=torch.tensor([173],dtype=torch.long)
  pred,b,cmsg=m.learned_execute(model,thresholds,si,so,q)
  fixed=m.sparse.sparse_population_execute(si,so,q)
  self.assertTrue(torch.equal(pred,fixed.predictions)); self.assertEqual(b,1); self.assertEqual(cmsg,int(m.sparse.byte_bits(q).sum()))
if __name__=='__main__': unittest.main()
