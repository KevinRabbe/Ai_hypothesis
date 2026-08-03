from __future__ import annotations
import importlib.util,pathlib,sys,unittest
ROOT=pathlib.Path(__file__).resolve().parents[1]
PATH=ROOT/'ai_hypothesis/population_compute/gate9d_answer_loss_router_feasibility_v8.py'

def load():
 spec=importlib.util.spec_from_file_location('gate9d_v8_contract',PATH)
 if spec is None or spec.loader is None: raise RuntimeError('cannot load v8')
 m=importlib.util.module_from_spec(spec); sys.modules[spec.name]=m; spec.loader.exec_module(m); return m

class Contract(unittest.TestCase):
 def test_contract(self):
  import torch
  m=load(); model=m.ContributionRouter()
  self.assertEqual(sum(p.numel() for p in model.parameters()),4)
  worker=torch.tensor([[0,1,3]],dtype=torch.long)
  output=torch.tensor([[5,7,11]],dtype=torch.long)
  query=torch.tensor([1],dtype=torch.long)
  pred,gates=m.soft_answer_signs(model,worker,output,query)
  self.assertEqual(tuple(pred.shape),(1,8)); self.assertEqual(tuple(gates.shape),(1,3))
  loss=pred.square().mean(); loss.backward()
  self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
  self.assertLess(m.TRAIN_COUNTER_START+m.TRAIN_OPERATORS,m.EVAL_COUNTER_START)
  source=PATH.read_text(encoding='utf-8')
  self.assertNotIn('routing_targets',source)
  self.assertIn('answer_loss',source)
  self.assertIn('bias_routing_learned":False',source)
  self.assertIn('automatic_coordinate_discovery_claimed":False',source)
  self.assertIn("CUBLAS_WORKSPACE_CONFIG=':4096:8'",(ROOT/'scripts/run_gate9d_answer_loss_router_feasibility_v8.ps1').read_text(encoding='utf-8'))

if __name__=='__main__': unittest.main()
