from __future__ import annotations
import importlib.util, pathlib, sys, unittest
import torch
ROOT=pathlib.Path(__file__).resolve().parents[1]
MODULE=ROOT/'ai_hypothesis/population_compute/gate9d_router_convergence_robustness_v6.py'

def load():
    spec=importlib.util.spec_from_file_location('gate9d_router_convergence_test',MODULE)
    if spec is None or spec.loader is None: raise RuntimeError('cannot load v6')
    m=importlib.util.module_from_spec(spec); sys.modules[spec.name]=m; spec.loader.exec_module(m); return m

class Contract(unittest.TestCase):
    def test_analytic_separator(self):
        m=load(); model=m.analytic_model(torch.device('cpu')); c=m.v2.calibrate_thresholds(model,torch.device('cpu'))
        self.assertTrue(c['separable']); self.assertGreater(c['gates']['bias']['margin'],0); self.assertGreater(c['gates']['contribution']['margin'],0)
        self.assertEqual(m.v0._parameter_count(model),8)
    def test_contract(self):
        m=load(); self.assertEqual(m.BASE_HEAD,'90315b5b078dd92c55d46698af1d1c0659d25f8c'); self.assertEqual(len(m.VARIANTS),3)
        source=MODULE.read_text(encoding='utf-8')
        for required in ('LBFGS','analytic_separator','adamw_2048'): self.assertIn(required,source)
        for forbidden in ('generate_gate9_test_world(','scientific_assignment_key','torch.save('): self.assertNotIn(forbidden,source)
if __name__=='__main__': unittest.main()
