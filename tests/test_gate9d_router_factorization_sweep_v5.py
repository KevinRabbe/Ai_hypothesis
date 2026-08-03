from __future__ import annotations
import importlib.util, pathlib, sys, unittest
import torch
ROOT=pathlib.Path(__file__).resolve().parents[1]
MODULE=ROOT/'ai_hypothesis/population_compute/gate9d_router_factorization_sweep_v5.py'

def load():
    spec=importlib.util.spec_from_file_location('gate9d_factorization_test',MODULE)
    if spec is None or spec.loader is None: raise RuntimeError('load failed')
    module=importlib.util.module_from_spec(spec); sys.modules[spec.name]=module; spec.loader.exec_module(module); return module

class Contract(unittest.TestCase):
    def test_geometry_and_exact_summary_features(self):
        m=load(); worker=torch.tensor([0,1,3,8]); query=torch.tensor([255,1,3,0])
        self.assertEqual(tuple(m.DecoupledRaw64()(worker,query).shape),(4,2))
        self.assertEqual(tuple(m.FactorizedOverlap16()(worker,query).shape),(4,2))
        summary=m.LocalSummaryLinear(); self.assertEqual(tuple(summary(worker,query).shape),(4,2))
        self.assertEqual(m.v0._parameter_count(m.DecoupledRaw64()),1314)
        self.assertEqual(m.v0._parameter_count(m.FactorizedOverlap16()),338)
        self.assertEqual(m.v0._parameter_count(summary),8)
        expected=torch.tensor([[1.,0.,0.],[0.,1.,1.],[0.,2.,2.],[0.,1.,0.]])
        self.assertTrue(torch.equal(summary.features(worker,query),expected))
    def test_finite_margin_backward(self):
        m=load(); worker,query,targets=m.v0.exhaustive_router_domain(torch.device('cpu')); model=m.LocalSummaryLinear(); loss=m.v3.exhaustive_margin_loss(model(worker,query),targets); loss.backward(); self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
    def test_boundaries(self):
        m=load(); self.assertEqual(m.BASE_HEAD,'ac7fc7ecaadcf46d4f55db6586984dd719eb5100'); self.assertEqual(m.VARIANTS,('decoupled_raw64','factorized_overlap16','local_summary_linear'))
        source=MODULE.read_text(encoding='utf-8')
        for forbidden in ('generate_gate9_test_world(','scientific_assignment_key','torch.save('): self.assertNotIn(forbidden,source)
if __name__=='__main__': unittest.main()
