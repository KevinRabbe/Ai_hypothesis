from __future__ import annotations
import importlib.util, pathlib, sys, unittest
import torch
ROOT=pathlib.Path(__file__).resolve().parents[1]
MODULE=ROOT/'ai_hypothesis/population_compute/gate9d_learned_shared_router_exhaustive_margin_v3.py'

def load():
    spec=importlib.util.spec_from_file_location('gate9d_router_margin_test',MODULE)
    if spec is None or spec.loader is None: raise RuntimeError('could not load exhaustive-margin router')
    module=importlib.util.module_from_spec(spec); sys.modules[spec.name]=module; spec.loader.exec_module(module); return module

class ExhaustiveMarginRouterTests(unittest.TestCase):
    def test_loss_rewards_strict_gate_margins(self):
        m=load(); worker,query,targets=m.v0.exhaustive_router_domain(torch.device('cpu'))
        exact=torch.where(targets.bool(),torch.full_like(targets,4.0),torch.full_like(targets,-4.0))
        overlap=torch.zeros_like(targets)
        self.assertLess(float(m.exhaustive_margin_loss(exact,targets)),float(m.exhaustive_margin_loss(overlap,targets)))
        exact.requires_grad_(); loss=m.exhaustive_margin_loss(exact,targets); loss.backward(); self.assertTrue(torch.isfinite(exact.grad).all())
    def test_worst_state_identities_are_exact(self):
        m=load()
        class Exact(torch.nn.Module):
            def forward(self,worker,query):
                targets=m.v0.routing_targets(worker,query)
                return torch.where(targets.bool(),torch.full_like(targets,4.0),torch.full_like(targets,-4.0))
        result=m.worst_states(Exact(),torch.device('cpu'))
        for gate in ('bias','contribution'):
            self.assertEqual(result[gate]['margin'],8.0)
            self.assertIn(result[gate]['min_positive_worker'],range(256))
            self.assertIn(result[gate]['min_positive_query'],range(256))
            self.assertIn(result[gate]['max_negative_worker'],range(256))
            self.assertIn(result[gate]['max_negative_query'],range(256))
    def test_contract_and_boundaries(self):
        m=load(); self.assertEqual(m.VERSION,'gate9d-learned-shared-router-exhaustive-margin-v3')
        self.assertEqual(m.BASE_HEAD,'1f3ba1e73dfe13d065880237536f72ff488d64c1')
        self.assertEqual(m.v0._parameter_count(m.SharedRouter()),1218)
        self.assertEqual(m.TRAIN_STEPS,512); self.assertEqual(m.TARGET_MARGIN,2.0)
        source=MODULE.read_text(encoding='utf-8')
        for required in ('exhaustive_router_domain','min_positive_worker','max_negative_query','exhaustive_margin_loss'):
            self.assertIn(required,source)
        for forbidden in ('generate_gate9_test_world(','scientific_assignment_key','torch.save('):
            self.assertNotIn(forbidden,source)
if __name__=='__main__': unittest.main()
