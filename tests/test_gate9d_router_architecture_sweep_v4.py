from __future__ import annotations
import importlib.util, pathlib, sys, unittest
import torch
ROOT=pathlib.Path(__file__).resolve().parents[1]
MODULE=ROOT/'ai_hypothesis/population_compute/gate9d_router_architecture_sweep_v4.py'

def load():
    spec=importlib.util.spec_from_file_location('gate9d_router_sweep_test',MODULE)
    if spec is None or spec.loader is None: raise RuntimeError('could not load router sweep')
    module=importlib.util.module_from_spec(spec); sys.modules[spec.name]=module; spec.loader.exec_module(module); return module

class RouterSweepTests(unittest.TestCase):
    def test_geometry_and_parameters(self):
        m=load(); worker=torch.tensor([0,1,3],dtype=torch.long); query=torch.tensor([0,1,2],dtype=torch.long)
        self.assertEqual(tuple(m.raw_features(worker,query).shape),(3,16))
        self.assertEqual(tuple(m.interaction_features(worker,query).shape),(3,24))
        self.assertEqual(m.parameter_count(m.RawWidth128()),2434)
        self.assertEqual(m.parameter_count(m.RawDeep64()),5378)
        self.assertEqual(m.parameter_count(m.Interaction16()),434)
        for variant in m.VARIANTS: self.assertEqual(tuple(m.make_model(variant)(worker,query).shape),(3,2))
    def test_classifier_order(self):
        m=load(); rows=[]
        for variant in m.VARIANTS:
            for seed in range(3): rows.append({'variant':variant,'separable':variant=='raw_deep64'})
        self.assertEqual(m.classify(rows),(m.PASS,'raw_deep64'))
    def test_boundaries(self):
        m=load(); self.assertEqual(m.BASE_HEAD,'b501783bababfa4fd82763441aea60620b2c2de9')
        source=MODULE.read_text(encoding='utf-8')
        for forbidden in ('generate_gate9_test_world(','scientific_assignment_key','torch.save('): self.assertNotIn(forbidden,source)
if __name__=='__main__': unittest.main()
