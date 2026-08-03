"""Development-only end-to-end population execution with the reliable LBFGS router."""
from __future__ import annotations
import importlib.util, json, pathlib, sys
from typing import Any
import torch

_ROOT = pathlib.Path(__file__).resolve().parent
_V6 = _ROOT / "gate9d_router_convergence_robustness_v6.py"
_SPARSE = _ROOT / "gate9d_sparse_affine_worker_population.py"
VERSION = "gate9d-lbfgs-router-population-execution-v7"
BRANCH = "agent/gate9d-lbfgs-router-population-execution-v7"
BASE_HEAD = "ae1dc4c916dbc361748c2205cb177038cf4e47ce"
PASS = "G9D_LBFGS_ROUTER_POPULATION_EXECUTION_PASSES"
FAIL = "G9D_LBFGS_ROUTER_POPULATION_EXECUTION_FAILED"
COUNTER_START = (1 << 57) + 0x5000
OPERATOR_COUNT = 128
POPULATION_SIZES = (9, 16, 64, 256)

def _load(name: str, path: pathlib.Path):
    if name in sys.modules: return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None: raise RuntimeError(f"cannot load {path.name}")
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module; spec.loader.exec_module(module); return module

v6 = _load("gate9d_v7_v6", _V6)
sparse = _load("gate9d_v7_sparse", _SPARSE)
v2, v0 = v6.v2, v6.v0

def materialize():
    si=[]; so=[]; q=[]; y=[]; counters=[]
    for counter in range(COUNTER_START, COUNTER_START + OPERATOR_COUNT):
        op = sparse.operators.operator_from_counter(counter)
        supports = sparse.operators.public_support_pairs(op)
        inputs = tuple(x for x,_ in supports); outputs = tuple(z for _,z in supports)
        for query in sparse.QUERY_VALUES:
            si.append(inputs); so.append(outputs); q.append(query); y.append(op.apply(query)); counters.append(counter)
    return tuple(torch.tensor(x,dtype=torch.long) for x in (si,so,q,y,counters))

def learned_execute(model, thresholds, worker_inputs, worker_outputs, query):
    batch,population = worker_inputs.shape
    flat_worker = worker_inputs.reshape(-1)
    flat_query = query.unsqueeze(1).expand(-1,population).reshape(-1)
    with torch.no_grad(): logits = model(flat_worker, flat_query).reshape(batch,population,2)
    bias_gate = logits[:,:,0] > thresholds["bias"]
    contribution_gate = logits[:,:,1] > thresholds["contribution"]
    if not bool(torch.all(bias_gate.sum(dim=1) == 1)):
        raise RuntimeError("learned router did not select exactly one bias worker")
    bias_slot = torch.argmax(bias_gate.to(torch.long),dim=1)
    bias_bytes = worker_outputs.gather(1,bias_slot.unsqueeze(1)).squeeze(1)
    bias_bits = sparse.byte_bits(bias_bytes)
    deltas = torch.bitwise_xor(sparse.byte_bits(worker_outputs), bias_bits.unsqueeze(1))
    parity = torch.remainder(torch.sum(deltas * contribution_gate.unsqueeze(-1).to(torch.long),dim=1),2)
    predictions = sparse.decode_bits(torch.bitwise_xor(parity,bias_bits))
    return predictions, int(bias_gate.sum().item()), int(contribution_gate.sum().item())

def run(output_root:pathlib.Path, execution_head:str)->dict[str,Any]:
    if output_root.exists(): raise FileExistsError(f"output exists: {output_root}")
    device=torch.device("cuda",0) if torch.cuda.is_available() else torch.device("cpu")
    si,so,q,y,counters=(x.to(device) for x in materialize())
    rows=[]
    for seed_index,seed in enumerate(v0.SEEDS):
        model,_=v6._lbfgs(seed,device)
        calibration=v2.calibrate_thresholds(model,device)
        if not calibration["separable"]: raise RuntimeError("LBFGS router not separable")
        thresholds={gate: calibration["gates"][gate]["threshold"] for gate in ("bias","contribution")}
        for size in POPULATION_SIZES:
            inputs,outputs=sparse.augment_population(si,so,counters,size)
            pred,bmsg,cmsg=learned_execute(model,thresholds,inputs,outputs,q)
            perm=sparse.deterministic_permutation(size).to(device)
            pp,_,_=learned_execute(model,thresholds,inputs[:,perm],outputs[:,perm],q)
            shuffled=torch.roll(outputs,shifts=247,dims=0)
            sp,_,_=learned_execute(model,thresholds,inputs,shuffled,q)
            expected_bias=int(q.numel()); expected_contrib=int(sparse.byte_bits(q).sum().item())
            rows.append({"seed_index":seed_index,"initialization_seed":seed,"population_size":size,"exact_accuracy":float((pred==y).to(torch.float64).mean().item()),"permuted_exact_accuracy":float((pp==y).to(torch.float64).mean().item()),"shuffled_exact_accuracy":float((sp==y).to(torch.float64).mean().item()),"bias_messages":bmsg,"expected_bias_messages":expected_bias,"contribution_messages":cmsg,"expected_contribution_messages":expected_contrib,"bias_margin":calibration["gates"]["bias"]["margin"],"contribution_margin":calibration["gates"]["contribution"]["margin"]})
    passed=all(r["exact_accuracy"]==1.0 and r["permuted_exact_accuracy"]==1.0 and r["shuffled_exact_accuracy"]<=0.02 and r["bias_messages"]==r["expected_bias_messages"] and r["contribution_messages"]==r["expected_contribution_messages"] for r in rows)
    summary={"status":"G9D_LBFGS_ROUTER_POPULATION_EXECUTION_COMPLETE_DEVELOPMENT_ONLY","version":VERSION,"diagnosis":PASS if passed else FAIL,"execution_head":execution_head,"base_head":BASE_HEAD,"counter_start":COUNTER_START,"operator_count":OPERATOR_COUNT,"population_sizes":list(POPULATION_SIZES),"rows":rows,"boundaries":{"supervised_routing_labels_used":True,"end_to_end_answer_loss_used":False,"automatic_coordinate_discovery_claimed":False,"population_confirmation_claimed":False,"frozen_result_modified":False}}
    output_root.mkdir(parents=True); (output_root/"aggregate-summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    with (output_root/"rows.jsonl").open("w",encoding="utf-8",newline="\n") as h:
        for row in rows: h.write(json.dumps(row,sort_keys=True,separators=(",",":"))+"\n")
    return summary
