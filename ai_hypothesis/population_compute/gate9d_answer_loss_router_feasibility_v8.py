"""Development-only answer-loss training for sparse contribution routing.

The unique zero-input worker remains the deterministic bias broadcaster. The
four-parameter contribution gate is trained only from final answer error, with
an optional global message-budget term based on popcount(query). No local
routing labels are used by the optimizer.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
from typing import Any

import torch
from torch import nn

_ROOT = pathlib.Path(__file__).resolve().parent
_V7 = _ROOT / "gate9d_lbfgs_router_population_execution_v7.py"
VERSION = "gate9d-answer-loss-router-feasibility-v8"
BRANCH = "agent/gate9d-answer-loss-router-feasibility-v8"
BASE_HEAD = "0a1b02e648852daf4e8a7d91b1eef8e302c74cb4"
VARIANTS = ("answer_only", "answer_plus_global_message_budget")
TRAIN_COUNTER_START = (1 << 57) + 0x6000
EVAL_COUNTER_START = (1 << 57) + 0x7000
TRAIN_OPERATORS = 64
EVAL_OPERATORS = 64
TRAIN_POPULATION = 16
EVAL_POPULATIONS = (9, 16, 64, 256)
STEPS = 1024
BATCH = 512
PASS = "G9D_ANSWER_LOSS_ROUTER_FEASIBILITY_PASSES"
FAIL = "G9D_ANSWER_LOSS_ROUTER_FEASIBILITY_FAILED"


def _load(name: str, path: pathlib.Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path.name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


v7 = _load("gate9d_v8_v7", _V7)
sparse = v7.sparse
v6 = v7.v6
v0 = v7.v0


class ContributionRouter(nn.Module):
    """One linear gate over zero, worker popcount, and local overlap."""

    def __init__(self) -> None:
        super().__init__()
        self.output = nn.Linear(3, 1)

    def features(self, worker: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        bits = sparse.byte_bits(worker).to(torch.float32)
        qbits = sparse.byte_bits(query).to(torch.float32)
        zero = (worker == 0).to(torch.float32).unsqueeze(1)
        popcount = bits.sum(dim=1, keepdim=True)
        overlap = (bits * qbits).sum(dim=1, keepdim=True)
        return torch.cat((zero, popcount, overlap), dim=1)

    def forward(self, worker: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        return self.output(self.features(worker, query)).squeeze(1)


def materialize(counter_start: int, operator_count: int):
    support_inputs=[]; support_outputs=[]; queries=[]; targets=[]; counters=[]
    for counter in range(counter_start, counter_start + operator_count):
        op = sparse.operators.operator_from_counter(counter)
        supports = sparse.operators.public_support_pairs(op)
        inputs = tuple(x for x,_ in supports); outputs = tuple(y for _,y in supports)
        for query in sparse.QUERY_VALUES:
            support_inputs.append(inputs); support_outputs.append(outputs)
            queries.append(query); targets.append(op.apply(query)); counters.append(counter)
    return tuple(torch.tensor(x,dtype=torch.long) for x in (support_inputs,support_outputs,queries,targets,counters))


def soft_answer_signs(model: nn.Module, worker_inputs: torch.Tensor, worker_outputs: torch.Tensor, query: torch.Tensor):
    batch,population=worker_inputs.shape
    zero_mask=worker_inputs==0
    zero_slot=torch.argmax(zero_mask.to(torch.long),dim=1)
    bias=worker_outputs.gather(1,zero_slot.unsqueeze(1)).squeeze(1)
    bias_bits=sparse.byte_bits(bias)
    bias_sign=1.0-2.0*bias_bits.to(torch.float32)
    delta_bits=torch.bitwise_xor(sparse.byte_bits(worker_outputs),bias_bits.unsqueeze(1))
    delta_sign=1.0-2.0*delta_bits.to(torch.float32)
    flat_worker=worker_inputs.reshape(-1)
    flat_query=query.unsqueeze(1).expand(-1,population).reshape(-1)
    gates=torch.sigmoid(model(flat_worker,flat_query)).reshape(batch,population)
    factors=1.0-gates.unsqueeze(-1)+gates.unsqueeze(-1)*delta_sign
    predicted_sign=bias_sign*torch.prod(factors,dim=1)
    return predicted_sign,gates


def hard_execute(model: nn.Module, worker_inputs: torch.Tensor, worker_outputs: torch.Tensor, query: torch.Tensor):
    batch,population=worker_inputs.shape
    flat_worker=worker_inputs.reshape(-1)
    flat_query=query.unsqueeze(1).expand(-1,population).reshape(-1)
    with torch.no_grad():
        gates=(model(flat_worker,flat_query)>0).reshape(batch,population)
    zero_mask=worker_inputs==0
    zero_slot=torch.argmax(zero_mask.to(torch.long),dim=1)
    bias=worker_outputs.gather(1,zero_slot.unsqueeze(1)).squeeze(1)
    bias_bits=sparse.byte_bits(bias)
    deltas=torch.bitwise_xor(sparse.byte_bits(worker_outputs),bias_bits.unsqueeze(1))
    parity=torch.remainder(torch.sum(deltas*gates.unsqueeze(-1).to(torch.long),dim=1),2)
    predictions=sparse.decode_bits(torch.bitwise_xor(parity,bias_bits))
    return predictions,int(gates.sum().item())


def train_variant(variant: str, seed: int, device: torch.device):
    v0._configure(seed)
    model=ContributionRouter().to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=0.01,weight_decay=0.0)
    si,so,q,y,counters=(x.to(device) for x in materialize(TRAIN_COUNTER_START,TRAIN_OPERATORS))
    si,so=sparse.augment_population(si,so,counters,TRAIN_POPULATION)
    target_sign=1.0-2.0*sparse.byte_bits(y).to(torch.float32)
    generator=torch.Generator(device="cpu"); generator.manual_seed(seed+880000)
    curves=[]
    checkpoints={1,64,256,512,1024}
    for step in range(1,STEPS+1):
        index=torch.randint(0,q.numel(),(BATCH,),generator=generator).to(device)
        pred,gates=soft_answer_signs(model,si[index],so[index],q[index])
        answer_loss=torch.mean((pred-target_sign[index])**2)
        budget_loss=torch.mean((gates.sum(dim=1)-sparse.byte_bits(q[index]).sum(dim=1).to(torch.float32))**2)
        loss=answer_loss + (0.05*budget_loss if variant=="answer_plus_global_message_budget" else 0.0)
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        if step in checkpoints:
            curves.append({"step":step,"loss":float(loss.detach().cpu()),"answer_loss":float(answer_loss.detach().cpu()),"budget_loss":float(budget_loss.detach().cpu())})
    return model,curves


def run(output_root:pathlib.Path,execution_head:str)->dict[str,Any]:
    if output_root.exists(): raise FileExistsError(f"output exists: {output_root}")
    device=torch.device("cuda",0) if torch.cuda.is_available() else torch.device("cpu")
    esi,eso,eq,ey,ec=(x.to(device) for x in materialize(EVAL_COUNTER_START,EVAL_OPERATORS))
    rows=[]; curves=[]
    for variant in VARIANTS:
        for seed_index,seed in enumerate(v0.SEEDS):
            model,history=train_variant(variant,seed,device)
            curves.extend({"variant":variant,"seed_index":seed_index,**row} for row in history)
            for size in EVAL_POPULATIONS:
                inputs,outputs=sparse.augment_population(esi,eso,ec,size)
                pred,messages=hard_execute(model,inputs,outputs,eq)
                perm=sparse.deterministic_permutation(size).to(device)
                pp,_=hard_execute(model,inputs[:,perm],outputs[:,perm],eq)
                shuffled=torch.roll(outputs,shifts=247,dims=0)
                sp,_=hard_execute(model,inputs,shuffled,eq)
                expected=int(sparse.byte_bits(eq).sum().item())
                rows.append({"variant":variant,"seed_index":seed_index,"initialization_seed":seed,"population_size":size,"parameter_count":sum(p.numel() for p in model.parameters()),"exact_accuracy":float((pred==ey).to(torch.float64).mean().item()),"permuted_exact_accuracy":float((pp==ey).to(torch.float64).mean().item()),"shuffled_exact_accuracy":float((sp==ey).to(torch.float64).mean().item()),"contribution_messages":messages,"expected_contribution_messages":expected})
    reliable=[variant for variant in VARIANTS if all(r["exact_accuracy"]==1.0 and r["permuted_exact_accuracy"]==1.0 and r["shuffled_exact_accuracy"]<=0.02 and r["contribution_messages"]==r["expected_contribution_messages"] for r in rows if r["variant"]==variant)]
    summary={"status":"G9D_ANSWER_LOSS_ROUTER_FEASIBILITY_COMPLETE_DEVELOPMENT_ONLY","version":VERSION,"diagnosis":PASS if reliable else FAIL,"execution_head":execution_head,"base_head":BASE_HEAD,"variants":list(VARIANTS),"reliable_variants":reliable,"train_counter_start":TRAIN_COUNTER_START,"eval_counter_start":EVAL_COUNTER_START,"rows":rows,"boundaries":{"local_routing_labels_used":False,"final_answer_loss_used":True,"global_message_budget_used_by_one_variant":True,"bias_routing_learned":False,"support_output_used_as_message_payload":True,"automatic_coordinate_discovery_claimed":False,"population_confirmation_claimed":False,"frozen_result_modified":False}}
    output_root.mkdir(parents=True)
    (output_root/"aggregate-summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n",encoding="utf-8",newline="\n")
    for name,records in (("rows.jsonl",rows),("curves.jsonl",curves)):
        with (output_root/name).open("w",encoding="utf-8",newline="\n") as h:
            for row in records: h.write(json.dumps(row,sort_keys=True,separators=(",",":"))+"\n")
    return summary
