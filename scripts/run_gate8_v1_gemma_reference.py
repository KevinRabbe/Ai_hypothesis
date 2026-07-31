#!/usr/bin/env python3
"""Run the frozen Gate-8 v1 Gemma 3 1B reference evaluation."""
from __future__ import annotations

import argparse, hashlib, importlib.metadata, importlib.util, json, os, pathlib
import platform, sqlite3, subprocess, sys, time
from collections import defaultdict
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
WORLD = ROOT / "ai_hypothesis/population_compute/gate8_distributed_transformation_worlds.py"
ENCODER = ROOT / "ai_hypothesis/population_compute/gate8_distributed_transformation_encoder.py"
CONTRACT = ROOT / "ai_hypothesis/population_compute/gate8_v1_gemma_reference_execution.py"
BRANCH = "agent/gate8-v1-gemma-reference-execution-v0"


def load(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod
    spec.loader.exec_module(mod); return mod


def sha(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()


def atomic_json(path: pathlib.Path, value: Any) -> None:
    tmp = pathlib.Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def files(root: pathlib.Path) -> tuple[str, ...]:
    return tuple(sorted(p.relative_to(root).as_posix() for p in root.rglob("*")
                        if p.is_file() and ".cache" not in p.relative_to(root).parts))


def validate_snapshot(root: pathlib.Path, expected: dict[str, str], label: str) -> None:
    if files(root) != tuple(expected):
        raise RuntimeError(f"{label} file set drifted")
    for name, digest in expected.items():
        if sha(root / name) != digest: raise RuntimeError(f"{label} hash drifted: {name}")


def software(c: Any, torch: Any) -> dict[str, str]:
    out = {"python": platform.python_version(), "torch": torch.__version__}
    for name in ("transformers", "tokenizers", "numpy", "huggingface-hub"):
        out[name] = importlib.metadata.version(name)
    for name, expected in c.GATE8_V1_REQUIRED_SOFTWARE.items():
        if out[name] != expected:
            raise RuntimeError(f"software drift: {name} expected={expected} observed={out[name]}")
    out["safetensors"] = importlib.metadata.version("safetensors")
    return out


def prompt_for(worlds: Any, encoder: Any, tokenizer: Any, demos: tuple[Any, ...],
               c: Any, population: int, depth: int, index: int):
    generated = worlds.generate_gate8_world(split="test", seed=0, world_index=index,
                                             population=population, depth=depth)
    oracle = worlds.gate8_exact_symbolic_oracle(generated.public)
    if oracle.answer_symbol != generated.truth.answer_symbol: raise RuntimeError("oracle drift")
    prompt = encoder.encode_gate8_reference_prompt(generated.public, demos)
    ids = tokenizer.apply_chat_template([{"role":"user","content":prompt}],
                                        add_generation_prompt=True, tokenize=True,
                                        return_dict=True)["input_ids"]
    row = c.Gate8V1ReferencePromptRow(
        sequence=c.gate8_v1_reference_sequence(population, depth, index),
        population=population, depth=depth, world_index=index,
        world_id=generated.public.world_id,
        prompt_sha256=hashlib.sha256(prompt.encode("ascii")).hexdigest(),
        ascii_bytes=len(prompt.encode("ascii")), input_tokens=len(ids),
        answer_symbol=oracle.answer_symbol)
    row.validate(); return prompt, row


def prompt_index(path: pathlib.Path, worlds: Any, encoder: Any, tokenizer: Any,
                 demos: tuple[Any, ...], c: Any) -> tuple[tuple[Any, ...], str]:
    tmp = pathlib.Path(str(path) + ".rebuild"); tmp.unlink(missing_ok=True); rows = []
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        for population, depth in c.GATE8_V1_VALID_CONDITIONS:
            for index in range(512):
                _, row = prompt_for(worlds, encoder, tokenizer, demos, c, population, depth, index)
                rows.append(row); f.write(c.canonical_json_line(row.to_dict()))
            print(f"prompt-index P={population} D={depth} complete", flush=True)
    matrix = c.validate_gate8_v1_prompt_matrix(rows); digest = sha(tmp)
    if path.exists():
        if path.read_bytes() != tmp.read_bytes(): raise RuntimeError("prompt index drifted")
        tmp.unlink()
    else: os.replace(tmp, path)
    return matrix, digest


def open_db(path: pathlib.Path, fingerprint: dict[str, Any], resume: bool) -> sqlite3.Connection:
    if path.exists() and not resume: raise FileExistsError("progress exists; use --resume")
    db = sqlite3.connect(path); db.execute("PRAGMA journal_mode=DELETE")
    db.execute("PRAGMA synchronous=FULL")
    db.execute("CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT NOT NULL)")
    db.execute("CREATE TABLE IF NOT EXISTS results(sequence INTEGER PRIMARY KEY,payload TEXT NOT NULL)")
    value = json.dumps(fingerprint, sort_keys=True, separators=(",", ":"))
    prior = db.execute("SELECT value FROM meta WHERE key='fingerprint'").fetchone()
    if prior is None: db.execute("INSERT INTO meta VALUES('fingerprint',?)", (value,)); db.commit()
    elif prior[0] != value: raise RuntimeError("resume fingerprint drifted")
    return db


def prefix(db: sqlite3.Connection, prompts: tuple[Any, ...], c: Any) -> tuple[Any, ...]:
    rows = []
    for sequence, text in db.execute("SELECT sequence,payload FROM results ORDER BY sequence"):
        data = json.loads(text); data["output_token_ids"] = tuple(data["output_token_ids"])
        row = c.Gate8V1ReferenceResultRow(**data)
        if row.sequence != sequence: raise RuntimeError("database sequence drifted")
        rows.append(row)
    return c.validate_gate8_v1_result_prefix(rows, prompts)


def load_model(model_root: pathlib.Path, c: Any, torch: Any):
    from transformers import AutoModelForCausalLM
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA BF16 is required")
    torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_root), local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    params = tuple(model.parameters())
    if sum(p.numel() for p in params) != c.GATE8_V1_REFERENCE_PARAMETER_COUNT:
        raise RuntimeError("loaded parameter count drifted")
    if any(p.dtype != torch.bfloat16 or p.device.type != "cuda" for p in params):
        raise RuntimeError("loaded parameter dtype/device drifted")
    return model


def generate(model: Any, tokenizer: Any, prompt: str, tokens: int, c: Any, torch: Any):
    encoded = tokenizer.apply_chat_template([{"role":"user","content":prompt}],
        add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
    if tuple(encoded["input_ids"].shape) != (1, tokens): raise RuntimeError("token count drifted")
    inputs = {k:v.to("cuda") for k,v in encoded.items()
              if k in ("input_ids","attention_mask") and hasattr(v,"to")}
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(**inputs, do_sample=False, num_beams=1,
            max_new_tokens=64, use_cache=True, return_dict_in_generate=False,
            pad_token_id=model.generation_config.pad_token_id,
            eos_token_id=model.generation_config.eos_token_id)
    torch.cuda.synchronize(); elapsed = time.perf_counter() - started
    ids = tuple(int(x) for x in output[0, tokens:].detach().cpu().tolist())
    text = tokenizer.decode(list(ids), skip_special_tokens=True,
                            clean_up_tokenization_spaces=False)
    return text, ids, elapsed, int(torch.cuda.max_memory_allocated())


def bootstrap(vector: Any, namespace: str, c: Any, np: Any) -> tuple[float,float]:
    unique, counts = np.unique(vector.astype(np.float64), return_counts=True)
    rng = np.random.Generator(np.random.PCG64(int.from_bytes(
        hashlib.sha256(namespace.encode("ascii")).digest()[:8], "big")))
    samples = rng.multinomial(512, counts / 512.0, size=c.GATE8_V1_BOOTSTRAP_SAMPLES)
    values = (samples @ unique) / 512.0
    low, high = np.quantile(values, (0.025,0.975), method="linear")
    return float(low), float(high)


def finish(db: sqlite3.Connection, prompts: tuple[Any, ...], out: pathlib.Path,
           c: Any, np: Any) -> tuple[tuple[Any,...], list[dict[str,Any]]]:
    rows = prefix(db, prompts, c)
    if len(rows) != 10_752: raise RuntimeError("reference execution incomplete")
    world_path = out / "gate8-v1-gemma-reference-per-world.jsonl"
    tmp = pathlib.Path(str(world_path)+".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows: f.write(c.canonical_json_line(row.to_dict()))
    os.replace(tmp, world_path)
    grouped: dict[tuple[int,int],list[Any]] = defaultdict(list)
    for row in rows: grouped[(row.population,row.depth)].append(row)
    metrics = []
    for population, depth in c.GATE8_V1_VALID_CONDITIONS:
        part = grouped[(population,depth)]; vec = np.array([r.correct for r in part], dtype=np.uint8)
        low, high = bootstrap(vec, f"{c.GATE8_V1_BOOTSTRAP_NAMESPACE}:{population}:{depth}", c, np)
        metric = c.Gate8V1ReferenceConditionMetric(
            population=population, depth=depth, accuracy=float(vec.mean()),
            bootstrap_ci_low=low, bootstrap_ci_high=high,
            valid_parse_rate=sum(r.parse_status=="valid" for r in part)/512,
            correct=int(vec.sum()), valid_outputs=sum(r.parse_status=="valid" for r in part),
            worlds=512, maximum_input_tokens=max(r.input_tokens for r in part),
            mean_input_tokens=sum(r.input_tokens for r in part)/512,
            mean_output_tokens=sum(len(r.output_token_ids) for r in part)/512,
            mean_wall_seconds=sum(r.wall_seconds for r in part)/512,
            peak_device_bytes=max(r.peak_device_bytes for r in part),
            correctness_vector_sha256=hashlib.sha256(vec.tobytes()).hexdigest())
        metrics.append(metric.to_dict())
    return rows, metrics


def run(tokenizer_root: pathlib.Path, model_root: pathlib.Path,
        output_root: pathlib.Path, resume: bool) -> int:
    worlds, encoder, c = load(WORLD,"g8ref_worlds"), load(ENCODER,"g8ref_encoder"), load(CONTRACT,"g8ref_contract")
    import numpy as np, torch
    from transformers import AutoTokenizer
    if git("branch","--show-current") != BRANCH or git("status","--porcelain"):
        raise RuntimeError("qualified branch and clean tree required")
    head = git("rev-parse","HEAD"); output_root = output_root.resolve()
    if output_root.exists() and not resume: raise FileExistsError(f"output exists: {output_root}")
    if not output_root.exists() and resume: raise FileNotFoundError("resume output missing")
    output_root.mkdir(parents=True, exist_ok=resume); ref = output_root / "reference"; ref.mkdir(exist_ok=resume)
    if (ref / "gate8-v1-gemma-reference-summary.json").exists(): raise FileExistsError("summary exists")
    tokenizer_root, model_root = tokenizer_root.resolve(), model_root.resolve()
    validate_snapshot(tokenizer_root, c.GATE8_V1_REQUIRED_TOKENIZER_FILE_SHA256, "tokenizer")
    validate_snapshot(model_root, c.GATE8_V1_REQUIRED_MODEL_FILE_SHA256, "model")
    soft = software(c, torch)
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_root), local_files_only=True,
                                               trust_remote_code=False, use_fast=True)
    if type(tokenizer).__name__ != "GemmaTokenizerFast": raise RuntimeError("tokenizer class drifted")
    if hashlib.sha256(tokenizer.chat_template.encode()).hexdigest() != c.GATE8_V1_CHAT_TEMPLATE_SHA256:
        raise RuntimeError("chat template drifted")
    demos = tuple(worlds.generate_gate8_world(split="demonstration",seed=0,world_index=i,
                                               population=32,depth=4) for i in range(8))
    index_path = ref / "gate8-v1-gemma-reference-prompt-index.jsonl"
    prompts, index_hash = prompt_index(index_path, worlds, encoder, tokenizer, demos, c)
    fingerprint = {"version":c.GATE8_V1_GEMMA_REFERENCE_EXECUTION_VERSION,"head":head,
        "population_result_head":c.GATE8_V1_POPULATION_RESULT_HEAD,"prompt_index_sha256":index_hash,
        "software":soft,"tokenizer_snapshot":str(tokenizer_root),"model_snapshot":str(model_root),
        "decoding":"greedy_temperature_0","rows":10_752}
    config = output_root / "run-config.json"
    if config.exists() and json.loads(config.read_text()) != fingerprint: raise RuntimeError("run config drifted")
    if not config.exists(): atomic_json(config, fingerprint)
    (output_root/"git-head.txt").write_text(head+"\n",encoding="ascii")
    (output_root/"git-status.txt").write_text("",encoding="ascii")
    db_path = ref / "gate8-v1-gemma-reference-progress.sqlite3"
    db = open_db(db_path, fingerprint, resume); done = prefix(db, prompts, c); start = len(done)
    print(json.dumps({"status":"G8_V1_GEMMA_REFERENCE_READY","completed":start,
        "remaining":10_752-start,"prompt_index_sha256":index_hash},indent=2),flush=True)
    model = load_model(model_root,c,torch); started = time.perf_counter()
    for sequence in range(start,10_752):
        expected = prompts[sequence]
        prompt, rebuilt = prompt_for(worlds,encoder,tokenizer,demos,c,expected.population,
                                     expected.depth,expected.world_index)
        if rebuilt != expected: raise RuntimeError("prompt regeneration drifted")
        text, ids, seconds, peak = generate(model,tokenizer,prompt,expected.input_tokens,c,torch)
        try: predicted, status = c.parse_gate8_v1_reference_answer(text), "valid"
        except ValueError: predicted, status = None, "invalid"
        row = c.Gate8V1ReferenceResultRow(**expected.to_dict(),generated_text=text,
            output_token_ids=ids,predicted_symbol=predicted,parse_status=status,
            correct=predicted is not None and predicted==expected.answer_symbol,
            wall_seconds=seconds,peak_device_bytes=peak); row.validate()
        db.execute("INSERT INTO results VALUES(?,?)",(sequence,json.dumps(row.to_dict(),sort_keys=True,separators=(",",":"))))
        db.commit()
        if (sequence+1)%32==0: print(f"rows={sequence+1}/10752 P={row.population} D={row.depth} I={row.world_index} valid={status=='valid'} correct={row.correct} elapsed={time.perf_counter()-started:.1f}s",flush=True)
    rows, metrics = finish(db,prompts,ref,c,np); db.close()
    world_path = ref / "gate8-v1-gemma-reference-per-world.jsonl"
    summary = {"experiment_version":c.GATE8_V1_GEMMA_REFERENCE_EXECUTION_VERSION,
        "scientific_status":"G8_V1_GEMMA_REFERENCE_EVALUATION_COMPLETE","execution_head":head,
        "population_result_head":c.GATE8_V1_POPULATION_RESULT_HEAD,
        "population_summary_sha256":c.GATE8_V1_POPULATION_SUMMARY_SHA256,
        "population_per_world_sha256":c.GATE8_V1_POPULATION_PER_WORLD_SHA256,
        "reference":{"repo_id":c.GATE8_V1_REFERENCE_REPO_ID,"revision":c.GATE8_V1_REFERENCE_REVISION,
            "parameter_count":c.GATE8_V1_REFERENCE_PARAMETER_COUNT,"dtype":"bfloat16",
            "decoding":"greedy_temperature_0","max_new_tokens":64,"demonstrations":8,
            "batch_size":1,"attention_implementation":"sdpa"},
        "prompt_matrix":{"artifact":index_path.name,"sha256":index_hash,"rows":10_752,
            "maximum_input_tokens":max(r.input_tokens for r in prompts),"input_token_limit":24_576},
        "per_world":{"artifact":world_path.name,"sha256":sha(world_path),"rows":10_752},
        "condition_metrics":metrics,"pooled_reference_accuracy":sum(m["accuracy"] for m in metrics)/21,
        "valid_parse_rate":sum(r.parse_status=="valid" for r in rows)/10_752,
        "environment":{"software":soft,"platform":platform.platform(),
            "cuda_device":torch.cuda.get_device_name(0),"cuda_capability":list(torch.cuda.get_device_capability(0))},
        "resume":{"transactional_sqlite":True,"progress_artifact":db_path.name,
            "resumed":resume,"rows_present_before_model_load":start},
        "reference_model_loaded":True,"reference_inference_performed":True,
        "scientific_test_worlds_generated":True,"population_execution_performed":False,
        "training_performed":False,"joint_reference_comparison_classified":False}
    summary_path = ref / "gate8-v1-gemma-reference-summary.json"; atomic_json(summary_path,summary)
    print(json.dumps({"status":summary["scientific_status"],"rows":10_752,
        "pooled_reference_accuracy":summary["pooled_reference_accuracy"],
        "valid_parse_rate":summary["valid_parse_rate"],"summary_sha256":sha(summary_path),
        "per_world_sha256":summary["per_world"]["sha256"],"prompt_index_sha256":index_hash,
        "output_root":str(output_root)},indent=2,sort_keys=True),flush=True); return 0


def main() -> int:
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--tokenizer-snapshot",type=pathlib.Path,required=True)
    p.add_argument("--model-snapshot",type=pathlib.Path,required=True); p.add_argument("--output-root",type=pathlib.Path,required=True)
    p.add_argument("--resume",action="store_true"); a=p.parse_args(); return run(a.tokenizer_snapshot,a.model_snapshot,a.output_root,a.resume)

if __name__ == "__main__": raise SystemExit(main())
