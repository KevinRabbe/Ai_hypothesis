"""Population Language L0 protocol and deterministic synthetic language world."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Literal

VERSION = "population-language-l0-protocol-v0"
BRANCH = "agent/population-language-l0-protocol-v0"
STATUS = "PROTOCOL_ONLY_NO_TRAINING_RESULT"
MAX_SEQUENCE_LENGTH = 32
TRAIN_WORKERS = 32
EVAL_WORKERS = (16, 32, 64, 128, 256)
REFERENCE_TRAIN_EPISODES = 131_072
REFERENCE_VALIDATION_EPISODES = 8_192
REFERENCE_TEST_EPISODES = 16_384
INITIALIZATION_SEEDS = (120100, 120101, 120102)
PARAMETER_TOLERANCE_FRACTION = 0.005

SPECIAL_TOKENS = (
    "<pad>", "<bos>", "<eos>", "<def>", "<query>", "<answer>", "<sep>", "<unk>",
)
NONCE_TOKENS = (
    "dax", "wug", "zorp", "kiki", "blicket", "toma", "fep", "luma",
    "nib", "vako", "suli", "pim", "rava", "mepo", "glar", "tiv",
)
COLOR_TOKENS = (
    "red", "blue", "green", "yellow", "orange", "purple", "white", "black",
)
SHAPE_TOKENS = (
    "triangle", "circle", "square", "star", "hexagon", "oval", "diamond", "cross",
)
RELATION_TOKENS = (
    "left_of", "right_of", "above", "below", "follows", "avoids", "sees", "likes",
)
FUNCTION_TOKENS = (
    "means", "the", "is", "what", "relation", "between", "and", "first",
    "second", "object", "describes", "with", "to", "from", "near", "far",
)
VOCABULARY = (
    SPECIAL_TOKENS + NONCE_TOKENS + COLOR_TOKENS + SHAPE_TOKENS
    + RELATION_TOKENS + FUNCTION_TOKENS
)
TOKEN_TO_ID = {token: index for index, token in enumerate(VOCABULARY)}
Split = Literal["train", "validation", "test"]
_SPLIT_RANGES = {"train": range(0, 80), "validation": range(80, 90), "test": range(90, 100)}
_SPLIT_SALT = {"train": 0x11, "validation": 0x22, "test": 0x33}


@dataclass(frozen=True)
class Episode:
    split: Split
    ordinal: int
    lhs_nonce: str
    rhs_nonce: str
    lhs_color: str
    lhs_shape: str
    relation: str
    rhs_color: str
    rhs_shape: str
    definition_order_swapped: bool
    tokens: tuple[str, ...]
    answer_start: int

    @property
    def token_ids(self) -> tuple[int, ...]:
        return tuple(TOKEN_TO_ID[token] for token in self.tokens)

    @property
    def answer_tokens(self) -> tuple[str, ...]:
        return self.tokens[self.answer_start : -1]


@dataclass(frozen=True)
class TransformerConfig:
    vocab_size: int = len(VOCABULARY)
    max_sequence_length: int = MAX_SEQUENCE_LENGTH
    d_model: int = 512
    layers: int = 6
    feed_forward: int = 2048
    heads: int = 8


@dataclass(frozen=True)
class PopulationConfig:
    vocab_size: int = len(VOCABULARY)
    max_sequence_length: int = MAX_SEQUENCE_LENGTH
    token_width: int = 512
    lexical_encoder_width: int = 14_544
    worker_width: int = 128
    worker_feed_forward: int = 512
    lexical_decoder_width: int = 14_544
    router_dim: int = 32
    training_workers: int = TRAIN_WORKERS


def _digest_bytes(*parts: object) -> bytes:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(payload).digest()


def _semantic_bucket(
    lhs_color: str,
    lhs_shape: str,
    relation: str,
    rhs_color: str,
    rhs_shape: str,
) -> int:
    digest = _digest_bytes(lhs_color, lhs_shape, relation, rhs_color, rhs_shape)
    return int.from_bytes(digest[:8], "little") % 100


def _candidate(split: Split, ordinal: int, attempt: int) -> tuple[int, ...]:
    digest = _digest_bytes(VERSION, _SPLIT_SALT[split], ordinal, attempt)
    return tuple(digest[index] for index in range(10))


def make_episode(split: Split, ordinal: int) -> Episode:
    if split not in _SPLIT_RANGES:
        raise ValueError(f"unknown split: {split}")
    if type(ordinal) is not int or ordinal < 0:
        raise ValueError("episode ordinal must be a nonnegative integer")

    for attempt in range(100_000):
        values = _candidate(split, ordinal, attempt)
        lhs_color = COLOR_TOKENS[values[0] % len(COLOR_TOKENS)]
        lhs_shape = SHAPE_TOKENS[values[1] % len(SHAPE_TOKENS)]
        relation = RELATION_TOKENS[values[2] % len(RELATION_TOKENS)]
        rhs_color = COLOR_TOKENS[values[3] % len(COLOR_TOKENS)]
        rhs_shape = SHAPE_TOKENS[values[4] % len(SHAPE_TOKENS)]
        if (lhs_color, lhs_shape) == (rhs_color, rhs_shape):
            continue
        if _semantic_bucket(lhs_color, lhs_shape, relation, rhs_color, rhs_shape) not in _SPLIT_RANGES[split]:
            continue

        lhs_index = values[5] % len(NONCE_TOKENS)
        rhs_index = values[6] % (len(NONCE_TOKENS) - 1)
        if rhs_index >= lhs_index:
            rhs_index += 1
        lhs_nonce = NONCE_TOKENS[lhs_index]
        rhs_nonce = NONCE_TOKENS[rhs_index]
        swapped = bool(values[7] & 1)

        lhs_definition = ("<def>", lhs_nonce, "means", lhs_color, lhs_shape, "<sep>")
        rhs_definition = ("<def>", rhs_nonce, "means", rhs_color, rhs_shape, "<sep>")
        definitions = rhs_definition + lhs_definition if swapped else lhs_definition + rhs_definition
        query = ("<query>", "the", lhs_nonce, "is", relation, "the", rhs_nonce, "<sep>")
        answer = ("<answer>", lhs_color, lhs_shape, relation, rhs_color, rhs_shape, "<eos>")
        tokens = ("<bos>",) + definitions + query + answer
        answer_start = len(tokens) - len(answer) + 1
        if len(tokens) > MAX_SEQUENCE_LENGTH:
            raise RuntimeError("L0 sequence exceeded the locked maximum")
        return Episode(
            split=split,
            ordinal=ordinal,
            lhs_nonce=lhs_nonce,
            rhs_nonce=rhs_nonce,
            lhs_color=lhs_color,
            lhs_shape=lhs_shape,
            relation=relation,
            rhs_color=rhs_color,
            rhs_shape=rhs_shape,
            definition_order_swapped=swapped,
            tokens=tokens,
            answer_start=answer_start,
        )
    raise RuntimeError("could not materialize requested L0 split episode")


def dataset_fingerprint(split: Split, count: int) -> str:
    if type(count) is not int or count <= 0:
        raise ValueError("fingerprint count must be positive")
    digest = hashlib.sha256()
    for ordinal in range(count):
        episode = make_episode(split, ordinal)
        digest.update(ordinal.to_bytes(8, "little"))
        digest.update(bytes(episode.token_ids))
    return digest.hexdigest()


def transformer_parameter_count(config: TransformerConfig = TransformerConfig()) -> int:
    if config.d_model % config.heads:
        raise ValueError("transformer width must divide evenly across heads")
    d = config.d_model
    ff = config.feed_forward
    embeddings = (config.vocab_size + config.max_sequence_length) * d
    per_layer = 4 * d * d + 2 * d * ff + 9 * d + ff
    final_norm = 2 * d
    tied_lm_bias = config.vocab_size
    return embeddings + config.layers * per_layer + final_norm + tied_lm_bias


def population_worker_core_parameter_count(config: PopulationConfig = PopulationConfig()) -> int:
    d = config.worker_width
    ff = config.worker_feed_forward
    route = config.router_dim
    initializer = d * d + d
    shared_gru = 6 * d * d + 3 * d
    message_encoder = d * d + d
    shared_router = 2 * d * route + 2 * route
    shared_feed_forward = 2 * d * ff + ff + d
    return initializer + shared_gru + message_encoder + shared_router + shared_feed_forward


def population_parameter_count(config: PopulationConfig = PopulationConfig()) -> int:
    token = config.token_width
    worker = config.worker_width
    encoder = config.lexical_encoder_width
    decoder = config.lexical_decoder_width
    embeddings = (config.vocab_size + config.max_sequence_length) * token
    lexical_encoder = token * encoder + encoder + encoder * worker + worker
    worker_core = population_worker_core_parameter_count(config)
    lexical_decoder = worker * decoder + decoder + decoder * token + token
    final_norm = 2 * token
    tied_lm_bias = config.vocab_size
    return embeddings + lexical_encoder + worker_core + lexical_decoder + final_norm + tied_lm_bias


def relative_parameter_delta() -> float:
    baseline = transformer_parameter_count()
    population = population_parameter_count()
    return abs(baseline - population) / baseline


def validate_protocol() -> dict[str, object]:
    transformer = transformer_parameter_count()
    population = population_parameter_count()
    worker_core = population_worker_core_parameter_count()
    checks = {
        "vocabulary_size_is_64": len(VOCABULARY) == 64,
        "vocabulary_is_unique": len(set(VOCABULARY)) == len(VOCABULARY),
        "training_worker_count_is_evaluated": TRAIN_WORKERS in EVAL_WORKERS,
        "worker_counts_are_strictly_increasing": tuple(sorted(set(EVAL_WORKERS))) == EVAL_WORKERS,
        "parameter_budget_is_matched": relative_parameter_delta() <= PARAMETER_TOLERANCE_FRACTION,
        "worker_core_is_small": worker_core / population < 0.05,
        "no_per_worker_parameter_term": population_parameter_count(PopulationConfig(training_workers=256)) == population,
        "sequence_fits": len(make_episode("train", 0).tokens) <= MAX_SEQUENCE_LENGTH,
        "splits_are_semantically_disjoint": all(
            _semantic_bucket(
                episode.lhs_color,
                episode.lhs_shape,
                episode.relation,
                episode.rhs_color,
                episode.rhs_shape,
            ) in _SPLIT_RANGES[split]
            for split in ("train", "validation", "test")
            for episode in (make_episode(split, 0), make_episode(split, 1))
        ),
    }
    return {
        "status": STATUS,
        "version": VERSION,
        "transformer_parameters": transformer,
        "population_parameters": population,
        "population_worker_core_parameters": worker_core,
        "relative_parameter_delta": relative_parameter_delta(),
        "training_workers": TRAIN_WORKERS,
        "evaluation_workers": list(EVAL_WORKERS),
        "checks": checks,
        "valid": all(checks.values()),
    }
