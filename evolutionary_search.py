# Evolutionary search over a simplified mov-only environment for discovering
# initial mössö configurations.
#
# Each candidate consists of:
#   - three free values for the mutable memory tape, and
#   - an initial mutable instruction block represented as up to five
#     destination-source pairs.
#
# In the search representation, memory-to-memory pairs are allowed. When such
# pairs are translated into simulatable instructions, they are implemented
# through eax as an intermediate register.
#
# The search evaluates candidates in a Numba simulation of the simplified
# environment and logs:
#   - all unique tried initial conditions
#   - all unique perfect initial conditions
#
# The run stops after a fixed number of evaluated candidates


from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import List, Optional, TextIO, Tuple

import numpy as np

from numba_simulation import (
    BIAS_OPERANDS,
    DATA0,
    DATA1,
    EAX_SYMBOL,
    EBX_SYMBOL,
    FREE0,
    FREE1,
    FREE2,
    INPUT,
    OP0,
    OP1,
    OPERAND_VOCAB_SIZE,
    OUTPUT,
    STOP_ID,
    TAPE_DWORDS,
    run_batch_env_4runs,
)


# =============================================================================
# Configuration
# =============================================================================

# !!!!!!!!!!!!!!!!!!!!
# These were the first values tried, and they were already good enough.
# The search could likely be simplified and tuned better.
# !!!!!!!!!!!!!!!!!!!!


CONFIG = {
    "TARGET_TRIES": 1_768_800,
    "RANDOM_SEED": 1,

    "POPULATION_SIZE": 220,
    "ELITE_COUNT": 28,
    "CHILDREN_PER_ELITE": 8,
    "IMMIGRANTS_PER_GENERATION": 40,

    "N_FREE_VALUES": 3,
    "N_INITIAL_PAIRS": 5,
    "N_GENERATIONS": 3,

    "P_MUTATE_INITIAL_BLOCK": 0.82,
    "P_MUTATE_FREE_VALUES": 0.55,
    "INITIAL_BLOCK_MUTATION_MAX_CHANGES": 4,
    "FREE_VALUE_MUTATION_MAX_CHANGES": 3,
    "P_PAIR_REPLACE": 0.72,
    "P_CROSSOVER": 0.55,

    "NOVELTY_ARCHIVE_MAX": 5000,
    "NOVELTY_K": 12,
    "NOVELTY_WEIGHT": 0.15,

    # reward weights
    "W_PIPELINE": 15.0,
    "W_SOURCE_USE": 1.5,
    "W_ROOT_DESTINATION": 1.5,
    "W_FREE_COVERAGE": 1.5,
    "W_SCENARIO_DIFFERENCE": 10.0,
    "W_CORRECTNESS": 10.0,

    # fitness shaping
    "RAW_REWARD_WEIGHT": 1.0,

    # discovery pressure
    # new correct initial mutable instruction block patterns matter more
    # than merely new free-value settings
    "DISCOVERY_BONUS_NEW_INITIAL_BLOCK": 1_000_000.0,
    "DISCOVERY_BONUS_NEW_SEQUENCE": 0.0,
    "CORRECT_KEEP_BONUS": 2_000.0,

    # reinjection
    "CORRECT_BLOCK_ARCHIVE_INJECT_PER_GENERATION": 16,
    "CORRECT_SEQUENCE_ARCHIVE_INJECT_PER_GENERATION": 8,

    # cap stored example lists used for reinjection
    "CORRECT_EXAMPLE_ARCHIVE_MAX": 20_000,
}

TRIED_INITIAL_CONDITIONS_LOG_PATH = "tried_initial_conditions.txt"
CORRECT_INITIAL_CONDITIONS_LOG_PATH = "correct_initial_conditions.txt"


# =============================================================================
# Formatting helpers for logs
# =============================================================================
def format_symbol(symbol: int) -> str:
    names = {
        FREE0: "free0",
        FREE1: "free1",
        FREE2: "free2",
        INPUT: "input",
        OUTPUT: "output",
        DATA0: "data0",
        DATA1: "data1",
        OP0: "op0",
        OP1: "op1",
        EAX_SYMBOL: "eax",
        EBX_SYMBOL: "ebx",
        STOP_ID: "STOP",
    }
    return names.get(int(symbol), f"sym{int(symbol)}")


def format_initial_pair(dst: int, src: int) -> str:
    if int(dst) == STOP_ID:
        return "STOP"
    text = f"{format_symbol(int(dst))} <- {format_symbol(int(src))}"
    if int(dst) <= OP1 and int(src) <= OP1:
        text += "  [via eax]"
    return text


def format_appended_pair(dst: int, src: int) -> str:
    return f"MOV {format_symbol(int(dst))}, {format_symbol(int(src))}"


def format_tape_row(tape: np.ndarray) -> str:
    x = [int(v) for v in tape.tolist()]
    return (
        f"free=[{x[0]},{x[1]},{x[2]}] "
        f"input={x[3]} output={x[4]} "
        f"data=[{x[5]},{x[6]}] "
        f"op=[{x[7]},{x[8]}] "
        f"regsym=[{x[9]},{x[10]}]"
    )


def format_source_mask(mask: int) -> str:
    names: List[str] = []
    if mask & 1:
        names.append("input")
    if mask & 2:
        names.append("data0")
    if mask & 4:
        names.append("data1")
    return ",".join(names) if names else "-"


def format_root_destination_mask(mask: int) -> str:
    names: List[str] = []
    if mask & 1:
        names.append("output")
    if mask & 2:
        names.append("op0")
    if mask & 4:
        names.append("op1")
    return ",".join(names) if names else "-"


def format_free_mask(mask: int) -> str:
    names: List[str] = []
    if mask & 1:
        names.append("free0")
    if mask & 2:
        names.append("free1")
    if mask & 4:
        names.append("free2")
    return ",".join(names) if names else "-"


def format_target_pipeline_mask(mask: int) -> str:
    names: List[str] = []
    if mask & (1 << 0):
        names.append("free0")
    if mask & (1 << 1):
        names.append("free1")
    if mask & (1 << 2):
        names.append("free2")
    if mask & (1 << 3):
        names.append("input")
    if mask & (1 << 5):
        names.append("data0")
    if mask & (1 << 6):
        names.append("data1")
    return ",".join(names) if names else "-"


def format_generated_initial_pair_line(dst: int, src: int) -> str:
    if int(dst) == STOP_ID:
        return "STOP"
    shortcut = ""
    if int(dst) <= OP1 and int(src) <= OP1:
        shortcut = "  [shortcut: mem->mem via eax]"
    return (
        f"{int(dst)} <- {int(src)}    "
        f"({format_symbol(int(dst))} <- {format_symbol(int(src))}){shortcut}"
    )


def concrete_initial_x86_lines(initial_pairs: List[Tuple[int, int]]) -> List[str]:
    lines: List[str] = []
    for i, (dst, src) in enumerate(initial_pairs):
        if int(dst) == STOP_ID:
            lines.append(f"  [{i:02d}] STOP")
            break

        dst_i = int(dst)
        src_i = int(src)

        if dst_i <= OP1 and src_i <= OP1:
            lines.append(f"  [{i:02d}.0] mov eax, {format_symbol(src_i)}")
            lines.append(f"  [{i:02d}.1] mov {format_symbol(dst_i)}, eax")
        else:
            lines.append(f"  [{i:02d}] mov {format_symbol(dst_i)}, {format_symbol(src_i)}")

    if not lines:
        lines.append("  (none)")
    return lines


def concrete_appended_x86_line(dst: int, src: int) -> str:
    return f"mov {format_symbol(int(dst))}, {format_symbol(int(src))}"


def write_run_block(
    log_file: TextIO,
    run_title: str,
    run_index: int,
    init_tape: np.ndarray,
    final_tape: np.ndarray,
    final_output: np.ndarray | np.uint32,
    history_length: np.ndarray | np.int32,
    history_dst_row: np.ndarray,
    history_src_row: np.ndarray,
    correct_run: np.ndarray | np.float32,
    source_used_mask: np.ndarray | np.uint8,
    root_destination_mask: np.ndarray | np.uint8,
    free_source_mask: np.ndarray | np.uint8,
    free_destination_mask: np.ndarray | np.uint8,
    target_pipeline_mask: np.ndarray | np.uint16,
) -> None:
    log_file.write(f"{run_title} (run_idx={run_index})\n")
    log_file.write(f"  init_tape:  {format_tape_row(init_tape)}\n")
    log_file.write(f"  final_tape: {format_tape_row(final_tape)}\n")
    log_file.write(f"  final_out: {int(final_output)}\n")
    log_file.write(f"  correct_run: {float(correct_run):.10f}\n")
    log_file.write(f"  src_used: {format_source_mask(int(source_used_mask))}\n")
    log_file.write(f"  root_dst: {format_root_destination_mask(int(root_destination_mask))}\n")
    log_file.write(f"  free_src: {format_free_mask(int(free_source_mask))}\n")
    log_file.write(f"  free_dst: {format_free_mask(int(free_destination_mask))}\n")
    log_file.write(f"  target_pipe: {format_target_pipeline_mask(int(target_pipeline_mask))}\n")
    log_file.write("  appended_rules_generated:\n")
    if int(history_length) <= 0:
        log_file.write("    (none)\n")
    else:
        for j in range(int(history_length)):
            dst = int(history_dst_row[j])
            src = int(history_src_row[j])
            log_file.write(f"    [{j:02d}] {format_appended_pair(dst, src)}\n")
    log_file.write("  appended_rules_x86:\n")
    if int(history_length) <= 0:
        log_file.write("    (none)\n")
    else:
        for j in range(int(history_length)):
            dst = int(history_dst_row[j])
            src = int(history_src_row[j])
            log_file.write(f"    [{j:02d}] {concrete_appended_x86_line(dst, src)}\n")
    log_file.write("\n")


def append_candidate_record(log_file: TextIO, record_title: str, record_index: int, candidate: "Candidate") -> None:
    assert candidate.payload is not None
    payload = candidate.payload

    log_file.write("=" * 80 + "\n")
    log_file.write(f"{record_title} #{record_index}\n")
    log_file.write("[generated]\n")
    log_file.write(f"free_values: {candidate.free_values}\n")
    log_file.write("initial_mutable_instruction_block_pairs:\n")
    printed_any = False
    for i, (dst, src) in enumerate(candidate.initial_pairs):
        log_file.write(f"  [{i:02d}] {format_generated_initial_pair_line(int(dst), int(src))}\n")
        printed_any = True
        if int(dst) == STOP_ID:
            break
    if not printed_any:
        log_file.write("  (none)\n")

    log_file.write("\n[concrete_initial_x86]\n")
    for line in concrete_initial_x86_lines(candidate.initial_pairs):
        log_file.write(line + "\n")

    log_file.write("\n[runs]\n")
    run_titles = [
        "data_case_0 / input=0",
        "data_case_0 / input=1",
        "data_case_1 / input=0",
        "data_case_1 / input=1",
    ]
    for run_index in range(4):
        write_run_block(
            log_file=log_file,
            run_title=run_titles[run_index],
            run_index=run_index,
            init_tape=payload["init_tapes_4"][run_index],
            final_tape=payload["final_tapes_4"][run_index],
            final_output=payload["final_outs_4"][run_index],
            history_length=payload["hist_len_4"][run_index],
            history_dst_row=payload["hist_dst_4"][run_index],
            history_src_row=payload["hist_src_4"][run_index],
            correct_run=payload["correct_run_4"][run_index],
            source_used_mask=payload["src_used_mask_4"][run_index],
            root_destination_mask=payload["root_dst_mask_4"][run_index],
            free_source_mask=payload["free_src_mask_4"][run_index],
            free_destination_mask=payload["free_dst_mask_4"][run_index],
            target_pipeline_mask=payload["target_pipe_mask_4"][run_index],
        )

    log_file.write("[reward]\n")
    log_file.write(f"raw_reward={candidate.raw_reward:.10f}\n")
    log_file.write(f"fitness={candidate.fitness:.10f}\n")
    log_file.write(f"novelty={candidate.novelty:.10f}\n")
    log_file.write(f"correct_keep_bonus={candidate.correct_keep_bonus:.10f}\n")
    log_file.write(f"unseen_initial_block_bonus={candidate.unseen_initial_block_bonus:.10f}\n")
    log_file.write(f"unseen_sequence_bonus={candidate.unseen_sequence_bonus:.10f}\n")
    log_file.write(f"case_pipeline={float(payload['case_pipeline']):.10f}\n")
    log_file.write(f"case_src_use={float(payload['case_src_use']):.10f}\n")
    log_file.write(f"case_root_dst={float(payload['case_root_dst']):.10f}\n")
    log_file.write(f"case_free_cov={float(payload['case_free_cov']):.10f}\n")
    log_file.write(f"case_hist_diff={float(payload['case_hist_diff']):.10f}\n")
    log_file.write(f"case_out_diff={float(payload['case_out_diff']):.10f}\n")
    log_file.write(f"case_scen_diff={float(payload['case_scen_diff']):.10f}\n")
    log_file.write(f"case_correct={float(payload['case_correct']):.10f}\n")

    log_file.write("\n[correctness]\n")
    log_file.write(f"correct_case={candidate.correct_case:.10f}\n")
    log_file.write(f"correct_run_4={[float(x) for x in payload['correct_run_4'].tolist()]}\n")
    log_file.write(f"is_perfect={int(bool(payload['is_perfect']))}\n")
    log_file.write("\n")


# =============================================================================
# Candidate representation
# =============================================================================
@dataclass
class Candidate:
    free_values: List[int]
    initial_pairs: List[Tuple[int, int]]

    fitness: float = -1e18
    raw_reward: float = -1e18
    novelty: float = 0.0
    correct_case: float = 0.0
    unseen_initial_block_bonus: float = 0.0
    unseen_sequence_bonus: float = 0.0
    correct_keep_bonus: float = 0.0
    payload: Optional[dict] = None


def clone_candidate(candidate: Candidate) -> Candidate:
    return Candidate(
        free_values=candidate.free_values[:],
        initial_pairs=candidate.initial_pairs[:],
        fitness=candidate.fitness,
        raw_reward=candidate.raw_reward,
        novelty=candidate.novelty,
        correct_case=candidate.correct_case,
        unseen_initial_block_bonus=candidate.unseen_initial_block_bonus,
        unseen_sequence_bonus=candidate.unseen_sequence_bonus,
        correct_keep_bonus=candidate.correct_keep_bonus,
        payload=None
        if candidate.payload is None
        else {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in candidate.payload.items()},
    )


# =============================================================================
# Reward terms
# =============================================================================
def correctness_reward_per_run(tape_templates_u32_4runs: np.ndarray, final_out_u32: np.ndarray) -> np.ndarray:
    input_values = tape_templates_u32_4runs[:, INPUT].astype(np.uint32)
    data0_values = tape_templates_u32_4runs[:, DATA0].astype(np.uint32)
    data1_values = tape_templates_u32_4runs[:, DATA1].astype(np.uint32)

    expected = np.where(input_values == np.uint32(0), data0_values, data1_values)
    return (final_out_u32.astype(np.uint32) == expected).astype(np.float32)


def correctness_reward_per_case(correct_run_4runs: np.ndarray, batch_size: int) -> np.ndarray:
    x = correct_run_4runs.reshape(batch_size, 2, 2)
    per_data_case = x.mean(axis=2)
    return per_data_case.min(axis=1).astype(np.float32)


def scenario_difference_reward_per_case(
    history_length_4runs: np.ndarray,
    history_dst_4runs: np.ndarray,
    history_src_4runs: np.ndarray,
    final_out_4runs: np.ndarray,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    history_difference = np.zeros((batch_size,), dtype=np.float32)
    output_difference = np.zeros((batch_size,), dtype=np.float32)
    total = np.zeros((batch_size,), dtype=np.float32)

    for i in range(batch_size):
        history_by_data_case = np.zeros((2,), dtype=np.float32)
        output_by_data_case = np.zeros((2,), dtype=np.float32)

        for data_case in range(2):
            a = 4 * i + 2 * data_case + 0
            b = 4 * i + 2 * data_case + 1

            same_history = int(history_length_4runs[a]) == int(history_length_4runs[b])
            if same_history:
                history_len = int(history_length_4runs[a])
                for j in range(history_len):
                    if (
                        int(history_dst_4runs[a, j]) != int(history_dst_4runs[b, j])
                        or int(history_src_4runs[a, j]) != int(history_src_4runs[b, j])
                    ):
                        same_history = False
                        break

            history_by_data_case[data_case] = np.float32(0.0 if same_history else 1.0)
            output_by_data_case[data_case] = np.float32(
                1.0 if int(final_out_4runs[a]) != int(final_out_4runs[b]) else 0.0
            )

        history_difference[i] = history_by_data_case.mean()
        output_difference[i] = output_by_data_case.mean()
        total[i] = np.float32((history_difference[i] + output_difference[i]) / 2.0)

    return history_difference, output_difference, total


def source_usage_reward_per_case(source_used_mask_4runs: np.ndarray, batch_size: int) -> np.ndarray:
    out = np.zeros((batch_size,), dtype=np.float32)
    for i in range(batch_size):
        per_data_case = np.zeros((2,), dtype=np.float32)
        for data_case in range(2):
            mask = int(source_used_mask_4runs[4 * i + 2 * data_case + 0]) | int(
                source_used_mask_4runs[4 * i + 2 * data_case + 1]
            )
            count = int(bool(mask & 1)) + int(bool(mask & 2)) + int(bool(mask & 4))
            per_data_case[data_case] = count / 3.0
        out[i] = np.float32(per_data_case.mean())
    return out


def root_destination_reward_per_case(root_destination_mask_4runs: np.ndarray, batch_size: int) -> np.ndarray:
    out = np.zeros((batch_size,), dtype=np.float32)
    for i in range(batch_size):
        per_data_case = np.zeros((2,), dtype=np.float32)
        for data_case in range(2):
            mask = int(root_destination_mask_4runs[4 * i + 2 * data_case + 0]) | int(
                root_destination_mask_4runs[4 * i + 2 * data_case + 1]
            )
            count = int(bool(mask & 1)) + int(bool(mask & 2)) + int(bool(mask & 4))
            per_data_case[data_case] = count / 3.0
        out[i] = np.float32(per_data_case.mean())
    return out


def free_coverage_reward_per_case(
    free_source_mask_4runs: np.ndarray,
    free_destination_mask_4runs: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    out = np.zeros((batch_size,), dtype=np.float32)
    for i in range(batch_size):
        per_data_case = np.zeros((2,), dtype=np.float32)
        for data_case in range(2):
            src_union = int(free_source_mask_4runs[4 * i + 2 * data_case + 0]) | int(
                free_source_mask_4runs[4 * i + 2 * data_case + 1]
            )
            dst_union = int(free_destination_mask_4runs[4 * i + 2 * data_case + 0]) | int(
                free_destination_mask_4runs[4 * i + 2 * data_case + 1]
            )

            src_cov = (int(bool(src_union & 1)) + int(bool(src_union & 2)) + int(bool(src_union & 4))) / 3.0
            dst_cov = (int(bool(dst_union & 1)) + int(bool(dst_union & 2)) + int(bool(dst_union & 4))) / 3.0
            per_data_case[data_case] = src_cov + dst_cov

        out[i] = np.float32(per_data_case.mean())
    return out


def target_pipeline_reward_per_case(target_pipeline_mask_4runs: np.ndarray, batch_size: int) -> np.ndarray:
    out = np.zeros((batch_size,), dtype=np.float32)
    relevant_bits = [0, 1, 2, 3, 5, 6]

    for i in range(batch_size):
        per_data_case = np.zeros((2,), dtype=np.float32)
        for data_case in range(2):
            union_mask = int(target_pipeline_mask_4runs[4 * i + 2 * data_case + 0]) | int(
                target_pipeline_mask_4runs[4 * i + 2 * data_case + 1]
            )

            count = 0
            for bit in relevant_bits:
                if union_mask & (1 << bit):
                    count += 1

            per_data_case[data_case] = count / 6.0

        out[i] = np.float32(per_data_case.mean())

    return out


# =============================================================================
# Data generation for evaluation
# =============================================================================
def sample_two_different_data_pairs(rng: random.Random) -> np.ndarray:
    values = rng.sample(list(range(2, 11)), 4)
    pair0 = [values[0], values[1]]
    pair1 = [values[2], values[3]]
    return np.array([pair0, pair1], dtype=np.int32)


def build_case_tapes_4runs(free_values: List[int], data_pairs: np.ndarray) -> np.ndarray:
    tapes = np.zeros((4, TAPE_DWORDS), dtype=np.uint32)

    for data_case in range(2):
        data0_value = int(data_pairs[data_case, 0])
        data1_value = int(data_pairs[data_case, 1])

        run0 = 2 * data_case + 0
        run1 = 2 * data_case + 1

        for run_index, input_value in [(run0, 0), (run1, 1)]:
            tape = np.zeros((TAPE_DWORDS,), dtype=np.uint32)
            tape[FREE0] = np.uint32(free_values[0])
            tape[FREE1] = np.uint32(free_values[1])
            tape[FREE2] = np.uint32(free_values[2])
            tape[INPUT] = np.uint32(input_value)
            tape[OUTPUT] = np.uint32(0)
            tape[DATA0] = np.uint32(data0_value)
            tape[DATA1] = np.uint32(data1_value)
            tape[OP0] = np.uint32(0)
            tape[OP1] = np.uint32(0)
            tape[EAX_SYMBOL] = np.uint32(0)
            tape[EBX_SYMBOL] = np.uint32(0)
            tapes[run_index] = tape

    return tapes


# =============================================================================
# Search-space operations
# =============================================================================
def sample_biased_operand(rng: random.Random) -> int:
    if rng.random() < 0.8:
        return rng.choice(BIAS_OPERANDS)
    return rng.randrange(0, OPERAND_VOCAB_SIZE)


def sample_initial_pair(rng: random.Random) -> Tuple[int, int]:
    dst = sample_biased_operand(rng)
    src = sample_biased_operand(rng)
    return dst, src


def canonicalize_initial_pairs(initial_pairs: List[Tuple[int, int]], n_pairs: int) -> List[Tuple[int, int]]:
    out = list(initial_pairs[:n_pairs])
    if len(out) < n_pairs:
        out.extend([(STOP_ID, 0)] * (n_pairs - len(out)))

    stop_seen = False
    fixed: List[Tuple[int, int]] = []
    for dst, src in out:
        if stop_seen:
            fixed.append((STOP_ID, 0))
            continue
        if int(dst) == STOP_ID:
            fixed.append((STOP_ID, 0))
            stop_seen = True
        else:
            fixed.append((int(dst) % OPERAND_VOCAB_SIZE, int(src) % OPERAND_VOCAB_SIZE))

    return fixed


def random_candidate(rng: random.Random, cfg: dict) -> Candidate:
    n_pairs = int(cfg["N_INITIAL_PAIRS"])
    active_length = rng.randrange(0, n_pairs + 1)

    initial_pairs: List[Tuple[int, int]] = []
    for i in range(n_pairs):
        if i < active_length:
            initial_pairs.append(sample_initial_pair(rng))
        elif i == active_length:
            initial_pairs.append((STOP_ID, 0))
        else:
            initial_pairs.append((STOP_ID, 0))

    initial_pairs = canonicalize_initial_pairs(initial_pairs, n_pairs)

    return Candidate(
        free_values=[rng.randrange(0, OPERAND_VOCAB_SIZE) for _ in range(cfg["N_FREE_VALUES"])],
        initial_pairs=initial_pairs,
    )


def mutate_candidate(candidate: Candidate, rng: random.Random, cfg: dict) -> Candidate:
    free_values = candidate.free_values[:]
    initial_pairs = candidate.initial_pairs[:]
    n_pairs = int(cfg["N_INITIAL_PAIRS"])

    if rng.random() < cfg["P_MUTATE_FREE_VALUES"]:
        n_changes = rng.randrange(1, int(cfg["FREE_VALUE_MUTATION_MAX_CHANGES"]) + 1)
        for _ in range(n_changes):
            idx = rng.randrange(len(free_values))
            free_values[idx] = rng.randrange(0, OPERAND_VOCAB_SIZE)

    if rng.random() < cfg["P_MUTATE_INITIAL_BLOCK"]:
        n_changes = rng.randrange(1, int(cfg["INITIAL_BLOCK_MUTATION_MAX_CHANGES"]) + 1)
        for _ in range(n_changes):
            idx = rng.randrange(n_pairs)

            if rng.random() < cfg["P_PAIR_REPLACE"]:
                if rng.random() < 0.15:
                    initial_pairs[idx] = (STOP_ID, 0)
                else:
                    initial_pairs[idx] = sample_initial_pair(rng)
            else:
                dst, src = initial_pairs[idx]
                which = rng.randrange(2)
                if which == 0:
                    if rng.random() < 0.15:
                        dst = STOP_ID
                    else:
                        dst = sample_biased_operand(rng)
                else:
                    src = sample_biased_operand(rng)
                initial_pairs[idx] = (dst, src)

    initial_pairs = canonicalize_initial_pairs(initial_pairs, n_pairs)
    return Candidate(free_values=free_values, initial_pairs=initial_pairs)


def crossover(a: Candidate, b: Candidate, rng: random.Random, cfg: dict) -> Candidate:
    cut_pairs = rng.randrange(1, int(cfg["N_INITIAL_PAIRS"]))
    cut_free_values = rng.randrange(1, int(cfg["N_FREE_VALUES"]))

    child_pairs = a.initial_pairs[:cut_pairs] + b.initial_pairs[cut_pairs:]
    child_free_values = a.free_values[:cut_free_values] + b.free_values[cut_free_values:]

    child_pairs = canonicalize_initial_pairs(child_pairs, int(cfg["N_INITIAL_PAIRS"]))
    return Candidate(free_values=child_free_values, initial_pairs=child_pairs)


# =============================================================================
# Candidate keys and novelty
# =============================================================================
def candidate_sequence_key(candidate: Candidate) -> Tuple[int, ...]:
    parts: List[int] = []
    parts.extend(int(x) for x in candidate.free_values)
    parts.append(-12345)

    stopped = False
    for dst, src in candidate.initial_pairs:
        if stopped:
            break
        parts.append(int(dst))
        if int(dst) == STOP_ID:
            stopped = True
            break
        parts.append(int(src))

    if not stopped:
        parts.append(-99999)
    return tuple(parts)


def candidate_initial_block_key(candidate: Candidate) -> Tuple[int, ...]:
    parts: List[int] = []
    stopped = False
    for dst, src in candidate.initial_pairs:
        if stopped:
            break
        parts.append(int(dst))
        if int(dst) == STOP_ID:
            stopped = True
            break
        parts.append(int(src))

    if not stopped:
        parts.append(-99999)
    return tuple(parts)


def history_fingerprint(history: List[Tuple[int, int]]) -> str:
    text = ";".join(f"{a},{b}" for a, b in history)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def novelty_score(history: List[Tuple[int, int]], archive_sets: List[set], k: int) -> float:
    if not history or not archive_sets:
        return 0.0
    history_set = set(history)
    distances = []
    for other in archive_sets:
        union = history_set | other
        if not union:
            distance = 0.0
        else:
            distance = 1.0 - (len(history_set & other) / len(union))
        distances.append(distance)
    distances.sort(reverse=True)
    kk = min(k, len(distances))
    return float(sum(distances[:kk]) / max(1, kk))


# =============================================================================
# Candidate evaluation
# =============================================================================
def candidate_initial_pair_arrays(candidate: Candidate, cfg: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_pairs = int(cfg["N_INITIAL_PAIRS"])
    initial_dst = np.zeros((1, n_pairs), dtype=np.int32)
    initial_src = np.zeros((1, n_pairs), dtype=np.int32)
    initial_len = np.zeros((1,), dtype=np.int32)

    k = 0
    for dst, src in candidate.initial_pairs:
        if int(dst) == STOP_ID:
            break
        initial_dst[0, k] = np.int32(int(dst))
        initial_src[0, k] = np.int32(int(src))
        k += 1

    initial_len[0] = np.int32(k)
    return initial_dst, initial_src, initial_len


def evaluate_candidate(
    candidate: Candidate,
    rng: random.Random,
    cfg: dict,
    behavior_archive_sets: List[set],
    unique_correct_sequence_keys: set,
    unique_correct_initial_block_keys: set,
) -> None:
    data_pairs = sample_two_different_data_pairs(rng)
    tapes_4runs = build_case_tapes_4runs(candidate.free_values, data_pairs)

    initial_dst_case, initial_src_case, initial_len_case = candidate_initial_pair_arrays(candidate, cfg)
    initial_dst_repeated = np.repeat(initial_dst_case, 4, axis=0)
    initial_src_repeated = np.repeat(initial_src_case, 4, axis=0)
    initial_len_repeated = np.repeat(initial_len_case, 4, axis=0)

    (
        history_length_np,
        history_dst_np,
        history_src_np,
        final_out_np,
        final_tape_np,
        source_used_mask_np,
        root_destination_mask_np,
        free_source_mask_np,
        free_destination_mask_np,
        target_pipeline_mask_np,
    ) = run_batch_env_4runs(
        tapes_4runs,
        initial_dst_repeated,
        initial_src_repeated,
        initial_len_repeated,
        int(cfg["N_GENERATIONS"]),
    )

    correct_run = correctness_reward_per_run(tapes_4runs, final_out_np)

    pipeline_case = target_pipeline_reward_per_case(target_pipeline_mask_np, batch_size=1)
    source_use_case = source_usage_reward_per_case(source_used_mask_np, batch_size=1)
    root_destination_case = root_destination_reward_per_case(root_destination_mask_np, batch_size=1)
    free_coverage_case = free_coverage_reward_per_case(
        free_source_mask_np, free_destination_mask_np, batch_size=1
    )
    correct_case = correctness_reward_per_case(correct_run, batch_size=1)
    history_diff_case, output_diff_case, scenario_diff_case = scenario_difference_reward_per_case(
        history_length_np, history_dst_np, history_src_np, final_out_np, batch_size=1
    )

    raw_reward = float(
        float(cfg["RAW_REWARD_WEIGHT"])
        * (
            float(cfg["W_PIPELINE"]) * float(pipeline_case[0])
            + float(cfg["W_SOURCE_USE"]) * float(source_use_case[0])
            + float(cfg["W_ROOT_DESTINATION"]) * float(root_destination_case[0])
            + float(cfg["W_FREE_COVERAGE"]) * float(free_coverage_case[0])
            + float(cfg["W_SCENARIO_DIFFERENCE"]) * float(scenario_diff_case[0])
            + float(cfg["W_CORRECTNESS"]) * float(correct_case[0])
        )
    )

    flat_history: List[Tuple[int, int]] = []
    for run_index in range(4):
        history_len = int(history_length_np[run_index])
        for j in range(history_len):
            flat_history.append((int(history_dst_np[run_index, j]), int(history_src_np[run_index, j])))

    novelty = float(cfg["NOVELTY_WEIGHT"]) * novelty_score(
        flat_history, behavior_archive_sets, int(cfg["NOVELTY_K"])
    )

    sequence_key = candidate_sequence_key(candidate)
    initial_block_key = candidate_initial_block_key(candidate)

    is_perfect = float(correct_case[0]) >= 0.999999
    unseen_sequence = is_perfect and (sequence_key not in unique_correct_sequence_keys)
    unseen_initial_block = is_perfect and (initial_block_key not in unique_correct_initial_block_keys)

    unseen_initial_block_bonus = (
        float(cfg["DISCOVERY_BONUS_NEW_INITIAL_BLOCK"]) if unseen_initial_block else 0.0
    )
    unseen_sequence_bonus = float(cfg["DISCOVERY_BONUS_NEW_SEQUENCE"]) if unseen_sequence else 0.0
    correct_keep_bonus = float(cfg["CORRECT_KEEP_BONUS"]) if is_perfect else 0.0

    fitness = raw_reward + novelty + correct_keep_bonus + unseen_initial_block_bonus + unseen_sequence_bonus

    candidate.raw_reward = raw_reward
    candidate.novelty = novelty
    candidate.correct_case = float(correct_case[0])
    candidate.unseen_initial_block_bonus = unseen_initial_block_bonus
    candidate.unseen_sequence_bonus = unseen_sequence_bonus
    candidate.correct_keep_bonus = correct_keep_bonus
    candidate.fitness = fitness

    candidate.payload = {
        "data_pairs": data_pairs.copy(),
        "init_tapes_4": tapes_4runs.copy(),
        "final_tapes_4": final_tape_np.copy(),
        "final_outs_4": final_out_np.copy(),
        "hist_len_4": history_length_np.copy(),
        "hist_dst_4": history_dst_np.copy(),
        "hist_src_4": history_src_np.copy(),
        "correct_run_4": correct_run.copy(),
        "src_used_mask_4": source_used_mask_np.copy(),
        "root_dst_mask_4": root_destination_mask_np.copy(),
        "free_src_mask_4": free_source_mask_np.copy(),
        "free_dst_mask_4": free_destination_mask_np.copy(),
        "target_pipe_mask_4": target_pipeline_mask_np.copy(),
        "case_pipeline": float(pipeline_case[0]),
        "case_src_use": float(source_use_case[0]),
        "case_root_dst": float(root_destination_case[0]),
        "case_free_cov": float(free_coverage_case[0]),
        "case_hist_diff": float(history_diff_case[0]),
        "case_out_diff": float(output_diff_case[0]),
        "case_scen_diff": float(scenario_diff_case[0]),
        "case_correct": float(correct_case[0]),
        "flat_hist": flat_history[:],
        "sequence_key": sequence_key,
        "initial_block_key": initial_block_key,
        "is_perfect": is_perfect,
        "unseen_sequence": unseen_sequence,
        "unseen_initial_block": unseen_initial_block,
    }


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    cfg = CONFIG
    rng = random.Random(int(cfg["RANDOM_SEED"]))

    population: List[Candidate] = [random_candidate(rng, cfg) for _ in range(int(cfg["POPULATION_SIZE"]))]

    behavior_archive_fingerprints = set()
    behavior_archive_sets: List[set] = []

    unique_correct_sequence_keys = set()
    unique_correct_initial_block_keys = set()

    unique_correct_sequence_examples: List[Candidate] = []
    unique_correct_initial_block_examples: List[Candidate] = []

    logged_tried_sequence_keys = set()
    logged_perfect_sequence_keys = set()

    total_candidates_evaluated = 0
    total_unique_sequences_logged = 0
    total_unique_perfect_sequences_logged = 0
    generation_index = 0
    target_tries = int(cfg["TARGET_TRIES"])

    progress_checkpoints = {
        int(target_tries * fraction / 10): f"{fraction * 10}%"
        for fraction in range(1, 10)
    }
    progress_checkpoints[target_tries] = "100%"

    with open(TRIED_INITIAL_CONDITIONS_LOG_PATH, "w", encoding="utf-8", buffering=1) as tried_log, open(
        CORRECT_INITIAL_CONDITIONS_LOG_PATH, "w", encoding="utf-8", buffering=1
    ) as perfect_log:
        while total_candidates_evaluated < target_tries:
            for candidate in population:
                if total_candidates_evaluated >= target_tries:
                    break

                total_candidates_evaluated += 1

                evaluate_candidate(
                    candidate,
                    rng,
                    cfg,
                    behavior_archive_sets,
                    unique_correct_sequence_keys,
                    unique_correct_initial_block_keys,
                )

                assert candidate.payload is not None
                payload = candidate.payload

                sequence_key = payload["sequence_key"]

                if sequence_key not in logged_tried_sequence_keys:
                    logged_tried_sequence_keys.add(sequence_key)
                    total_unique_sequences_logged += 1
                    append_candidate_record(
                        tried_log,
                        "tried_initial_condition",
                        total_unique_sequences_logged,
                        candidate,
                    )

                if bool(payload["is_perfect"]) and sequence_key not in logged_perfect_sequence_keys:
                    logged_perfect_sequence_keys.add(sequence_key)
                    total_unique_perfect_sequences_logged += 1
                    append_candidate_record(
                        perfect_log,
                        "perfect_initial_condition",
                        total_unique_perfect_sequences_logged,
                        candidate,
                    )

                flat_history = payload["flat_hist"]
                if flat_history:
                    fingerprint = history_fingerprint(flat_history)
                    if fingerprint not in behavior_archive_fingerprints:
                        behavior_archive_fingerprints.add(fingerprint)
                        behavior_archive_sets.append(set(flat_history))
                        if len(behavior_archive_sets) > int(cfg["NOVELTY_ARCHIVE_MAX"]):
                            behavior_archive_sets.pop(0)

                if payload["is_perfect"]:
                    initial_block_key = payload["initial_block_key"]

                    if sequence_key not in unique_correct_sequence_keys:
                        unique_correct_sequence_keys.add(sequence_key)
                        unique_correct_sequence_examples.append(clone_candidate(candidate))
                        if len(unique_correct_sequence_examples) > int(cfg["CORRECT_EXAMPLE_ARCHIVE_MAX"]):
                            unique_correct_sequence_examples.pop(0)

                    if initial_block_key not in unique_correct_initial_block_keys:
                        unique_correct_initial_block_keys.add(initial_block_key)
                        unique_correct_initial_block_examples.append(clone_candidate(candidate))
                        if len(unique_correct_initial_block_examples) > int(cfg["CORRECT_EXAMPLE_ARCHIVE_MAX"]):
                            unique_correct_initial_block_examples.pop(0)

                if total_candidates_evaluated in progress_checkpoints:
                    label = progress_checkpoints[total_candidates_evaluated]
                    print(
                        f"{label} complete: tried={total_candidates_evaluated} "
                        f"unique_logged={total_unique_sequences_logged} "
                        f"unique_correct_sequences={len(unique_correct_sequence_keys)} "
                        f"unique_correct_initial_blocks={len(unique_correct_initial_block_keys)}"
                    )

            if total_candidates_evaluated >= target_tries:
                break

            population.sort(key=lambda cand: cand.fitness, reverse=True)
            elites = population[: int(cfg["ELITE_COUNT"])]

            next_population: List[Candidate] = []

            for elite in elites:
                next_population.append(Candidate(elite.free_values[:], elite.initial_pairs[:]))

            if unique_correct_initial_block_examples:
                n_inject = min(
                    int(cfg["CORRECT_BLOCK_ARCHIVE_INJECT_PER_GENERATION"]),
                    len(unique_correct_initial_block_examples),
                )
                inject_indices = rng.sample(range(len(unique_correct_initial_block_examples)), n_inject)
                for idx in inject_indices:
                    example = unique_correct_initial_block_examples[idx]
                    next_population.append(Candidate(example.free_values[:], example.initial_pairs[:]))

            if unique_correct_sequence_examples:
                n_inject = min(
                    int(cfg["CORRECT_SEQUENCE_ARCHIVE_INJECT_PER_GENERATION"]),
                    len(unique_correct_sequence_examples),
                )
                inject_indices = rng.sample(range(len(unique_correct_sequence_examples)), n_inject)
                for idx in inject_indices:
                    example = unique_correct_sequence_examples[idx]
                    next_population.append(Candidate(example.free_values[:], example.initial_pairs[:]))

            parent_pool = elites[:]
            if unique_correct_initial_block_examples:
                extra_pool = rng.sample(
                    unique_correct_initial_block_examples,
                    min(len(unique_correct_initial_block_examples), max(1, int(cfg["ELITE_COUNT"]))),
                )
                parent_pool = parent_pool + extra_pool

            if unique_correct_sequence_examples:
                extra_pool = rng.sample(
                    unique_correct_sequence_examples,
                    min(len(unique_correct_sequence_examples), max(1, int(cfg["ELITE_COUNT"]) // 2)),
                )
                parent_pool = parent_pool + extra_pool

            for elite in elites:
                for _ in range(int(cfg["CHILDREN_PER_ELITE"])):
                    if rng.random() < float(cfg["P_CROSSOVER"]) and len(parent_pool) >= 2:
                        mate = rng.choice(parent_pool)
                        child = crossover(elite, mate, rng, cfg)
                        child = mutate_candidate(child, rng, cfg)
                    else:
                        child = mutate_candidate(elite, rng, cfg)
                    next_population.append(child)

            for _ in range(int(cfg["IMMIGRANTS_PER_GENERATION"])):
                next_population.append(random_candidate(rng, cfg))

            while len(next_population) < int(cfg["POPULATION_SIZE"]):
                parent = rng.choice(parent_pool)
                next_population.append(mutate_candidate(parent, rng, cfg))

            population = next_population[: int(cfg["POPULATION_SIZE"])]
            generation_index += 1


    print("\nDONE")
    print(f"total candidates tried: {total_candidates_evaluated}")
    print(f"unique sequences logged to tried file: {total_unique_sequences_logged}")
    print(f"unique correct sequences logged to perfect file: {total_unique_perfect_sequences_logged}")
    print(f"unique sequences with correctness 1.0: {len(unique_correct_sequence_keys)}")
    print(f"unique initial mutable instruction block patterns among correctness 1.0: {len(unique_correct_initial_block_keys)}")
    print(f"saved tried initial conditions to: {TRIED_INITIAL_CONDITIONS_LOG_PATH}")
    print(f"saved perfect initial conditions to: {CORRECT_INITIAL_CONDITIONS_LOG_PATH}")


if __name__ == "__main__":
    main()
