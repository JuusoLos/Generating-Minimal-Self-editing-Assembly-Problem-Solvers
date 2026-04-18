# Minimal x86 assembly mössö demonstration and emulator runner.
#
# This file builds and runs a small self-editing x86 assembly program in Unicorn.
# The program has a self-editing part that consists of:
#   1) a mutable instruction block that can grow by calling an append routine that
#      appends one new mov instruction per generation, and
#   2) a mutable memory tape that stores the program state and the symbolic
#      operands used to decide what instruction gets appended next.
#
# The append routine reads op0 and op1 from the mutable memory tape, interprets
# them as symbolic operand indices, emits the corresponding x86 mov instruction,
# and inserts that instruction into the mutable instruction block.
#
# The demo mössö matches the example described in the article. It consists of
# an initial mutable instruction block and an initial mutable memory tape.
#
# When run, this code prints the initial mutable memory tape, the initial eax
# and ebx, the final mutable memory tape after executing 3 generations, and
# eax and ebx.
#
#
# TABLE OF CONTENTS
#
# - Constants for the emulator, mutable memory tape, and header
# - x86 assembly mössö program
# - Small helper functions for formatting and address handling
# - Unicorn hook for stopping at HLT
# - Execution
# - A demo mössö matching the article example
# - Main
#
#

import struct
from typing import List, Tuple

from unicorn import Uc, UC_ARCH_X86, UC_MODE_32, UC_HOOK_CODE, UC_PROT_ALL
from unicorn.x86_const import UC_X86_REG_EAX, UC_X86_REG_EBX
from keystone import Ks, KS_ARCH_X86, KS_MODE_32


# =============================================================================
# Emulator configuration
# =============================================================================
BASE_ADDR = 0x01000000
STACK_ADDR = 0x09000000
MEM_SIZE = 4 * 1024 * 1024

STACK_MAP_START = STACK_ADDR - MEM_SIZE
STACK_MAP_SIZE = MEM_SIZE * 2

DEFAULT_GENERATIONS = 3


# =============================================================================
# Defining mutable memory tape
# =============================================================================
# The mutable memory tape is a fixed array of 11 dwords whose *contents* change
# during execution. The instruction-selection cells op0 and op1 contain symbolic
# indices that refer either to tape cells or to eax/ebx.
#
# The tape cells eax_sym and ebx_sym are symbolic register references used by
# op0/op1 when selecting operands for appended instructions. They do not store
# the live values of the real eax and ebx registers.
TAPE_DWORDS = 11
TAPE_BYTES = TAPE_DWORDS * 4

FREE0 = 0
FREE1 = 1
FREE2 = 2
INPUT = 3
OUTPUT = 4
DATA0 = 5
DATA1 = 6
OP0 = 7
OP1 = 8
EAX_SYMBOL = 9
EBX_SYMBOL = 10

MEMORY_SYMBOL_MAX = 8


# =============================================================================
# Defining header layout
# =============================================================================
# The generated binary starts with a small header that the x86 assembly uses to
# track the mutable instruction block, the append location, and the generation count.
# Python also relies on the layout implicitly when reading and writing memory offsets.
OFF_HEADER = 0
OFF_TOTAL_SIZE = OFF_HEADER + 0
OFF_MUTABLE_INSTRUCTION_BLOCK_SIZE = OFF_HEADER + 4
OFF_INSERT_OFFSET = OFF_HEADER + 8
OFF_GENERATIONS_LEFT = OFF_HEADER + 12
OFF_SAVED_EAX = OFF_HEADER + 16
OFF_SAVED_EBX = OFF_HEADER + 20

HEADER_DWORDS = 6
HEADER_BYTES = HEADER_DWORDS * 4

OFF_MUTABLE_MEMORY_TAPE = HEADER_BYTES
EMIT_SCRATCH_BYTES = 128


# =============================================================================
# x86 assembly mössö program
# =============================================================================
# The created assembly contains:
#   - a header and mutable memory tape region at the start of the mapped memory,
#   - a mutable instruction block that grows across generations at the end,
#   - a fixed append routine that emits and inserts one new mov instruction,
#   - a HLT instruction used by Python/Unicorn to detect completed execution.
def make_asm(initial_mutable_instruction_block_asm: str, generations: int) -> str:
    return f"""
.intel_syntax noprefix

# ------------------------------------------------------------------
# Header shared by the x86 assembly helper routines.
# Python also relies on the layout implicitly when reading and writing
# memory offsets.
# ------------------------------------------------------------------
.long 0
.long 0
.long 0
.long 0
.long 0
.long 0

# ------------------------------------------------------------------
# Mutable memory tape
# ------------------------------------------------------------------
.rept {TAPE_DWORDS}
.long 0
.endr

# ------------------------------------------------------------------
# Scratch area used by the append routine when emitting x86 bytes
# ------------------------------------------------------------------
_emit_scratch:
.rept {EMIT_SCRATCH_BYTES}
.byte 0
.endr

# ------------------------------------------------------------------
# Entry point used by Python/Unicorn
#
# This section initializes the stack pointer, sets ebp to the base of the
# mapped memory region, fills the header fields needed by the program, and
# then jumps to the mutable instruction block.
# ------------------------------------------------------------------
_entry:
    mov esp, {STACK_ADDR}
    mov ebp, {BASE_ADDR}

    mov eax, 0
    mov ebx, 0
    mov ecx, 0
    mov edx, 0

    mov eax, _program_end
    sub eax, ebp
    mov dword ptr [ebp + {OFF_TOTAL_SIZE}], eax

    mov eax, _mutable_instruction_block_after_initial_block
    sub eax, _mutable_instruction_block_entry
    mov dword ptr [ebp + {OFF_MUTABLE_INSTRUCTION_BLOCK_SIZE}], eax

    mov eax, _append_call_site
    sub eax, ebp
    mov dword ptr [ebp + {OFF_INSERT_OFFSET}], eax

    mov dword ptr [ebp + {OFF_GENERATIONS_LEFT}], {generations}
    mov dword ptr [ebp + {OFF_SAVED_EAX}], 0
    mov dword ptr [ebp + {OFF_SAVED_EBX}], 0

    jmp _mutable_instruction_block_entry


# ------------------------------------------------------------------
# _insert_bytes
#
# Low-level byte insertion helper.
# Inputs on stack:
#   [esp+0]  inserted_bytes_ptr
#   [esp+4]  inserted_len
#   [esp+8]  insert_offset
#   [esp+12] current_total_size
#
# Effect:
#   Shifts the old bytes upward and copies the inserted instruction bytes into
#   the chosen insertion point.
# ------------------------------------------------------------------
_insert_bytes:
    push esi
    push edi
    push ebx

    push eax
    push edx
    push ecx
    push esi

    mov eax, dword ptr [esp + 12]
    sub eax, dword ptr [esp + 8]
    jle _insert_bytes_copy_insert

    mov esi, ebp
    add esi, dword ptr [esp + 12]
    dec esi

    mov edi, esi
    add edi, dword ptr [esp + 4]

_insert_bytes_shift_loop:
    mov bl, byte ptr [esi]
    mov byte ptr [edi], bl
    dec esi
    dec edi
    dec eax
    jnz _insert_bytes_shift_loop

_insert_bytes_copy_insert:
    mov ecx, dword ptr [esp + 4]
    mov esi, dword ptr [esp + 0]
    mov edi, ebp
    add edi, dword ptr [esp + 8]

_insert_bytes_copy_loop:
    test ecx, ecx
    jz _insert_bytes_done
    mov bl, byte ptr [esi]
    mov byte ptr [edi], bl
    inc esi
    inc edi
    dec ecx
    jmp _insert_bytes_copy_loop

_insert_bytes_done:
    add esp, 16
    pop ebx
    pop edi
    pop esi
    ret


# ------------------------------------------------------------------
# _commit_append_insert
#
# Higher-level insertion helper used by _append.
# It inserts the emitted bytes into the mutable instruction block
# and updates the tracked offsets.
# ------------------------------------------------------------------
_commit_append_insert:
    mov edx, dword ptr [ebp + {OFF_INSERT_OFFSET}]

    mov eax, dword ptr [ebp + {OFF_TOTAL_SIZE}]
    mov ecx, ebx
    mov esi, edi
    call _insert_bytes

    add dword ptr [ebp + {OFF_TOTAL_SIZE}], ebx
    add dword ptr [ebp + {OFF_MUTABLE_INSTRUCTION_BLOCK_SIZE}], ebx

    # The append call boundary moves forward as the block grows.
    add dword ptr [ebp + {OFF_INSERT_OFFSET}], ebx

    # The wrapper around the append call returns into the mutable instruction block.
    # After insertion, that return address must be moved forward as well.
    add dword ptr [esp + 16], ebx
    ret


# ------------------------------------------------------------------
# _append
#
# Reads op0/op1 from the mutable memory tape, interprets them as symbolic
# destination/source operands, emits the corresponding x86 mov instruction,
# and inserts the instruction into the mutable instruction block.
#
# If the derived instruction would be memory <- memory, appending aborts and
# execution jumps to the final done HLT.
# ------------------------------------------------------------------
_append:
    pushfd
    push eax
    push ebx

    mov dword ptr [ebp + {OFF_SAVED_EAX}], eax
    mov dword ptr [ebp + {OFF_SAVED_EBX}], ebx

    mov eax, dword ptr [ebp + {OFF_MUTABLE_MEMORY_TAPE} + {OP0 * 4}]
    mov ebx, dword ptr [ebp + {OFF_MUTABLE_MEMORY_TAPE} + {OP1 * 4}]

    # Scratch metadata layout in _emit_scratch:
    #   [esi+0]  dst_kind   (0 = tape memory, 1 = register)
    #   [esi+4]  dst_regid
    #   [esi+8]  dst_disp
    #   [esi+12] src_kind   (0 = tape memory, 1 = register)
    #   [esi+16] src_regid
    #   [esi+20] src_disp

    lea esi, [_emit_scratch]
    mov dword ptr [esi+0],  0
    mov dword ptr [esi+4],  0
    mov dword ptr [esi+8],  0
    mov dword ptr [esi+12], 0
    mov dword ptr [esi+16], 0
    mov dword ptr [esi+20], 0

    # --------------------------------------------------------------
    # Decode destination from EAX
    # --------------------------------------------------------------
    cmp eax, {MEMORY_SYMBOL_MAX}
    jbe _append_decode_dst_tape

    cmp eax, {EAX_SYMBOL}
    je _append_decode_dst_eax
    cmp eax, {EBX_SYMBOL}
    je _append_decode_dst_ebx
    jmp _append_abort_to_done

_append_decode_dst_tape:
    mov dword ptr [esi+0], 0
    mov ecx, eax
    shl ecx, 2
    add ecx, {OFF_MUTABLE_MEMORY_TAPE}
    mov dword ptr [esi+8], ecx
    jmp _append_decode_src

_append_decode_dst_eax:
    mov dword ptr [esi+0], 1
    mov dword ptr [esi+4], 0
    jmp _append_decode_src

_append_decode_dst_ebx:
    mov dword ptr [esi+0], 1
    mov dword ptr [esi+4], 3
    jmp _append_decode_src

    # --------------------------------------------------------------
    # Decode source from EBX
    # --------------------------------------------------------------
_append_decode_src:
    cmp ebx, {MEMORY_SYMBOL_MAX}
    jbe _append_decode_src_tape

    cmp ebx, {EAX_SYMBOL}
    je _append_decode_src_eax
    cmp ebx, {EBX_SYMBOL}
    je _append_decode_src_ebx
    jmp _append_abort_to_done

_append_decode_src_tape:
    mov dword ptr [esi+12], 0
    mov ecx, ebx
    shl ecx, 2
    add ecx, {OFF_MUTABLE_MEMORY_TAPE}
    mov dword ptr [esi+20], ecx
    jmp _append_emit_mov

_append_decode_src_eax:
    mov dword ptr [esi+12], 1
    mov dword ptr [esi+16], 0
    jmp _append_emit_mov

_append_decode_src_ebx:
    mov dword ptr [esi+12], 1
    mov dword ptr [esi+16], 3
    jmp _append_emit_mov

_append_emit_mov:
    mov eax, dword ptr [esi+0]
    mov ebx, dword ptr [esi+12]

    # Invalid append: memory <- memory is not allowed.
    cmp eax, 0
    jne _append_emit_mov_dst_reg
    cmp ebx, 0
    je _append_abort_to_done
    jmp _append_emit_mov_mem_reg

_append_emit_mov_dst_reg:
    cmp ebx, 1
    je _append_emit_mov_reg_reg
    jmp _append_emit_mov_reg_mem

_append_emit_mov_reg_reg:
    # 8B /r : mov reg, r/m
    lea edi, [_emit_scratch + 32]
    mov byte ptr [edi+0], 0x8B

    mov eax, 0xC0
    mov ecx, dword ptr [esi+4]
    shl ecx, 3
    or eax, ecx
    or eax, dword ptr [esi+16]
    mov byte ptr [edi+1], al

    mov ebx, 2
    call _commit_append_insert
    jmp _append_restore_and_return

_append_emit_mov_reg_mem:
    # 8B /r : mov reg, [disp32]
    lea edi, [_emit_scratch + 32]
    mov byte ptr [edi+0], 0x8B

    mov eax, 0x85
    mov ecx, dword ptr [esi+4]
    shl ecx, 3
    or eax, ecx
    mov byte ptr [edi+1], al

    mov eax, dword ptr [esi+20]
    mov dword ptr [edi+2], eax

    mov ebx, 6
    call _commit_append_insert
    jmp _append_restore_and_return

_append_emit_mov_mem_reg:
    # 89 /r : mov [disp32], reg
    lea edi, [_emit_scratch + 32]
    mov byte ptr [edi+0], 0x89

    mov eax, 0x85
    mov ecx, dword ptr [esi+16]
    shl ecx, 3
    or eax, ecx
    mov byte ptr [edi+1], al

    mov eax, dword ptr [esi+8]
    mov dword ptr [edi+2], eax

    mov ebx, 6
    call _commit_append_insert
    jmp _append_restore_and_return

_append_abort_to_done:
    pop ebx
    pop eax
    popfd
    jmp _done_hlt

_append_restore_and_return:
    pop ebx
    pop eax
    popfd
    ret


# ------------------------------------------------------------------
# _generation_counter
#
# Decrements the remaining generation count. If the count reaches zero,
# execution jumps to the final done HLT. Otherwise it returns.
# ------------------------------------------------------------------
_generation_counter:
    pushfd
    push eax
    push ebx

    mov eax, dword ptr [ebp + {OFF_GENERATIONS_LEFT}]
    dec eax
    mov dword ptr [ebp + {OFF_GENERATIONS_LEFT}], eax

    cmp eax, 0
    je _generation_counter_to_done

    pop ebx
    pop eax
    popfd
    ret

_generation_counter_to_done:
    pop ebx
    pop eax
    popfd
    jmp _done_hlt


# ------------------------------------------------------------------
# Final HLT used by Python/Unicorn to detect completed execution
# ------------------------------------------------------------------
_done_hlt:
    lea esp, [esp + 4]
    pop eax
    hlt


# ------------------------------------------------------------------
# Start of the mutable instruction block
# ------------------------------------------------------------------
_mutable_instruction_block_entry:
{initial_mutable_instruction_block_asm}

_mutable_instruction_block_after_initial_block:
_append_call_site:
    push eax
    mov eax, _append
    call eax
    pop eax

    push eax
    mov eax, _generation_counter
    call eax
    pop eax

    jmp _mutable_instruction_block_entry

_program_end:
"""


# =============================================================================
# Small helper functions for formatting and address handling
# =============================================================================
def s32(x: int) -> int:
    x &= 0xFFFFFFFF
    return x if x < 0x80000000 else x - 0x100000000


def format_tape_row(tape: List[int]) -> str:
    return (
        f"free0={tape[0]} free1={tape[1]} free2={tape[2]} "
        f"input={tape[3]} output={tape[4]} "
        f"data0={tape[5]} data1={tape[6]} "
        f"op0={tape[7]} op1={tape[8]}"
    )


def mutable_memory_tape_disp(index: int) -> int:
    return OFF_MUTABLE_MEMORY_TAPE + index * 4


# =============================================================================
# Unicorn hook for stopping at HLT
# =============================================================================
def hook_code(mu: Uc, address: int, size: int, user_data: None) -> None:
    del size, user_data
    if mu.mem_read(address, 1) == b"\xF4":
        mu.emu_stop()


# =============================================================================
# Execution
# =============================================================================
def run_case(
    initial_mutable_instruction_block_lines: List[str],
    initial_tape: List[int],
    generations: int = DEFAULT_GENERATIONS,
) -> Tuple[List[int], int, int]:
    if initial_mutable_instruction_block_lines:
        initial_mutable_instruction_block_asm = "\n".join(
            f"    {line}" for line in initial_mutable_instruction_block_lines
        )
    else:
        initial_mutable_instruction_block_asm = "    nop"

    asm = make_asm(initial_mutable_instruction_block_asm, generations)
    binary = bytes(Ks(KS_ARCH_X86, KS_MODE_32).asm(asm, addr=BASE_ADDR)[0])

    mu = Uc(UC_ARCH_X86, UC_MODE_32)
    mu.mem_map(BASE_ADDR, MEM_SIZE, UC_PROT_ALL)
    mu.mem_map(STACK_MAP_START, STACK_MAP_SIZE, UC_PROT_ALL)
    mu.mem_write(BASE_ADDR, binary)
    mu.mem_write(
        BASE_ADDR + OFF_MUTABLE_MEMORY_TAPE,
        struct.pack("<" + "I" * TAPE_DWORDS, *initial_tape),
    )

    mu.hook_add(UC_HOOK_CODE, hook_code)
    mu.emu_start(
        BASE_ADDR + HEADER_BYTES + TAPE_BYTES + EMIT_SCRATCH_BYTES,
        BASE_ADDR + MEM_SIZE,
    )

    tape_bytes = mu.mem_read(BASE_ADDR + OFF_MUTABLE_MEMORY_TAPE, TAPE_BYTES)
    final_tape = [s32(x) for x in struct.unpack("<" + "I" * TAPE_DWORDS, tape_bytes)]

    final_eax = s32(mu.reg_read(UC_X86_REG_EAX))
    final_ebx = s32(mu.reg_read(UC_X86_REG_EBX))

    return final_tape, final_eax, final_ebx


# =============================================================================
# Demo mössö matching the article example
# =============================================================================
def demo_initial_mutable_instruction_block_lines() -> List[str]:
    return [
        f"mov eax, dword ptr [ebp + 0x{mutable_memory_tape_disp(DATA0):X}]",
        f"mov dword ptr [ebp + 0x{mutable_memory_tape_disp(FREE0):X}], eax",
        f"mov eax, dword ptr [ebp + 0x{mutable_memory_tape_disp(DATA1):X}]",
        f"mov dword ptr [ebp + 0x{mutable_memory_tape_disp(FREE1):X}], eax",
        f"mov eax, dword ptr [ebp + 0x{mutable_memory_tape_disp(INPUT):X}]",
        f"mov dword ptr [ebp + 0x{mutable_memory_tape_disp(OP1):X}], eax",
        f"mov eax, dword ptr [ebp + 0x{mutable_memory_tape_disp(FREE2):X}]",
        f"mov dword ptr [ebp + 0x{mutable_memory_tape_disp(OP0):X}], eax",
        f"mov dword ptr [ebp + 0x{mutable_memory_tape_disp(OUTPUT):X}], ebx",
    ]


def demo_tape() -> List[int]:
    return [0, 0, 10, 1, 0, 3, 7, 0, 0, 0, 0]


# =============================================================================
# Main
# =============================================================================
# When executed, this code prints the initial mutable memory tape, the
# mutable memory tape after executing 3 generations, and the final values
# of the real registers eax and ebx.
def main() -> None:
    tape = demo_tape()

    print("Initial mutable memory tape:")
    print(format_tape_row(tape))
    print("\nInitial real registers:")
    print("eax=0 ebx=0")

    final_tape, final_eax, final_ebx = run_case(
        demo_initial_mutable_instruction_block_lines(),
        tape,
        generations=DEFAULT_GENERATIONS,
    )

    print(f"\nMutable memory tape after {DEFAULT_GENERATIONS} generations:")
    print(format_tape_row(final_tape))

    print("\nFinal real registers:")
    print(f"eax={final_eax} ebx={final_ebx}")


if __name__ == "__main__":
    main()
