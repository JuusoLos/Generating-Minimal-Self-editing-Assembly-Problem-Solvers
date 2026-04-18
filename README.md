“Generating Minimal Self-editing Assembly Problem Solvers” Juuso Lösönen. 18.04.2026.

## Abstract

We present a minimal implementation of an x86 assembly program that edits itself to
solve a problem. In the implementation, a mutable instruction block restricted to mov
instructions reads and modifies a memory state and calls a fixed append routine that adds a
new instruction to the end of the mutable instruction block. Using evolutionary search, we
generate a dataset of initial instruction-block and memory-state configurations that solve a
simple conditional selection task: copying one of two data values to the output depending
on a binary input. The implementation together with evolutionary search serves as a proof
of concept that such programs can be automatically generated. We use the term mössö to
refer to self-editing assembly-level problem-solver programs whose structure is intended to
support automatic generation.



```text

FORMATTING OF verified_correct_initial_conditions.txt:
[free0, free1, free2, dst0, src0, dst1, src1, ...]

Symbol IDs:
0 = free0, 1 = free1, 2 = free2, 3 = input, 4 = output, 5 = data0, 6 = data1, 7 = op0, 8 = op1, 9 = eax, 10 = ebx

- Tape starts with:
  [free0, free1, free2, input, output, data0, data1, op0, op1, eax, ebx]
- output, op0, op1, eax, ebx start at 0 during evaluation
