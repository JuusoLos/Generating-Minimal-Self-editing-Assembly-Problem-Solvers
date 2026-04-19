## Generating minimal self-editing x86 assembly problem solvers

Juuso Lösönen

Code and dataset posted on this public Github repository on 18.04.2026.


A minimal implementation of an x86 assembly program that edits itself to
solve a toy problem. In the implementation, a mutable instruction block restricted to mov
instructions reads and modifies a memory state and calls a fixed append routine at the end
of every generation. The fixed append routine selects operands based on the memory tape and adds a
new mov instruction to the end of the mutable instruction block. Using evolutionary search, we
generate a dataset of initial instruction-block and memory-state configurations that solve a
simple conditional selection task: copying one of two data values to the output depending
on an input that's either 0 or 1. I call the generated self-editing assembly programs mössö.

In future work, mössö maybe could be scaled by training larger ML models on generated mössö program-data
to generate more capable mössö, because ML systems often benefit from simpler training cases first.



```text

FORMATTING OF verified_correct_initial_conditions.txt:
[free0, free1, free2, dst0, src0, dst1, src1, ...]

Symbol IDs:
0 = free0, 1 = free1, 2 = free2, 3 = input, 4 = output, 5 = data0, 6 = data1, 7 = op0, 8 = op1, 9 = eax, 10 = ebx

- Tape starts with:
  [free0, free1, free2, input, output, data0, data1, op0, op1, eax, ebx]
- output, op0, op1, eax, ebx start at 0 during evaluation
