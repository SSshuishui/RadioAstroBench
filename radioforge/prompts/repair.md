Repair or improve the previous candidate.

Task signature:
{signature}

Relevant memory:
{memory}

Optimization knowledge:
{knowledge}

Previous evaluation feedback:
{feedback}

Reference source excerpt:
```python
{source}
```

Requirements:
- If the candidate failed to compile, fix the compile error first.
- If it compiled but was incorrect, make it conservative and match PyTorch semantics.
- If it was correct but slow, fuse epilogues or reduce memory traffic.
- Avoid producing nearly identical code to the previous failed candidate.
- Return only the complete Python solution file.
