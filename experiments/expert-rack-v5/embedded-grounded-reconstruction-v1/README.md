# Embedded grounded reconstruction v1

This materialization is an honest domain-adaptation view of the selected
`reference_cpt` passages. Each assistant target is one contiguous, verbatim
slice of the recorded source text. It is not synthetic QA, and successful
reconstruction is not evidence of source correctness or answer quality.

The user request contains only the repository, file path, assigned leaf, and
zero-based chunk index. The source split and repository group are preserved in
provenance. Exact-content, upstream-identity, and opposite-split group guards
fail closed before publication.

Long passages use deterministic character boundaries verified by the pinned
Qwen3-0.6B chat tokenizer. Every rendered chat is at most 4,096 tokens. There is
no truncation or deferred tier, and concatenating assistant targets in chunk
order exactly reproduces every source row.

## Build

From the repository root with `PYTHONPATH=src`:

```powershell
.\.venv\Scripts\python.exe experiments\expert-rack-v5\build_embedded_grounded_reconstruction.py `
  --source-manifest 'C:\Users\engin\Documents\datasets\processed\molor-embedded-source-experts-v1\manifest.json' `
  --source-manifest-sha256 '672d621eaf7c3ad74a2aefcae810c7ce5927714bd3d32835ac5f176ddc1fe389' `
  --tokenizer 'C:\Users\engin\Documents\molor\models\Qwen3-0.6B' `
  --tokenizer-aggregate-sha256 '4c6d93851307c525ca8fa64bb5f0dcaf61b4b854a4d4123a1b83aedde94d973c' `
  --chat-template-sha256 'a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8' `
  --output 'C:\Users\engin\Documents\datasets\processed\molor-embedded-grounded-reconstruction-v1'
```

Add `--compare-to <completed-output>` on an independent rebuild to require an
exact file-name and SHA-256 match.

## Result

- 2,749 selected source rows
- 3,956 messages-only SFT records
- 27,228,532 source and emitted characters
- zero omitted, reordered, duplicated, truncated, rejected, or deferred rows
- 28 files and 40,613,769 bytes per build
- primary and independent rebuild byte-identical
- manifest semantic SHA-256:
  `4e676208b5e26460367bd7b0badd6f798c007bc7fbe27c6dc44143637da8f348`
- manifest file SHA-256:
  `46c109be7a780a7cb0732f819d10a4ec1b013a0c17928622d3775f61c20936db`

`registration-fragment.json` is a conflict-free integration hook. It does not
modify shared source-assignment or mixture-recipe artifacts.
