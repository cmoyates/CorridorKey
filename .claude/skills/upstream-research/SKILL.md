# upstream-research

Read-only research skill for exploring upstream CorridorKey repos, MLX source, and MLX examples to find pipeline optimization opportunities.

## When to use
- Before implementing a pipeline optimization, to check if upstream has relevant changes
- When investigating MLX API patterns, async I/O, or video decode optimizations
- When looking for prior art in CorridorKey forks (postprocessing, output selection, threading)
- When researching numpy/OpenCV alternatives for color_utils bottlenecks

## Behavior
- Read-only: do not modify any files
- Search local code, git history, and BTCA resources if available
- Classify findings as: `pipeline-applicable`, `mlx-engine-only`, `concept-only`, `pytorch-only`

## BTCA resources (if available)
Check `btca.config.jsonc` for configured repos:
- `mlx` — MLX framework source
- `mlxExamples` — MLX examples repo
- `corridorKey` — upstream CorridorKey repo (nikopueringer)
- `corridorKeyEngine` — CorridorKey Engine (99oblivius)
- `corridorKeyMarcel` — Marcel Lieb's CorridorKey fork
- `ezCorridorKey` — EZ-CorridorKey (edenaion)

## Without BTCA
Fall back to:
- Local git log analysis
- Grep/glob on local files
- Web search for MLX documentation, PyAV/VideoToolbox, async pipeline patterns

## Focus areas for this repo
- **Async I/O**: threading patterns for overlapping read/infer/write in frame loops
- **Output reduction**: configurable output selection, lazy comp generation
- **Postprocessing**: checkerboard caching, GPU-side colorspace ops, skipping unused work
- **Video decode**: PyAV hw-accelerated decode, frame pre-extraction
- **Tile sizing**: larger tile support, memory-aware tile selection

## Output format
For each finding, report:
- **Source**: repo/PR/file
- **Summary**: one sentence
- **Classification**: pipeline-applicable / mlx-engine-only / concept-only / pytorch-only
- **Relevance**: how it maps to a priority from `docs/HANDOFF_TO_CORRIDORKEY.md`
