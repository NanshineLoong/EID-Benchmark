## Analysis Helpers

Modular pieces for result stats and plotting:

- `loaders.py`: read dataset facts, result.json, and record traces.
- `metrics.py`: atomic metric math and coverage curves.
- `runner.py`: aggregate a run into table-friendly rows.
- `plots.py`: plotting utilities (labels in English only).
- `cli.py`: entry for E1–E4 scripts.

### Configuration

Make sure your `.env` file is configured:

```bash
OPENAI_API_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=your-api-key-here
```

### Run Examples

Pass dataset/mode/model lists to generate all combinations in one go:

```bash
eid-analyze e1 \
  --datasets derm medqa diagnosisarena \
  --modes roleplay react cot \
  --models gpt-5-mini \
  --max-turns 8 16 \
  --fig-dir figures \
  --excel figures/e1_summary.xlsx
```

Key subcommands:

- `e1`: overview table + success vs coverage scatter plots (optionally per dataset). Use `--excel` to save a styled grouped Excel table.
- `e2`: success/coverage vs max-turn trends and coverage-vs-turn curves (`--metric`, `--curve-metric`).
- `e3`: coverage bars for success vs failure cases.
- `e4`: ablation deltas vs `--baseline` (use `--md` to save text table if needed).
- `turns`: print/save average interaction turns per dataset/mode/model (`--md` for markdown).
- `turn_time`: average per-turn duration by mode/model across datasets (`--by-dataset` for per-dataset rows).

Default dataset paths live in `loaders.DEFAULT_DATASET_FILES`; override globally with `--dataset-path` if needed.

---

### Full Command Examples

**E1: All models summary**
```bash
eid-analyze e1 \
  --datasets derm medqa diagnosisarena rarearena clinicalbench \
  --modes cot roleplay \
  --models gpt-5-mini \
  --max-turns 16 \
  --fig-dir figures \
  --excel figures/e1_summary_models.xlsx
```

**E1: Mode comparison**
```bash
eid-analyze e1 \
  --datasets derm medqa diagnosisarena rarearena clinicalbench \
  --modes roleplay react sc refine \
  --models gpt-5-mini \
  --max-turns 16 \
  --fig-dir figures \
  --excel figures/e1_summary_modes.xlsx
```

**E2: Turn limit experiments**
```bash
eid-analyze e2 \
  --datasets diagnosisarena \
  --modes roleplay refine \
  --models gpt-5-mini \
  --max-turns 4 8 12 16 20 \
  --fig-dir figures
```

**E3: Coverage vs outcome analysis**
```bash
eid-analyze e3 \
  --datasets derm medqa clinicalbench \
  --modes roleplay \
  --models gpt-5-mini \
  --max-turns 16 \
  --fig-dir figures
```

**E4: Ablation study**
```bash
eid-analyze e4 \
  --datasets diagnosisarena \
  --modes roleplay react sc refine \
  --models gpt-5-mini \
  --max-turns 16 \
  --baseline roleplay \
  --md e4.md
```

**Average turns**
```bash
eid-analyze turns \
  --datasets derm medqa diagnosisarena rarearena clinicalbench \
  --modes roleplay \
  --models gpt-5-mini \
  --max-turns 16 \
  --md figures/turns.md
```

**Per-turn duration**
```bash
eid-analyze turn_time \
  --datasets derm medqa diagnosisarena rarearena clinicalbench \
  --modes roleplay react sc refine \
  --models gpt-5-mini \
  --max-turns 16 \
  --md figures/turn_time.md
```
