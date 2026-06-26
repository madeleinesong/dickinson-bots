# Authors

One directory per writer. Each has the same shape:

```
authors/<name>/
  data/
    raw/         # original source texts (.txt) — input
    processed/   # cleaned text ready for training — generated
    README.md    # where the data came from + licensing notes
  model/         # fine-tuned weights land here (gitignored)
```

| Author    | Died | Status        | Data on hand                          |
|-----------|------|---------------|---------------------------------------|
| Dickinson | 1886 | Public domain | ✅ poems + letters (~134k words)       |
| Weil      | 1943 | Personal use  | ✅ combined corpus (~560k words)       |
| Le Guin   | 2018 | Copyrighted   | ⚠️ none yet — see `le_guin/data/README.md` |

The legacy `dickinson/`, `weil/`, `jesus/` dirs at the repo root are the original
GPT-2 experiment, kept for reference. The fresh project lives under `authors/`.
