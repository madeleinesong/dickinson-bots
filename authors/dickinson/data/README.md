# Dickinson data

Emily Dickinson (d. 1886) — **public domain**.

`raw/`
- `poems.txt`   — her poems (Higginson/Todd editions; em-dash heavy style intact)
- `letters.txt` — her collected letters (OCR'd; has some double-spacing artifacts
  that the cleaning step normalizes)

Source: carried over from the original `dickinson-bots` repo.

To regenerate `processed/`:
```
python scripts/prepare_data.py dickinson
```
