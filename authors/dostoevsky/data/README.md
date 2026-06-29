# Dostoevsky data

Fyodor Dostoevsky (1821–1881) — **public domain**. Public-domain English
translations (Constance Garnett) from Project Gutenberg.

`raw/`
- `crime_and_punishment.txt` — Gutenberg #2554 (~204k words)
- `brothers_karamazov.txt`   — Gutenberg #28054 (~351k words)

`strip_gutenberg_novel` skips the translator's preface / table of contents to the
first chapter and drops structural headings. To regenerate `processed/`:
```
python scripts/prepare_data.py dostoevsky
```
