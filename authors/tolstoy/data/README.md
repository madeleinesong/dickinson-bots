# Tolstoy data

Leo Tolstoy (1828–1910) — **public domain**. Public-domain English translations
(Constance Garnett / Maude) from Project Gutenberg.

`raw/`
- `anna_karenina.txt` — Gutenberg #1399 (~350k words)
- `war_and_peace.txt` — Gutenberg #2600 (~566k words)

`strip_gutenberg_novel` skips front matter / table of contents to the first
chapter and drops structural headings. To regenerate `processed/`:
```
python scripts/prepare_data.py tolstoy
```
