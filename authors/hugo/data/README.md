# Hugo data

Victor Hugo (1802–1885) — **public domain**. These are public-domain **English
translations** (Isabel F. Hapgood, 1887/1888) from Project Gutenberg, chosen for
consistency with the other (English) bots. The originals are French.

`raw/`
- `les_miserables.txt`      — *Les Misérables*, Gutenberg #135 (~566k words)
- `notre_dame_de_paris.txt` — *Notre-Dame de Paris* (The Hunchback), Gutenberg #2610 (~188k words)

Note: the "voice" learned is Hugo-via-translator English. To regenerate `processed/`:
```
python scripts/prepare_data.py hugo
```
