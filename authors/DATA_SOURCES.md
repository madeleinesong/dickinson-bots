# Data sources

All raw text was pulled from online sources. Provenance + licensing per file.

## Dickinson (public domain — d. 1886)
| File | Source | Words |
|------|--------|-------|
| `dickinson/data/raw/gutenberg_poems_complete.txt` | Project Gutenberg #12242, *Poems, Three Series, Complete* | ~31k |
| `dickinson/data/raw/letters_todd_1894.txt` | *Letters of Emily Dickinson*, ed. Mabel Loomis Todd (1894) | ~104k |

## Weil (d. 1943 — translations may be in copyright; personal use)
| File | Source | Words |
|------|--------|-------|
| `weil/data/raw/weil_combined.txt` | Combined English corpus (essays, notebooks, *Gravity and Grace*, etc.) — carried from original repo | ~560k |
| `weil/data/raw/abolition_political_parties.muse` | The Anarchist Library, *On the Abolition of All Political Parties* | ~15k |

Note: `weil_combined.txt` already contains *Gravity and Grace*, so the standalone
download was dropped to avoid overweighting it during training.

## Le Guin (d. 2018 — copyrighted; only freely-circulated / permitted texts used)
| File | Source | Words |
|------|--------|-------|
| `le_guin/data/raw/bryn_mawr_commencement_1986.txt` | Bryn Mawr Commencement Address, 1986 — hosted **with Le Guin's permission** at serendipstudio.org (later collected in *Dancing at the Edge of the World*) | ~5.8k |
| `le_guin/data/raw/left_handed_commencement_1983.txt` | "A Left-Handed Commencement Address", Mills College 1983 (americanrhetoric.com) | ~1.2k |
| `le_guin/data/raw/nba_speech_2014.txt` | National Book Award acceptance speech, 2014 (americanrhetoric.com) | ~0.5k |

⚠️ **Le Guin corpus is still small (~7.5k words)** — this is close to the honest
ceiling of what's freely usable, since her books are under copyright. For a real
voice model you'll want to add essays/fiction you personally own (see
`le_guin/data/README.md`). Those go in `raw/` and are gitignored.

## Hugo (d. 1885 — public domain)
| File | Source | Words |
|------|--------|-------|
| `hugo/data/raw/les_miserables.txt` | *Les Misérables*, Gutenberg #135 (Hapgood trans.) | ~566k |
| `hugo/data/raw/notre_dame_de_paris.txt` | *Notre-Dame de Paris*, Gutenberg #2610 (Hapgood trans.) | ~188k |

Public-domain English translations (chosen for consistency with the other English
bots; the originals are French). `strip_hugo` drops the title page, table of
contents, VOLUME/BOOK/CHAPTER headings, and the list of illustrations, leaving
~741k words of prose.

## De-duplication notes
- Dropped `dickinson/poems.txt` (subset of the Gutenberg complete edition).
- Dropped `dickinson/archive_letters_1894.txt` (OCR-noisy subset of the Todd letters).
- Dropped standalone `weil/gravity_and_grace.muse` (already in `weil_combined.txt`).
