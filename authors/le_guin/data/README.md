# Le Guin data

Ursula K. Le Guin died in **2018**, so essentially all of her published work
(novels, stories, poetry, most essays) is **under copyright**. Unlike Dickinson
(d. 1886, public domain) and Weil (d. 1943), we can't just download her books.

To train on her voice you need text you can legitimately use. Drop `.txt` files
into `raw/` here. Good, defensible sources:

## Freely available / quotable
- **Public speeches** — e.g. her 2014 National Book Awards "Freedom" speech
  (widely transcribed online). Short but very high signal.
- **Blog posts** — she blogged at `ursulakleguin.com` (the "Book View Cafe" blog,
  2010–2017). Many posts were public; some collected in *No Time to Spare* (2017).
- **Interviews** — transcripts of public interviews (Paris Review, etc.) capture
  her conversational voice. Note: interviewer text should be stripped out.
- **Essays posted in full** by the author/estate.

## Texts you personally own
If you own ebooks/PDFs of her essay collections (*The Wave in the Mind*,
*Words Are My Matter*, *Dancing at the Edge of the World*) or fiction, you can
extract text locally for **personal, non-distributed** experimentation. Keep
these out of any public commit (see `.gitignore`).

## How much you need
LoRA fine-tuning for "voice" works with surprisingly little — even
5,000–20,000 words of clean prose is enough to shift style. You do not need a
whole corpus. Prioritize **essays/nonfiction** for a consistent first-person voice.

## After adding raw text
Run the shared cleaning step to produce `processed/`:
```
python scripts/prepare_data.py le_guin
```

---
⚠️ Don't commit copyrighted full texts to a public GitHub repo. `raw/` for Le Guin
is gitignored by default.
