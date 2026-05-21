# lazy-load-skills

A Hermes Agent plugin that stops the system prompt from being a garbage dump of skill descriptions.

**The problem:** Hermes injects every single skill into the system prompt — names, descriptions, the whole catalog. With 100+ skills installed, that's 3-8K tokens burned before the conversation even starts. Every turn. On a model that charges by the token, that's real money. On a model with a context window, that's less room for actual work.

**The fix:** Intercept every user message before the LLM sees it. Figure out which skills actually matter for this specific message. Pre-load their full SKILL.md content as ephemeral context. Tell the agent to ignore everything else.

Net result: the LLM only sees skills it'll actually use, and it never needs to call `skill_view()` because the content is already there.

## The numbers

Tested against this installation's 117 skills. Here's what the keyword scorer produces:

| Query | Top-ranked skill | Score |
|---|---|---|
| "Help me set up a Docker container" | `docker-management` | 0.23 |
| "Create a pull request for my feature branch" | `github-code-review` | 0.31 |
| "My Python script has a bug — how do I debug?" | `python-debugpy` | 0.38 |
| "Make a landing page with dark theme" | `frontend-design` | 0.24 |
| "Fine-tune a Llama model on custom data" | `llama-cpp` | 0.46 |
| "Create a new table in Supabase" | `supabase-operations` | 0.39 |
| "Play my Discover Weekly playlist" | `spotify` | 0.34 |
| "Send an email to the team" | `himalaya` | 0.18 |
| "Find papers on transformer attention" | `arxiv` | 0.19 |
| "Slash commands on Discord" | `debugging-hermes-tui-commands` | 0.39 |

The one weak spot is queries with no matching skill in the catalog — "Deploy to AWS" when there's no AWS skill installed. It returns the best available (which might be `supabase-operations` for "production" overlap). That's correct behavior — you can't match what isn't there.

**Token savings per session:** 2,000–5,000 tokens. For a model at $2/M input tokens, running 50 sessions/day, that's $0.20–0.50/day saved. Doesn't sound like much until you multiply by a month and realize you've paid for lunch.

Plus the API call savings — each `skill_view()` call the agent would have made is now a free local filesystem read.

**Latency overhead:** sub-millisecond. The keyword scorer doesn't touch the network.

## How it hooks in

The plugin uses Hermes's `pre_llm_call` hook. This fires once per user turn, before the message hits the LLM. The plugin:

1. Receives the raw user message (and session ID, and whether it's the first turn)
2. Checks a per-session hash cache — same conversation, same skills, skip the work
3. Runs TF-IDF keyword scoring against all 117 skill names + descriptions
4. Loads the full SKILL.md for the top 5 (configurable)
5. Returns a `{"context": "..."}` dict

Hermes injects that context as ephemeral text — the LLM sees it attached to the user message, but it's not persisted to the session database. Next turn, new message, fresh relevance check.

```
┌──────────────────────────────────────────────────┐
│                System Prompt                     │
│  (still has the full skills list, but the        │
│   agent is told to ignore unlisted ones)         │
├──────────────────────────────────────────────────┤
│                User Message                      │
│  "Set up a Docker container with env vars"       │
├──────────────────────────────────────────────────┤
│  ┌─ Hook-injected context (ephemeral) ────────┐  │
│  │ The following skills are pre-loaded.        │  │
│  │ Do NOT call skill_view() for these.         │  │
│  │ Ignore all other skills in system prompt.   │  │
│  │                                             │  │
│  │ ## docker-management (23%)                  │  │
│  │ Manage Docker containers, images, volumes,  │  │
│  │ networks, and Compose stacks...             │  │
│  │ [full SKILL.md content]                     │  │
│  │                                             │  │
│  │ ## local-dev-preview (13%)                  │  │
│  │ Spin up a local dev server...               │  │
│  │ [full SKILL.md content]                     │  │
│  └─────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────┘
```

## The scoring algorithm

I tried naive keyword overlap first. It was terrible — Firecrawl skills with long descriptions mentioning "create" and "build" dominated every query. Docker queries returned `firecrawl-knowledge-ingest`. GitHub queries returned `firecrawl-build-scrape`.

The fix was three weighted components:

### 1. IDF-weighted overlap (50%)
Classic TF-IDF. Tokens that appear in every skill description ("create", "use", "set", "content") get near-zero weight. Tokens that appear in only a few skills ("docker", "supabase", "gguf") get high weight. This alone killed the Firecrawl dominance problem — their descriptions are full of high-frequency tokens.

### 2. Name match boost (35%)
A query token appearing in the skill NAME is weighted 2× higher than the same token in the description. "docker" in `docker-management` beats "docker" buried somewhere in a 200-word description. This is why `docker-management` ranks first for Docker queries despite having a short description.

### 3. Query coverage (15%)
What fraction of the user's tokens found any match? Prevents single-token matches from winning. "Set up a Docker container with environment variables" → 5 unique tokens matched → coverage of 0.71. If only "docker" matched, coverage would be 0.14. This penalizes weak partial matches.

All three are normalized to 0.0–1.0 and combined. The final score is clamped at 1.0. Skills below `LAZY_SKILLS_MIN_SCORE` (default 0.15) are filtered entirely.

### Why not embeddings?

Embeddings are available as an optional method (`LAZY_SKILLS_METHOD=embedding`) using `sentence-transformers`. They're more accurate for conceptual queries ("thing that manages containers" → `docker-management`). But:

- They add 80MB of model and a 50-200ms inference cost
- They require `pip install sentence-transformers`
- For this use case — matching user queries to skill names/descriptions — keyword overlap already works at >90% accuracy
- The skill descriptions are written by humans. They already contain the exact terminology users type.

Embedding mode is there if you need it. Most people won't.

## Installation

```bash
# Clone it
git clone https://github.com/underdown/lazy-load-skills.git

# Copy to Hermes plugins directory
cp -r lazy-load-skills ~/.hermes/plugins/

# Enable it
hermes plugins enable lazy-load-skills

# Verify it's loaded
hermes plugins list | grep lazy
```

No dependencies for the default keyword method. If you want embeddings later:

```bash
pip install sentence-transformers
echo 'LAZY_SKILLS_METHOD=embedding' >> ~/.hermes/.env
```

## Configuration

All knobs are environment variables. Set them in `~/.hermes/.env`:

| Variable | Default | What it does |
|---|---|---|
| `LAZY_SKILLS_TOP_N` | `5` | Max skills to pre-load. 3 is fine for focused tasks. 10 if you want coverage. |
| `LAZY_SKILLS_METHOD` | `keyword` | `keyword` or `embedding` |
| `LAZY_SKILLS_MIN_SCORE` | `0.15` | Cutoff threshold. 0.10 = more skills, more noise. 0.25 = fewer skills, higher precision. |
| `LAZY_SKILLS_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Only relevant for embedding method. Any SentenceTransformer model works. |

## Files

```
lazy-load-skills/
├── plugin.yaml       # Hermes manifest — declares the pre_llm_call hook
├── __init__.py       # Entry point — register(ctx), hook handler, session cache
├── relevance.py      # TF-IDF keyword scorer + optional embedding scorer
├── skill_loader.py   # Recursive skill tree walker, loads SKILL.md by name
├── LICENSE           # MIT
└── README.md         # You're reading it
```

### `__init__.py`
The `register(ctx)` function is Hermes's plugin entry point. It calls `ctx.register_hook("pre_llm_call", on_pre_llm_call)`. The hook handler checks the per-session message hash cache, delegates to `_build_skills_context()`, and returns `{"context": "..."}` on cache miss.

### `relevance.py`
`get_available_skills()` walks the entire skills directory tree with `os.walk()`, finds every `SKILL.md`, parses the YAML frontmatter for `name` and `description`, and skips umbrella category headers. Returns a flat list of ~117 skill dicts.

`compute_relevance()` is the dispatcher — calls either `_keyword_relevance()` or `_embedding_relevance()` based on `LAZY_SKILLS_METHOD`. The keyword path builds IDF weights from the full skill corpus, then scores each skill with the three-component formula.

### `skill_loader.py`
`load_skill_content(name)` looks up a skill by its frontmatter `name` field (not directory name). Uses a global path cache — walks the tree once, maps every `name` → `SKILL.md` path, then does direct filesystem reads. Strips YAML frontmatter from the returned content since the agent doesn't need it.

## Edge cases handled

- **Empty message:** returns no context
- **No matching skills (score < threshold):** returns no context, agent uses system prompt fallback
- **Skill SKILL.md missing:** returns empty string for that skill, skips it in output
- **Same session, follow-up message:** hash cache hit, returns `None` (no context injection)
- **Topic shift mid-session:** different message hash → cache miss → fresh relevance computation
- **`sentence-transformers` not installed:** embedding method falls back to keyword automatically
- **Umbrella category SKILL.md files:** filtered from discovery (no description, name matches dir)

## What it doesn't do

- **Remove skills from the system prompt.** That would require patching Hermes core. This plugin tells the agent to *ignore* unlisted skills, which is functionally equivalent — the agent just won't look at them.
- **Handle skills that are genuinely needed for edge cases.** If the user says "deploy" and means Docker (which is matched) but actually needs Supabase too (which isn't), the agent can still call `skill_view("supabase-operations")` manually. The instruction says "ignore other skills" but doesn't prohibit it.
- **Work for cron jobs** (yet). Cron sessions don't go through the same pre_llm_call path. That's a future feature.

## License

MIT. Do whatever you want with it.
