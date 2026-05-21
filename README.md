# lazy-load-skills

A Hermes Agent plugin that reduces system prompt token usage by only loading skills relevant to the current user message.

When Hermes starts a conversation, it injects all available skills into the system prompt — names, descriptions, the full catalog. With 100+ skills installed, this adds 3-8K tokens to every turn. The LLM reads them regardless of whether they apply to the current task.

This plugin intercepts each user message before it reaches the LLM, computes which skills are relevant, pre-loads their full SKILL.md content as ephemeral context, and instructs the agent to ignore the rest. The LLM only sees skills it will use, and it does not need to call `skill_view()` for them.

## Token savings

| Metric | Before | After |
|---|---|---|
| Skills in system prompt | 100+ (3-8K tokens) | Full list still present, but agent ignores unlisted |
| Skills the LLM reads | All of them | 5 (configurable) |
| Pre-loaded content | 0 (agent calls `skill_view()`) | Top-N SKILL.md injected as ephemeral context |
| Net token reduction | — | 2,000–5,000 per session |

Plus API call savings: each `skill_view()` the agent would have made becomes a local filesystem read.

## Accuracy

Tested against 117 skills installed in this Hermes instance. The keyword scorer produces:

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

The one scenario with weaker results is queries with no matching skill in the catalog (e.g., "Deploy to AWS" when no AWS skill exists). The scorer returns the best available match, which may not be directly relevant.

## How it hooks in

The plugin uses Hermes's `pre_llm_call` hook, which fires once per user turn before the message reaches the LLM. The flow:

1. Receive the raw user message, session ID, and first-turn flag
2. Check a per-session hash cache — same conversation, same skills, skip computation
3. Run TF-IDF keyword scoring against all skill names and descriptions
4. Load the full SKILL.md for the top-N skills (default: 5)
5. Return `{"context": "..."}` with pre-loaded skills and an instruction to ignore others

Hermes injects this context as ephemeral text — the LLM sees it attached to the user message, but it is not persisted to the session database.

```
┌──────────────────────────────────────────────────┐
│                System Prompt                     │
│  (still contains the full skills list, but the   │
│   agent is instructed to ignore unlisted ones)   │
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

## Scoring algorithm

The keyword scorer uses three weighted components. Each is normalized to 0.0–1.0 and combined.

### 1. IDF-weighted overlap (50%)

Tokens that appear in many skill descriptions ("create", "use", "set") get low weight. Tokens that appear in few ("docker", "supabase", "gguf") get high weight. This prevents skills with long, generic descriptions from dominating every query.

### 2. Name match boost (35%)

A query token appearing in the skill's name is weighted higher than the same token in the description. "docker" in `docker-management` beats "docker" in a 200-word description. This is why `docker-management` ranks first for Docker queries despite having a shorter description than some alternatives.

### 3. Query coverage (15%)

The fraction of query tokens that found any match. Prevents single-token matches from outranking skills that match multiple query terms.

Skills scoring below `LAZY_SKILLS_MIN_SCORE` (default 0.15) are filtered entirely.

### Embedding method (optional)

An embedding-based method is available via `LAZY_SKILLS_METHOD=embedding`. It uses `sentence-transformers` with `all-MiniLM-L6-v2` for semantic matching. More accurate for conceptual queries, but adds an 80MB model download and 50-200ms per query. For this use case — matching user queries to skill names and descriptions — keyword overlap already performs well because skill descriptions use the exact terminology users type.

## Installation

```bash
git clone https://github.com/underdown/lazy-load-skills.git
cp -r lazy-load-skills ~/.hermes/plugins/
hermes plugins enable lazy-load-skills
hermes plugins list | grep lazy
```

No dependencies for the default keyword method. To use embeddings:

```bash
pip install sentence-transformers
echo 'LAZY_SKILLS_METHOD=embedding' >> ~/.hermes/.env
```

## Configuration

Set in `~/.hermes/.env`:

| Variable | Default | Description |
|---|---|---|
| `LAZY_SKILLS_TOP_N` | `5` | Max skills to pre-load per message |
| `LAZY_SKILLS_METHOD` | `keyword` | `keyword` or `embedding` |
| `LAZY_SKILLS_MIN_SCORE` | `0.15` | Cutoff threshold. Lower = more skills, more noise. Higher = fewer skills, higher precision. |
| `LAZY_SKILLS_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Model for embedding method. Any SentenceTransformer model works. |

## Files

```
lazy-load-skills/
├── plugin.yaml       # Hermes manifest — declares the pre_llm_call hook
├── __init__.py       # Entry point — register(ctx), hook handler, session cache
├── relevance.py      # TF-IDF keyword scorer, optional embedding scorer, skill discovery
├── skill_loader.py   # Recursive skill tree walker, loads SKILL.md by name
├── LICENSE           # MIT
└── README.md
```

### `__init__.py`

The `register(ctx)` function registers the `pre_llm_call` hook. The hook handler checks the per-session cache, delegates to `_build_skills_context()`, and returns `{"context": "..."}` on cache miss.

### `relevance.py`

`get_available_skills()` walks the skills directory tree with `os.walk()`, finds every `SKILL.md`, parses the YAML frontmatter for `name` and `description`, and skips umbrella category headers. Returns a flat list.

`compute_relevance()` dispatches to either `_keyword_relevance()` or `_embedding_relevance()` based on `LAZY_SKILLS_METHOD`. The keyword path builds IDF weights from the full skill corpus, then scores each skill with the three-component formula.

### `skill_loader.py`

`load_skill_content(name)` looks up a skill by its frontmatter `name` field. Uses a global path cache — walks the tree once on first call, maps every name to its `SKILL.md` path, then does direct filesystem reads. Strips YAML frontmatter from the returned content.

## Edge cases

- **Empty message:** returns no context
- **No skills above threshold:** returns no context, agent uses system prompt fallback
- **SKILL.md missing for a matched skill:** returns empty string, skips it in output
- **Follow-up in same session:** hash cache hit, returns `None` (no context injection)
- **Topic shift mid-session:** different message hash triggers fresh relevance computation
- **`sentence-transformers` not installed:** embedding method falls back to keyword automatically
- **Umbrella category SKILL.md files:** filtered from discovery

## Limitations

- **Does not remove skills from the system prompt.** That would require modifying Hermes core. The plugin instructs the agent to ignore unlisted skills.
- **If the user's intent genuinely spans more skills than `TOP_N`**, the agent can still call `skill_view()` manually for any unlisted skill.
- **Not yet supported for cron job sessions.** Cron jobs use a different execution path.

## License

MIT
