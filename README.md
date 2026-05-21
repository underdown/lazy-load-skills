# lazy-load-skills

**Hermes Agent plugin** — reduces system prompt token usage by **2-5K tokens** per session by only loading skills relevant to the current conversation.

Instead of dumping all 100+ available skills into the system prompt (where the LLM has to read them every turn), this plugin:

1. **Intercepts** every user message via the `pre_llm_call` hook
2. **Scores** all available skills against the message content using TF-IDF-style keyword relevance
3. **Pre-loads** the full SKILL.md for only the top-N most relevant skills
4. **Injects** their content as ephemeral context with instructions to ignore the rest

The result: the LLM never wastes tokens reading irrelevant skill descriptions, never calls `skill_view()` for pre-loaded skills, and stays focused on what actually matters.

## Installation

```bash
# Install the plugin
hermes plugins install lazy-load-skills

# Enable it
hermes plugins enable lazy-load-skills
```

Or manually:
```bash
cp -r lazy-load-skills ~/.hermes/plugins/lazy-load-skills
# Then add to config.yaml:
#   plugins:
#     enabled:
#       - lazy-load-skills
```

## Configuration

All configuration is via environment variables in `~/.hermes/.env`:

| Variable | Default | Description |
|---|---|---|
| `LAZY_SKILLS_TOP_N` | `5` | Max skills to pre-load per message |
| `LAZY_SKILLS_METHOD` | `keyword` | `keyword` (fast, no deps) or `embedding` (more accurate, needs `sentence-transformers`) |
| `LAZY_SKILLS_MIN_SCORE` | `0.15` | Minimum relevance score 0.0–1.0 |
| `LAZY_SKILLS_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Embedding model name (only for `embedding` method) |

## How It Works

```
User sends: "Help me set up a Docker container"
         │
         ▼
┌──────────────────────────────────────┐
│  lazy-load-skills (pre_llm_call)     │
│                                      │
│  1. Tokenize & compute IDF weights   │
│  2. Score all 117 skills:            │
│     docker-management    0.23  ✓     │
│     office-hours         0.17        │
│     1password            0.15        │
│     spotify              0.03  ✗     │
│     arxiv                0.02  ✗     │
│     ...106 more filtered out         │
│  3. Load top-5 SKILL.md files        │
│  4. Inject ephemeral context         │
└──────────────────────────────────────┘
         │
         ▼
    Agent receives:
    "These skills are pre-loaded — do NOT call skill_view().
     Ignore all other skills in the system prompt.
     
     ## Skill: docker-management (relevance: 23%)
     Manage Docker containers, images, volumes, networks..."
```

## Relevance Scoring

The keyword method uses three weighted components:

| Component | Weight | Description |
|---|---|---|
| **IDF overlap** | 50% | Rare matching terms count more than common ones (TF-IDF style) |
| **Name match** | 35% | Direct hits in the skill NAME are heavily boosted |
| **Query coverage** | 15% | What fraction of query tokens found a match |

### Real-world accuracy (tested on 117 skills)

| Query | Top-ranked skill | Score |
|---|---|---|
| "Docker container" | `docker-management` | 0.23 |
| "Create a pull request" | `github-code-review` | 0.31 |
| "Python script bug debug breakpoints" | `python-debugpy` | 0.38 |
| "Landing page dark theme gradient" | `frontend-design` | 0.24 |
| "Send email to team" | `himalaya` | 0.18 |
| "Fine-tune Llama model" | `llama-cpp` | 0.46 |
| "Supabase create table" | `supabase-operations` | 0.39 |
| "Play Discover Weekly" | `spotify` | 0.34 |

## Token Savings

For a typical Hermes installation with 100+ skills:
- System prompt skills list: ~3-8K tokens
- With lazy loading: the agent is told to ignore unlisted skills
- 5 pre-loaded skills in ephemeral context: ~1-3K tokens
- **Net savings: 2-5K tokens per session**

Plus API call savings from skipping `skill_view()` for pre-loaded skills.

## Caching

Per-session message hash cache avoids recomputing relevance when the user sends follow-ups in the same conversation. A significant topic shift (detected by message hash change) triggers a fresh relevance computation.

## Embedding Method (Optional)

For ambiguous queries where keywords miss semantic meaning, use the embedding method:

```bash
pip install sentence-transformers
# Set in ~/.hermes/.env:
LAZY_SKILLS_METHOD=embedding
```

First call downloads the model (~80MB). Subsequent calls use the cached model.

## License

MIT — see [LICENSE](LICENSE)
