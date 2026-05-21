# lazy-load-skills

**Hermes Agent plugin** — reduces system prompt token usage by **2-5K tokens** per session by only loading skills relevant to the current conversation.

Instead of dumping all available skills into the system prompt (where the LLM has to read them every turn), this plugin:

1. **Intercepts** every user message via the `pre_llm_call` hook
2. **Scores** all available skills against the message content
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

## Relevance Methods

### Keyword (default)
Fast, zero-dependency TF-IDF-like scoring. Sub-millisecond per query. Handles direct word overlap, partial matches, and normalizes for description length. Works well for 90%+ of queries.

### Embedding (optional)
Semantic similarity via sentence-transformers. Better for ambiguous or conceptual queries. Requires:

```bash
pip install sentence-transformers
```

First call will download the model (~80MB). Subsequent calls use the cached model instance.

## How It Works

```
User sends message: "Help me deploy to AWS"
         │
         ▼
┌─────────────────────────────────────┐
│ lazy-load-skills pre_llm_call hook  │
│                                     │
│ 1. Tokenize: [deploy, aws]          │
│ 2. Score against all skills:        │
│    devops/docker        0.12  ✗     │
│    devops/supabase      0.08  ✗     │
│    mlops/serving        0.05  ✗     │
│    devops/aws-deploy    0.72  ✓     │
│    devops/terraform     0.45  ✓     │
│    security/1password   0.02  ✗     │
│ 3. Pre-load top 5 SKILL.md files    │
│ 4. Inject as ephemeral context      │
└─────────────────────────────────────┘
         │
         ▼
    Agent receives:
    - System prompt (without skills list bloat)
    - User message + injected skills context
    - Agent reads pre-loaded skills directly
    - Agent does NOT call skill_view()
```

## Token Savings

For a typical Hermes installation with 50+ skills:
- System prompt skills list: ~3-8K tokens
- With lazy loading: ~0K tokens (removed by agent instruction)
- 5 pre-loaded skills in ephemeral context: ~1-3K tokens
- **Net savings: 2-5K tokens per session**

Plus API call savings from skipping `skill_view()` for pre-loaded skills.

## Caching

Per-session message hash cache avoids recomputing relevance when the user sends follow-ups in the same conversation. A significant topic shift (detected by message hash change) triggers a fresh relevance computation.

## Debugging

Enable verbose logging:
```bash
export HERMES_PLUGINS_DEBUG=1
```

Or set in `~/.hermes/.env`:
```
LAZY_SKILLS_DEBUG=1
```

Logs appear in `~/.hermes/logs/agent.log` and stderr.

## License

MIT — see [LICENSE](LICENSE)
