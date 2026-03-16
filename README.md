# AcolyteRAG

**AcolyteRAG** is the open-source RAG (Retrieval-Augmented Generation) engine powering [AcolyteAI](https://www.acolyteai.net). It retrieves semantically relevant messages from a conversation history to provide context for language model generation.

## Features

- **Two-phase retrieval** — fast Jaccard pre-filter → detailed TF-IDF + concept-overlap scoring with 10 blended signals
- **Narrative element extraction** — automatically identifies emotions, actions, locations, and named entities in text
- **Bidirectional typo correction** — OSA distance-based fuzzy matching corrects misspellings in both queries and candidates
- **Diversity clustering** — concept-based clustering ensures retrieved memories cover different topics
- **Extensible concept registry** — 36 built-in semantic groups; add any domain in one line
- **Token-budget mode** — fill a context window to a target token count instead of a fixed count
- **Concept Manager GUI** — local web app for managing concepts and tuning scoring weights with live preview
- **Sync + async** — `retrieve_related_messages` and `retrieve_related_messages_async`
- **Zero external dependencies** — pure Python stdlib

## Installation

```bash
# Clone and install in editable mode
git clone https://github.com/pastor0711/AcolyteRAG.git
cd AcolyteRAG
pip install -e .
```

## Quick Start

```python
from acolyterag import retrieve_related_messages

history = [
    {"role": "user",      "content": "Tell me about Python."},
    {"role": "assistant", "content": "Python is a high-level language known for readability."},
    # ... more messages
]

results = retrieve_related_messages(history, query_text="Python web frameworks", max_retrieved=3)
for msg in results:
    print(msg["content"])  # prefixed with [RELATED_MEMORY]
```

## Adding Your Own Concepts

The concept registry drives semantic matching. Extend it at any time:

```python
from acolyterag import register_concepts

register_concepts("cooking",  ["bake", "roast", "saute", "simmer", "fry", "broil"])
register_concepts("legal",    ["contract", "lawsuit", "plaintiff", "verdict", "appeal"])
register_concepts("medical",  ["diagnosis", "symptom", "treatment", "prognosis", "dosage"])
```

Changes take effect immediately for all subsequent retrieval calls. Remove a group with `unregister_concepts("cooking")`.

Multi-word phrases are also supported — registering `"best friend"` will match the phrase as a single token during tokenization.

## Concept Manager GUI

A local web app for managing concept groups and tuning scoring weights without touching code.

```bash
python concept_manager.py
# Opens http://localhost:7842
```

### Concepts Tab
- **Sidebar** — lists all concept groups with word counts; searchable
- **Main panel** — shows words as removable tags; add new words via the input field
- **New group** — click "＋ New group", name it, add optional seed words
- **Delete group** — removes the group and its words; automatically cleans up any scoring dimension references

### Scoring Tab
- **Narrative Scoring Dimensions** — manage which concept groups map to each scoring dimension (emotions, actions, locations) with adjustable weights per dimension
- **Blend Weights** — sliders for all 10 scoring signals (TF-IDF, concept overlap, narrative, etc.)
- **Live Preview** — enter a query and candidate message to compute a real-time similarity score using your current (including unsaved) weight configuration
- **Save All** — persists updated weights to `scoring.py` and dimension groups to `narrative_elements.py`

All changes are written directly to the source files (`concepts.py`, `scoring.py`, `narrative_elements.py`) with automatic rollback on failure.

## API Reference

### Retrieval

```python
retrieve_related_messages(
    history,                        # List[Dict[str, str]] — full conversation
    query_text,                     # str — current query
    max_retrieved=4,                # max memories to return
    exclude_last_n=6,               # skip the N most recent messages
    enable_clustering=True,         # diversity clustering
    importance_weight=0.3,          # blend of relevance vs message importance (0–1)
    enable_token_based_retrieval=False,  # fill a token budget instead of fixed count
    target_token_count=14000,       # token budget target
    current_token_count=0,          # tokens already used
    max_retrieved_for_token_target=50,
)
```

`retrieve_related_messages_async()` accepts the same arguments and runs in a thread-pool executor.

### Concept Registry

| Function | Description |
|---|---|
| `register_concepts(group, words)` | Add words to a concept group (creates if new) |
| `unregister_concepts(group)` | Remove a concept group |
| `list_concept_groups()` | List all registered groups |

### Analysis Utilities

| Function | Description |
|---|---|
| `summarize_conversation_window(history, start, end)` | Short text summary of a history slice |
| `extract_conversation_topics(history, min_frequency)` | Token frequency map of recurring topics |
| `get_memory_statistics(history)` | Message counts, entities, avg importance, topics |
| `create_memory_index(history, chunk_size)` | Chunked index with summaries and metadata |

## How It Works

```
Query
  │
  ▼
Tokenize (normalize → stem → filter stopwords → detect multi-word phrases)
  │
  ▼
Expand concepts (map tokens to semantic groups)
  │
  ▼
Extract narrative elements (emotions, actions, locations, entities)
  │
  ▼
Fast pass (Jaccard overlap + importance + recency) → top 30 candidates
  │                                                  (60 for complex queries)
  ▼
Bidirectional typo correction (OSA distance, both query ↔ candidate)
  │
  ▼
Build IDF over fast-pass candidates
  │
  ▼
Detailed pass (10-signal blend: TF-IDF + concepts + narrative + bigrams + bonuses)
  │
  ▼
Semantic signal filter (require at least one token/concept/entity overlap)
  │
  ▼
Diversity clustering → round-robin pick from each cluster
  │
  ▼
Return tagged memories [RELATED_MEMORY] in chronological order
```

## Scoring Mechanics (Blend Weights)

AcolyteRAG uses 10 configurable blend weights during the detailed scoring pass to determine semantic relevance. These can be adjusted dynamically via the Concept Manager UI:

1. **`tfidf` (20%)**: TF-IDF Cosine Similarity. Measures keyword frequency, giving higher importance to rare/unique words over common ones.
2. **`idf_overlap` (15%)**: IDF Overlap Ratio. Calculates what percentage of the "important" (rare) meaning in the query is captured by the candidate text.
3. **`concept` (20%)**: Concept Matching. Evaluates the Jaccard overlap of semantic concept groups between the query and candidate, finding matching ideas even with different terminology.
4. **`bigram` (5%)**: Bigram Overlap. Checks for exact 2-word phrase matches, rewarding candidates that preserve word order.
5. **`token_coef` (5%)**: Token Overlap Coefficient. Shared tokens divided by the shorter text's token count — prevents penalizing longer texts.
6. **`narrative` (25%)**: Narrative Similarity. Extracts narrative elements and checks for shared emotions (30%), actions (30%), entities/characters (25%), and locations (20%). Ensures thematic and emotional alignment.
7. **`substring` (10%)**: Substring Bonus. A flat +0.2 bonus if the entire query (>10 chars) is found perfectly intact within the candidate.
8. **`entity_bonus` (5%)**: Entity Matching Bonus. Awards up to +0.3 bonus if both texts reference the same named entities (capitalized words, with sentence-start false-positive filtering).
9. **`temporal_bonus` (10%)**: Temporal Context Bonus. Adds +0.1 flat bonus if both texts contain time-anchoring words (e.g., "tomorrow", "yesterday", "later").
10. **`action_bonus` (10%)**: Action/Event Bonus. Adds +0.1 flat bonus if both texts contain event-driven action verbs (e.g., "arrives", "happened", "took place").

The final retrieval score blends relevance with message importance: `(1 - importance_weight) × detailed_score + importance_weight × importance_score`.

## Module Structure

| Module | Purpose |
|---|---|
| `constants.py` | Stopwords, irregular verb lemma map, regex patterns |
| `text_processing.py` | Normalization, tokenization, custom stemmer/lemmatizer |
| `concepts.py` | Concept registry (36 groups, ~525 words), dynamic registration API |
| `concept_expansion.py` | Maps tokens to their semantic concept groups |
| `narrative_elements.py` | Extracts emotions, actions, locations, and named entities from text |
| `importance.py` | Heuristic message importance scoring (length, emotions, actions, entities) |
| `scoring.py` | TF-IDF, Jaccard, overlap coefficient, 10-signal blended scoring engine |
| `retrieval.py` | Two-phase retrieval pipeline, typo correction, clustering, token-budget mode |
| `analysis.py` | Conversation summarization, topic extraction, memory statistics, indexing |
| `concept_manager.py` | Local HTTP server + REST API for the management GUI |
| `manager.html/css/js` | Web frontend for the Concept Manager |

## Built-in Concept Groups

36 semantic groups organized into 6 categories:

| Category | Groups |
|---|---|
| **Emotions** | love, anger, fear, sadness, happiness, trust, betrayal, surprise, guilt |
| **Actions** | meeting, conflict, help, travel, work, create, destroy, investigate |
| **Locations** | home, workplace, school, outdoors, social_venue |
| **Relationships** | family, friendship, romance, authority, adversary |
| **Themes** | mental_state, psychology, health, past, future, communication, knowledge, morality, power, mystery |
| **Genres** | fantasy, sci_fi, horror |

See [`concepts.py`](concepts.py) for the full word lists.

## Testing

```bash
# Run the full test suite
pytest

# Run a quick smoke test
python smoke_test.py
```

The test suite includes ~97 tests covering retrieval accuracy, scoring mechanics, typo correction, concept manager HTTP API, file persistence with rollback, cache invalidation, and input validation.

## License

MIT — see [LICENSE](LICENSE).

## Credits

AcolyteRAG is the open-source core of [AcolyteAI](https://www.acolyteai.net).

## AI Disclosure

Please note that Artificial Intelligence (AI) was used in the development and generation of code within this repository.
