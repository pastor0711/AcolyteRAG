"""
A basic validation script to ensure all public components of AcolyteRAG load successfully 
and the core retrieval pipeline executes without crashing.
"""

import asyncio
import sys
from pathlib import Path

_PACKAGE_ROOT = str(Path(__file__).resolve().parent.parent)
if _PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, _PACKAGE_ROOT)


from acolyterag import (
    retrieve_related_messages,
    retrieve_related_messages_async,
    summarize_conversation_window,
    extract_conversation_topics,
    get_memory_statistics,
    create_memory_index,
    register_concepts,
    list_concept_groups,
)
print("OK: All imports succeeded")


register_concepts("cooking", ["bake", "roast", "saute", "simmer", "fry", "boil"])
groups = list_concept_groups()
print(f"OK: Registered cooking concepts. Groups now: {len(groups)}")
assert "cooking" in groups, "cooking group missing"


history = [
    {"role": "user",      "content": "Tell me about Python programming."},
    {"role": "assistant", "content": "Python is a high-level programming language known for readability."},
    {"role": "user",      "content": "What about web development with Python?"},
    {"role": "assistant", "content": "Flask and Django are popular Python web frameworks."},
    {"role": "user",      "content": "Can you compare Flask and Django?"},
    {"role": "assistant", "content": "Flask is lightweight, Django is batteries-included."},
    {"role": "user",      "content": "Which is better for beginners?"},
    {"role": "assistant", "content": "Flask is simpler for beginners learning web development."},
    {"role": "user",      "content": "What other languages work for web development?"},
    {"role": "assistant", "content": "JavaScript, Ruby, and Go are also popular for web development."},
]

results = retrieve_related_messages(history, "Python web framework comparison", max_retrieved=3)
assert len(results) > 0, "Expected at least one result"
print(f"OK: Retrieved {len(results)} memories:")
for r in results:
    print("  -", r["content"][:90])

assert all("[RELATED_MEMORY]" in r["content"] for r in results), "Missing [RELATED_MEMORY] tag"
print("OK: All results tagged with [RELATED_MEMORY]")

async_results = asyncio.run(
    retrieve_related_messages_async(history, "Python web framework comparison", max_retrieved=2)
)
assert len(async_results) > 0, "Expected at least one async result"
assert all("[RELATED_MEMORY]" in r["content"] for r in async_results), "Missing [RELATED_MEMORY] tag in async results"
print(f"OK: Async retrieval returned {len(async_results)} tagged memories")


stats = get_memory_statistics(history)
assert stats["total_messages"] == 10
total = stats["total_messages"]
avg_imp = stats["average_importance"]
print(f"OK: Stats - {total} msgs, average_importance={avg_imp}")

topics = extract_conversation_topics(history, min_topic_frequency=2)
topic_keys = list(topics.keys())[:5]
print(f"OK: Topics - {topic_keys}")

summary = summarize_conversation_window(history, 0, 5)
print(f"OK: Summary - {summary[:100]}")

index = create_memory_index(history, chunk_size=5)
assert len(index) > 0
print(f"OK: Memory index created with {len(index)} chunks")

print("\nAll tests passed!")
