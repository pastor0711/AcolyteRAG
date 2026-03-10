"""
Defines foundational concept groupings and categories, mapping related words 
(like "happy", "joy", "cheerful") into consistent thematic buckets. 
Also manages the registration and lookup of custom concept sets.
"""
from __future__ import annotations
import sys
from typing import Dict, FrozenSet, List, Set
from functools import lru_cache

from .constants import _PUNCTUATION_PATTERN, _WHITESPACE_PATTERN

_CORE_CONCEPTS: Dict[str, List[str]] = {
    'love'              : ['love', 'loved', 'romance', 'affection', 'passion', 'heart', 'crush',
                     'dating', 'relationship', 'partner', 'devotion', 'intimate', 'kiss',
                     'hug', 'sweetheart'],
    'anger'             : ['angry', 'mad', 'furious', 'rage', 'frustrated', 'annoyed',
                     'irritated', 'outraged', 'hostile', 'wrath', 'resentful', 'bitter',
                     'grudge', 'vindictive'],
    'fear'              : ['afraid', 'scared', 'fear', 'terrified', 'worried', 'anxious',
                     'nervous', 'frightened', 'panicked', 'dread', 'phobia', 'trembling',
                     'vulnerable'],
    'sadness'           : ['sad', 'sorrow', 'grief', 'depressed', 'melancholy', 'crying', 'tears',
                     'heartbroken', 'miserable', 'despair', 'hopeless', 'lonely',
                     'mourning', 'regret'],
    'happiness'         : ['happy', 'joy', 'cheerful', 'excited', 'thrilled', 'delighted',
                     'pleased', 'ecstatic', 'content', 'satisfied', 'grateful', 'blissful',
                     'optimistic', 'laughing', 'proud'],
    'trust'             : ['trust', 'trustworthy', 'reliable', 'loyal', 'faithful', 'honest',
                     'sincere', 'credible', 'confidence', 'faith'],
    'betrayal'          : ['betray', 'betrayed', 'traitor', 'deceive', 'lie', 'cheat',
                     'unfaithful', 'disloyal', 'double-cross', 'manipulate'],
    'surprise'          : ['surprised', 'shocked', 'amazed', 'astonished', 'stunned',
                     'unexpected', 'sudden', 'startled', 'bewildered'],
    'guilt'             : ['guilt', 'guilty', 'ashamed', 'shame', 'regret', 'remorse', 'sorry',
                     'apologize', 'conscience', 'fault'],
    'meeting'           : ['meet', 'met', 'encounter', 'bumped', 'introduced', 'greeted',
                     'reunion', 'rendezvous', 'appointment', 'gathering'],
    'conflict'          : ['fight', 'argue', 'disagreement', 'battle', 'struggle', 'quarrel',
                     'clash', 'feud', 'war', 'confrontation', 'brawl'],
    'help'              : ['help', 'assist', 'support', 'aid', 'rescue', 'save', 'protect',
                     'defend', 'guide', 'encourage', 'cooperate'],
    'travel'            : ['travel', 'journey', 'trip', 'voyage', 'adventure', 'explore', 'drive',
                     'fly', 'sail', 'walk', 'expedition', 'departure', 'arrival',
                     'destination'],
    'work'              : ['work', 'job', 'employment', 'career', 'task', 'duty', 'project',
                     'assignment', 'mission', 'contract', 'shift'],
    'create'            : ['create', 'make', 'build', 'design', 'invent', 'discover', 'art',
                     'write', 'compose', 'produce', 'develop', 'craft'],
    'destroy'           : ['destroy', 'break', 'ruin', 'demolish', 'wreck', 'damage', 'sabotage',
                     'shatter', 'smash', 'annihilate'],
    'investigate'       : ['investigate', 'search', 'seek', 'probe', 'examine', 'research',
                     'inquire', 'question', 'uncover', 'solve'],
    'home'              : ['home', 'house', 'apartment', 'room', 'bedroom', 'kitchen', 'living',
                     'bathroom', 'garden', 'residence', 'flat', 'condo'],
    'workplace'         : ['office', 'business', 'company', 'desk', 'computer', 'meeting', 'boss',
                     'employee', 'workplace', 'boardroom'],
    'school'            : ['school', 'university', 'college', 'classroom', 'library', 'campus',
                     'lecture', 'exam', 'grade', 'professor', 'student'],
    'outdoors'          : ['park', 'forest', 'beach', 'mountain', 'city', 'street', 'wilderness',
                     'nature', 'trail', 'river', 'valley'],
    'social_venue'      : ['bar', 'restaurant', 'cafe', 'club', 'party', 'pub', 'tavern',
                     'lounge', 'store', 'mall', 'market'],
    'family'            : ['family', 'parent', 'mother', 'father', 'child', 'son', 'daughter',
                     'sibling', 'brother', 'sister', 'relative', 'grandmother',
                     'grandfather', 'aunt', 'uncle', 'cousin'],
    'friendship'        : ['friend', 'buddy', 'pal', 'companion', 'ally', 'comrade',
                     'acquaintance', 'colleague', 'teammate', 'best friend', 'loyalty',
                     'bond', 'trust', 'camaraderie'],
    'romance'           : ['lover', 'sweetheart', 'partner', 'boyfriend', 'girlfriend', 'husband',
                     'wife', 'spouse', 'fiancé', 'crush', 'affair', 'engagement', 'wedding',
                     'dating', 'breakup'],
    'authority'         : ['boss', 'leader', 'chief', 'captain', 'commander', 'officer',
                     'authority', 'hierarchy', 'command', 'control', 'dominance', 'master',
                     'king', 'queen', 'president', 'manager'],
    'adversary'         : ['enemy', 'foe', 'rival', 'adversary', 'opponent', 'nemesis', 'villain',
                     'criminal', 'thug', 'gangster', 'traitor'],
    'mental_state'      : ['sane', 'mad', 'rational', 'irrational', 'lucid', 'confused',
                     'focused', 'distracted', 'aware', 'mindful', 'stable', 'unstable',
                     'sharp', 'dull', 'motivated', 'inspired'],
    'psychology'        : ['depression', 'anxiety', 'stress', 'trauma', 'ptsd', 'panic', 'phobia',
                     'paranoia', 'delusion', 'hallucination', 'therapy', 'counseling',
                     'addiction', 'recovery', 'resilience'],
    'health'            : ['sick', 'ill', 'disease', 'fever', 'cough', 'cold', 'flu', 'injury',
                     'wound', 'pain', 'recovery', 'healing', 'healthy', 'fit', 'strong',
                     'energetic', 'exercise'],
    'past'              : ['yesterday', 'ago', 'before', 'earlier', 'previous', 'last', 'former',
                     'old', 'ancient', 'history', 'memory', 'flashback', 'nostalgia',
                     'used to', 'once'],
    'future'            : ['tomorrow', 'later', 'soon', 'next', 'upcoming', 'destiny', 'plan',
                     'intend', 'goal', 'dream', 'aspiration', 'hope'],
    'communication'     : ['talk', 'speak', 'chat', 'discuss', 'conversation', 'dialogue',
                     'whisper', 'shout', 'message', 'call', 'write', 'read', 'email',
                     'letter', 'announce'],
    'knowledge'         : ['learn', 'study', 'education', 'school', 'lesson', 'teacher',
                     'student', 'knowledge', 'wisdom', 'understand', 'analyze', 'research',
                     'science', 'data', 'theory'],
    'morality'          : ['moral', 'ethics', 'right', 'wrong', 'good', 'bad', 'virtue', 'vice',
                     'honor', 'integrity', 'honesty', 'justice', 'fairness', 'compassion',
                     'courage', 'loyalty'],
    'power'             : ['power', 'strength', 'force', 'control', 'dominance', 'submission',
                     'influence', 'persuasion', 'manipulation', 'privilege', 'oppression',
                     'freedom', 'authority'],
    'mystery'           : ['secret', 'hidden', 'concealed', 'mysterious', 'enigmatic', 'clue',
                     'evidence', 'reveal', 'expose', 'spy', 'conspiracy', 'plot', 'scheme',
                     'intrigue', 'unknown'],
    'fantasy'           : ['magic', 'spell', 'wizard', 'witch', 'dragon', 'elf', 'dwarf',
                     'monster', 'creature', 'supernatural', 'enchantment', 'potion',
                     'quest', 'kingdom', 'castle'],
    'sci_fi'            : ['spaceship', 'alien', 'robot', 'android', 'ai', 'cyborg', 'clone',
                     'warp', 'laser', 'hologram', 'quantum', 'colony'],
    'horror'            : ['ghost', 'demon', 'haunted', 'curse', 'possession', 'terror',
                     'darkness', 'nightmare', 'paranormal', 'creature', 'evil'],
}


def _normalize_group_name(group_name: str) -> str:
    return str(group_name).strip().lower()


def _normalize_concept_term(term: str) -> str:
    normalized = str(term).strip().lower()
    normalized = _PUNCTUATION_PATTERN.sub(" ", normalized)
    normalized = _WHITESPACE_PATTERN.sub(" ", normalized).strip()
    return normalized


_registry: Dict[str, Set[str]] = {
    _normalize_group_name(group): set(words) for group, words in _CORE_CONCEPTS.items()
}


def _clear_runtime_concept_caches() -> None:
    _get_word_to_groups.cache_clear()
    _get_group_sets.cache_clear()

    from .concept_expansion import _expand_concepts
    from .narrative_elements import _extract_narrative_elements
    from .text_processing import _tokenize

    _expand_concepts.cache_clear()
    _extract_narrative_elements.cache_clear()
    _tokenize.cache_clear()

    for module_name, cache_names in {
        f"{__package__}.analysis": (
            "_tokenize",
            "_extract_narrative_elements",
        ),
        f"{__package__}.retrieval": (
            "_tokenize",
            "_expand_concepts",
            "_extract_narrative_elements",
            "_correct_token_typo",
            "_get_typo_reference_terms",
        ),
    }.items():
        module = sys.modules.get(module_name)
        if module is None:
            continue
        for cache_name in cache_names:
            cached_fn = getattr(module, cache_name, None)
            if cached_fn is not None and hasattr(cached_fn, "cache_clear"):
                cached_fn.cache_clear()


def register_concepts(group_name: str, words: List[str]) -> None:
    """
    Dynamically adds a new concept group or updates an existing one with new words.
    Clears internal caches to ensure the new words are recognized immediately.
    """
    group_name = _normalize_group_name(group_name)
    normalized_words = {str(word).strip().lower() for word in words if str(word).strip()}
    if group_name not in _registry:
        _registry[group_name] = set()
    _registry[group_name].update(normalized_words)
    _clear_runtime_concept_caches()


def unregister_concepts(group_name: str) -> bool:
    """
    Removes a concept group dynamically and clears internal caches.
    """
    group_name = _normalize_group_name(group_name)
    existed = group_name in _registry
    if existed:
        del _registry[group_name]
        _clear_runtime_concept_caches()
    return existed


def list_concept_groups() -> List[str]:
    
    return sorted(_registry.keys())


@lru_cache(maxsize=1)
def _get_word_to_groups() -> Dict[str, FrozenSet[str]]:
    index: Dict[str, Set[str]] = {}
    for group, words in _registry.items():
        for word in words:
            normalized_variants = {word.lower(), _normalize_concept_term(word)}
            for variant in normalized_variants:
                if not variant:
                    continue
                if variant not in index:
                    index[variant] = set()
                index[variant].add(group)
    return {w: frozenset(gs) for w, gs in index.items()}


@lru_cache(maxsize=1)
def _get_group_sets() -> Dict[str, FrozenSet[str]]:
    return {group: frozenset(words) for group, words in _registry.items()}


def get_word_to_groups() -> Dict[str, FrozenSet[str]]:
    return _get_word_to_groups()


def get_group_sets() -> Dict[str, FrozenSet[str]]:
    return _get_group_sets()
