"""
Stores globally utilized constant values such as text processing stopwords, 
irregular verb lemmas, and common regex patterns.
"""
import re
from typing import Dict, Set

_STOPWORDS: Set[str] = {
    "the", "a", "an", "and", "or", "but", "if", "then", "else",
    "as", "while", "to", "of", "in", "on", "for", "with", "by",
    "up", "at", "from", "into", "about", "above", "below", "between",
    "when", "during", "before", "after", "since", "until",
    "that", "this", "these", "those", "it", "its",
    "i", "you", "he", "she", "we", "they",
    "me", "him", "her", "us", "them",
    "my", "your", "his", "hers", "our", "their",
    "is", "are", "was", "were", "be", "been", "am",
    "do", "does", "did", "doing",
    "have", "has", "had", "having",
    "will", "would", "shall", "should", "may", "might", "must", "can", "could",
    "yes", "yeah", "yep", "nope", "ok", "okay", "well", "um", "uh", "hmm", "huh", "hey", "yo",
    "say", "says", "said", "saying",
    "tell", "tells", "told", "telling",
    "ask", "asks", "asked", "asking",
    "look", "looks", "looked", "looking",
    "see", "sees", "saw", "seeing",
    "go", "goes", "went", "going",
    "come", "comes", "came", "coming",
    "get", "gets", "got", "getting",
    "make", "makes", "made", "making",
    "take", "takes", "took", "taking",
    "give", "gives", "gave", "giving",
    "put", "puts", "putting",
    "not", "now", "then", "here", "there", "just", "also", "so",
    "very", "really", "quite", "rather", "already",
}

_IRREGULAR_LEMMA_MAP: Dict[str, str] = {
    "was": "be", "were": "be", "been": "be", "am": "be", "is": "be", "are": "be",
    "had": "have", "has": "have", "having": "have",
    "did": "do", "does": "do", "doing": "do", "done": "do",
    "went": "go", "gone": "go", "going": "go",
    "came": "come", "coming": "come",
    "made": "make", "making": "make",
    "took": "take", "taken": "take", "taking": "take",
    "gave": "give", "given": "give", "giving": "give",
    "ran": "run", "running": "run",
    "knew": "know", "known": "know", "knowing": "know",
    "saw": "see", "seen": "see", "seeing": "see",
    "thought": "think", "thinking": "think",
    "felt": "feel", "feeling": "feel",
    "found": "find", "finding": "find",
    "got": "get", "gotten": "get", "getting": "get",
    "said": "say", "saying": "say",
    "told": "tell", "telling": "tell",
    "put": "put",
    "bought": "buy", "buying": "buy",
    "brought": "bring", "bringing": "bring",
    "spoke": "speak", "spoken": "speak", "speaking": "speak",
    "talked": "talk", "talking": "talk",
    "wrote": "write", "written": "write", "writing": "write",
    "read": "read",
    "sat": "sit", "sitting": "sit",
    "stood": "stand", "standing": "stand",
    "began": "begin", "begun": "begin", "beginning": "begin",
    "started": "start", "starting": "start",
    "met": "meet", "meeting": "meet",
    "left": "leave", "leaving": "leave",
    "fought": "fight", "fighting": "fight",
    "lost": "lose", "losing": "lose",
    "won": "win", "winning": "win",
}

_PUNCTUATION_PATTERN = re.compile(r"[^\w\s]")
_WHITESPACE_PATTERN  = re.compile(r"\s+")
_NAME_PATTERN        = re.compile(r"\b[A-Z][a-z]{1,20}\b")
