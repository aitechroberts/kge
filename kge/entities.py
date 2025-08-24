"""
Entity and relationship data structures.
"""

import hashlib
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


@dataclass
class Entity:
    """Represents an extracted entity."""
    name: str
    type: str
    description: Optional[str] = None
    confidence: float = 1.0
    source_text: Optional[str] = None
    entity_id: Optional[str] = None
    
    def __post_init__(self):
        if self.entity_id is None:
            self.entity_id = self._generate_id()
    
    def _generate_id(self) -> str:
        """Generate unique entity ID based on name and type."""
        key = f"{self.type}|{self.name.strip().lower()}"
        return "ent:" + hashlib.sha1(key.encode()).hexdigest()[:12]


@dataclass
class Relationship:
    """Represents an extracted relationship."""
    source: str
    relation: str
    target: str
    confidence: float = 1.0
    source_text: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}