"""
Documentation Version Control System for DocuTutor.
Handles versioning, change tracking, and evolution of documentation.
"""

from typing import Dict, List, Optional
import time
import hashlib
import json

class DocVersion:
    def __init__(self, content: str, metadata: Dict):
        self.content = content
        self.metadata = metadata
        self.timestamp = time.time()
        self.version_hash = self._generate_hash()

    def _generate_hash(self) -> str:
        """Generate a unique hash for this version."""
        content_str = f"{self.content}{json.dumps(self.metadata)}{self.timestamp}"
        return hashlib.sha256(content_str.encode()).hexdigest()

class VersionHistory:
    def __init__(self):
        self.versions: List[DocVersion] = []
        self.current_version: Optional[DocVersion] = None

    def add_version(self, content: str, metadata: Dict) -> DocVersion:
        """Add a new version to the history."""
        version = DocVersion(content, metadata)
        self.versions.append(version)
        self.current_version = version
        return version

    def get_version(self, version_hash: str) -> Optional[DocVersion]:
        """Retrieve a specific version by its hash."""
        return next((v for v in self.versions if v.version_hash == version_hash), None)

    def list_versions(self) -> List[Dict]:
        """List all versions with their metadata."""
        return [
            {
                'version_hash': v.version_hash,
                'timestamp': v.timestamp,
                'metadata': v.metadata
            }
            for v in self.versions
        ]

class DocumentVersionControl:
    def __init__(self):
        self.documents: Dict[str, VersionHistory] = {}

    def create_document(self, doc_id: str, content: str, metadata: Dict) -> DocVersion:
        """Create a new document with version control."""
        if doc_id not in self.documents:
            self.documents[doc_id] = VersionHistory()
        return self.documents[doc_id].add_version(content, metadata)

    def update_document(self, doc_id: str, content: str, metadata: Dict) -> DocVersion:
        """Update an existing document, creating a new version."""
        if doc_id not in self.documents:
            raise KeyError(f"Document {doc_id} does not exist")
        return self.documents[doc_id].add_version(content, metadata)

    def get_document_history(self, doc_id: str) -> List[Dict]:
        """Get the version history for a document."""
        if doc_id not in self.documents:
            raise KeyError(f"Document {doc_id} does not exist")
        return self.documents[doc_id].list_versions()

    def get_document_version(self, doc_id: str, version_hash: str) -> Optional[DocVersion]:
        """Get a specific version of a document."""
        if doc_id not in self.documents:
            raise KeyError(f"Document {doc_id} does not exist")
        return self.documents[doc_id].get_version(version_hash)
