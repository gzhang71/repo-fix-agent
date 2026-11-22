from dataclasses import dataclass


@dataclass
class Document:
    id: int
    project_id: str
    title: str


class DocumentService:
    def __init__(self, authorizer):
        self.authz = authorizer
        self.docs = {}

    def add(self, doc):
        self.docs[doc.id] = doc

    def open(self, user_id, doc_id):
        doc = self.docs[doc_id]
        self.authz.require(user_id, "read", doc.project_id)
        return doc

    def delete(self, user_id, doc_id):
        doc = self.docs[doc_id]
        self.authz.require(user_id, "delete", doc.project_id)
        del self.docs[doc_id]
