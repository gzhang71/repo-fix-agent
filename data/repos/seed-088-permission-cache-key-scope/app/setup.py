from authz.cache import PermissionCache
from authz.memberships import MembershipStore
from authz.policy import Authorizer

from .documents import DocumentService


def build_app():
    cache = PermissionCache()
    members = MembershipStore(cache)
    docs = DocumentService(Authorizer(members, cache))
    return members, docs
