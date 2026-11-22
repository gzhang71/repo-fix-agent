from .base import Model


class AuditMixin:
    """Records every save in store["audit_log"] as (table, id)."""

    def save(self):
        result = super().save()
        self.store.setdefault("audit_log", []).append((self.table, self.fields.get("id")))
        return result


class ValidationMixin:
    """Checks that every field named in `required` is present before saving."""

    required = ()

    def save(self):
        missing = [f for f in self.required if self.fields.get(f) in (None, "")]
        if missing:
            raise ValueError(f"{type(self).__name__} missing required fields: {', '.join(missing)}")
        return Model.save(self)
