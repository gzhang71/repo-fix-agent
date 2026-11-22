from .base import Model
from .mixins import AuditMixin, ValidationMixin


class Customer(AuditMixin, Model):
    table = "customers"


class Order(ValidationMixin, AuditMixin, Model):
    table = "orders"
    required = ("id", "customer_id")


class Note(ValidationMixin, Model):
    table = "notes"
    required = ("text",)
