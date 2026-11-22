"""Seed instances 086-100: business logic in multi-module apps (the hard batch)."""
from ._helpers import B, F, I, R, S, SUBMIT, T, TEST_CMD, V

SPECS = [
    # 86 ---------------------------------------------------------------------------------------
    dict(
        id="seed-086-tax-on-pre-discount-subtotal",
        bug_type="wrong-calculation-base", difficulty="medium",
        issue="""\
            Sales tax charged on the full price even when a coupon is used

            A customer in CA checked out an $80.00 cart with coupon SAVE10:

                subtotal  80.00
                discount  -8.00
                shipping   0.00
                tax        5.80
                total     77.80

            Tax should be 7.25% of what they actually paid for the goods ($72.00), i.e. $5.22,
            for a total of $77.22. Orders without a coupon look correct. Support has had a few
            of these escalated and finance wants to know which code path is wrong.
            """,
        files={
            "shop/__init__.py": "",
            "shop/models.py": '''\
                from dataclasses import dataclass, field
                from typing import Optional


                @dataclass
                class LineItem:
                    sku: str
                    unit_cents: int
                    qty: int = 1

                    @property
                    def total_cents(self):
                        return self.unit_cents * self.qty


                @dataclass
                class Order:
                    lines: list = field(default_factory=list)
                    region: str = "CA"
                    coupon_code: Optional[str] = None
                    discount_cents: int = 0
                    shipping_cents: int = 0
                    tax_cents: int = 0

                    @property
                    def subtotal_cents(self):
                        return sum(line.total_cents for line in self.lines)

                    @property
                    def discounted_subtotal_cents(self):
                        """What the customer pays for goods, after order-level discounts."""
                        return self.subtotal_cents - self.discount_cents

                    @property
                    def total_cents(self):
                        return self.discounted_subtotal_cents + self.shipping_cents + self.tax_cents
                ''',
            "shop/coupons.py": '''\
                COUPONS = {
                    "SAVE10": {"kind": "percent", "value": 10, "min_subtotal_cents": 0},
                    "FIVEOFF": {"kind": "fixed", "value": 500, "min_subtotal_cents": 2500},
                }


                class CouponError(ValueError):
                    pass


                def discount_for(code, subtotal_cents):
                    coupon = COUPONS.get(code.upper())
                    if coupon is None:
                        raise CouponError(f"unknown coupon {code!r}")
                    if subtotal_cents < coupon["min_subtotal_cents"]:
                        raise CouponError(f"{code} requires a larger order")
                    if coupon["kind"] == "percent":
                        return subtotal_cents * coupon["value"] // 100
                    return min(coupon["value"], subtotal_cents)
                ''',
            "shop/tax.py": '''\
                # Rates in basis points (1/100 of a percent).
                RATES_BPS = {"CA": 725, "NY": 400, "OR": 0}


                def tax_rate_bps(region):
                    try:
                        return RATES_BPS[region]
                    except KeyError:
                        raise ValueError(f"no tax rate for region {region!r}") from None


                def compute_tax(order):
                    """Sales tax on goods: charged on the discounted price, never on shipping."""
                    base = order.subtotal_cents
                    return (base * tax_rate_bps(order.region) + 5000) // 10000
                ''',
            "shop/shipping.py": '''\
                FREE_SHIPPING_CENTS = 5000
                FLAT_RATE_CENTS = 599


                def shipping_for(order):
                    if order.discounted_subtotal_cents >= FREE_SHIPPING_CENTS:
                        return 0
                    return FLAT_RATE_CENTS
                ''',
            "shop/checkout.py": '''\
                from .coupons import discount_for
                from .models import Order
                from .shipping import shipping_for
                from .tax import compute_tax


                def checkout(lines, region, coupon_code=None):
                    order = Order(lines=list(lines), region=region, coupon_code=coupon_code)
                    if coupon_code:
                        order.discount_cents = discount_for(coupon_code, order.subtotal_cents)
                    order.shipping_cents = shipping_for(order)
                    order.tax_cents = compute_tax(order)
                    return order


                def receipt(order):
                    return {
                        "subtotal": order.subtotal_cents,
                        "discount": order.discount_cents,
                        "shipping": order.shipping_cents,
                        "tax": order.tax_cents,
                        "total": order.total_cents,
                    }
                ''',
            "tests/test_checkout.py": '''\
                import pytest

                from shop.checkout import checkout, receipt
                from shop.coupons import CouponError, discount_for
                from shop.models import LineItem


                def test_no_coupon_ca():
                    order = checkout([LineItem("mug", 2000, 2)], "CA")
                    assert receipt(order) == {"subtotal": 4000, "discount": 0, "shipping": 599,
                                              "tax": 290, "total": 4889}


                def test_coupon_amounts():
                    assert discount_for("save10", 8000) == 800
                    assert discount_for("FIVEOFF", 3000) == 500


                def test_coupon_minimum():
                    with pytest.raises(CouponError):
                        discount_for("FIVEOFF", 1000)


                def test_free_shipping_threshold():
                    assert checkout([LineItem("lamp", 5000)], "OR").shipping_cents == 0
                ''',
        },
        hidden_test='''\
            from shop.checkout import checkout, receipt
            from shop.models import LineItem


            def test_tax_uses_discounted_subtotal():
                order = checkout([LineItem("jacket", 8000)], "CA", "SAVE10")
                assert receipt(order) == {"subtotal": 8000, "discount": 800, "shipping": 0,
                                          "tax": 522, "total": 7722}


            def test_fixed_coupon_tax_ny():
                order = checkout([LineItem("book", 1500, 2)], "NY", "FIVEOFF")
                assert order.tax_cents == 100
                assert order.total_cents == 2500 + 599 + 100


            def test_no_coupon_unchanged():
                order = checkout([LineItem("mug", 2000, 2)], "CA")
                assert order.tax_cents == 290
            ''',
        p2p=["tests/test_checkout.py::test_no_coupon_ca", "tests/test_checkout.py::test_coupon_amounts",
             "tests/test_checkout.py::test_coupon_minimum", "tests/test_checkout.py::test_free_shipping_threshold"],
        steps=[
            ("Start at the checkout orchestration to see the order totals are assembled.", V("shop/checkout.py")),
            ("Reproduce the customer's order.",
             B("python -c 'from shop.checkout import checkout, receipt; from shop.models import LineItem; print(receipt(checkout([LineItem(\"x\", 8000)], \"CA\", \"SAVE10\")))'")),
            ("The discount (800) is right but tax 580 is 7.25% of 8000, the pre-discount amount. Check the tax module.",
             V("shop/tax.py")),
            ("The docstring says discounted price, but it reads subtotal_cents. Confirm the model has a post-discount property.",
             S("def (subtotal_cents|discounted_subtotal_cents)", "shop")),
            ("Use the discounted subtotal as the tax base; shipping is already excluded.",
             R("shop/tax.py", "    base = order.subtotal_cents\n", "    base = order.discounted_subtotal_cents\n")),
            ("Re-run the repro; expect tax 522 and total 7722.",
             B("python -c 'from shop.checkout import checkout, receipt; from shop.models import LineItem; print(receipt(checkout([LineItem(\"x\", 8000)], \"CA\", \"SAVE10\")))'")),
            ("Run the full test suite.", T()),
            ("Tax is now computed after discounts.", SUBMIT),
        ],
    ),
    # 87 ---------------------------------------------------------------------------------------
    dict(
        id="seed-087-reservation-double-release",
        bug_type="state-not-cleared", difficulty="medium",
        issue="""\
            Available stock goes above on-hand after a cancelled order

            We have 10 units of SKU `TEE-M` on hand. A customer placed an order for 3 and
            cancelled it a minute later. After the nightly reservation sweep the storefront
            shows **13** available, and the cancelled order's status flipped from `cancelled`
            to `expired` in the admin. Looks like the cancelled reservation is being given back
            to the pool a second time.
            """,
        files={
            "warehouse/__init__.py": "",
            "warehouse/stock.py": '''\
                class Stock:
                    def __init__(self, on_hand=None):
                        self.on_hand = dict(on_hand or {})
                        self.reserved = {}

                    def available(self, sku):
                        return self.on_hand.get(sku, 0) - self.reserved.get(sku, 0)
                ''',
            "warehouse/reservations.py": '''\
                from dataclasses import dataclass


                class OutOfStock(Exception):
                    pass


                @dataclass
                class Reservation:
                    order_id: str
                    items: dict
                    expires_at: int


                class ReservationBook:
                    def __init__(self, stock):
                        self.stock = stock
                        self._by_order = {}

                    def reserve(self, order_id, items, expires_at):
                        for sku, qty in items.items():
                            if self.stock.available(sku) < qty:
                                raise OutOfStock(sku)
                        for sku, qty in items.items():
                            self.stock.reserved[sku] = self.stock.reserved.get(sku, 0) + qty
                        self._by_order[order_id] = Reservation(order_id, dict(items), expires_at)

                    def release(self, order_id):
                        """Give reserved units back to the pool (cancellation or timeout)."""
                        res = self._by_order.get(order_id)
                        if res is None:
                            return False
                        for sku, qty in res.items.items():
                            self.stock.reserved[sku] -= qty
                        return True

                    def commit(self, order_id):
                        """Turn a reservation into a real stock decrement when the order ships."""
                        res = self._by_order.pop(order_id)
                        for sku, qty in res.items.items():
                            self.stock.reserved[sku] -= qty
                            self.stock.on_hand[sku] -= qty

                    def expire(self, now):
                        """Release every reservation whose hold has run out; return their order ids."""
                        expired = [oid for oid, r in self._by_order.items() if r.expires_at <= now]
                        for oid in expired:
                            self.release(oid)
                        return expired
                ''',
            "orders/__init__.py": "",
            "orders/models.py": '''\
                from dataclasses import dataclass, field


                @dataclass
                class Order:
                    id: str
                    items: dict = field(default_factory=dict)
                    status: str = "pending"
                ''',
            "orders/service.py": '''\
                from .models import Order

                HOLD_TICKS = 30


                class OrderService:
                    def __init__(self, book):
                        self.book = book
                        self.orders = {}

                    def place(self, order_id, items, now):
                        self.book.reserve(order_id, items, expires_at=now + HOLD_TICKS)
                        order = Order(order_id, dict(items))
                        self.orders[order_id] = order
                        return order

                    def cancel(self, order_id):
                        order = self.orders[order_id]
                        if order.status != "pending":
                            raise ValueError(f"cannot cancel a {order.status} order")
                        self.book.release(order_id)
                        order.status = "cancelled"
                        return order

                    def ship(self, order_id):
                        order = self.orders[order_id]
                        self.book.commit(order_id)
                        order.status = "shipped"
                        return order

                    def sweep(self, now):
                        for order_id in self.book.expire(now):
                            self.orders[order_id].status = "expired"
                ''',
            "tests/test_orders.py": '''\
                import pytest

                from orders.service import OrderService
                from warehouse.reservations import OutOfStock, ReservationBook
                from warehouse.stock import Stock


                def make():
                    stock = Stock({"TEE-M": 10})
                    return stock, OrderService(ReservationBook(stock))


                def test_place_reserves():
                    stock, svc = make()
                    svc.place("o1", {"TEE-M": 3}, now=0)
                    assert stock.available("TEE-M") == 7


                def test_cannot_oversell():
                    stock, svc = make()
                    with pytest.raises(OutOfStock):
                        svc.place("o1", {"TEE-M": 11}, now=0)


                def test_ship_decrements_on_hand():
                    stock, svc = make()
                    svc.place("o1", {"TEE-M": 4}, now=0)
                    svc.ship("o1")
                    assert stock.on_hand["TEE-M"] == 6 and stock.available("TEE-M") == 6


                def test_sweep_expires_stale_holds():
                    stock, svc = make()
                    svc.place("o1", {"TEE-M": 2}, now=0)
                    svc.sweep(now=100)
                    assert svc.orders["o1"].status == "expired"
                    assert stock.available("TEE-M") == 10
                ''',
        },
        hidden_test='''\
            from orders.service import OrderService
            from warehouse.reservations import ReservationBook
            from warehouse.stock import Stock


            def make():
                stock = Stock({"TEE-M": 10})
                return stock, OrderService(ReservationBook(stock))


            def test_cancel_then_sweep_releases_once():
                stock, svc = make()
                svc.place("o1", {"TEE-M": 3}, now=0)
                svc.cancel("o1")
                svc.sweep(now=1000)
                assert stock.available("TEE-M") == 10
                assert svc.orders["o1"].status == "cancelled"


            def test_release_twice_is_noop():
                stock = Stock({"TEE-M": 10})
                book = ReservationBook(stock)
                book.reserve("o1", {"TEE-M": 3}, expires_at=5)
                assert book.release("o1") is True
                assert book.release("o1") is False
                assert stock.available("TEE-M") == 10


            def test_sweep_keeps_fresh_holds():
                stock, svc = make()
                svc.place("o1", {"TEE-M": 2}, now=0)
                svc.sweep(now=10)
                assert stock.available("TEE-M") == 8
            ''',
        p2p=["tests/test_orders.py::test_place_reserves", "tests/test_orders.py::test_cannot_oversell",
             "tests/test_orders.py::test_ship_decrements_on_hand", "tests/test_orders.py::test_sweep_expires_stale_holds"],
        steps=[
            ("Get an overview of the modules involved.", F("*.py")),
            ("Look at how cancel and the sweep interact on the order side.", V("orders/service.py")),
            ("Reproduce: place 3, cancel, sweep, then check availability and status.",
             B("python - <<'EOF'\nfrom orders.service import OrderService\nfrom warehouse.reservations import ReservationBook\nfrom warehouse.stock import Stock\nstock = Stock({'TEE-M': 10}); svc = OrderService(ReservationBook(stock))\nsvc.place('o1', {'TEE-M': 3}, now=0); svc.cancel('o1'); svc.sweep(now=1000)\nprint(stock.available('TEE-M'), svc.orders['o1'].status)\nEOF")),
            ("The service only delegates; the sweep must still see the cancelled hold. Check the reservation book.",
             V("warehouse/reservations.py", 28, 49)),
            ("release() reads the reservation with get() and never removes it, so expire() finds it again and releases a second time. Pop it instead.",
             R("warehouse/reservations.py", "        res = self._by_order.get(order_id)\n",
               "        res = self._by_order.pop(order_id, None)\n")),
            ("Re-run the repro; expect 10 available and status cancelled.",
             B("python - <<'EOF'\nfrom orders.service import OrderService\nfrom warehouse.reservations import ReservationBook\nfrom warehouse.stock import Stock\nstock = Stock({'TEE-M': 10}); svc = OrderService(ReservationBook(stock))\nsvc.place('o1', {'TEE-M': 3}, now=0); svc.cancel('o1'); svc.sweep(now=1000)\nprint(stock.available('TEE-M'), svc.orders['o1'].status)\nEOF")),
            ("Run the test suite.", B(TEST_CMD)),
            ("Releases are now idempotent.", SUBMIT),
        ],
    ),
    # 88 ---------------------------------------------------------------------------------------
    dict(
        id="seed-088-permission-cache-key-scope",
        bug_type="cache-key-collision", difficulty="hard",
        issue="""\
            [security] Viewer was able to delete a document in another project

            Carol is `admin` on the **marketing** project and only `viewer` on **finance**.
            This morning she deleted a finance document (doc 7) without any error. We can
            reproduce it in a fresh process: if she first opens anything in marketing, her
            later delete in finance goes through. If the first thing she does after a restart
            is the finance delete, it is correctly refused with PermissionDenied.

            Role definitions haven't changed in months, so I don't think it's the role table.
            """,
        files={
            "authz/__init__.py": "",
            "authz/roles.py": '''\
                BASE_PERMISSIONS = {
                    "viewer": {"read"},
                    "editor": {"write"},
                    "admin": {"delete", "manage_members"},
                }
                INHERITS = {"viewer": None, "editor": "viewer", "admin": "editor"}


                def permissions_for(role):
                    perms = set()
                    while role is not None:
                        perms |= BASE_PERMISSIONS[role]
                        role = INHERITS[role]
                    return frozenset(perms)
                ''',
            "authz/memberships.py": '''\
                class MembershipStore:
                    def __init__(self, cache=None):
                        self._roles = {}
                        self._cache = cache

                    def set_role(self, user_id, project_id, role):
                        self._roles[(user_id, project_id)] = role
                        if self._cache is not None:
                            self._cache.invalidate_user(user_id)

                    def role_of(self, user_id, project_id):
                        return self._roles.get((user_id, project_id))
                ''',
            "authz/cache.py": '''\
                class PermissionCache:
                    """Memoizes a user's effective permission set within a project."""

                    def __init__(self):
                        self._data = {}
                        self.hits = 0

                    def get_or_compute(self, user_id, project_id, compute):
                        key = user_id
                        if key in self._data:
                            self.hits += 1
                            return self._data[key]
                        perms = compute()
                        self._data[key] = perms
                        return perms

                    def invalidate_user(self, user_id):
                        self._data.pop(user_id, None)
                ''',
            "authz/policy.py": '''\
                from .roles import permissions_for


                class PermissionDenied(Exception):
                    pass


                class Authorizer:
                    def __init__(self, memberships, cache):
                        self.memberships = memberships
                        self.cache = cache

                    def _compute(self, user_id, project_id):
                        role = self.memberships.role_of(user_id, project_id)
                        return permissions_for(role) if role else frozenset()

                    def can(self, user_id, action, project_id):
                        perms = self.cache.get_or_compute(
                            user_id, project_id, lambda: self._compute(user_id, project_id))
                        return action in perms

                    def require(self, user_id, action, project_id):
                        if not self.can(user_id, action, project_id):
                            raise PermissionDenied(f"{user_id} may not {action} in {project_id}")
                ''',
            "app/__init__.py": "",
            "app/documents.py": '''\
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
                ''',
            "app/setup.py": '''\
                from authz.cache import PermissionCache
                from authz.memberships import MembershipStore
                from authz.policy import Authorizer

                from .documents import DocumentService


                def build_app():
                    cache = PermissionCache()
                    members = MembershipStore(cache)
                    docs = DocumentService(Authorizer(members, cache))
                    return members, docs
                ''',
            "tests/test_authz.py": '''\
                import pytest

                from app.documents import Document
                from app.setup import build_app
                from authz.policy import PermissionDenied
                from authz.roles import permissions_for


                def test_role_inheritance():
                    assert permissions_for("admin") == {"read", "write", "delete", "manage_members"}
                    assert permissions_for("viewer") == {"read"}


                def test_viewer_cannot_delete():
                    members, docs = build_app()
                    members.set_role("bob", "finance", "viewer")
                    docs.add(Document(1, "finance", "Q3"))
                    assert docs.open("bob", 1).title == "Q3"
                    with pytest.raises(PermissionDenied):
                        docs.delete("bob", 1)


                def test_promotion_takes_effect():
                    members, docs = build_app()
                    members.set_role("bob", "finance", "viewer")
                    docs.add(Document(1, "finance", "Q3"))
                    docs.open("bob", 1)
                    members.set_role("bob", "finance", "admin")
                    docs.delete("bob", 1)
                    assert 1 not in docs.docs


                def test_cache_hits():
                    members, docs = build_app()
                    members.set_role("bob", "finance", "viewer")
                    docs.add(Document(1, "finance", "Q3"))
                    docs.open("bob", 1); docs.open("bob", 1)
                    assert docs.authz.cache.hits == 1
                ''',
        },
        hidden_test='''\
            import pytest

            from app.documents import Document
            from app.setup import build_app
            from authz.policy import PermissionDenied


            def setup_carol():
                members, docs = build_app()
                members.set_role("carol", "marketing", "admin")
                members.set_role("carol", "finance", "viewer")
                docs.add(Document(3, "marketing", "Launch plan"))
                docs.add(Document(7, "finance", "Budget"))
                return members, docs


            def test_admin_elsewhere_cannot_delete_in_viewer_project():
                members, docs = setup_carol()
                docs.open("carol", 3)
                with pytest.raises(PermissionDenied):
                    docs.delete("carol", 7)
                assert 7 in docs.docs


            def test_viewer_project_first_does_not_block_admin_project():
                members, docs = setup_carol()
                docs.open("carol", 7)
                docs.delete("carol", 3)
                assert 3 not in docs.docs


            def test_role_change_invalidates_every_project():
                members, docs = setup_carol()
                docs.open("carol", 3)
                docs.open("carol", 7)
                members.set_role("carol", "marketing", "viewer")
                with pytest.raises(PermissionDenied):
                    docs.delete("carol", 3)
            ''',
        p2p=["tests/test_authz.py::test_role_inheritance", "tests/test_authz.py::test_viewer_cannot_delete",
             "tests/test_authz.py::test_promotion_takes_effect", "tests/test_authz.py::test_cache_hits"],
        steps=[
            ("Find where delete permission is enforced.", S("delete", ".", "*.py")),
            ("Reproduce Carol's sequence: open a marketing doc, then delete a finance doc.",
             B("python - <<'EOF'\nfrom app.documents import Document\nfrom app.setup import build_app\nmembers, docs = build_app()\nmembers.set_role('carol', 'marketing', 'admin'); members.set_role('carol', 'finance', 'viewer')\ndocs.add(Document(3, 'marketing', 'a')); docs.add(Document(7, 'finance', 'b'))\ndocs.open('carol', 3); docs.delete('carol', 7)\nprint('deleted finance doc:', 7 not in docs.docs)\nEOF")),
            ("First suspicion is role inheritance granting delete too broadly; check the role table.", V("authz/roles.py")),
            ("Roles are fine: viewer only gets read. Order dependence points at state, so look at the authorizer.",
             V("authz/policy.py")),
            ("The authorizer passes project_id to the cache; check how the cache keys it.", V("authz/cache.py")),
            ("The cache key is only user_id, so whichever project is checked first wins for every project. Key by (user, project).",
             R("authz/cache.py", "        key = user_id\n", "        key = (user_id, project_id)\n")),
            ("invalidate_user popped the bare user_id key; it must now drop every project entry for that user.",
             R("authz/cache.py", "        self._data.pop(user_id, None)\n",
               "        for key in [k for k in self._data if k[0] == user_id]:\n            del self._data[key]\n")),
            ("Re-run the repro; the finance delete should now raise PermissionDenied.",
             B("python - <<'EOF' 2>&1 | tail -1\nfrom app.documents import Document\nfrom app.setup import build_app\nmembers, docs = build_app()\nmembers.set_role('carol', 'marketing', 'admin'); members.set_role('carol', 'finance', 'viewer')\ndocs.add(Document(3, 'marketing', 'a')); docs.add(Document(7, 'finance', 'b'))\ndocs.open('carol', 3); docs.delete('carol', 7)\nEOF")),
            ("Run the suite, including the promotion and cache-hit tests that depend on invalidation.", T()),
            ("Permissions are now cached per project.", SUBMIT),
        ],
    ),
    # 89 ---------------------------------------------------------------------------------------
    dict(
        id="seed-089-ticket-transition-tuple",
        bug_type="missing-tuple-comma", difficulty="easy",
        issue="""\
            Pending tickets show four garbage action buttons

            On a ticket in `pending` status the action bar renders buttons labelled
            "o", "p", "e", "n". Every other status shows proper actions ("Resolve", "Close", ...).
            The API response for a pending ticket has `"actions": ["o", "p", "e", "n"]`; it
            should just be `["open"]`.
            """,
        files={
            "helpdesk/__init__.py": "",
            "helpdesk/states.py": '''\
                TRANSITIONS = {
                    "new": ("open", "closed"),
                    "open": ("pending", "resolved", "closed"),
                    "pending": ("open"),
                    "resolved": ("open", "closed"),
                    "closed": (),
                }


                def can_transition(src, dst):
                    return dst in TRANSITIONS[src]


                def next_states(src):
                    return list(TRANSITIONS[src])
                ''',
            "helpdesk/models.py": '''\
                from dataclasses import dataclass, field


                @dataclass
                class Ticket:
                    id: int
                    title: str
                    status: str = "new"
                    history: list = field(default_factory=list)
                ''',
            "helpdesk/service.py": '''\
                from .models import Ticket
                from .states import can_transition


                class InvalidTransition(Exception):
                    pass


                class TicketService:
                    def __init__(self):
                        self.tickets = {}
                        self._next = 1

                    def create(self, title):
                        t = Ticket(self._next, title)
                        self.tickets[t.id] = t
                        self._next += 1
                        return t

                    def move(self, ticket_id, dst):
                        t = self.tickets[ticket_id]
                        if not can_transition(t.status, dst):
                            raise InvalidTransition(f"{t.status} -> {dst}")
                        t.history.append((t.status, dst))
                        t.status = dst
                        return t
                ''',
            "helpdesk/api.py": '''\
                from .states import next_states


                def ticket_view(service, ticket_id):
                    t = service.tickets[ticket_id]
                    return {"id": t.id, "title": t.title, "status": t.status,
                            "actions": next_states(t.status)}
                ''',
            "tests/test_helpdesk.py": '''\
                import pytest

                from helpdesk.api import ticket_view
                from helpdesk.service import InvalidTransition, TicketService


                def test_happy_path():
                    svc = TicketService()
                    t = svc.create("printer on fire")
                    for dst in ("open", "pending", "open", "resolved", "closed"):
                        svc.move(t.id, dst)
                    assert t.status == "closed" and len(t.history) == 5


                def test_invalid_transition():
                    svc = TicketService()
                    t = svc.create("x")
                    with pytest.raises(InvalidTransition):
                        svc.move(t.id, "resolved")


                def test_view_open_ticket():
                    svc = TicketService()
                    t = svc.create("x")
                    svc.move(t.id, "open")
                    assert ticket_view(svc, t.id)["actions"] == ["pending", "resolved", "closed"]
                ''',
        },
        hidden_test='''\
            import pytest

            from helpdesk.api import ticket_view
            from helpdesk.service import InvalidTransition, TicketService
            from helpdesk.states import can_transition, next_states


            def test_pending_actions():
                svc = TicketService()
                t = svc.create("x")
                svc.move(t.id, "open"); svc.move(t.id, "pending")
                assert ticket_view(svc, t.id)["actions"] == ["open"]


            def test_pending_rejects_substrings():
                assert not can_transition("pending", "pen")
                assert not can_transition("pending", "")
                svc = TicketService()
                t = svc.create("x")
                svc.move(t.id, "open"); svc.move(t.id, "pending")
                with pytest.raises(InvalidTransition):
                    svc.move(t.id, "o")


            def test_closed_is_terminal():
                assert next_states("closed") == []
            ''',
        p2p=["tests/test_helpdesk.py::test_happy_path", "tests/test_helpdesk.py::test_invalid_transition",
             "tests/test_helpdesk.py::test_view_open_ticket"],
        steps=[
            ("Find where the actions list is produced.", S("actions", "helpdesk")),
            ("The API delegates to next_states; look at the transition table.", V("helpdesk/states.py")),
            ("`(\"open\")` is just a parenthesised string, not a tuple, so list() splits it into characters. Confirm.",
             B("python -c 'from helpdesk.states import next_states, can_transition; print(next_states(\"pending\"), can_transition(\"pending\", \"pen\"))'")),
            ("Add the trailing comma so it is a one-element tuple.",
             R("helpdesk/states.py", '    "pending": ("open"),\n', '    "pending": ("open",),\n')),
            ("Re-check both the action list and the substring transition.",
             B("python -c 'from helpdesk.states import next_states, can_transition; print(next_states(\"pending\"), can_transition(\"pending\", \"pen\"))'")),
            ("Run the tests.", B(TEST_CMD)),
            ("Done.", SUBMIT),
        ],
    ),
    # 90 ---------------------------------------------------------------------------------------
    dict(
        id="seed-090-event-bus-once-skips-next",
        bug_type="mutation-during-iteration", difficulty="medium",
        issue="""\
            First account on a fresh install never gets the welcome email

            On a brand new deployment the very first user to sign up (who also becomes the
            workspace admin) never receives the welcome email. The audit log has their
            `user.created` entry and they are correctly made admin. Every signup after the
            first one gets the email normally. Reproducible every time on an empty database.
            """,
        files={
            "events/__init__.py": "from .bus import EventBus\n",
            "events/bus.py": '''\
                from collections import defaultdict
                from dataclasses import dataclass
                from typing import Callable


                @dataclass
                class Subscription:
                    handler: Callable
                    once: bool = False


                class EventBus:
                    """Synchronous pub/sub. Handlers run in subscription order."""

                    def __init__(self):
                        self._subs = defaultdict(list)

                    def subscribe(self, event, handler, once=False):
                        self._subs[event].append(Subscription(handler, once))

                    def unsubscribe(self, event, handler):
                        self._subs[event] = [s for s in self._subs[event] if s.handler != handler]

                    def publish(self, event, payload):
                        results = []
                        for sub in self._subs[event]:
                            if sub.once:
                                self._subs[event].remove(sub)
                            results.append(sub.handler(payload))
                        return results
                ''',
            "app/__init__.py": "",
            "app/audit.py": '''\
                class AuditLog:
                    def __init__(self):
                        self.entries = []

                    def record(self, user):
                        self.entries.append(("user.created", user["email"]))
                ''',
            "app/bootstrap.py": '''\
                class Bootstrap:
                    """Makes the first user of a fresh install the workspace admin."""

                    def __init__(self):
                        self.admin = None

                    def make_first_user_admin(self, user):
                        user["role"] = "admin"
                        self.admin = user["email"]
                ''',
            "app/mailer.py": '''\
                class Mailer:
                    def __init__(self):
                        self.outbox = []

                    def send_welcome(self, user):
                        self.outbox.append((user["email"], "Welcome aboard!"))
                ''',
            "app/wiring.py": '''\
                from events import EventBus

                from .audit import AuditLog
                from .bootstrap import Bootstrap
                from .mailer import Mailer


                class App:
                    def __init__(self):
                        self.bus = EventBus()
                        self.audit = AuditLog()
                        self.bootstrap = Bootstrap()
                        self.mailer = Mailer()
                        self.users = []
                        self.bus.subscribe("user.created", self.audit.record)
                        self.bus.subscribe("user.created", self.bootstrap.make_first_user_admin, once=True)
                        self.bus.subscribe("user.created", self.mailer.send_welcome)

                    def signup(self, email):
                        user = {"email": email, "role": "member"}
                        self.users.append(user)
                        self.bus.publish("user.created", user)
                        return user
                ''',
            "tests/test_bus.py": '''\
                from events import EventBus


                def test_handlers_run_in_order():
                    bus, seen = EventBus(), []
                    bus.subscribe("e", lambda p: seen.append(("a", p)))
                    bus.subscribe("e", lambda p: seen.append(("b", p)))
                    bus.publish("e", 1)
                    assert seen == [("a", 1), ("b", 1)]


                def test_once_handler_runs_once():
                    bus, seen = EventBus(), []
                    bus.subscribe("e", seen.append, once=True)
                    bus.publish("e", 1); bus.publish("e", 2)
                    assert seen == [1]


                def test_unsubscribe():
                    bus, seen = EventBus(), []
                    bus.subscribe("e", seen.append)
                    bus.unsubscribe("e", seen.append)
                    bus.publish("e", 1)
                    assert seen == []
                ''',
            "tests/test_signup.py": '''\
                from app.wiring import App


                def test_first_user_is_admin():
                    app = App()
                    assert app.signup("a@x.io")["role"] == "admin"
                    assert app.signup("b@x.io")["role"] == "member"


                def test_signup_is_audited():
                    app = App()
                    app.signup("a@x.io"); app.signup("b@x.io")
                    assert [e for _, e in app.audit.entries] == ["a@x.io", "b@x.io"]
                ''',
        },
        hidden_test='''\
            from app.wiring import App
            from events import EventBus


            def test_first_user_gets_welcome_email():
                app = App()
                app.signup("a@x.io"); app.signup("b@x.io")
                assert [to for to, _ in app.mailer.outbox] == ["a@x.io", "b@x.io"]


            def test_once_handler_does_not_skip_followers():
                bus, seen = EventBus(), []
                bus.subscribe("e", lambda p: seen.append("once"), once=True)
                bus.subscribe("e", lambda p: seen.append("always"))
                bus.publish("e", None); bus.publish("e", None)
                assert seen == ["once", "always", "always"]


            def test_publish_returns_all_results():
                bus = EventBus()
                bus.subscribe("e", lambda p: 1, once=True)
                bus.subscribe("e", lambda p: 2, once=True)
                bus.subscribe("e", lambda p: 3)
                assert bus.publish("e", None) == [1, 2, 3]
                assert bus.publish("e", None) == [3]
            ''',
        p2p=["tests/test_bus.py::test_handlers_run_in_order", "tests/test_bus.py::test_once_handler_runs_once",
             "tests/test_bus.py::test_unsubscribe", "tests/test_signup.py::test_first_user_is_admin",
             "tests/test_signup.py::test_signup_is_audited"],
        steps=[
            ("See how signup fans out to the mailer.", V("app/wiring.py")),
            ("Reproduce on a fresh App: two signups, then inspect the outbox.",
             B("python -c 'from app.wiring import App; a=App(); a.signup(\"a@x.io\"); a.signup(\"b@x.io\"); print(a.mailer.outbox)'")),
            ("The mailer is subscribed right after the once=True bootstrap handler, and only the first publish misses it. Look at publish().",
             V("events/bus.py", 24, 30)),
            ("publish() removes the once-subscription from the list it is iterating, which shifts the next handler into the visited slot. Iterate over a snapshot.",
             R("events/bus.py", "        for sub in self._subs[event]:\n", "        for sub in list(self._subs[event]):\n")),
            ("Re-run the repro.",
             B("python -c 'from app.wiring import App; a=App(); a.signup(\"a@x.io\"); a.signup(\"b@x.io\"); print(a.mailer.outbox)'")),
            ("Check nothing else iterates the subscriber list while mutating it.", S("_subs\\[", "events")),
            ("Run the test suite.", T()),
            ("Done.", SUBMIT),
        ],
    ),
    # 91 ---------------------------------------------------------------------------------------
    dict(
        id="seed-091-macro-undo-order",
        bug_type="wrong-iteration-order", difficulty="medium",
        issue="""\
            Undo after Bold / Replace All corrupts or ignores the text

            Two things I noticed in the editor:

            1. Type `hello`, select it, press Bold (wraps in `**`). Document is `**hello**`.
               Ctrl+Z leaves `hel**` instead of `hello`.
            2. In `cat hat cat`, Replace All "cat" -> "dog" gives `dog hat dog` (correct), but
               Ctrl+Z does nothing; the text stays `dog hat dog`.

            Undo of plain typing and deleting works fine. Redo seems to have the same problem.
            """,
        files={
            "editor/__init__.py": "",
            "editor/document.py": '''\
                class Document:
                    def __init__(self, text=""):
                        self.text = text

                    def __repr__(self):
                        return f"Document({self.text!r})"
                ''',
            "editor/commands.py": '''\
                class Insert:
                    def __init__(self, pos, text):
                        self.pos, self.text = pos, text

                    def do(self, doc):
                        doc.text = doc.text[:self.pos] + self.text + doc.text[self.pos:]

                    def undo(self, doc):
                        doc.text = doc.text[:self.pos] + doc.text[self.pos + len(self.text):]


                class Delete:
                    def __init__(self, pos, length):
                        self.pos, self.length = pos, length
                        self.removed = None

                    def do(self, doc):
                        self.removed = doc.text[self.pos:self.pos + self.length]
                        doc.text = doc.text[:self.pos] + doc.text[self.pos + self.length:]

                    def undo(self, doc):
                        doc.text = doc.text[:self.pos] + self.removed + doc.text[self.pos:]


                class Macro:
                    """Several commands applied (and undone) as one history entry."""

                    def __init__(self, commands, label="macro"):
                        self.commands = list(commands)
                        self.label = label

                    def do(self, doc):
                        for cmd in self.commands:
                            cmd.do(doc)

                    def undo(self, doc):
                        for cmd in self.commands:
                            cmd.undo(doc)
                ''',
            "editor/history.py": '''\
                class History:
                    def __init__(self, doc):
                        self.doc = doc
                        self._undo = []
                        self._redo = []

                    def execute(self, cmd):
                        cmd.do(self.doc)
                        self._undo.append(cmd)
                        self._redo.clear()

                    def undo(self):
                        if not self._undo:
                            return False
                        cmd = self._undo.pop()
                        cmd.undo(self.doc)
                        self._redo.append(cmd)
                        return True

                    def redo(self):
                        if not self._redo:
                            return False
                        cmd = self._redo.pop()
                        cmd.do(self.doc)
                        self._undo.append(cmd)
                        return True
                ''',
            "editor/ops.py": '''\
                from .commands import Delete, Insert, Macro


                def find_all(text, needle):
                    out, i = [], text.find(needle)
                    while needle and i != -1:
                        out.append(i)
                        i = text.find(needle, i + len(needle))
                    return out


                def type_text(history, pos, text):
                    history.execute(Insert(pos, text))


                def delete_range(history, start, end):
                    history.execute(Delete(start, end - start))


                def wrap_selection(history, start, end, before, after):
                    """E.g. bold: wrap_selection(h, 0, 5, "**", "**")."""
                    history.execute(Macro([Insert(end, after), Insert(start, before)], label="wrap"))


                def replace_all(history, old, new):
                    positions = find_all(history.doc.text, old)
                    if not positions:
                        return 0
                    cmds = []
                    for pos in reversed(positions):  # right to left keeps earlier offsets valid
                        cmds += [Delete(pos, len(old)), Insert(pos, new)]
                    history.execute(Macro(cmds, label=f"replace {old!r}"))
                    return len(positions)
                ''',
            "tests/test_editor.py": '''\
                from editor.document import Document
                from editor.history import History
                from editor.ops import delete_range, replace_all, type_text, wrap_selection


                def test_type_undo_redo():
                    h = History(Document())
                    type_text(h, 0, "hello")
                    type_text(h, 5, " world")
                    h.undo()
                    assert h.doc.text == "hello"
                    h.redo()
                    assert h.doc.text == "hello world"


                def test_delete_undo():
                    h = History(Document("abcdef"))
                    delete_range(h, 1, 3)
                    assert h.doc.text == "adef"
                    h.undo()
                    assert h.doc.text == "abcdef"


                def test_replace_all_and_wrap():
                    h = History(Document("cat hat cat"))
                    assert replace_all(h, "cat", "dog") == 2
                    assert h.doc.text == "dog hat dog"
                    wrap_selection(h, 4, 7, "**", "**")
                    assert h.doc.text == "dog **hat** dog"


                def test_new_edit_clears_redo():
                    h = History(Document())
                    type_text(h, 0, "a")
                    h.undo()
                    type_text(h, 0, "b")
                    assert h.redo() is False
                ''',
        },
        hidden_test='''\
            from editor.commands import Insert, Macro
            from editor.document import Document
            from editor.history import History
            from editor.ops import replace_all, type_text, wrap_selection


            def test_undo_wrap_restores_text():
                h = History(Document("hello"))
                wrap_selection(h, 0, 5, "**", "**")
                assert h.doc.text == "**hello**"
                h.undo()
                assert h.doc.text == "hello"


            def test_undo_redo_replace_all():
                h = History(Document("cat hat cat"))
                replace_all(h, "cat", "dog")
                h.undo()
                assert h.doc.text == "cat hat cat"
                h.redo()
                assert h.doc.text == "dog hat dog"


            def test_macro_undo_is_reverse_of_do():
                doc = Document("")
                m = Macro([Insert(0, "ab"), Insert(1, "XY")])
                m.do(doc)
                assert doc.text == "aXYb"
                m.undo(doc)
                assert doc.text == ""


            def test_plain_typing_still_undoes():
                h = History(Document())
                type_text(h, 0, "hi")
                h.undo()
                assert h.doc.text == ""
            ''',
        p2p=["tests/test_editor.py::test_type_undo_redo", "tests/test_editor.py::test_delete_undo",
             "tests/test_editor.py::test_replace_all_and_wrap", "tests/test_editor.py::test_new_edit_clears_redo"],
        steps=[
            ("List the editor modules.", F("editor/*.py")),
            ("Bold and Replace All are both built in ops.py; see how.", V("editor/ops.py")),
            ("Reproduce both reports.",
             B("python - <<'EOF'\nfrom editor.document import Document\nfrom editor.history import History\nfrom editor.ops import replace_all, wrap_selection\nh = History(Document('hello')); wrap_selection(h, 0, 5, '**', '**'); h.undo(); print(repr(h.doc.text))\nh = History(Document('cat hat cat')); replace_all(h, 'cat', 'dog'); h.undo(); print(repr(h.doc.text))\nEOF")),
            ("Both use Macro while plain typing works. First check whether History mishandles multi-step entries.",
             V("editor/history.py")),
            ("History just calls cmd.undo once per entry, so it is fine. The problem must be Macro.undo.",
             V("editor/commands.py", 25, 38)),
            ("Macro.undo replays sub-commands in the same order as do(); later commands depend on the state left by earlier ones, so undo has to run in reverse.",
             R("editor/commands.py", "    def undo(self, doc):\n        for cmd in self.commands:\n            cmd.undo(doc)\n",
               "    def undo(self, doc):\n        for cmd in reversed(self.commands):\n            cmd.undo(doc)\n")),
            ("Re-run the repro; expect 'hello' and 'cat hat cat'.",
             B("python - <<'EOF'\nfrom editor.document import Document\nfrom editor.history import History\nfrom editor.ops import replace_all, wrap_selection\nh = History(Document('hello')); wrap_selection(h, 0, 5, '**', '**'); h.undo(); print(repr(h.doc.text))\nh = History(Document('cat hat cat')); replace_all(h, 'cat', 'dog'); h.undo(); print(repr(h.doc.text)); h.redo(); print(repr(h.doc.text))\nEOF")),
            ("Run the test suite.", T()),
            ("Done.", SUBMIT),
        ],
    ),
    # 92 ---------------------------------------------------------------------------------------
    dict(
        id="seed-092-overnight-booking-overlap",
        bug_type="boundary-crossing-query", difficulty="hard",
        issue="""\
            Studio double-booked across midnight

            Booking #1: Studio A, 2024-05-10 22:00 -> 2024-05-11 02:00 (overnight recording).
            Someone then booked Studio A for 2024-05-11 01:00 -> 03:00 and it was accepted, so
            two bands showed up at 1am.

            Also, the May 11 day view for Studio A doesn't list the overnight session at all,
            which is probably why the second person thought the room was free. Normal
            same-day conflicts are rejected correctly, and back-to-back bookings (10-11, 11-12)
            are allowed as they should be.
            """,
        files={
            "booking/__init__.py": "",
            "booking/models.py": '''\
                from dataclasses import dataclass
                from datetime import datetime


                @dataclass(frozen=True)
                class Booking:
                    id: int
                    room: str
                    start: datetime
                    end: datetime

                    def overlaps(self, other):
                        """Half-open intervals: back-to-back bookings do not overlap."""
                        return self.start < other.end and other.start < self.end
                ''',
            "booking/repository.py": '''\
                from datetime import datetime

                from .models import Booking


                class BookingRepository:
                    def __init__(self):
                        self._rows = []
                        self._next_id = 1

                    def add(self, room, start, end):
                        booking = Booking(self._next_id, room, start, end)
                        self._next_id += 1
                        self._rows.append(booking)
                        return booking

                    def for_room(self, room):
                        return sorted((b for b in self._rows if b.room == room), key=lambda b: b.start)

                    def for_room_on(self, room, day):
                        """Bookings for `room` that touch calendar day `day` (a date), by start time."""
                        return [b for b in self.for_room(room) if b.start.date() == day]
                ''',
            "booking/policy.py": '''\
                from datetime import timedelta

                MAX_DURATION = timedelta(hours=12)


                class BookingError(ValueError):
                    pass


                def validate(start, end):
                    if end <= start:
                        raise BookingError("booking must end after it starts")
                    if end - start > MAX_DURATION:
                        raise BookingError("bookings are limited to 12 hours")
                ''',
            "booking/service.py": '''\
                from .models import Booking
                from .policy import validate


                class Conflict(Exception):
                    pass


                class BookingService:
                    def __init__(self, repo):
                        self.repo = repo

                    def book(self, room, start, end):
                        validate(start, end)
                        candidate = Booking(0, room, start, end)
                        for existing in self.repo.for_room_on(room, start.date()):
                            if existing.overlaps(candidate):
                                raise Conflict(f"{room} is taken by booking {existing.id}")
                        return self.repo.add(room, start, end)
                ''',
            "booking/schedule.py": '''\
                def day_schedule(repo, room, day):
                    """Rows for the day view: (booking id, start, end) as HH:MM strings."""
                    return [(b.id, b.start.strftime("%H:%M"), b.end.strftime("%H:%M"))
                            for b in repo.for_room_on(room, day)]
                ''',
            "tests/test_booking.py": '''\
                from datetime import date, datetime

                import pytest

                from booking.policy import BookingError
                from booking.repository import BookingRepository
                from booking.schedule import day_schedule
                from booking.service import BookingService, Conflict


                def dt(d, h, m=0):
                    return datetime(2024, 5, d, h, m)


                def test_same_day_conflict():
                    svc = BookingService(BookingRepository())
                    svc.book("A", dt(10, 9), dt(10, 11))
                    with pytest.raises(Conflict):
                        svc.book("A", dt(10, 10), dt(10, 12))


                def test_back_to_back_allowed():
                    svc = BookingService(BookingRepository())
                    svc.book("A", dt(10, 10), dt(10, 11))
                    svc.book("A", dt(10, 11), dt(10, 12))


                def test_other_room_free():
                    svc = BookingService(BookingRepository())
                    svc.book("A", dt(10, 9), dt(10, 11))
                    svc.book("B", dt(10, 9), dt(10, 11))


                def test_policy():
                    svc = BookingService(BookingRepository())
                    with pytest.raises(BookingError):
                        svc.book("A", dt(10, 9), dt(10, 23))


                def test_day_schedule():
                    repo = BookingRepository()
                    BookingService(repo).book("A", dt(10, 14), dt(10, 15, 30))
                    assert day_schedule(repo, "A", date(2024, 5, 10)) == [(1, "14:00", "15:30")]
                ''',
        },
        hidden_test='''\
            from datetime import date, datetime

            import pytest

            from booking.repository import BookingRepository
            from booking.schedule import day_schedule
            from booking.service import BookingService, Conflict


            def dt(d, h, m=0):
                return datetime(2024, 5, d, h, m)


            def test_existing_overnight_blocks_next_morning():
                svc = BookingService(BookingRepository())
                svc.book("A", dt(10, 22), dt(11, 2))
                with pytest.raises(Conflict):
                    svc.book("A", dt(11, 1), dt(11, 3))


            def test_new_overnight_blocked_by_next_morning():
                svc = BookingService(BookingRepository())
                svc.book("A", dt(11, 0, 30), dt(11, 2))
                with pytest.raises(Conflict):
                    svc.book("A", dt(10, 23), dt(11, 1))


            def test_overnight_shows_on_both_days():
                repo = BookingRepository()
                BookingService(repo).book("A", dt(10, 22), dt(11, 2))
                assert day_schedule(repo, "A", date(2024, 5, 10)) == [(1, "22:00", "02:00")]
                assert day_schedule(repo, "A", date(2024, 5, 11)) == [(1, "22:00", "02:00")]
                assert day_schedule(repo, "A", date(2024, 5, 12)) == []


            def test_ending_at_midnight_allows_next_day_start():
                svc = BookingService(BookingRepository())
                svc.book("A", dt(10, 20), dt(11, 0))
                svc.book("A", dt(11, 0), dt(11, 2))
                assert day_schedule(svc.repo, "A", date(2024, 5, 11)) == [(2, "00:00", "02:00")]
            ''',
        p2p=["tests/test_booking.py::test_same_day_conflict", "tests/test_booking.py::test_back_to_back_allowed",
             "tests/test_booking.py::test_other_room_free", "tests/test_booking.py::test_policy",
             "tests/test_booking.py::test_day_schedule"],
        steps=[
            ("Locate the booking flow and the overlap check.", S("def (book|for_room_on|overlaps|day_schedule)", "booking")),
            ("Read the service that accepts or rejects bookings.", V("booking/service.py")),
            ("Reproduce the overnight double booking.",
             B("python - <<'EOF'\nfrom datetime import datetime as D\nfrom booking.repository import BookingRepository\nfrom booking.service import BookingService\nsvc = BookingService(BookingRepository())\nsvc.book('A', D(2024,5,10,22), D(2024,5,11,2))\nprint(svc.book('A', D(2024,5,11,1), D(2024,5,11,3)))\nEOF")),
            ("Maybe overlaps() is wrong for intervals that span midnight; check it.", V("booking/models.py")),
            ("overlaps() compares full datetimes and is correct. The candidates come from for_room_on; look at the repository.",
             V("booking/repository.py")),
            ("for_room_on only matches bookings that *start* on the day, contradicting its docstring. Make it match any booking intersecting [day 00:00, next day 00:00).",
             R("booking/repository.py", "from datetime import datetime\n", "from datetime import datetime, time, timedelta\n")),
            ("Now rewrite the filter.",
             R("booking/repository.py",
               "        return [b for b in self.for_room(room) if b.start.date() == day]\n",
               "        day_start = datetime.combine(day, time.min)\n        day_end = day_start + timedelta(days=1)\n        return [b for b in self.for_room(room) if b.start < day_end and b.end > day_start]\n")),
            ("Re-run the repro; it should raise Conflict now.",
             B("python - <<'EOF' 2>&1 | tail -1\nfrom datetime import datetime as D\nfrom booking.repository import BookingRepository\nfrom booking.service import BookingService\nsvc = BookingService(BookingRepository())\nsvc.book('A', D(2024,5,10,22), D(2024,5,11,2))\nprint(svc.book('A', D(2024,5,11,1), D(2024,5,11,3)))\nEOF")),
            ("The service still only asks for the candidate's start day. Check the mirror case: a new overnight booking against an existing early-morning one.",
             B("python - <<'EOF'\nfrom datetime import datetime as D\nfrom booking.repository import BookingRepository\nfrom booking.service import BookingService\nsvc = BookingService(BookingRepository())\nsvc.book('A', D(2024,5,11,0,30), D(2024,5,11,2))\nprint(svc.book('A', D(2024,5,10,23), D(2024,5,11,1)))\nEOF")),
            ("Accepted, which is wrong. Bookings are capped at 12h so they touch at most two days; check both.",
             R("booking/service.py",
               "        for existing in self.repo.for_room_on(room, start.date()):\n            if existing.overlaps(candidate):\n                raise Conflict(f\"{room} is taken by booking {existing.id}\")\n",
               "        # A booking may run past midnight (max 12h), so check every day it touches.\n        for day in sorted({start.date(), end.date()}):\n            for existing in self.repo.for_room_on(room, day):\n                if existing.overlaps(candidate):\n                    raise Conflict(f\"{room} is taken by booking {existing.id}\")\n")),
            ("Re-run the mirror case.",
             B("python - <<'EOF' 2>&1 | tail -1\nfrom datetime import datetime as D\nfrom booking.repository import BookingRepository\nfrom booking.service import BookingService\nsvc = BookingService(BookingRepository())\nsvc.book('A', D(2024,5,11,0,30), D(2024,5,11,2))\nprint(svc.book('A', D(2024,5,10,23), D(2024,5,11,1)))\nEOF")),
            ("Run all tests, including back-to-back and day schedule.", B(TEST_CMD)),
            ("Overnight bookings are now checked and listed on both days.", SUBMIT),
        ],
    ),
    # 93 ---------------------------------------------------------------------------------------
    dict(
        id="seed-093-validate-before-clean",
        bug_type="pipeline-order", difficulty="easy",
        issue="""\
            Signup rejects emails with capitals / surrounding spaces

            People pasting their address from an email client get "is not a valid email" for
            things like ` Bob@Example.COM `. The form help text says we trim and lowercase the
            email, and that usernames are trimmed. A username of `alice ` (trailing space from
            autocomplete) is also rejected with "must not contain spaces".

                accounts.signup.SignupError: {'email': ['is not a valid email']}
            """,
        files={
            "forms/__init__.py": "",
            "forms/rules.py": '''\
                import re

                EMAIL_RE = re.compile(r"^[a-z0-9._+-]+@[a-z0-9-]+(\\.[a-z0-9-]+)+$")


                def required(value):
                    return "is required" if value in (None, "") else None


                def email(value):
                    if value in (None, ""):
                        return None
                    return None if EMAIL_RE.match(value) else "is not a valid email"


                def length(lo, hi):
                    def rule(value):
                        if value in (None, ""):
                            return None
                        return None if lo <= len(value) <= hi else f"must be {lo}-{hi} characters"
                    return rule


                def no_spaces(value):
                    return "must not contain spaces" if value and " " in value else None
                ''',
            "forms/cleaners.py": '''\
                def strip(value):
                    return value.strip()


                def lower(value):
                    return value.lower()


                def collapse_spaces(value):
                    return " ".join(value.split())
                ''',
            "forms/pipeline.py": '''\
                class ValidationError(Exception):
                    def __init__(self, errors):
                        super().__init__(errors)
                        self.errors = errors


                class Pipeline:
                    """Normalise input with `cleaners`, then check the cleaned values with `validators`.

                    Returns the cleaned dict or raises ValidationError({field: [messages]}).
                    """

                    def __init__(self, cleaners, validators):
                        self.cleaners = cleaners
                        self.validators = validators

                    def run(self, data):
                        errors = {}
                        for field, rules in self.validators.items():
                            value = data.get(field)
                            for rule in rules:
                                msg = rule(value)
                                if msg:
                                    errors.setdefault(field, []).append(msg)
                        cleaned = dict(data)
                        for field, fns in self.cleaners.items():
                            if cleaned.get(field) is not None:
                                for fn in fns:
                                    cleaned[field] = fn(cleaned[field])
                        if errors:
                            raise ValidationError(errors)
                        return cleaned
                ''',
            "accounts/__init__.py": "",
            "accounts/store.py": '''\
                class DuplicateUser(Exception):
                    pass


                class UserStore:
                    def __init__(self):
                        self.users = {}

                    def create(self, data):
                        if data["email"] in self.users:
                            raise DuplicateUser(data["email"])
                        self.users[data["email"]] = dict(data)
                        return self.users[data["email"]]
                ''',
            "accounts/signup.py": '''\
                from forms.cleaners import collapse_spaces, lower, strip
                from forms.pipeline import Pipeline, ValidationError
                from forms.rules import email, length, no_spaces, required


                class SignupError(Exception):
                    pass


                SIGNUP = Pipeline(
                    cleaners={
                        "email": [strip, lower],
                        "username": [strip],
                        "display_name": [strip, collapse_spaces],
                    },
                    validators={
                        "email": [required, email],
                        "username": [required, length(3, 20), no_spaces],
                        "display_name": [length(1, 40)],
                    },
                )


                def signup(store, form):
                    try:
                        data = SIGNUP.run(form)
                    except ValidationError as e:
                        raise SignupError(e.errors) from None
                    return store.create(data)
                ''',
            "tests/test_signup.py": '''\
                import pytest

                from accounts.signup import SignupError, signup
                from accounts.store import DuplicateUser, UserStore


                def test_valid_signup():
                    store = UserStore()
                    user = signup(store, {"email": "bob@example.com", "username": "bob"})
                    assert user["username"] == "bob"


                def test_missing_email():
                    with pytest.raises(SignupError) as e:
                        signup(UserStore(), {"username": "bob"})
                    assert e.value.args[0] == {"email": ["is required"]}


                def test_short_username():
                    with pytest.raises(SignupError) as e:
                        signup(UserStore(), {"email": "a@b.io", "username": "bo"})
                    assert e.value.args[0] == {"username": ["must be 3-20 characters"]}


                def test_duplicate():
                    store = UserStore()
                    signup(store, {"email": "a@b.io", "username": "abc"})
                    with pytest.raises(DuplicateUser):
                        signup(store, {"email": "a@b.io", "username": "abd"})
                ''',
        },
        hidden_test='''\
            import pytest

            from accounts.signup import SignupError, signup
            from accounts.store import DuplicateUser, UserStore


            def test_messy_email_and_username_are_cleaned_first():
                store = UserStore()
                user = signup(store, {"email": "  Bob@Example.COM ", "username": "alice ",
                                      "display_name": "  Alice   Smith "})
                assert user == {"email": "bob@example.com", "username": "alice",
                                "display_name": "Alice Smith"}


            def test_blank_username_is_required():
                with pytest.raises(SignupError) as e:
                    signup(UserStore(), {"email": "a@b.io", "username": "   "})
                assert e.value.args[0] == {"username": ["is required"]}


            def test_duplicate_detected_after_normalising():
                store = UserStore()
                signup(store, {"email": "a@b.io", "username": "abc"})
                with pytest.raises(DuplicateUser):
                    signup(store, {"email": " A@B.io", "username": "abd"})
            ''',
        p2p=["tests/test_signup.py::test_valid_signup", "tests/test_signup.py::test_missing_email",
             "tests/test_signup.py::test_short_username", "tests/test_signup.py::test_duplicate"],
        steps=[
            ("Look at the signup entry point and its pipeline config.", V("accounts/signup.py")),
            ("Reproduce with the pasted address.",
             B("python -c 'from accounts.signup import signup; from accounts.store import UserStore; print(signup(UserStore(), {\"email\": \" Bob@Example.COM \", \"username\": \"alice \"}))' 2>&1 | tail -1")),
            ("Cleaners are configured correctly, so check the order in which the pipeline applies them.", V("forms/pipeline.py")),
            ("run() validates the raw input and only cleans afterwards, contrary to its docstring. Clean first, then validate the cleaned values.",
             R("forms/pipeline.py",
               "        errors = {}\n        for field, rules in self.validators.items():\n            value = data.get(field)\n            for rule in rules:\n                msg = rule(value)\n                if msg:\n                    errors.setdefault(field, []).append(msg)\n        cleaned = dict(data)\n        for field, fns in self.cleaners.items():\n            if cleaned.get(field) is not None:\n                for fn in fns:\n                    cleaned[field] = fn(cleaned[field])\n",
               "        cleaned = dict(data)\n        for field, fns in self.cleaners.items():\n            if cleaned.get(field) is not None:\n                for fn in fns:\n                    cleaned[field] = fn(cleaned[field])\n        errors = {}\n        for field, rules in self.validators.items():\n            value = cleaned.get(field)\n            for rule in rules:\n                msg = rule(value)\n                if msg:\n                    errors.setdefault(field, []).append(msg)\n")),
            ("Re-run the repro.",
             B("python -c 'from accounts.signup import signup; from accounts.store import UserStore; print(signup(UserStore(), {\"email\": \" Bob@Example.COM \", \"username\": \"alice \"}))' 2>&1 | tail -1")),
            ("Run the tests.", B(TEST_CMD)),
            ("Done.", SUBMIT),
        ],
    ),
    # 94 ---------------------------------------------------------------------------------------
    dict(
        id="seed-094-falsy-filter-dropped",
        bug_type="truthiness-check", difficulty="medium",
        issue="""\
            "Out of stock" admin tab lists every product

            `GET /products?in_stock=false` returns all 8 products (`total: 8`) instead of the 3
            that are out of stock. `?in_stock=true` works. Same with `?max_price=0`, which
            merchandising uses to list free samples: it returns the whole catalog.
            """,
        files={
            "catalog/__init__.py": "",
            "catalog/models.py": '''\
                from dataclasses import dataclass


                @dataclass(frozen=True)
                class Product:
                    id: int
                    name: str
                    category: str
                    price_cents: int
                    in_stock: bool
                    rating: int
                ''',
            "catalog/fixtures.py": '''\
                from .models import Product

                PRODUCTS = [
                    Product(1, "Kettle", "kitchen", 3999, True, 4),
                    Product(2, "Toaster", "kitchen", 2999, False, 3),
                    Product(3, "Desk lamp", "office", 1999, True, 5),
                    Product(4, "Stapler", "office", 899, False, 2),
                    Product(5, "Chef knife", "kitchen", 5999, True, 5),
                    Product(6, "Monitor arm", "office", 7999, False, 4),
                    Product(7, "Notebook", "office", 499, True, 3),
                    Product(8, "Tea sampler", "kitchen", 0, True, 4),
                ]
                ''',
            "catalog/query.py": '''\
                class UnknownFilter(KeyError):
                    pass


                OPERATORS = {
                    "category": lambda p, v: p.category == v,
                    "in_stock": lambda p, v: p.in_stock == v,
                    "min_price": lambda p, v: p.price_cents >= v,
                    "max_price": lambda p, v: p.price_cents <= v,
                    "min_rating": lambda p, v: p.rating >= v,
                }

                SORTS = {
                    "id": (lambda p: p.id, False),
                    "price": (lambda p: (p.price_cents, p.id), False),
                    "-price": (lambda p: (-p.price_cents, p.id), False),
                    "-rating": (lambda p: (-p.rating, p.id), False),
                    "name": (lambda p: (p.name.lower(), p.id), False),
                }


                def build_predicate(filters):
                    checks = []
                    for key, value in filters.items():
                        if not value:  # unset filter
                            continue
                        if key not in OPERATORS:
                            raise UnknownFilter(key)
                        checks.append((OPERATORS[key], value))
                    return lambda product: all(fn(product, v) for fn, v in checks)


                def sort_spec(order_by):
                    try:
                        return SORTS[order_by]
                    except KeyError:
                        raise ValueError(f"cannot sort by {order_by!r}") from None
                ''',
            "catalog/repository.py": '''\
                from dataclasses import dataclass

                from .query import build_predicate, sort_spec


                @dataclass
                class Page:
                    items: list
                    total: int
                    page: int
                    per_page: int


                class ProductRepository:
                    def __init__(self, products):
                        self._products = list(products)

                    def search(self, filters=None, order_by="id", page=1, per_page=20):
                        pred = build_predicate(filters or {})
                        rows = [p for p in self._products if pred(p)]
                        key, reverse = sort_spec(order_by)
                        rows.sort(key=key, reverse=reverse)
                        start = (page - 1) * per_page
                        return Page(rows[start:start + per_page], len(rows), page, per_page)
                ''',
            "catalog/api.py": '''\
                INT_FILTERS = {"min_price", "max_price", "min_rating"}


                def parse_params(params):
                    """Turn raw query-string values into typed filters and paging options."""
                    filters, opts = {}, {}
                    for key, raw in params.items():
                        raw = raw.strip()
                        if raw == "":
                            continue
                        if key in ("page", "per_page"):
                            opts[key] = int(raw)
                        elif key == "sort":
                            opts["order_by"] = raw
                        elif key == "in_stock":
                            filters[key] = raw.lower() in ("1", "true", "yes")
                        elif key in INT_FILTERS:
                            filters[key] = int(raw)
                        else:
                            filters[key] = raw
                    return filters, opts


                def list_products(repo, params):
                    filters, opts = parse_params(params)
                    page = repo.search(filters, **opts)
                    return {"total": page.total, "page": page.page, "ids": [p.id for p in page.items]}
                ''',
            "tests/test_catalog.py": '''\
                import pytest

                from catalog.api import list_products, parse_params
                from catalog.fixtures import PRODUCTS
                from catalog.query import UnknownFilter
                from catalog.repository import ProductRepository


                @pytest.fixture
                def repo():
                    return ProductRepository(PRODUCTS)


                def test_in_stock_true(repo):
                    assert list_products(repo, {"in_stock": "true"})["ids"] == [1, 3, 5, 7, 8]


                def test_category_sorted_by_price(repo):
                    out = list_products(repo, {"category": "office", "sort": "-price"})
                    assert out["ids"] == [6, 3, 4, 7]


                def test_paging(repo):
                    out = list_products(repo, {"sort": "price", "per_page": "3", "page": "2"})
                    assert out == {"total": 8, "page": 2, "ids": [3, 2, 1]}


                def test_blank_params_ignored(repo):
                    assert list_products(repo, {"category": " ", "min_rating": "5"})["ids"] == [3, 5]


                def test_parse_params_types():
                    assert parse_params({"in_stock": "no", "max_price": "100"}) == (
                        {"in_stock": False, "max_price": 100}, {})


                def test_unknown_filter(repo):
                    with pytest.raises(UnknownFilter):
                        list_products(repo, {"colour": "red"})
                ''',
        },
        hidden_test='''\
            from catalog.api import list_products
            from catalog.fixtures import PRODUCTS
            from catalog.repository import ProductRepository


            def test_out_of_stock_filter():
                out = list_products(ProductRepository(PRODUCTS), {"in_stock": "false"})
                assert out["total"] == 3
                assert out["ids"] == [2, 4, 6]


            def test_max_price_zero():
                out = list_products(ProductRepository(PRODUCTS), {"max_price": "0"})
                assert out["ids"] == [8]


            def test_out_of_stock_office_sorted():
                out = list_products(ProductRepository(PRODUCTS),
                                    {"in_stock": "false", "category": "office", "sort": "price"})
                assert out["ids"] == [4, 6]


            def test_none_filter_still_ignored():
                repo = ProductRepository(PRODUCTS)
                assert repo.search({"category": None}).total == 8
            ''',
        p2p=["tests/test_catalog.py::test_in_stock_true", "tests/test_catalog.py::test_category_sorted_by_price",
             "tests/test_catalog.py::test_paging", "tests/test_catalog.py::test_blank_params_ignored",
             "tests/test_catalog.py::test_parse_params_types", "tests/test_catalog.py::test_unknown_filter"],
        steps=[
            ("Trace the in_stock filter from the API down.", S("in_stock", "catalog")),
            ("Check how the query string is parsed first.", V("catalog/api.py")),
            ("Reproduce: parse the params and run the search.",
             B("python -c 'from catalog.api import list_products, parse_params; from catalog.fixtures import PRODUCTS; from catalog.repository import ProductRepository as R; print(parse_params({\"in_stock\": \"false\"}), list_products(R(PRODUCTS), {\"in_stock\": \"false\"}))'")),
            ("Parsing yields in_stock=False correctly, yet nothing is filtered. Look at how the predicate is built.",
             V("catalog/query.py", 22, 30)),
            ("`if not value` treats legitimate False and 0 filter values as unset. Only skip None.",
             R("catalog/query.py", "        if not value:  # unset filter\n", "        if value is None:  # unset filter\n")),
            ("Re-run for both in_stock=false and max_price=0.",
             B("python -c 'from catalog.api import list_products; from catalog.fixtures import PRODUCTS; from catalog.repository import ProductRepository as R; print(list_products(R(PRODUCTS), {\"in_stock\": \"false\"}), list_products(R(PRODUCTS), {\"max_price\": \"0\"}))'")),
            ("Run the tests; blank query params are dropped in the API layer so they are unaffected.", T()),
            ("Done.", SUBMIT),
        ],
    ),
    # 95 ---------------------------------------------------------------------------------------
    dict(
        id="seed-095-plugin-registry-inherited-key",
        bug_type="registry-inheritance", difficulty="hard",
        issue="""\
            JSON report exports suddenly pretty-printed

            Since last week's release, `export_report(rows, "json")` returns indented,
            multi-line JSON instead of the compact single-line form. Our downstream ingestion
            job splits responses on newlines and is now failing, and payloads are ~3x larger.
            CSV is unaffected. Nobody touched the JSON exporter itself as far as I can tell;
            the only related change in the release was the new debug console.
            """,
        files={
            "exporters/__init__.py": '''\
                from . import csv_exporter, debug, json_exporter  # noqa: F401  (import registers plugins)
                from .base import REGISTRY, Exporter, UnknownFormat, get_exporter  # noqa: F401
                ''',
            "exporters/base.py": '''\
                REGISTRY = {}


                class UnknownFormat(KeyError):
                    pass


                class Exporter:
                    """Base class for export plugins.

                    Subclasses that declare a `format` are registered automatically so the API can
                    look them up by name.
                    """

                    format = None
                    content_type = "application/octet-stream"

                    def __init_subclass__(cls, **kwargs):
                        super().__init_subclass__(**kwargs)
                        if cls.format is not None:
                            REGISTRY[cls.format] = cls

                    def export(self, rows):
                        raise NotImplementedError


                def get_exporter(fmt):
                    try:
                        return REGISTRY[fmt]()
                    except KeyError:
                        raise UnknownFormat(fmt) from None
                ''',
            "exporters/json_exporter.py": '''\
                import json

                from .base import Exporter


                class JsonExporter(Exporter):
                    format = "json"
                    content_type = "application/json"

                    def export(self, rows):
                        return json.dumps(rows, separators=(",", ":"), sort_keys=True)
                ''',
            "exporters/csv_exporter.py": '''\
                import csv
                import io

                from .base import Exporter


                class CsvExporter(Exporter):
                    format = "csv"
                    content_type = "text/csv"

                    def export(self, rows):
                        if not rows:
                            return ""
                        buf = io.StringIO()
                        writer = csv.DictWriter(buf, fieldnames=list(rows[0]), lineterminator="\\n")
                        writer.writeheader()
                        writer.writerows(rows)
                        return buf.getvalue()
                ''',
            "exporters/debug.py": '''\
                import json

                from .json_exporter import JsonExporter


                class PrettyJsonExporter(JsonExporter):
                    """Indented JSON for the internal debug console. Not an API format."""

                    def export(self, rows):
                        return json.dumps(rows, indent=2, sort_keys=True)


                def dump_for_console(rows):
                    return PrettyJsonExporter().export(rows)
                ''',
            "reports/__init__.py": "",
            "reports/api.py": '''\
                from exporters import get_exporter


                def export_report(rows, fmt="json"):
                    exporter = get_exporter(fmt)
                    return {"content_type": exporter.content_type, "body": exporter.export(rows)}
                ''',
            "tests/test_exports.py": '''\
                import json

                import pytest

                from exporters import UnknownFormat
                from exporters.debug import dump_for_console
                from reports.api import export_report

                ROWS = [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}]


                def test_json_roundtrip():
                    out = export_report(ROWS, "json")
                    assert out["content_type"] == "application/json"
                    assert json.loads(out["body"]) == ROWS


                def test_csv():
                    assert export_report(ROWS, "csv")["body"] == "id,name\\n1,a\\n2,b\\n"


                def test_unknown_format():
                    with pytest.raises(UnknownFormat):
                        export_report(ROWS, "xml")


                def test_debug_console_is_indented():
                    assert dump_for_console([{"a": 1}]) == '[\\n  {\\n    "a": 1\\n  }\\n]'
                ''',
        },
        hidden_test='''\
            from exporters import REGISTRY
            from exporters.debug import PrettyJsonExporter, dump_for_console
            from exporters.json_exporter import JsonExporter
            from reports.api import export_report


            def test_json_export_is_compact():
                out = export_report([{"id": 1, "name": "a"}], "json")
                assert out["body"] == '[{"id":1,"name":"a"}]'


            def test_registry_only_has_declared_formats():
                assert REGISTRY["json"] is JsonExporter
                assert PrettyJsonExporter not in REGISTRY.values()
                assert sorted(REGISTRY) == ["csv", "json"]


            def test_debug_console_still_pretty():
                assert "\\n" in dump_for_console([{"a": 1}])
            ''',
        p2p=["tests/test_exports.py::test_json_roundtrip", "tests/test_exports.py::test_csv",
             "tests/test_exports.py::test_unknown_format", "tests/test_exports.py::test_debug_console_is_indented"],
        steps=[
            ("Start from the report API.", V("reports/api.py")),
            ("Reproduce and see which class the registry hands back for json.",
             B("python -c 'from reports.api import export_report; from exporters import REGISTRY; print(REGISTRY); print(export_report([{\"id\": 1}], \"json\")[\"body\"])'")),
            ("json maps to PrettyJsonExporter. List the plugin modules.", F("exporters/*.py")),
            ("The JSON exporter itself is compact, so see how registration decides the key.", V("exporters/base.py")),
            ("__init_subclass__ registers any subclass whose (possibly inherited) format is set. Find subclasses of JsonExporter.",
             S("\\(JsonExporter\\)", "exporters")),
            ("PrettyJsonExporter declares no format of its own, inherits 'json', and overwrites the real exporter because debug.py is imported after json_exporter.py.",
             V("exporters/debug.py")),
            ("Only register classes that declare `format` in their own body, as the base docstring describes.",
             R("exporters/base.py", "        if cls.format is not None:\n",
               "        if \"format\" in cls.__dict__ and cls.format is not None:\n")),
            ("Re-run the repro; expect compact output and only csv/json registered.",
             B("python -c 'from reports.api import export_report; from exporters import REGISTRY; print(REGISTRY); print(export_report([{\"id\": 1}], \"json\")[\"body\"])'")),
            ("Run the tests, including the debug console one.", B(TEST_CMD)),
            ("Done.", SUBMIT),
        ],
    ),
    # 96 ---------------------------------------------------------------------------------------
    dict(
        id="seed-096-query-builder-shared-state",
        bug_type="aliasing-mutable-state", difficulty="medium",
        issue="""\
            Team summary numbers are wrong for engineering

            The team dashboard for `engineering` shows:

                {'members': 6, 'admins': 2, 'active': 1, 'newest': 'gus'}

            We have 4 active engineers (ana, ben, dee, gus); only ana is both active and an
            admin, if that's a clue. `members`, `admins` and `newest` look right. The sales team summary looks fine, but it only
            has one person so maybe it's just not visible there.
            """,
        files={
            "minidb/__init__.py": "from .database import Database\n",
            "minidb/query.py": '''\
                import operator

                LOOKUPS = {"eq": operator.eq, "gt": operator.gt, "gte": operator.ge, "lt": operator.lt, "lte": operator.le}


                def _parse(key):
                    field, _, op = key.partition("__")
                    return field, LOOKUPS[op or "eq"]


                class Query:
                    def __init__(self, table):
                        self.table = table
                        self._filters = []
                        self._order = None
                        self._limit = None

                    def where(self, **conds):
                        for key, value in conds.items():
                            field, op = _parse(key)
                            self._filters.append((field, op, value))
                        return self

                    def order_by(self, field):
                        self._order = field
                        return self

                    def limit(self, n):
                        self._limit = n
                        return self

                    def _matching(self):
                        return [r for r in self.table.rows
                                if all(op(r.get(f), v) for f, op, v in self._filters)]

                    def all(self):
                        rows = self._matching()
                        if self._order:
                            field = self._order.lstrip("-")
                            rows.sort(key=lambda r: r[field], reverse=self._order.startswith("-"))
                        return rows[:self._limit] if self._limit is not None else rows

                    def count(self):
                        return len(self._matching())

                    def first(self):
                        rows = self.limit(1).all()
                        return rows[0] if rows else None
                ''',
            "minidb/table.py": '''\
                from .query import Query


                class Table:
                    def __init__(self, name):
                        self.name = name
                        self.rows = []
                        self._next_id = 1

                    def insert(self, **fields):
                        row = {"id": self._next_id, **fields}
                        self._next_id += 1
                        self.rows.append(row)
                        return row

                    def query(self):
                        return Query(self)
                ''',
            "minidb/database.py": '''\
                from .table import Table


                class Database:
                    def __init__(self):
                        self._tables = {}

                    def table(self, name):
                        if name not in self._tables:
                            self._tables[name] = Table(name)
                        return self._tables[name]
                ''',
            "app/__init__.py": "",
            "app/seed.py": '''\
                from minidb import Database

                USERS = [
                    ("ana", "engineering", "admin", True, 12000),
                    ("ben", "engineering", "member", True, 4500),
                    ("cy", "engineering", "admin", False, 800),
                    ("dee", "engineering", "member", True, 15000),
                    ("eve", "engineering", "member", False, 0),
                    ("fay", "sales", "admin", True, 9900),
                    ("gus", "engineering", "member", True, 3000),
                ]


                def demo_db():
                    db = Database()
                    users = db.table("users")
                    for name, team, role, active, total in USERS:
                        users.insert(name=name, team=team, role=role, active=active, total_cents=total)
                    return db
                ''',
            "app/reports.py": '''\
                def team_summary(db, team):
                    members = db.table("users").query().where(team=team)
                    newest = members.order_by("-id").first()
                    return {
                        "members": members.count(),
                        "admins": members.where(role="admin").count(),
                        "active": members.where(active=True).count(),
                        "newest": newest["name"] if newest else None,
                    }


                def top_spenders(db, n=3, min_total=0):
                    q = db.table("users").query().where(total_cents__gte=min_total)
                    return [r["name"] for r in q.order_by("-total_cents").limit(n).all()]
                ''',
            "tests/test_minidb.py": '''\
                from app.reports import team_summary, top_spenders
                from app.seed import demo_db


                def test_where_and_order():
                    q = demo_db().table("users").query().where(team="engineering", active=True)
                    assert [r["name"] for r in q.order_by("name").all()] == ["ana", "ben", "dee", "gus"]


                def test_lookups_and_limit():
                    q = demo_db().table("users").query().where(total_cents__gt=5000).order_by("-total_cents").limit(2)
                    assert [r["name"] for r in q.all()] == ["dee", "ana"]


                def test_top_spenders():
                    assert top_spenders(demo_db()) == ["dee", "ana", "fay"]


                def test_sales_summary():
                    assert team_summary(demo_db(), "sales") == {
                        "members": 1, "admins": 1, "active": 1, "newest": "fay"}
                ''',
        },
        hidden_test='''\
            from app.reports import team_summary
            from app.seed import demo_db


            def test_engineering_summary():
                assert team_summary(demo_db(), "engineering") == {
                    "members": 6, "admins": 2, "active": 4, "newest": "gus"}


            def test_refining_does_not_change_base_query():
                base = demo_db().table("users").query().where(team="engineering")
                admins = base.where(role="admin")
                ordered = base.order_by("name")
                assert admins.count() == 2
                assert base.count() == 6
                assert ordered.first()["name"] == "ana"
                assert len(base.all()) == 6


            def test_first_does_not_limit_query():
                q = demo_db().table("users").query().order_by("id")
                assert q.first()["name"] == "ana"
                assert len(q.all()) == 7
            ''',
        p2p=["tests/test_minidb.py::test_where_and_order", "tests/test_minidb.py::test_lookups_and_limit",
             "tests/test_minidb.py::test_top_spenders", "tests/test_minidb.py::test_sales_summary"],
        steps=[
            ("Start with the report that builds the summary.", V("app/reports.py")),
            ("Reproduce the engineering numbers.",
             B("python -c 'from app.reports import team_summary; from app.seed import demo_db; print(team_summary(demo_db(), \"engineering\"))'")),
            ("Each figure refines the same `members` query; if refining mutates it, filters pile up. Check the query builder.",
             V("minidb/query.py")),
            ("where/order_by/limit mutate self and return it, and first() calls limit(1) on the caller's query too. Builder methods should return a modified copy. Add a clone helper after __init__.",
             I("minidb/query.py", 16, "\n    def _clone(self):\n        q = Query(self.table)\n        q._filters = list(self._filters)\n        q._order, q._limit = self._order, self._limit\n        return q\n")),
            ("Make the three builder methods work on a clone.",
             R("minidb/query.py",
               "    def where(self, **conds):\n        for key, value in conds.items():\n            field, op = _parse(key)\n            self._filters.append((field, op, value))\n        return self\n\n    def order_by(self, field):\n        self._order = field\n        return self\n\n    def limit(self, n):\n        self._limit = n\n        return self\n",
               "    def where(self, **conds):\n        q = self._clone()\n        for key, value in conds.items():\n            field, op = _parse(key)\n            q._filters.append((field, op, value))\n        return q\n\n    def order_by(self, field):\n        q = self._clone()\n        q._order = field\n        return q\n\n    def limit(self, n):\n        q = self._clone()\n        q._limit = n\n        return q\n")),
            ("Make sure no other method still returns a mutated self.", S("return self", "minidb")),
            ("Re-run the repro; expect active 4.",
             B("python -c 'from app.reports import team_summary; from app.seed import demo_db; print(team_summary(demo_db(), \"engineering\"))'")),
            ("Run the tests.", T()),
            ("Query builder is now immutable.", SUBMIT),
        ],
    ),
    # 97 ---------------------------------------------------------------------------------------
    dict(
        id="seed-097-graduated-tier-bounds",
        bug_type="cumulative-vs-incremental", difficulty="hard",
        issue="""\
            Customer overbilled on api-pro usage

            ACME (plan `api-pro`) made 15,000 API calls in 2024-10. Their invoice:

                Base fee                 $20.00
                Usage (15,000 calls)    $240.00
                Total                   $260.00

            Per the pricing page api-pro is graduated: first 1,000 calls free, calls 1,001 to
            10,000 at $0.02, everything above 10,000 at $0.01. That's 9,000 x 0.02 + 5,000 x 0.01
            = $230.00 usage, $250.00 total. Smaller customers (a few thousand calls) seem to
            be billed correctly. Finance needs this fixed before the next invoice run.
            """,
        files={
            "billing/__init__.py": "",
            "billing/models.py": '''\
                from dataclasses import dataclass
                from typing import Optional


                @dataclass(frozen=True)
                class Tier:
                    up_to: Optional[int]  # inclusive bound on *cumulative* units; None = no limit
                    unit_cents: int


                @dataclass(frozen=True)
                class Plan:
                    name: str
                    base_cents: int
                    tiers: tuple
                    mode: str = "graduated"  # or "volume"
                ''',
            "billing/catalog.py": '''\
                from .models import Plan, Tier

                PLANS = {
                    "api-pro": Plan("api-pro", 2000, (Tier(1000, 0), Tier(10000, 2), Tier(None, 1)), "graduated"),
                    "api-volume": Plan("api-volume", 0, (Tier(5000, 3), Tier(50000, 2), Tier(None, 1)), "volume"),
                }


                def get_plan(name):
                    try:
                        return PLANS[name]
                    except KeyError:
                        raise ValueError(f"unknown plan {name!r}") from None
                ''',
            "billing/tiers.py": '''\
                def graduated_cost(units, tiers):
                    """Price each unit at the rate of the tier it falls in (like tax brackets).

                    Tier bounds are cumulative: after Tier(1000, 0), Tier(10000, 2) covers units
                    1,001-10,000.
                    """
                    total = 0
                    remaining = units
                    for tier in tiers:
                        if remaining <= 0:
                            break
                        in_tier = remaining if tier.up_to is None else min(remaining, tier.up_to)
                        total += in_tier * tier.unit_cents
                        remaining -= in_tier
                    return total


                def volume_cost(units, tiers):
                    """All units are priced at the rate of the single tier the total falls in."""
                    for tier in tiers:
                        if tier.up_to is None or units <= tier.up_to:
                            return units * tier.unit_cents
                    raise ValueError("tiers must end with an unbounded tier")


                def usage_cost(units, plan):
                    if plan.mode == "graduated":
                        return graduated_cost(units, plan.tiers)
                    if plan.mode == "volume":
                        return volume_cost(units, plan.tiers)
                    raise ValueError(f"unknown pricing mode {plan.mode!r}")
                ''',
            "billing/usage.py": '''\
                def total_units(events, customer, period):
                    """events: iterable of (customer, period, units) tuples."""
                    return sum(units for c, p, units in events if c == customer and p == period)
                ''',
            "billing/invoice.py": '''\
                from .catalog import get_plan
                from .tiers import usage_cost
                from .usage import total_units


                def build_invoice(customer, plan_name, events, period):
                    plan = get_plan(plan_name)
                    units = total_units(events, customer, period)
                    lines = [("Base fee", plan.base_cents),
                             (f"Usage ({units:,} calls)", usage_cost(units, plan))]
                    return {"customer": customer, "period": period, "lines": lines,
                            "total_cents": sum(cents for _, cents in lines)}


                def format_cents(cents):
                    return f"${cents // 100:,}.{cents % 100:02d}"
                ''',
            "tests/test_billing.py": '''\
                from billing.catalog import get_plan
                from billing.invoice import build_invoice, format_cents
                from billing.tiers import usage_cost
                from billing.usage import total_units

                EVENTS = [("acme", "2024-10", 2000), ("acme", "2024-10", 1000),
                          ("acme", "2024-09", 50000), ("globex", "2024-10", 700)]


                def test_total_units():
                    assert total_units(EVENTS, "acme", "2024-10") == 3000


                def test_free_tier():
                    assert usage_cost(700, get_plan("api-pro")) == 0
                    assert usage_cost(1000, get_plan("api-pro")) == 0


                def test_second_tier_invoice():
                    inv = build_invoice("acme", "api-pro", EVENTS, "2024-10")
                    assert inv["lines"] == [("Base fee", 2000), ("Usage (3,000 calls)", 4000)]
                    assert format_cents(inv["total_cents"]) == "$60.00"


                def test_volume_pricing():
                    plan = get_plan("api-volume")
                    assert usage_cost(5000, plan) == 15000
                    assert usage_cost(20000, plan) == 40000
                    assert usage_cost(60000, plan) == 60000
                ''',
        },
        hidden_test='''\
            from billing.catalog import get_plan
            from billing.invoice import build_invoice
            from billing.models import Tier
            from billing.tiers import graduated_cost, usage_cost


            def test_acme_october_invoice():
                events = [("acme", "2024-10", 15000)]
                inv = build_invoice("acme", "api-pro", events, "2024-10")
                assert inv["lines"][1] == ("Usage (15,000 calls)", 23000)
                assert inv["total_cents"] == 25000


            def test_exact_tier_boundaries():
                plan = get_plan("api-pro")
                assert usage_cost(10000, plan) == 18000
                assert usage_cost(10001, plan) == 18001


            def test_four_tiers():
                tiers = (Tier(10, 5), Tier(20, 4), Tier(30, 3), Tier(None, 1))
                assert graduated_cost(35, tiers) == 50 + 40 + 30 + 5


            def test_within_second_tier_unchanged():
                assert usage_cost(3000, get_plan("api-pro")) == 4000
            ''',
        p2p=["tests/test_billing.py::test_total_units", "tests/test_billing.py::test_free_tier",
             "tests/test_billing.py::test_second_tier_invoice", "tests/test_billing.py::test_volume_pricing"],
        steps=[
            ("List the billing modules.", F("billing/*.py")),
            ("Follow the invoice construction.", V("billing/invoice.py")),
            ("Reproduce ACME's invoice.",
             B("python -c 'from billing.invoice import build_invoice; print(build_invoice(\"acme\", \"api-pro\", [(\"acme\", \"2024-10\", 15000)], \"2024-10\"))'")),
            ("Units are right (15,000) but usage is 24000 instead of 23000. Possibly api-pro is misconfigured in the catalog (wrong mode or bounds).",
             V("billing/catalog.py")),
            ("The catalog matches the pricing page: graduated, bounds 1000/10000. So the tier math itself must be off.",
             V("billing/tiers.py", 1, 15)),
            ("The loop treats `up_to` as a per-tier width, but bounds are cumulative: tier 2 takes min(14000, 10000) units instead of 9,000. Confirm at the boundary.",
             B("python -c 'from billing.catalog import get_plan; from billing.tiers import usage_cost; p=get_plan(\"api-pro\"); print([usage_cost(u, p) for u in (3000, 10000, 15000)])'")),
            ("Track the previous cumulative bound and price only the units between it and this tier's bound.",
             R("billing/tiers.py",
               "    total = 0\n    remaining = units\n    for tier in tiers:\n        if remaining <= 0:\n            break\n        in_tier = remaining if tier.up_to is None else min(remaining, tier.up_to)\n        total += in_tier * tier.unit_cents\n        remaining -= in_tier\n    return total\n",
               "    total = 0\n    prev_bound = 0\n    for tier in tiers:\n        if units <= prev_bound:\n            break\n        upper = units if tier.up_to is None else min(units, tier.up_to)\n        total += (upper - prev_bound) * tier.unit_cents\n        prev_bound = upper\n    return total\n")),
            ("Re-check: expect 4000, 18000, 23000.",
             B("python -c 'from billing.catalog import get_plan; from billing.tiers import usage_cost; p=get_plan(\"api-pro\"); print([usage_cost(u, p) for u in (3000, 10000, 15000)])'")),
            ("Run the test suite.", B(TEST_CMD)),
            ("Graduated pricing now uses incremental tier widths.", SUBMIT),
        ],
    ),
    # 98 ---------------------------------------------------------------------------------------
    dict(
        id="seed-098-enum-status-string-compare",
        bug_type="enum-vs-string-comparison", difficulty="hard",
        issue="""\
            Shipped orders can be cancelled; returns refunded in full

            Since the order-status refactor (statuses moved to the `OrderStatus` enum) two
            things are off:

            1. Support cancelled order 1042 *after* it shipped. The system allowed it,
               refunded the full amount and put the 2 units back into stock while they were
               on the truck. Cancel is supposed to refuse shipped/delivered orders.
            2. Returns of delivered orders are refunded 100%. The 15% restocking fee is no
               longer deducted.

            No errors anywhere; it just silently does the wrong thing.
            """,
        files={
            "orders/__init__.py": "",
            "orders/status.py": '''\
                from enum import Enum


                class OrderStatus(Enum):
                    PENDING = "pending"
                    PAID = "paid"
                    SHIPPED = "shipped"
                    DELIVERED = "delivered"
                    CANCELLED = "cancelled"
                    RETURNED = "returned"
                ''',
            "orders/models.py": '''\
                from dataclasses import dataclass, field

                from .status import OrderStatus


                @dataclass
                class Order:
                    id: int
                    items: dict = field(default_factory=dict)
                    total_cents: int = 0
                    status: OrderStatus = OrderStatus.PENDING
                ''',
            "orders/refunds.py": '''\
                RESTOCKING_FEE_PCT = 15


                def refund_amount(order):
                    """Full refund for orders that never left the warehouse; returned
                    (delivered) orders are refunded minus the restocking fee."""
                    if order.status == "delivered":
                        return order.total_cents - order.total_cents * RESTOCKING_FEE_PCT // 100
                    return order.total_cents
                ''',
            "orders/service.py": '''\
                from .refunds import refund_amount
                from .status import OrderStatus


                class CannotCancel(Exception):
                    pass


                class CannotReturn(Exception):
                    pass


                class OrderService:
                    def __init__(self, stock, gateway):
                        self.stock = stock
                        self.gateway = gateway
                        self.orders = {}

                    def create(self, order):
                        self.stock.take(order.items)
                        self.orders[order.id] = order
                        return order

                    def pay(self, order_id):
                        order = self.orders[order_id]
                        self.gateway.charge(order.id, order.total_cents)
                        order.status = OrderStatus.PAID

                    def ship(self, order_id):
                        self.orders[order_id].status = OrderStatus.SHIPPED

                    def deliver(self, order_id):
                        self.orders[order_id].status = OrderStatus.DELIVERED

                    def cancel(self, order_id):
                        order = self.orders[order_id]
                        if order.status in ("shipped", "delivered"):
                            raise CannotCancel(f"order {order_id} has already shipped")
                        if order.status == OrderStatus.CANCELLED:
                            return order
                        if order.status == OrderStatus.PAID:
                            self.gateway.refund(order.id, refund_amount(order))
                        self.stock.restock(order.items)
                        order.status = OrderStatus.CANCELLED
                        return order

                    def return_order(self, order_id):
                        order = self.orders[order_id]
                        if order.status != OrderStatus.DELIVERED:
                            raise CannotReturn(f"order {order_id} was not delivered")
                        self.gateway.refund(order.id, refund_amount(order))
                        self.stock.restock(order.items)
                        order.status = OrderStatus.RETURNED
                        return order
                ''',
            "inventory/__init__.py": "",
            "inventory/stock.py": '''\
                class Stock:
                    def __init__(self, levels):
                        self.levels = dict(levels)

                    def take(self, items):
                        for sku, qty in items.items():
                            if self.levels.get(sku, 0) < qty:
                                raise ValueError(f"not enough {sku}")
                        for sku, qty in items.items():
                            self.levels[sku] -= qty

                    def restock(self, items):
                        for sku, qty in items.items():
                            self.levels[sku] = self.levels.get(sku, 0) + qty
                ''',
            "payments/__init__.py": "",
            "payments/gateway.py": '''\
                class FakeGateway:
                    def __init__(self):
                        self.charges = []
                        self.refunds = []

                    def charge(self, order_id, cents):
                        self.charges.append((order_id, cents))

                    def refund(self, order_id, cents):
                        self.refunds.append((order_id, cents))
                ''',
            "tests/test_orders.py": '''\
                import pytest

                from inventory.stock import Stock
                from orders.models import Order
                from orders.service import CannotReturn, OrderService
                from orders.status import OrderStatus
                from payments.gateway import FakeGateway


                def make():
                    svc = OrderService(Stock({"MUG": 5}), FakeGateway())
                    svc.create(Order(1042, {"MUG": 2}, 4000))
                    return svc


                def test_cancel_paid_order_refunds_and_restocks():
                    svc = make()
                    svc.pay(1042)
                    svc.cancel(1042)
                    assert svc.orders[1042].status is OrderStatus.CANCELLED
                    assert svc.gateway.refunds == [(1042, 4000)]
                    assert svc.stock.levels["MUG"] == 5


                def test_cancel_pending_order_no_refund():
                    svc = make()
                    svc.cancel(1042)
                    assert svc.gateway.refunds == []
                    assert svc.stock.levels["MUG"] == 5


                def test_cannot_return_undelivered():
                    svc = make()
                    svc.pay(1042)
                    with pytest.raises(CannotReturn):
                        svc.return_order(1042)
                ''',
        },
        hidden_test='''\
            import pytest

            from inventory.stock import Stock
            from orders.models import Order
            from orders.service import CannotCancel, OrderService
            from orders.status import OrderStatus
            from payments.gateway import FakeGateway


            def make():
                svc = OrderService(Stock({"MUG": 5}), FakeGateway())
                svc.create(Order(1042, {"MUG": 2}, 4000))
                svc.pay(1042)
                return svc


            @pytest.mark.parametrize("final", ["ship", "deliver"])
            def test_cannot_cancel_after_shipping(final):
                svc = make()
                svc.ship(1042)
                if final == "deliver":
                    svc.deliver(1042)
                with pytest.raises(CannotCancel):
                    svc.cancel(1042)
                assert svc.gateway.refunds == []
                assert svc.stock.levels["MUG"] == 3


            def test_return_deducts_restocking_fee():
                svc = make()
                svc.ship(1042); svc.deliver(1042)
                svc.return_order(1042)
                assert svc.gateway.refunds == [(1042, 3400)]
                assert svc.orders[1042].status is OrderStatus.RETURNED


            def test_cancel_paid_still_full_refund():
                svc = make()
                svc.cancel(1042)
                assert svc.gateway.refunds == [(1042, 4000)]
            ''',
        p2p=["tests/test_orders.py::test_cancel_paid_order_refunds_and_restocks",
             "tests/test_orders.py::test_cancel_pending_order_no_refund",
             "tests/test_orders.py::test_cannot_return_undelivered"],
        steps=[
            ("Get the layout of the order package.", F("orders/*.py")),
            ("Read the cancel and return flows.", V("orders/service.py", 35, 54)),
            ("Reproduce both: cancel after shipping, and return after delivery.",
             B("python - <<'EOF'\nfrom inventory.stock import Stock\nfrom orders.models import Order\nfrom orders.service import OrderService\nfrom payments.gateway import FakeGateway\nfor flow in ('cancel', 'return'):\n    svc = OrderService(Stock({'MUG': 5}), FakeGateway())\n    svc.create(Order(1042, {'MUG': 2}, 4000)); svc.pay(1042); svc.ship(1042)\n    if flow == 'cancel':\n        svc.cancel(1042)\n    else:\n        svc.deliver(1042); svc.return_order(1042)\n    print(flow, svc.orders[1042].status, svc.gateway.refunds, svc.stock.levels)\nEOF")),
            ("The cancel guard compares an OrderStatus member with plain strings, which is never equal for a plain Enum. Find every leftover string comparison from before the refactor.",
             S("status (==|!=|in) \\(?[\"']", ".")),
            ("refunds.py has the same problem, which explains the missing restocking fee.", V("orders/refunds.py")),
            ("Compare against enum members in the cancel guard.",
             R("orders/service.py", '        if order.status in ("shipped", "delivered"):\n',
               "        if order.status in (OrderStatus.SHIPPED, OrderStatus.DELIVERED):\n")),
            ("Import OrderStatus at the top of refunds.py.",
             I("orders/refunds.py", 0, "from .status import OrderStatus\n\n")),
            ("And compare against the enum member there.",
             R("orders/refunds.py", '    if order.status == "delivered":\n',
               "    if order.status == OrderStatus.DELIVERED:\n")),
            ("Re-run the repro: cancel should raise CannotCancel, return should refund 3400.",
             B("python - <<'EOF'\nfrom inventory.stock import Stock\nfrom orders.models import Order\nfrom orders.service import CannotCancel, OrderService\nfrom payments.gateway import FakeGateway\nfor flow in ('cancel', 'return'):\n    svc = OrderService(Stock({'MUG': 5}), FakeGateway())\n    svc.create(Order(1042, {'MUG': 2}, 4000)); svc.pay(1042); svc.ship(1042)\n    try:\n        if flow == 'cancel':\n            svc.cancel(1042)\n        else:\n            svc.deliver(1042); svc.return_order(1042)\n    except CannotCancel as e:\n        print(flow, 'refused:', e)\n    print(flow, svc.orders[1042].status, svc.gateway.refunds, svc.stock.levels)\nEOF")),
            ("Confirm no string status comparisons remain.", S("status (==|!=|in) \\(?[\"']", ".")),
            ("Run the test suite.", T()),
            ("Both checks now use OrderStatus members.", SUBMIT),
        ],
    ),
    # 99 ---------------------------------------------------------------------------------------
    dict(
        id="seed-099-discount-allocation-rounding",
        bug_type="rounding-drift", difficulty="hard",
        issue="""\
            Can't refund the last item of a discounted order

            Order 5521: three items at $10.00 each with a $10.00 coupon, customer paid $20.00.
            Support refunded items 1 and 2 ($6.67 each) fine, but refunding item 3 blows up:

                Traceback (most recent call last):
                  File "support_tool.py", line 31, in refund_items
                    refund_line(order, i, ledger)
                  File "refunds/service.py", line 15, in refund_line
                    ledger.refund(order.id, amount)
                  File "payments/ledger.py", line 18, in refund
                    raise RefundError(
                payments.ledger.RefundError: refund of 667 exceeds remaining captured amount 666 for order 5521

            We've also seen the opposite on other orders (customer ends up 1 cent short after a
            full refund). Is the ledger check off by one?
            """,
        files={
            "orders/__init__.py": "",
            "orders/models.py": '''\
                from dataclasses import dataclass, field


                @dataclass
                class Line:
                    sku: str
                    unit_cents: int
                    qty: int = 1
                    refunded: bool = False

                    @property
                    def total_cents(self):
                        return self.unit_cents * self.qty


                @dataclass
                class Order:
                    id: int
                    lines: list
                    discount_cents: int = 0
                    allocations: list = field(default_factory=list)  # discount share per line

                    @property
                    def subtotal_cents(self):
                        return sum(line.total_cents for line in self.lines)

                    @property
                    def paid_cents(self):
                        return self.subtotal_cents - self.discount_cents
                ''',
            "pricing/__init__.py": "",
            "pricing/allocation.py": '''\
                def allocate(total_cents, weights):
                    """Split `total_cents` across lines in proportion to `weights`.

                    The parts always add up to exactly `total_cents`.
                    """
                    weight_sum = sum(weights)
                    if weight_sum == 0:
                        return [0] * len(weights)
                    return [round(total_cents * w / weight_sum) for w in weights]
                ''',
            "checkout/__init__.py": "",
            "checkout/place.py": '''\
                from orders.models import Order
                from pricing.allocation import allocate


                def place_order(order_id, lines, ledger, discount_cents=0):
                    order = Order(order_id, list(lines), discount_cents)
                    if discount_cents > order.subtotal_cents:
                        raise ValueError("discount exceeds order subtotal")
                    order.allocations = allocate(discount_cents, [l.total_cents for l in order.lines])
                    ledger.capture(order.id, order.paid_cents)
                    return order
                ''',
            "refunds/__init__.py": "",
            "refunds/service.py": '''\
                class AlreadyRefunded(Exception):
                    pass


                def refundable_amount(order, index):
                    """What the customer actually paid for a line: its price minus its discount share."""
                    return order.lines[index].total_cents - order.allocations[index]


                def refund_line(order, index, ledger):
                    line = order.lines[index]
                    if line.refunded:
                        raise AlreadyRefunded(f"line {index} of order {order.id}")
                    amount = refundable_amount(order, index)
                    ledger.refund(order.id, amount)
                    line.refunded = True
                    return amount
                ''',
            "payments/__init__.py": "",
            "payments/ledger.py": '''\
                class RefundError(Exception):
                    pass


                class Ledger:
                    def __init__(self):
                        self.captured = {}
                        self.refunded = {}

                    def capture(self, order_id, cents):
                        self.captured[order_id] = self.captured.get(order_id, 0) + cents

                    def remaining(self, order_id):
                        return self.captured.get(order_id, 0) - self.refunded.get(order_id, 0)

                    def refund(self, order_id, cents):
                        if cents > self.remaining(order_id):
                            raise RefundError(
                                f"refund of {cents} exceeds remaining captured amount "
                                f"{self.remaining(order_id)} for order {order_id}")
                        self.refunded[order_id] = self.refunded.get(order_id, 0) + cents
                ''',
            "tests/test_refunds.py": '''\
                import pytest

                from checkout.place import place_order
                from orders.models import Line
                from payments.ledger import Ledger, RefundError
                from pricing.allocation import allocate
                from refunds.service import AlreadyRefunded, refund_line


                def test_allocate_even_split():
                    assert allocate(600, [100, 200, 300]) == [100, 200, 300]
                    assert allocate(0, [5, 5]) == [0, 0]
                    assert allocate(10, [0, 0]) == [0, 0]


                def test_refund_single_line():
                    ledger = Ledger()
                    order = place_order(1, [Line("a", 1000), Line("b", 3000)], ledger, discount_cents=400)
                    assert refund_line(order, 1, ledger) == 2700
                    assert ledger.remaining(1) == 900


                def test_cannot_refund_twice():
                    ledger = Ledger()
                    order = place_order(1, [Line("a", 1000)], ledger)
                    refund_line(order, 0, ledger)
                    with pytest.raises(AlreadyRefunded):
                        refund_line(order, 0, ledger)


                def test_ledger_rejects_over_refund():
                    ledger = Ledger()
                    ledger.capture(9, 100)
                    with pytest.raises(RefundError):
                        ledger.refund(9, 101)
                ''',
        },
        hidden_test='''\
            from checkout.place import place_order
            from orders.models import Line
            from payments.ledger import Ledger
            from pricing.allocation import allocate
            from refunds.service import refund_line


            def test_full_refund_of_discounted_order():
                ledger = Ledger()
                order = place_order(5521, [Line("a", 1000), Line("b", 1000), Line("c", 1000)],
                                    ledger, discount_cents=1000)
                refunds = [refund_line(order, i, ledger) for i in range(3)]
                assert sum(refunds) == 2000
                assert ledger.remaining(5521) == 0


            def test_allocations_always_sum_to_discount():
                for total, weights in [(1000, [1000, 1000, 1000]), (500, [1000, 1000, 1000]),
                                       (7, [1, 1, 1, 1]), (999, [333, 333, 334]), (101, [50, 50])]:
                    parts = allocate(total, weights)
                    assert sum(parts) == total, (total, weights, parts)
                    assert all(p >= 0 for p in parts)


            def test_allocation_stays_proportional():
                assert allocate(1000, [1000, 1000, 1000]) == [334, 333, 333]
                assert allocate(500, [1000, 1000, 1000]) == [167, 167, 166]


            def test_even_split_unchanged():
                assert allocate(600, [100, 200, 300]) == [100, 200, 300]
            ''',
        p2p=["tests/test_refunds.py::test_allocate_even_split", "tests/test_refunds.py::test_refund_single_line",
             "tests/test_refunds.py::test_cannot_refund_twice", "tests/test_refunds.py::test_ledger_rejects_over_refund"],
        steps=[
            ("Start where the traceback ends: the ledger check the reporter suspects.", V("payments/ledger.py")),
            ("The check is correct: refunding more than what's left must fail. Reproduce order 5521 and look at the numbers going in.",
             B("python - <<'EOF'\nfrom checkout.place import place_order\nfrom orders.models import Line\nfrom payments.ledger import Ledger\nfrom refunds.service import refund_line\nledger = Ledger()\norder = place_order(5521, [Line('a', 1000), Line('b', 1000), Line('c', 1000)], ledger, discount_cents=1000)\nprint('paid', order.paid_cents, 'allocations', order.allocations, 'sum', sum(order.allocations))\nfor i in range(3):\n    print(refund_line(order, i, ledger))\nEOF")),
            ("Allocations sum to 999, not 1000, so per-line refunds add up to 2001 against 2000 paid. Find where allocations are computed.",
             S("allocat", ".", "*.py")),
            ("Read the allocation helper.", V("pricing/allocation.py")),
            ("Rounding each share independently can drift from the total in either direction. Floor each share, then hand the leftover cents to the largest remainders (ties to the earliest line).",
             R("pricing/allocation.py",
               "    return [round(total_cents * w / weight_sum) for w in weights]\n",
               "    parts = [total_cents * w // weight_sum for w in weights]\n    leftover = total_cents - sum(parts)\n    by_remainder = sorted(range(len(weights)),\n                          key=lambda i: (-(total_cents * weights[i] % weight_sum), i))\n    for i in by_remainder[:leftover]:\n        parts[i] += 1\n    return parts\n")),
            ("Re-run the order 5521 scenario; all three refunds should succeed and total 2000.",
             B("python - <<'EOF'\nfrom checkout.place import place_order\nfrom orders.models import Line\nfrom payments.ledger import Ledger\nfrom refunds.service import refund_line\nledger = Ledger()\norder = place_order(5521, [Line('a', 1000), Line('b', 1000), Line('c', 1000)], ledger, discount_cents=1000)\nprint('allocations', order.allocations)\nprint([refund_line(order, i, ledger) for i in range(3)], 'remaining', ledger.remaining(5521))\nEOF")),
            ("Also check the under-refund direction ($5 coupon) mentioned in the issue.",
             B("python -c 'from pricing.allocation import allocate; print(allocate(500, [1000, 1000, 1000]), allocate(7, [1, 1, 1, 1]))'")),
            ("Run the test suite.", T()),
            ("The ledger was right; the discount allocation now sums exactly.", SUBMIT),
        ],
    ),
    # 100 --------------------------------------------------------------------------------------
    dict(
        id="seed-100-exclusive-coupon-stacking",
        bug_type="order-dependent-rule", difficulty="medium",
        issue="""\
            Exclusive coupon combined with another code depending on entry order

            A customer with a $100.00 cart entered `WELCOME5` and then `VIP20`. They were
            charged $75.00: both codes applied. VIP20 is marked "cannot be combined with other
            offers". If you enter them the other way round (VIP20 first) you correctly get
            $80.00 with only VIP20 applied. Marketing noticed a spike in VIP20 + WELCOME5
            combos in last week's orders.
            """,
        files={
            "promo/__init__.py": "",
            "promo/coupons.py": '''\
                from dataclasses import dataclass


                @dataclass(frozen=True)
                class Coupon:
                    code: str
                    percent: int = 0
                    amount_cents: int = 0
                    exclusive: bool = False


                CATALOG = {c.code: c for c in [
                    Coupon("WELCOME5", amount_cents=500),
                    Coupon("SPRING10", percent=10),
                    Coupon("VIP20", percent=20, exclusive=True),
                    Coupon("STAFF30", percent=30, exclusive=True),
                ]}


                class InvalidCoupon(KeyError):
                    pass


                def lookup(code):
                    try:
                        return CATALOG[code.strip().upper()]
                    except KeyError:
                        raise InvalidCoupon(code) from None
                ''',
            "promo/stacking.py": '''\
                def resolve(coupons):
                    """Pick the coupons that apply to a cart, in the order they were entered.

                    Duplicates are ignored. Exclusive coupons cannot be combined with anything:
                    if any exclusive coupon was entered, only the first exclusive one applies.
                    """
                    unique = list({c.code: c for c in coupons}.values())
                    if unique and unique[0].exclusive:
                        return [unique[0]]
                    return unique
                ''',
            "checkout/__init__.py": "",
            "checkout/cart.py": '''\
                from promo.coupons import lookup


                class Cart:
                    def __init__(self):
                        self.items = []
                        self.codes = []

                    def add(self, sku, unit_cents, qty=1):
                        self.items.append((sku, unit_cents, qty))

                    def add_code(self, code):
                        lookup(code)  # validate early so the UI can show an error
                        self.codes.append(code)

                    @property
                    def subtotal_cents(self):
                        return sum(unit * qty for _, unit, qty in self.items)
                ''',
            "checkout/pricing.py": '''\
                from promo.coupons import lookup
                from promo.stacking import resolve


                def price(cart):
                    applied = resolve([lookup(code) for code in cart.codes])
                    subtotal = cart.subtotal_cents
                    discount = subtotal * sum(c.percent for c in applied) // 100
                    discount += sum(c.amount_cents for c in applied)
                    discount = min(discount, subtotal)
                    return {"subtotal": subtotal, "applied": [c.code for c in applied],
                            "discount": discount, "total": subtotal - discount}
                ''',
            "tests/test_pricing.py": '''\
                import pytest

                from checkout.cart import Cart
                from checkout.pricing import price
                from promo.coupons import InvalidCoupon


                def cart_with(*codes):
                    cart = Cart()
                    cart.add("shoes", 6000)
                    cart.add("socks", 2000, 2)
                    for code in codes:
                        cart.add_code(code)
                    return cart


                def test_no_coupons():
                    assert price(cart_with())["total"] == 10000


                def test_non_exclusive_stack():
                    out = price(cart_with("WELCOME5", "spring10"))
                    assert out["applied"] == ["WELCOME5", "SPRING10"]
                    assert out["total"] == 8500


                def test_exclusive_first_blocks_others():
                    out = price(cart_with("VIP20", "WELCOME5"))
                    assert out["applied"] == ["VIP20"] and out["total"] == 8000


                def test_duplicates_ignored():
                    assert price(cart_with("WELCOME5", "WELCOME5"))["total"] == 9500


                def test_invalid_code():
                    with pytest.raises(InvalidCoupon):
                        cart_with("BOGUS")
                ''',
        },
        hidden_test='''\
            from checkout.cart import Cart
            from checkout.pricing import price
            from promo.coupons import lookup
            from promo.stacking import resolve


            def cart_with(*codes):
                cart = Cart()
                cart.add("shoes", 6000)
                cart.add("socks", 2000, 2)
                for code in codes:
                    cart.add_code(code)
                return cart


            def test_exclusive_entered_second_still_exclusive():
                out = price(cart_with("WELCOME5", "VIP20"))
                assert out["applied"] == ["VIP20"]
                assert out["total"] == 8000


            def test_first_exclusive_wins():
                codes = ["WELCOME5", "STAFF30", "SPRING10", "VIP20"]
                assert [c.code for c in resolve([lookup(c) for c in codes])] == ["STAFF30"]


            def test_non_exclusive_still_stack():
                assert [c.code for c in resolve([lookup("SPRING10"), lookup("WELCOME5")])] == [
                    "SPRING10", "WELCOME5"]
            ''',
        p2p=["tests/test_pricing.py::test_no_coupons", "tests/test_pricing.py::test_non_exclusive_stack",
             "tests/test_pricing.py::test_exclusive_first_blocks_others", "tests/test_pricing.py::test_duplicates_ignored",
             "tests/test_pricing.py::test_invalid_code"],
        steps=[
            ("Find where exclusivity is handled.", S("exclusive", ".", "*.py")),
            ("Check how pricing applies the resolved coupons.", V("checkout/pricing.py")),
            ("Reproduce both entry orders on a $100 cart.",
             B("python - <<'EOF'\nfrom checkout.cart import Cart\nfrom checkout.pricing import price\nfor codes in (['WELCOME5', 'VIP20'], ['VIP20', 'WELCOME5']):\n    cart = Cart(); cart.add('x', 10000)\n    for c in codes:\n        cart.add_code(c)\n    print(codes, price(cart))\nEOF")),
            ("Pricing just sums whatever resolve() returns; the order dependence must come from resolve.", V("promo/stacking.py")),
            ("It only checks whether the *first* coupon is exclusive. Look for any exclusive coupon and keep the first one entered.",
             R("promo/stacking.py",
               "    if unique and unique[0].exclusive:\n        return [unique[0]]\n",
               "    exclusive = [c for c in unique if c.exclusive]\n    if exclusive:\n        return exclusive[:1]\n")),
            ("Re-run; both orders should give 8000 with only VIP20.",
             B("python - <<'EOF'\nfrom checkout.cart import Cart\nfrom checkout.pricing import price\nfor codes in (['WELCOME5', 'VIP20'], ['VIP20', 'WELCOME5']):\n    cart = Cart(); cart.add('x', 10000)\n    for c in codes:\n        cart.add_code(c)\n    print(codes, price(cart))\nEOF")),
            ("Run the tests.", T()),
            ("Done.", SUBMIT),
        ],
    ),
]
