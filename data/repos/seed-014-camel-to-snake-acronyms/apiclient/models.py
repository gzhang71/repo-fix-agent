from dataclasses import dataclass

from .payload import normalize_keys


@dataclass
class User:
    user_id: int
    display_name: str
    avatar_url: str | None = None

    @classmethod
    def from_api(cls, raw):
        data = normalize_keys(raw)
        return cls(user_id=data["user_id"], display_name=data["display_name"],
                   avatar_url=data.get("avatar_url"))
