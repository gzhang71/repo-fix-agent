from .truncate import truncate

TITLE_COLUMN_WIDTH = 40  # notifications.title is VARCHAR(40)


def card_title(title):
    return truncate(" ".join(title.split()), TITLE_COLUMN_WIDTH)


def preview(body, width=80):
    return truncate(body.replace("\n", " "), width, ellipsis="...")
