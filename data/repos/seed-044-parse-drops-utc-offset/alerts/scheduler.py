from datetime import timedelta

from .timeparse import parse_ts


class Reminder:
    def __init__(self, name, at, lead_minutes=0):
        self.name = name
        self.at = parse_ts(at)
        self.lead = timedelta(minutes=lead_minutes)

    def fire_time(self):
        return self.at - self.lead

    def is_due(self, now):
        return now >= self.fire_time()


def due_reminders(reminders, now):
    return [r.name for r in reminders if r.is_due(now)]
