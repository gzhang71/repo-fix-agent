import logging

log = logging.getLogger("applog.worker")


def process(jobs):
    for job in jobs:
        log.info("processing %s", job)
    log.warning("done: %d jobs", len(jobs))
