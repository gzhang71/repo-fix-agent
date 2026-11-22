def run_all(queue, handler):
    """Pop every job in priority order and return the handler results."""
    return [handler(job) for job in queue.drain()]
