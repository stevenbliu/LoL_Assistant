import time
from collections import deque


class RiotRateLimiter:
    def __init__(self, per_second=20, per_2min=100):
        self.per_second = per_second
        self.per_2min = per_2min
        self.second_window = deque()
        self.long_window = deque()

    def wait(self):
        now = time.time()

        # Remove expired timestamps
        while self.second_window and now - self.second_window[0] > 1:
            self.second_window.popleft()

        while self.long_window and now - self.long_window[0] > 120:
            self.long_window.popleft()

        # Check if limits exceeded
        while (
            len(self.second_window) >= self.per_second
            or len(self.long_window) >= self.per_2min
        ):
            time.sleep(0.1)
            now = time.time()
            self.wait()  # recursive retry
            return

        # Record the request timestamp
        self.second_window.append(now)
        self.long_window.append(now)
