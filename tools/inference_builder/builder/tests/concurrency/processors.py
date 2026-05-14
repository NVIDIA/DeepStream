class RequestCounter:
    name = "request-counter"

    def __init__(self, config):
        self.config = config
        self.num_requests = 0

    def __call__(self, *args, **kwargs):
        if len(args) != 1:
            raise ValueError(
                "request-counter expects exactly one argument"
            )
        self.num_requests += 1
        return args[0], self.num_requests


class ResponseChecker:
    name = "response-checker"

    def __init__(self, config):
        self.config = config
        self.last_id = -1

    def __call__(self, *args, **kwargs):
        if self.last_id < 0:
            self.last_id = args[1]
        elif args[1] != self.last_id + 1:
            raise ValueError(
                f"response-checker expects request id to be {self.last_id + 1} vs {args[1]}"
            )
        self.last_id = args[1]
        return args[0]
