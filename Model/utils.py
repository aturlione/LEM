class Result(object):
    OK = "OK"
    FAIL = "ERROR"

    def __init__(self, status, value):

        self.status = status
        self.value = value

    def to_json(self):
        return {
            "status": self.status,
            "min": float(self.value["min"]),
            "max": float(self.value["max"]),
            "mean": float(self.value["mean"]),
        }
