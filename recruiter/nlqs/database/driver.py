
from typing import overload


class AbstractDriver:

    def __init__(self, config):
        raise NotImplementedError("This method must be implemented by the subclass")

    def connect(self):
        raise NotImplementedError("This method must be implemented by the subclass")

    def disconnect(self):
        raise NotImplementedError("This method must be implemented by the subclass")

    def execute_query(self, query):
        raise NotImplementedError("This method must be implemented by the subclass")