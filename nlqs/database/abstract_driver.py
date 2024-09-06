from typing import overload
from abc import ABC, abstractmethod
import pandas as pd

class AbstractDriver(ABC):

    def __init__(self, config):
        raise NotImplementedError("This method must be implemented by the subclass")

    @abstractmethod
    def connect(self):
        raise NotImplementedError("This method must be implemented by the subclass")

    @abstractmethod
    def disconnect(self):
        raise NotImplementedError("This method must be implemented by the subclass")
    
    @abstractmethod
    def execute_query(self, query):
        raise NotImplementedError("This method must be implemented by the subclass")

    @abstractmethod
    def retrieve_descriptions_and_types_from_db(self):
        raise NotImplementedError("This method must be implemented by the subclass")

    @abstractmethod
    def fetch_data_from_database(self, table_name: str) -> pd.DataFrame:
        raise NotImplementedError("This method must be implemented by the subclass")
    
    @abstractmethod
    def validate_query(self, query):
        raise NotImplementedError("This method must be implemented by the subclass")
    