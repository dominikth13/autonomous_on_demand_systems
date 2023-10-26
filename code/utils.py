class IdProvider:
    def __init__(self) -> None:
        self._current_id = 0
    
    def get_id(self):
        current_id = self._current_id
        self._current_id += 1
        return current_id