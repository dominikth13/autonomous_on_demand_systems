from utils import IdProvider
from state import Location

ID_PROVIDER = IdProvider()

class Driver:
    def __init__(self, start_position: Location) -> None:
        self.id = ID_PROVIDER.get_id()
        self.current_position = start_position
        self.job : tuple[int, Location] = None
    
    def is_occupied(self) -> bool:
        return self.job != None
    
    # Total driving time in seconds
    def set_new_job(self, total_driving_time: int, new_position: Location) -> None:
        self.job = (total_driving_time, new_position)
    
    # Duration in seconds
    def update_job_status(self, duration: int) -> None:
        if self.job[0] - duration < 0:
            # Job is finished by next interval
            self.current_position = self.job[1]
            self.job = None
        else:
            self.job[0] -= duration