from __future__ import annotations
from driver.driver_job import DriverJob
from location.location import Location
from utils import IdProvider
import csv

ID_PROVIDER = IdProvider()

class Driver:

    def __init__(self, start_position: Location) -> None:
        self.id = ID_PROVIDER.get_id()
        self.current_position = start_position
        self.job: DriverJob = None
        # Time that passed since drivers last job
        self.idle_time = 0

    def is_occupied(self) -> bool:
        return self.job != None

    def set_new_job(self, total_driving_time: int, new_position: Location) -> None:
        self.job = DriverJob(total_driving_time, new_position, False)
    
    def set_new_relocation_job(self, total_driving_time: int, new_position: Location) -> None:
        self.job = DriverJob(total_driving_time, new_position, True)

    # Duration in seconds
    def update_job_status(self, duration: int) -> None:
        if self.job == None:
            self.idle_time += duration
            return

        self.idle_time = 0
        if self.job.total_driving_time - duration < 0:
            # Job is finished by next interval
            self.current_position = self.job.new_position
            self.job = None
        else:
            self.job.total_driving_time -= duration
