"""Event-driven home automation routines."""

from .models import (
    Routine,
    RoutineAction,
    RoutineRun,
    ScheduledTask,
    VehicleEvent,
)
from .planning import ScheduledActionInput, ScheduledRoutineInput, VehicleRoutineInput
from .routines import (
    build_arrival_routine,
    build_scheduled_routine,
    build_vehicle_routine,
    build_wake_routine,
)
from .scheduler import RoutineScheduler
from .store import AutomationStore
from .vehicle import ArrivalOrchestrator, VehicleSimulator

__all__ = [
    "ArrivalOrchestrator",
    "AutomationStore",
    "Routine",
    "RoutineAction",
    "RoutineRun",
    "RoutineScheduler",
    "ScheduledTask",
    "ScheduledActionInput",
    "ScheduledRoutineInput",
    "VehicleRoutineInput",
    "VehicleEvent",
    "VehicleSimulator",
    "build_arrival_routine",
    "build_scheduled_routine",
    "build_vehicle_routine",
    "build_wake_routine",
]
