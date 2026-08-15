"""Event-driven home automation routines."""

from .models import (
    Routine,
    RoutineAction,
    RoutineRun,
    ScheduledTask,
    VehicleEvent,
)
from .store import AutomationStore
from .scheduler import RoutineScheduler
from .vehicle import ArrivalOrchestrator, VehicleSimulator
from .planning import ScheduledActionInput, ScheduledRoutineInput, VehicleRoutineInput
from .routines import (
    build_arrival_routine,
    build_scheduled_routine,
    build_vehicle_routine,
    build_wake_routine,
)

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
