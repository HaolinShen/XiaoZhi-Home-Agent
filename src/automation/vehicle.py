"""Vehicle event adapter and arrival routine orchestration."""

from __future__ import annotations

from datetime import timedelta

from .models import VehicleEvent
from .scheduler import RoutineScheduler
from .store import AutomationStore


class VehicleSimulator:
    def __init__(self, vehicle_id: str, home_id: str):
        self.vehicle_id = vehicle_id
        self.home_id = home_id

    def eta_event(
        self,
        eta_minutes: int,
        *,
        latitude: float = 31.2304,
        longitude: float = 121.4737,
        trip_id: str = "demo-trip",
    ) -> VehicleEvent:
        return VehicleEvent(
            vehicle_id=self.vehicle_id,
            home_id=self.home_id,
            event_type="eta_update",
            latitude=latitude,
            longitude=longitude,
            eta_minutes=eta_minutes,
            metadata={"trip_id": trip_id},
        )

    def geofence_enter(self, *, trip_id: str = "demo-trip") -> VehicleEvent:
        return VehicleEvent(
            vehicle_id=self.vehicle_id,
            home_id=self.home_id,
            event_type="geofence_enter",
            latitude=0,
            longitude=0,
            eta_minutes=0,
            metadata={"trip_id": trip_id},
        )


class ArrivalOrchestrator:
    def __init__(self, store: AutomationStore, scheduler: RoutineScheduler):
        self.store = store
        self.scheduler = scheduler

    def handle(self, event: VehicleEvent) -> int:
        if event.event_type not in {"eta_update", "geofence_enter"}:
            return 0
        routines = [
            routine for routine in self.store.list_routines(event.home_id, enabled_only=True)
            if routine.trigger_type in {"vehicle_eta", "vehicle_geofence"}
            and routine.metadata.get("vehicle_id") == event.vehicle_id
        ]
        trip_id = str(event.metadata.get("trip_id") or event.occurred_at.date().isoformat())
        eta = event.eta_minutes or 0
        anchor_at = event.occurred_at + timedelta(minutes=eta)
        for routine in routines:
            self.scheduler.schedule(
                routine,
                anchor_at=anchor_at,
                trigger_key=f"vehicle:{event.vehicle_id}:{trip_id}",
                now=event.occurred_at,
            )
        return len(routines)
