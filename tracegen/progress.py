"""Live counters and readable progress logs with bounded update cost."""
from __future__ import annotations

import asyncio
from collections import Counter
from datetime import datetime, timezone
import time
import sys

from tqdm import tqdm

from .plan import write_json


class Progress:
    def __init__(self, root, manifest, slots, records, selected_ids, session_id):
        self.root, self.run_id, self.session_id = root, manifest["run_id"], session_id
        self.targets = Counter(slot["problem_id"] for slot in slots)
        self.saved_by_task = Counter(row["problem_id"] for row in records)
        self.status_counts = Counter(row["status"] for row in records)
        self.selected_ids = selected_ids
        self.selected_total = sum(self.targets[task_id] for task_id in selected_ids)
        self.selected_saved = sum(self.saved_by_task[task_id] for task_id in selected_ids)
        self.completed_tasks = sum(self.saved_by_task[task_id] == target for task_id, target in self.targets.items())
        self.active = Counter()
        self.planned = len(slots)
        self.session_saved = 0
        self.started = time.monotonic()
        self.state = "starting"
        self.bar = tqdm(total=self.selected_total, initial=self.selected_saved,
                        desc="Traces", unit="trace", dynamic_ncols=True,
                        file=sys.stderr, disable=not sys.stderr.isatty())

    def snapshot(self):
        elapsed = time.monotonic() - self.started
        rate = self.session_saved / elapsed if self.session_saved else None
        remaining = self.selected_total - self.selected_saved
        eta = 0 if remaining == 0 else remaining / rate if rate else None
        saved = sum(self.status_counts.values())
        return {"run_id": self.run_id, "session_id": self.session_id, "state": self.state,
                "updated_at": datetime.now(timezone.utc).isoformat(), "planned_traces": self.planned,
                "saved_traces": saved, "ok_traces": self.status_counts["ok"],
                "error_traces": saved - self.status_counts["ok"],
                "planned_tasks": len(self.targets), "completed_tasks": self.completed_tasks,
                "selected_tasks": len(self.selected_ids),
                "selected_traces": self.selected_total, "selected_saved_traces": self.selected_saved,
                "active_requests": sum(self.active.values()), "active_by_server": dict(self.active),
                "session_saved_traces": self.session_saved, "session_elapsed_seconds": elapsed,
                "traces_per_second": rate, "eta_seconds": eta}

    def publish(self, *, state=None, log=False):
        if state is not None:
            self.state = state
        snapshot = self.snapshot()
        write_json(self.root / "progress.json", snapshot)
        if not self.bar.disable:
            self.bar.set_description(f"Traces ({self.state})", refresh=False)
            self.bar.set_postfix(active=snapshot["active_requests"],
                                 errors=snapshot["error_traces"],
                                 tasks=f"{self.completed_tasks}/{len(self.targets)}",
                                 refresh=False)
            self.bar.update(self.selected_saved - self.bar.n)
            if log:
                self.bar.refresh()
        if self.state in {"finished", "interrupted", "failed"}:
            self.bar.close()
        if log:
            line = (f"{snapshot['updated_at']} {self.state} "
                    f"tasks={self.completed_tasks}/{len(self.targets)} "
                    f"traces={snapshot['saved_traces']}/{self.planned} "
                    f"ok={snapshot['ok_traces']} errors={snapshot['error_traces']} "
                    f"selected={self.selected_saved}/{self.selected_total} "
                    f"active={snapshot['active_requests']}")
            with (self.root / "progress.log").open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")
                stream.flush()
            if self.bar.disable and not sys.stderr.isatty():
                print(line, flush=True)

    def saved(self, row):
        task_id = row["problem_id"]
        self.saved_by_task[task_id] += 1
        self.status_counts[row["status"]] += 1
        self.session_saved += 1
        self.selected_saved += 1
        if self.saved_by_task[task_id] == self.targets[task_id]:
            self.completed_tasks += 1
        if row["error"] is not None:
            with (self.root / "progress.log").open("a", encoding="utf-8") as stream:
                stream.write(f"{row['timing']['completed_at']} error trace={row['trace_id']} "
                             f"server={row['server_id']} {row['error']['message']}\n")
        self.publish()

    async def heartbeat(self):
        while True:
            await asyncio.sleep(5)
            self.publish(log=True)
