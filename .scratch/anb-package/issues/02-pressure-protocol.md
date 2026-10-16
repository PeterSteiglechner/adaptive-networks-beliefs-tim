# 02: Pressure protocol (`PressureSchedule` + `TestbedConfig`)

**What to build:** A researcher describes the experiment protocol as data. A `PressureSchedule` holds a strength and inclusive `(start, end)` events and answers `strength_at(t)`. It's built with `single_event(s)` (101–150) or `two_events(s)` (101–150, 201–250), and ramps or pulses can later be subclasses. A `TestbedConfig` (`schedule`, `window_len = 10`, `recovery = 50`, `detail`) derives everything time-related from the events: `T`, `windows()` (`before`, `during{j}`/`after{j}`, with bare names for the first event), `snapshot_times()` and `track_times()`. This is seam 3 in the spec (`docu/spec-anb-package.md`, "`testbed` module" and "Testing Decisions").

**Blocked by:** 01 (Test scaffold and dependencies)

**Status:** done

- [ ] `strength_at` is 0 outside the events and `strength` inside them, inclusive at both bounds
- [ ] For `single_event`: `T` = 200, windows 91–100 / 141–150 / 191–200, snapshots 0 / 100 / 200
- [ ] For `two_events`: also `T` = 300, windows 241–250 / 291–300 and snapshot 300
- [ ] `track_times()` is every step in detail mode, and {0, 1} ∪ window steps ∪ snapshots in ensemble mode
- [ ] All derived values are checked against legacy's numbers (through the ticket-01 helper) for both schedules in both modes
