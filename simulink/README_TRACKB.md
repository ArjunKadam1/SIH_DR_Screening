# Track B simulation in plain words

## The one question it answers
At what patient arrival rate does the camp doctor start drowning in a queue?

## The model (`simulink/DR_Backlog_Model.slx`, built by `createBacklogModel.m`)
Follow the blocks left to right — each is labeled, colored, and carries a
description (double-click any block → Help/Description):

1. **ArrivalPerHour** (blue) — patients walking in per hour. THE knob you turn.
   `20` = normal day, `40` = stress day.
2. **PerSecond + AI_Processing** (blue/orange) — unit conversion and the 2.5 s
   AI grading delay (negligible at day scale, kept for the story).
3. **ServiceRate** (orange) — doctor speed: 1 patient per 105 s on average
   (= 5 min per referable case x 35% referable = ~34 patients/hour).
4. **NetFlow → BacklogInt → FloorZero** (yellow/green) — the heart of it:
   `queue change = arrivals − served`, added up over the day, never below zero.
5. **BacklogScope + BacklogDisplay** (gray) — THE RESULT. Flat zero = camp
   copes. Rising ramp = camp breaks. Double-click the Scope after a run.

## The two runs (already simulated, traces in `reports/`)
| Run | End-of-day queue | Picture |
|---|---|---|
| 20/hr (`reports/backlog_20hr.png`) | 0 patients | flat green line |
| 40/hr (`reports/backlog_40hr.png`) | ~46 patients | red ramp all afternoon |

Breaking point: ~34/hr (doctor capacity). Quote for the slide:
*"Below ~34 patients/hour the queue never forms; at 40/hour, ~46 patients
are still waiting at 6pm."*

## Reproduce (2 commands)
```
addpath('simulink'); createBacklogModel(20); sim('DR_Backlog_Model')
```
Change arrival: edit the `ArrivalPerHour` block value (or `createBacklogModel(40)`),
press Run, double-click `BacklogScope`.

## Honest limits (put on the slide)
Averaged continuous flow, NOT individual patients; single doctor; flat arrival
rate all day; AI time fixed at 2.5 s. Capacity illustration only — it says
nothing about diagnostic quality (see classifier/lesion metrics instead).
Raw runs saved in `simulink/backlog_runs.mat`.
