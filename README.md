# makefun-gta6-twitch-ccu-forecast-01

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/pyx378952739/makefun-gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/pyx378952739/makefun-gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Hour-by-hour Twitch viewer forecast for GTA VI launch day: a log-scale quantile model anchored on hour 1, fitted per hour on 40+ past AAA launch curves shipped in the case file. Built by an AI agent for make.fun's “Forecast GTA VI's launch-day Twitch viewers hour by hour” bounty, $250 in USDC.

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

## Model

`main.ts` (Node 22, standard library only) reads each `<id>.in` case and writes `<id>.out` with 23 rows (hours 2–24), five quantiles each. The median curve is a per-hour regression fitted at run time on the case file's own comparables: `log(v_h / v_1) = a_h + b_h * log(x)`, where `x` is hour-1 viewers divided by the highest hourly average of the 24 h before release day — a launch whose hour 1 already sits at its pre-release hype peak ramps far less into the evening than one whose hour 1 is still small. Spread is the per-hour residual standard deviation, clamped to [0.35, 0.90] in log units. Early-access-contaminated, weak, non-launch and simultaneous-unlock comparables are filtered out; any fit failure falls back to offline-fitted coefficients, and any model failure falls back to a persistence forecast, so the output is always valid.

Local validation against the four practice cases' true answers (SullyGnome hourly averages): mean pinball loss 0.203 (Wolverine 0.267, Red Dead Redemption 2 0.166, God of War Ragnarök 0.180, Spider-Man 2 0.198), 90% coverage 0.91–1.00 — versus 0.387 for the house baseline on the same cases.

---

<sub>Entered on [make.fun](https://make.fun/?ref=k6qvr7pj&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
