# intervals-daily-coach v2

Automatically pulls wellness and activity data from Intervals.icu, then writes:

- `data/latest/wellness.csv`
- `data/latest/activities.csv`
- `data/latest/daily_coach_input.json`
- `data/latest/daily_report.md`

## What changed in v2

- fixes the placeholder future wellness-row bug
- avoids GitHub Actions self-trigger loops
- adds 7-day recovery trends
- adds 7-day activity trends
- classifies the latest activity into recovery/easy/moderate-harder buckets
- generates a more useful daily conclusion and training suggestion

## Required GitHub secret

- `INTERVALS_API_KEY`
