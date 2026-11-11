# Local selectable A tracker profiles

The Pi integration UI exposes three versioned profiles. Default remains D0.
This enables local candidate testing, not an unconditional production promotion.

| Request tuning.tracker_variant | Behavior |
| --- | --- |
| d0 | Original tracker; no temporal feature retention/central restriction/handoff repair |
| central_v1 | Frozen zigzag-reduction candidate: central displacement, retained KLT points, no contrast recenter pull |
| central_handoff_v1 | central_v1 plus validated low-contrast motion handoff instead of blending disagreeing centers |

Unknown values are rejected. Omitted fields default to d0. Selected profile is
persisted in metadata.json and returned in API metadata. Existing TrackSequence
schema and RF/MiniRocket checkpoints, B/C processing and thresholds are unchanged.
The candidate can still drift, switch targets or fail on occlusions; it is not
a guarantee of body-center accuracy or clean training data.

## UI and rollback

Open http://127.0.0.1:5178/#/upload and select the A tracker under tracking
settings. The third option is the new motion-handoff candidate. Choose D0
again (or the tuning reset button on the upload page) to return to the original
algorithm for the next request. Changing a selector does not rewrite existing
results. The result label and history retain the requested/used profile.
The live clip panel also provides the selector. Spring clips use the AI-direct
route; legacy local clips pass it through the local bridge and persist it in
request_json. Existing bridge requests without this field retain D0.

For MiniRocket keep camera stabilization enabled as required by its existing
input contract. Selecting an A profile does not override other tuning values,
ROI, duration, resolution or classification-model selection.

## Deployment boundaries

Only docker-compose.pi.yml mounts the shared tracking_schemas.py and
manual_roi_tracker.py read-only into both model containers. The normal Compose
file and model worktrees/images are not edited. This avoids duplicating candidate
code or changing the RF/MiniRocket branches. The local override depends on the
same existing tracker interfaces in those pinned images; revalidate when updating
model branches. Recreate the AI containers after changing mount configuration;
restart them after changing mounted Python code.

```sh
cd docker
docker compose --project-name skydetect-pi-local --env-file .env --env-file .env.pi -f docker-compose.yml -f docker-compose.pi.yml up -d --no-deps --no-build ai-rf ai-minirocket
docker compose --project-name skydetect-pi-local --env-file .env --env-file .env.pi -f docker-compose.yml -f docker-compose.pi.yml build frontend
docker compose --project-name skydetect-pi-local --env-file .env --env-file .env.pi -f docker-compose.yml -f docker-compose.pi.yml up -d --no-deps --no-build frontend
docker compose --project-name skydetect-pi-local --env-file .env --env-file .env.pi -f docker-compose.yml -f docker-compose.pi.yml restart clip-bridge
```

Do not remove volumes or existing results. For an infrastructure rollback,
remove only the two A-code mounts from each AI service override, recreate those
two services, and use a frontend compatible with the old API (which does not
accept tracker_variant). Routine user rollback needs only the D0 selection.

## Verification (2026-11-11)

- Tracker/API-focused tests: 13 passed in the actual RF serving image, including
  all three profiles via ManualTrackingRequest -> execute_manual_tracking.
- Frontend tests: 5 passed. Bridge/local tests: 17 passed.
- Native Vite and Compose frontend builds passed (existing large-chunk warning).
- Port parity: Oido179 D0 64, central 17, handoff 64 observed frames; each ported
  run exactly matches its saved TrackSequence and raw/source/visibility output.
  This establishes transplant parity, not a new accuracy benchmark.
- Real HTTP synthetic smoke: both RF and MiniRocket, all three profiles followed
  by D0 again, eight successful requests; metadata, >=40 observations, JSON and
  overlay downloads verified. Unknown profiles returned 422.
- Browser: both selectors visible, D0 reset verified, upload -> ROI -> handoff
  -> MiniRocket response/overlay/download links completed with 48 observations.
  A first test deliberately disabled CMC and MiniRocket abstained as required;
  with CMC enabled it accepted one window. Synthetic labels are not accuracy
  evidence and must not enter training/evaluation data.
- Pi live stream was unavailable during browser verification; no live camera
  capture or hardware reconnection was attempted. Local clip selector/bridge
  wiring is present; this turn did not create a new physical camera clip.
- Initial verification was local. Publication includes only this selectable
  tracker feature; pre-existing clip-gap and hardware changes are excluded.
- The exact staged publication snapshot was exported to a temporary directory
  and separately verified: tracker tests 13 passed, bridge tests 14 passed,
  frontend tests 5 passed, Vite build passed. The earlier 17 bridge tests included
  three pre-existing uncommitted hardware tests, which are not in this commit.

Local-only evidence is in ../validation/tracker-profile-port/ and
../validation/tracker-profile-api/ relative to the repository root, including
summary JSON and ui_profile_verified.jpg. Frozen experimental source/results
remain in the sibling skydetect-ai-a-center-probe worktree.
