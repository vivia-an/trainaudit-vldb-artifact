# Recorded GPU launchers

These scripts preserve the commands used for the fault-injection and collector
measurements. They depend on external Megatron-LM and VTimeline checkouts and on
the distributed environment used by the original runs.

For a new environment, create a localized scratch copy before running them:

```bash
bash scripts/localize_paths.sh --base /path/to/workspace --out /tmp/trainaudit-launchers
```

The stable offline artifact checks consume the recorded outputs and do not
execute these launchers.
