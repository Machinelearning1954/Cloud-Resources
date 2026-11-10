# Source/ — NOT RECOVERED

The original merged C++/UHT tree (Core, Player, Save, UI, World, AI, Missions,
Vehicles, plugins) was lost in the sandbox recycle before `tools/publish.sh` ran.
This directory intentionally contains **no code**: fabricating "source" that merely
looks like the original would be dishonest and would poison the compiler-loop.

Regeneration path: `docs/RECOVERY_LOG.md` (restore-from-local protocol, then
generate-from-ADR slices M0 → M1 → M2 → M3/3.5). The compile loop stands:

1. Generate VS project
2. Compile
3. Paste the full error log
