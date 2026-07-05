# Island of Jamaica 3D


**🎮 Play it live: [machinelearning1954.github.io/cloud-resources](https://machinelearning1954.github.io/cloud-resources/)**

A single-file, browser-based 3D open-world game engine built from scratch in JavaScript and Three.js (WebGPU) — no game engine, no external assets, no build step. One `.html` file, ~120 KB, runs on PC, Android, and iOS.

Genre homage to top-down open-world driving games; **all code, art, map, and audio are original**.

---

## Highlights

- **Procedural island generation** — a full Jamaica-shaped landmass (Kingston metro plus Spanish Town, Mandeville, Ocho Rios, Montego Bay, and Negril) generated from a superellipse coastline, terrain classification passes, and a highway graph routed around impassable karst mountains.
- **A\* pursuit AI** — police plan shortest paths over a weighted navigation grid (roads cheap, dirt/sand costly, sea and mountains impassable), replanning on a stagger and following waypoints until close, then switching to direct pursuit. Verified: a cross-island chase plans a 118-waypoint route in ~80 ms with zero invalid tiles.
- **WebGPU rendering with graceful fallback** — Three.js `WebGPURenderer` with automatic WebGL 2 fallback, ACES filmic tone mapping, shadow-mapped sun, and TSL-based HDR bloom post-processing.
- **Performance scaling** — four quality presets plus dynamic resolution scaling that watches frame time and adjusts render resolution to hold a target framerate.
- **Cross-platform input** — keyboard/mouse with pointer lock on desktop; an on-screen virtual joystick, look-drag, and action buttons on touch devices, auto-detected. Installable as a PWA.
- **Cinematic systems** — an opening camera flyover, a four-angle cinematic chase camera, and physics-driven vehicle body roll, squat, and dive.

---

## Technical systems

### World generation
The island is a `Uint8Array` terrain grid classified into sea / land / beach / mountain. The western landmass is defined by a superellipse with a coastline wobble term; beaches are derived by a sea-adjacency pass; a karst mountain core is carved from a radial field. Towns are grid-blocked and connected by a hand-authored highway graph that lays causeways across water and marks a road set consumed by both the renderer and the AI.

### Navigation & AI
A cost grid `NAV` is derived from terrain and road membership. `aStar()` implements A\* with a Manhattan heuristic and a bounded node budget, returning a tile path. Police entities re-plan on a staggered timer, advance through waypoints with look-ahead skipping, and fall back to steering toward the target inside a close radius. Traffic uses lane-keeping heuristics; pedestrians and gang NPCs use wander/flee state with terrain-blocked movement.

### Rendering
`InstancedMesh` is used for buildings, palms, and lamp poles to keep draw calls low. Lighting uses a pooled point-light system that repositions a small fixed set of lights to the nearest street lamps at night — the standard light-pool discipline for large scenes. A day/night cycle drives sun elevation, sky/fog color, emissive building windows, headlight spotlights, and a fading star field. HDR bloom is composed via Three.js TSL nodes with a try/catch fallback to plain rendering.

### Gameplay
State-machine wanted system (0–5 stars with heat decay), carjacking, hitscan shooting with building occlusion, delivery mission loop, dual playable characters with hot-swap, a dynamic stick-up mechanic, a follower/"fame" economy, six switchable render+time "dimensions," and pickup landmarks (health, funding, AR glasses).

### Audio
A synthesized reggae radio built on the Web Audio API — multiple stations (roots one-drop, dancehall, dub) scheduled bar-by-bar, plus an engine oscillator that pitches with speed, synthesized SFX, and an AUX slot for live audio streams. No audio files.

---

## Controls

| | Desktop | Mobile |
|---|---|---|
| Move / drive | WASD | Left joystick |
| Look / aim | Mouse | Right-side drag |
| Shoot | Left-click | FIRE |
| Enter / exit car | E | ENTER |
| Handbrake | Space | BRAKE |
| Switch character | Tab | SWAP |
| Cinematic camera | C | — |
| Radio / AR / dimension | R / G / V | on-screen |

Cheat codes: type in-game (e.g. `IRIEHEART`, `BULLETPROOF`, `YAADSTORM`).

---

## Running it

Play the live link above, or open `index.html` in a recent Chrome, Edge, or Firefox. It needs internet on first launch to load the Three.js module from a CDN, then runs locally. The start screen reports whether it obtained the WebGPU or WebGL 2 backend.

On mobile: open in the browser, then "Add to Home Screen" for a full-screen installable experience.

---

## Tech stack

JavaScript (ES modules) · Three.js `WebGPURenderer` · TSL post-processing · Web Audio API · HTML5 Canvas (2D minimap/HUD) · zero dependencies beyond Three.js · zero build step.

---

## Notes

Original work and a genre homage; it does not use any assets, code, or trademarks from existing commercial titles.
