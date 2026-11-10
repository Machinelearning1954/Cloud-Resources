#!/usr/bin/env python3
"""Static check for Config/DefaultEngine.ini.

This is NOT a rendering test: no Unreal Engine is involved. It checks that the
file is well-formed and that every ACTIVE setting is on the verified allowlist
below, with a value of the expected kind. Each allowlist entry was confirmed
against a public Unreal source on 2026-11-10 (docs/RUMOR_LEDGER.md, row 70).

Usage: python3 tools/check_engine_config.py     (exit 0 = pass, 1 = fail)
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INI = os.path.join(ROOT, "Config", "DefaultEngine.ini")

BOOL = {"True", "False"}
RENDERER = "/Script/Engine.RendererSettings"
WINDOWS = "/Script/WindowsTargetPlatform.WindowsTargetSettings"

# section -> key -> (allowed values, required?)
VERIFIED = {
    RENDERER: {
        "r.DynamicGlobalIlluminationMethod": ({"1"}, True),   # 1 = Lumen
        "r.ReflectionMethod": ({"1"}, True),                  # 1 = Lumen
        "r.RayTracing": (BOOL, True),
        "r.Lumen.HardwareRayTracing": (BOOL, True),
        "r.Shadow.Virtual.Enable": ({"0", "1"}, True),
        "r.Nanite.ProjectEnabled": (BOOL, True),
        "r.GenerateMeshDistanceFields": (BOOL, True),
        "r.SkinCache.CompileShaders": (BOOL, True),
        "r.AntiAliasingMethod": ({"0", "1", "2", "3", "4"}, True),
    },
    WINDOWS: {
        "DefaultGraphicsRHI": ({"DefaultGraphicsRHI_DX12"}, True),
        "+D3D12TargetedShaderFormats": ({"PCD3D_SM6"}, True),
    },
}
# Verified names that must stay OFF by default (experimental / value unsettled).
MUST_STAY_COMMENTED = [
    "r.MegaLights.EnableForProject",
    "r.Lumen.HardwareRayTracing.LightingMode",
]


def parse(path):
    """Return ({section: [(key, value, line_no)]}, [problems])."""
    sections, problems, current = {}, [], None
    with open(path, encoding="utf-8") as fh:
        for n, raw in enumerate(fh, 1):
            line = raw.rstrip("\n")
            if raw.endswith("\r\n"):
                problems.append(f"line {n}: Windows line ending")
            if not line.strip() or line.lstrip().startswith(";"):
                continue
            m = re.fullmatch(r"\[(.+)\]", line.strip())
            if m:
                current = m.group(1)
                if current in sections:
                    problems.append(f"line {n}: duplicate section [{current}]")
                sections.setdefault(current, [])
                continue
            if "=" not in line or current is None:
                problems.append(f"line {n}: not a 'key=value' line inside a section: {line!r}")
                continue
            key, value = line.split("=", 1)
            if key != key.strip() or value != value.strip():
                problems.append(f"line {n}: stray spaces around key or value")
            sections[current].append((key.strip(), value.strip(), n))
    return sections, problems


def main():
    results = []

    def check(name, ok, detail=""):
        results.append((ok, name, detail))

    if not os.path.isfile(INI):
        print(f"FAIL  file missing: {INI}")
        return 1
    sections, problems = parse(INI)
    check("E1 file parses as ini", not problems, "; ".join(problems))

    unknown_sections = sorted(set(sections) - set(VERIFIED))
    check("E2 only verified sections", not unknown_sections, str(unknown_sections))

    bad, dupes, seen = [], [], set()
    for sec, rows in sections.items():
        for key, value, n in rows:
            if (sec, key) in seen:
                dupes.append(f"{key} (line {n})")
            seen.add((sec, key))
            allowed = VERIFIED.get(sec, {}).get(key)
            if allowed is None:
                bad.append(f"line {n}: {key} is not on the verified list")
            elif value not in allowed[0]:
                bad.append(f"line {n}: {key}={value} not in {sorted(allowed[0])}")
    check("E3 every active setting is verified, with a valid value", not bad, "; ".join(bad))
    check("E4 no duplicate keys", not dupes, ", ".join(dupes))

    missing = [k for sec, keys in VERIFIED.items() for k, (_, req) in keys.items()
               if req and (sec, k) not in seen]
    check("E5 all required settings present", not missing, ", ".join(missing))

    text = open(INI, encoding="utf-8").read()
    live = [k for k in MUST_STAY_COMMENTED
            if re.search(rf"^\s*{re.escape(k)}\s*=", text, re.M)]
    check("E6 experimental settings stay commented out", not live, ", ".join(live))

    vals = {k: v for rows in sections.values() for k, v, _ in rows}
    rt_ok = not (vals.get("r.Lumen.HardwareRayTracing") == "True"
                 and vals.get("r.RayTracing") != "True")
    check("E7 Lumen hardware ray tracing implies r.RayTracing=True", rt_ok)
    dx_ok = not (vals.get("r.RayTracing") == "True"
                 and vals.get("DefaultGraphicsRHI") != "DefaultGraphicsRHI_DX12")
    check("E8 ray tracing implies DirectX 12", dx_ok)
    skin_ok = not (vals.get("r.RayTracing") == "True"
                   and vals.get("r.SkinCache.CompileShaders") != "True")
    check("E9 ray tracing implies skin cache shaders", skin_ok)
    check("E10 status banner says rendering is untested", "RENDERING UNTESTED" in text)

    failed = 0
    for ok, name, detail in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f" — {detail}" if detail and not ok else ""))
        failed += not ok
    print(f"\nEngine config check: {len(results) - failed} passed, {failed} failed "
          f"(of {len(results)}). Static check only — not a rendering test.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
