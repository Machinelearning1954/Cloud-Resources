#!/usr/bin/env node
/* Build the one self-contained page from src/ and vendor/. No dependencies.
     node build.js                       -> index.html
     node build.js --check               -> exit 1 if index.html is not what the sources produce
     node build.js --solo people --out dev/out/people.html
                                         -> that one module from src/, the other three from dev/baseline/, which holds
                                            the last integrated version of each visual module
                                            (so a module can be worked on without the others' unfinished changes) */
"use strict";
const fs = require("node:fs"), path = require("node:path"), crypto = require("node:crypto");
const here = (...p) => path.join(__dirname, ...p);
const args = process.argv.slice(2), arg = (k) => { const i = args.indexOf(k); return i < 0 ? null : args[i + 1]; };
const MODULES = ["world", "people", "props", "lighting"], solo = arg("--solo"), out = arg("--out") || here("index.html");
if (solo && !MODULES.includes(solo)) { console.error(`--solo must be one of ${MODULES.join(", ")}`); process.exit(2); }

const want = fs.readFileSync(here("vendor", "CHECKSUMS.sha256"), "utf8").split("\n").filter(Boolean).map((l) => l.trim().split(/\s+/));
for (const [sum, name] of want) { // the vendored library must be exactly the pinned file
  const got = crypto.createHash("sha256").update(fs.readFileSync(here("vendor", name))).digest("hex");
  if (got !== sum) { console.error(`vendor/${name} does not match its recorded checksum`); process.exit(1); }
}
const parts = [["vendor/three.min.js (three.js r160, MIT)", here("vendor", "three.min.js")], ["src/sim.js", here("src", "sim.js")], ["src/contract.js", here("src", "contract.js")]]
  .concat(MODULES.map((m) => [`src/${m}.js`, solo && m !== solo ? here("dev", "baseline", `${m}.js`) : here("src", `${m}.js`)]))
  .concat([["src/lock.js", here("src", "lock.js")], ["src/save.js", here("src", "save.js")], ["src/main.js", here("src", "main.js")]]);
const scripts = parts.map(([label, file]) => {
  const code = fs.readFileSync(file, "utf8");
  if (/<\/script/i.test(code)) { console.error(`${label} contains a closing script tag and cannot be inlined`); process.exit(1); }
  return `<script>/* ${label} */\n${code}\n</script>`;
}).join("\n");
const page = fs.readFileSync(here("src", "shell.html"), "utf8").replace("<!--BUILD:SCRIPTS-->", () => scripts);   // a function, so "$" sequences inside the library are not treated as replace patterns
if (args.includes("--check")) {
  const same = fs.existsSync(out) && fs.readFileSync(out, "utf8") === page;
  console.log(same ? "index.html is up to date with src/" : "index.html is STALE: run `node build.js`"); process.exit(same ? 0 : 1);
}
fs.mkdirSync(path.dirname(out), { recursive: true }); fs.writeFileSync(out, page);
console.log(`built ${path.relative(process.cwd(), out)}: ${(page.length / 1024).toFixed(0)} KB${solo ? ` (solo: ${solo})` : ""}`);
