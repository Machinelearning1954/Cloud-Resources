/* lock: an optional lock on the small local save (best runs), opened with the device's own screen lock.

   What it does
     - The save is a few numbers kept in this browser only. It never leaves the device.
     - With the lock on, the save is stored sealed with AES-256-GCM. The key is never stored: it is
       derived each time from a passkey on this device (the WebAuthn "prf" extension), and the device
       only releases it after its own user check.
   What it does not do, said plainly
     - "User check" means whatever unlocks the device: fingerprint, face, or the PIN, pattern or password.
       A web page cannot demand a fingerprint specifically, and cannot tell which one was used.
     - It needs a page served from its own https address (or localhost). It cannot work from a file
       opened off disk, or inside another site's frame; available() says so instead of failing later.
     - AES-256-GCM is a symmetric cipher, the kind that is not expected to be broken by quantum computers.
       The passkey around it is ordinary elliptic-curve cryptography and is NOT post-quantum. This is a
       device lock on local data, not a post-quantum system.
     - Tested with a software authenticator in a desktop browser. Not yet tested on a real phone.
     - If the passkey is deleted or the device is lost, a sealed save cannot be recovered. forget() starts over.

   API (browser: NH.lock, Node: module.exports), every method async unless noted:
     core.seal(prfBytes, obj)        -> blob      core.unseal(prfBytes, blob) -> obj   (pure crypto; used by the tests)
     available()                     -> { ok, why }            can this page use the lock here?
     status()            (sync)      -> "none" | "plain" | "locked" | "open"
     read()              (sync)      -> the save if it is readable now (plain, or locked and already opened), else null
     write(obj)                      -> stores it (sealed if the lock is on and open; refused if locked and not opened)
     enable(obj)                     -> creates the passkey, seals obj, turns the lock on
     open()                          -> asks the device to unlock; returns the save
     disable()                       -> lock off: the opened save is stored plain again
     forget()            (sync)      -> deletes the save, sealed or not */
(function (root) {
  "use strict";
  const KEY = "nh.k3d.save", VERSION = 1, INFO = "nh-crate-run-3d save v1", te = new TextEncoder(), td = new TextDecoder();
  const subtle = () => (root.crypto && root.crypto.subtle) || null;
  const rand = (n) => root.crypto.getRandomValues(new Uint8Array(n));
  const b64 = (buf) => { const u = new Uint8Array(buf); let s = ""; for (let i = 0; i < u.length; i++) s += String.fromCharCode(u[i]); return btoa(s); };
  const unb64 = (s) => { const bin = atob(s), u = new Uint8Array(bin.length); for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i); return u; };

  /* ---- pure crypto: secret bytes in, sealed blob out ---- */
  async function aesKey(secret, salt) { // HKDF-SHA-256 over the passkey's secret, then a non-extractable AES-256-GCM key
    if (!secret || secret.byteLength < 32) throw new Error("the device returned no usable secret");
    const ikm = await subtle().importKey("raw", secret, "HKDF", false, ["deriveKey"]);
    return subtle().deriveKey({ name: "HKDF", hash: "SHA-256", salt, info: te.encode(INFO) }, ikm, { name: "AES-GCM", length: 256 }, false, ["encrypt", "decrypt"]);
  }
  const core = {
    async seal(secret, obj) {
      const salt = rand(16), iv = rand(12), key = await aesKey(secret, salt);   // fresh salt and nonce every time: a nonce is never reused under one key
      const ct = await subtle().encrypt({ name: "AES-GCM", iv, additionalData: te.encode(INFO) }, key, te.encode(JSON.stringify(obj)));
      return { v: VERSION, salt: b64(salt), iv: b64(iv), ct: b64(ct) };
    },
    async unseal(secret, blob) {
      if (!blob || blob.v !== VERSION) throw new Error("this save was written by a different version");
      const key = await aesKey(secret, unb64(blob.salt));
      let plain;
      try { plain = await subtle().decrypt({ name: "AES-GCM", iv: unb64(blob.iv), additionalData: te.encode(INFO) }, key, unb64(blob.ct)); }
      catch (e) { throw new Error("the save could not be opened: wrong key, or the stored data was changed"); }
      return JSON.parse(td.decode(plain));
    },
  };

  /* ---- storage: one small record, and never a crash if storage is blocked ---- */
  let store = null, memory = null;      // `memory` holds the opened save and the secret for this visit only
  const storage = () => { if (store) return store; try { return root.localStorage || null; } catch (e) { return null; } };
  function load() { try { const s = storage(), raw = s && s.getItem(KEY); return raw ? JSON.parse(raw) : null; } catch (e) { return null; } }
  function keep(rec) { try { const s = storage(); if (!s) return false; s.setItem(KEY, JSON.stringify(rec)); return true; } catch (e) { return false; } }

  /* ---- the passkey ---- */
  const pk = () => root.PublicKeyCredential, creds = () => root.navigator && root.navigator.credentials;
  async function available() {
    const no = (why) => ({ ok: false, why });
    if (!subtle() || !pk() || !creds()) return no("This browser has no passkey support.");
    if (root.location && root.location.protocol === "file:") return no("The lock needs the game served from its own https address. It cannot work from a file.");
    if (root.isSecureContext === false) return no("The lock needs an https address.");
    try { if (root.self !== root.top) { const fp = root.document.featurePolicy;
      if (!fp || !fp.allowsFeature("publickey-credentials-create") || !fp.allowsFeature("publickey-credentials-get")) return no("The lock is switched off inside this viewer. Open the game at its own address to use it."); } }
    catch (e) { return no("The lock is switched off inside this viewer. Open the game at its own address to use it."); }
    try { if (!(await pk().isUserVerifyingPlatformAuthenticatorAvailable())) return no("Set up a screen lock on this device first (fingerprint, face or PIN)."); }
    catch (e) { return no("This browser could not check for a screen lock."); }
    try { if (pk().getClientCapabilities) { const caps = await pk().getClientCapabilities(); if (caps && caps["extension:prf"] === false) return no("This browser cannot derive a key from a passkey yet."); } } catch (e) { /* older browsers: find out when we try */ }
    return { ok: true, why: "" };
  }
  const reason = (e) => e && e.name === "NotAllowedError" ? "The device check was cancelled or refused." : e && e.name === "SecurityError" ? "This address is not allowed to use passkeys." : (e && e.message) || "The device check failed.";
  async function secretFor(credId, prfSalt) { // ask the device to verify the user and hand back the passkey's secret
    let got;
    try { got = await creds().get({ publicKey: { challenge: rand(32), rpId: root.location.hostname, timeout: 60000, userVerification: "required",
      allowCredentials: [{ type: "public-key", id: credId }], extensions: { prf: { eval: { first: prfSalt } } } } }); }
    catch (e) { throw new Error(reason(e)); }
    const ext = got.getClientExtensionResults(), first = ext && ext.prf && ext.prf.results && ext.prf.results.first;
    if (!first) throw new Error("This device's passkey cannot derive a key (no prf result).");
    return new Uint8Array(first);
  }

  const lock = {
    core, available,
    use(s) { store = s; memory = null; },                    // tests swap in their own storage
    status() { const r = load(); return !r ? "none" : r.mode === "sealed" ? (memory ? "open" : "locked") : "plain"; },
    read() { const r = load(); return !r ? null : r.mode === "sealed" ? (memory ? memory.data : null) : r.data; },
    async write(obj) {
      const r = load();
      if (r && r.mode === "sealed") {
        if (!memory) throw new Error("The save is locked. Unlock it first.");
        memory.data = obj; keep({ mode: "sealed", cred: r.cred, prfSalt: r.prfSalt, blob: await core.seal(memory.secret, obj) }); return;
      }
      keep({ mode: "plain", data: obj });
    },
    async enable(obj) {
      const a = await available(); if (!a.ok) throw new Error(a.why);
      const prfSalt = rand(32); let made;
      try { made = await creds().create({ publicKey: {
        rp: { name: "Crate Run 3D", id: root.location.hostname },
        user: { id: rand(16), name: "crate-run-save", displayName: "Crate Run save" },   // a label only: no personal details are asked for or stored
        challenge: rand(32), timeout: 60000,
        pubKeyCredParams: [{ type: "public-key", alg: -7 }, { type: "public-key", alg: -257 }],
        authenticatorSelection: { authenticatorAttachment: "platform", residentKey: "preferred", userVerification: "required" },
        extensions: { prf: { eval: { first: prfSalt } } } } }); }
      catch (e) { throw new Error(reason(e)); }
      const ext = made.getClientExtensionResults(), prf = ext && ext.prf;
      if (!prf || prf.enabled === false) throw new Error("This device made a passkey but cannot derive a key from it, so the lock stays off. You can delete the unused \"Crate Run save\" passkey in your password manager.");
      const credId = new Uint8Array(made.rawId);
      const secret = prf.results && prf.results.first ? new Uint8Array(prf.results.first) : await secretFor(credId, prfSalt);   // some devices only answer on the second step
      memory = { secret, data: obj };
      if (!keep({ mode: "sealed", cred: b64(credId), prfSalt: b64(prfSalt), blob: await core.seal(secret, obj) })) { memory = null; throw new Error("This browser would not store the save."); }
    },
    async open() {
      const r = load(); if (!r || r.mode !== "sealed") return lock.read();
      if (memory) return memory.data;
      const secret = await secretFor(unb64(r.cred), unb64(r.prfSalt)), data = await core.unseal(secret, r.blob);
      memory = { secret, data }; return data;
    },
    async disable() { const r = load(); if (!r || r.mode !== "sealed") return; const data = await lock.open(); memory = null; keep({ mode: "plain", data }); },
    forget() { memory = null; try { const s = storage(); if (s) s.removeItem(KEY); } catch (e) { /* nothing to forget */ } },
  };
  if (typeof module !== "undefined" && module.exports) module.exports = lock; else (root.NH = root.NH || {}).lock = lock;
})(typeof window !== "undefined" ? window : globalThis);
