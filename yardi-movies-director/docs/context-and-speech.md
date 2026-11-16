# Context box and speech input: UX and UI design

The context box lets someone add extra description to a scene: mood, setting,
who is in it, style. The Speak button dictates into it, so the words can be
spoken instead of typed. This page records how the feature behaves and why, so
that later changes keep the same rules.

## Where it sits

- Under the **Scene** box and above **Lyrics**, so the scene is read first.
- Shown for **Video**, **Image** and **Music**.
- Hidden for **Voice**: a voiceover speaks the scene itself, and extra context
  would be read aloud.
- On a phone (under 860 px) the slate stacks above the reel and the box takes
  the full width. Checked at 390 px: no horizontal scrolling.

## Anatomy

1. **Label row:** `CONTEXT optional` on the left, the **Speak** button on the right.
2. **Textarea:** two rows high, resizable, 1,000 characters at most.
3. **Hint row:** one sentence on how speech is handled (left) and the counter
   `used / 1000` (right).
4. **Status line:** appears only when there is something to say. It is never
   shown by default, so an idle form stays quiet.

## States

| State | How it starts | Button | Box | Status line | Counter |
|---|---|---|---|---|---|
| Idle | Default | Outline, "Speak" | Editable | Hidden | `used / 1000` |
| Listening | Speak pressed | Solid red, pulsing dot, "Stop" | Red border, read-only | "Listening. Press Stop when you are done." (live, red dot) | Updates as words arrive |
| Stopping | Stop pressed | "Stop" until the last phrase arrives | Read-only | Still "Listening" until the end | Updates |
| Ended | Recogniser stops on its own | "Speak" | Editable | Hidden | Final count |
| Near limit | 900 characters or more | Unchanged | Unchanged | Unchanged | Amber |
| Full | Dictation would pass 1,000 characters | "Speak" | Editable | Amber: "The context box is full, so dictation has stopped." | Red at 1,000 |
| Error | Browser reports a problem | "Speak" | Editable | Red, with what to do next | Unchanged |
| Unsupported | No speech recogniser (Firefox) | Disabled, tooltip explains | Editable | Hint says speech needs Chrome, Edge or Safari | Unchanged |
| Hidden | Voice tab | Not shown | Not shown | Not shown | Not shown |

### Error messages

| Browser error | Message shown |
|---|---|
| Microphone blocked | "The microphone is blocked. Allow it for this page, on localhost or an https address." |
| Speech service blocked | "This browser does not allow speech recognition here." |
| No microphone | "No microphone was found." |
| Speech service unreachable | "The browser's speech service could not be reached. Check the connection and try again." |
| Nothing heard | "No speech was heard. Press Speak and try again." |

## Interaction rules

- **Speak toggles.** Pressing it starts listening; pressing Stop ends it and keeps
  every word already written.
- **Typing is paused while listening.** The box is read-only, so typed text
  cannot be overwritten by a late speech result.
- **Speech goes where the caret is.** Text before and after the caret is kept, and
  a space is added only where one is needed.
- **Speech stops at the limit.** Typed text is capped by the browser's `maxlength`;
  speech is capped in code, because a value set in code is not limited by it.
- **Action is held while listening** and says "Press Stop first, so the last words
  reach the context box." The hold does not apply on the Voice tab, which sends no
  context.
- **Switching tabs stops listening.** The words already written stay in the box.
- **Reuse stops listening** and refills the box from the take. Recognition results
  that arrive after that are ignored.
- **Ctrl + Enter** in the context box calls Action, as it does in the scene box.

## Copy

- Label: **Context**, with *optional* beside it.
- Placeholder: "Mood, setting, who is in it, style. Added to the end of your scene."
- Hint: "Speak uses your browser's speech recogniser. In Chrome and Edge the audio
  goes to their speech service."
- Button: **Speak** and **Stop**. Both are words, not just an icon.

Copy is written as plain sentences that say what happens next. Errors say how to
fix the problem.

## Accessibility

- The button's accessible name is its visible text, so it reads "Speak" or "Stop"
  as the state changes.
- The textarea is labelled by "Context" and described by the hint, the counter and
  the status line, so a screen reader announces the limit when the box is focused.
- The status line is a polite live region. Errors and limit messages use the same
  polite region and are announced without interrupting.
- Listening is shown with text, a red button and a dot. Colour is never the only
  signal. The dot's pulse is turned off for users who ask for reduced motion.
- Contrast: white text on the red button is about 5.3 to 1. Status text uses the
  normal text colour; red is used only for the dot and the border.
- Keyboard: Speak and the textarea are reachable by Tab; the focus ring is the
  app's accent outline.

## Privacy

- Speech is recognised by the browser. In Chrome and Edge the audio is sent to
  the browser maker's speech service. The studio never receives or stores the
  audio.
- Only the text is kept: it is saved with each take in `outputs/library.json` on
  this machine.

## Known gaps

- **Touch size:** the Speak button is about 25 px tall, below the 44 px that touch
  guidance recommends. It is not yet enlarged on touch screens.
- **No Escape to stop:** only the Stop button ends listening.
- **Language:** recognition uses the browser's language. There is no picker.
- **On-device recognition:** a recogniser that never leaves this machine would be a
  separate install. The studio does not include one.
- **Real microphone not tested here:** the browser checks use a stand-in
  recogniser, because the test environment has no microphone or speech service.
  The state machine, the layout and the error handling were exercised; the actual
  recognition quality was not.
