# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A toolkit that turns long videos into short, Instagram-ready clips:

1. **detect → downscale → rename** finds the good moments in a long video and cuts them into a carousel of 1080p clips (c1, c2, …).
2. **rank** trims a finished carousel down to the clips worth posting.
3. **reel** turns a clip into a vertical 9:16 reel with a virtual camera that follows the DJ, ending on the DNB Chile logo.

`detect.py` handles two kinds of content, chosen by a prompt at startup (or `--mode`):

- **Music sets** — finds the high-energy drops. Fully offline: `ffmpeg` plus audio energy analysis via `librosa`.
- **Interviews** — transcribes the speech with Whisper (local) and asks Claude which moments are worth clipping. Only the transcript text leaves the machine; the video never does.

Its main real-world use is the **El Sótano** sets of DNBCL (drum and bass, filmed in 4K from a fixed camera in front of LED-tube walls) — see [The El Sótano workflow](#the-el-sótano-workflow).

Repository: https://github.com/matiasbn/drop-detector (private). Branch `main`, no CI.

## Requirements and setup

- **macOS on Apple Silicon.** Every encode uses `h264_videotoolbox`, and interview mode runs Whisper through `mlx-whisper` on the Apple GPU. Porting means swapping the codec (e.g. `libx264`) and the Whisper backend.
- **`ffmpeg` and `ffprobe`** on PATH.
- **Python venv** in `.venv/` (the npm scripts call `.venv/bin/python` directly):

  ```bash
  python3 -m venv .venv
  .venv/bin/pip install -r requirements.txt
  ```

  `requirements.txt`: `librosa`, `numpy`, `ffmpeg-python` (listed but unused — everything shells out to the `ffmpeg` binary), `mlx-whisper` (pulls in `torch`), `torchvision` and `scipy` (reel.py). `anthropic` is commented out: only needed for `--llm api`.
- **Downloads on first use:** ~3 GB of Whisper weights (interview mode, to the Hugging Face cache) and ~74 MB of person-detector weights (reel.py, to `~/.cache/torch/hub/checkpoints/`).
- **Interview mode only:** the `claude` CLI on PATH (default `--llm cli`), or `ANTHROPIC_API_KEY` + `pip install anthropic` for `--llm api`.
- **`assets/outro-dnbchile.mp4`** for reel.py's outro. `assets/` is gitignored, so a fresh clone doesn't have it — see [the outro](#the-outro-assetsoutro-dnbchilemp4) to rebuild it.
- **Disk:** a set's long video is ~20–24 GB. Copying it into `input/` needs that much free space on top of the clips; the machine has run with ~30–50 GB free, so either empty `input/` right after each set or symlink the video instead of copying it (`detect.py` follows symlinks, `clean.py` removes only the link).

There are no tests, linters, or build step. `package.json` exists only for the `npm run` shortcuts.

## Commands

The npm scripts are thin wrappers that invoke the project's virtualenv Python:

```bash
npm run detect      # input/*.{mp4,mov,mkv,avi}  -> output/<name>_{drop,moment}_NN_MMmSSs.mp4
npm run downscale   # output/*                   -> downscaled/* (1080p, Instagram)
npm run rename      # output/* and downscaled/*  -> c1, c2, ... (sequential, in place, per folder)
npm run rank "<set folder>" -- [--keep 15] [--gap 45] [--video NAME]
                    # <set folder>/carrusel: best clips by light show stay as c1..cN,
                    # the rest -> carrusel/repetidos/ or carrusel/descartados/
npm run reel "<set folder>" c7 [c3 ...] -- [--out PATH] [--zoom ...] [--tilt ...]
                    # -> <set folder>/reel-c7.mp4: vertical follow-cam reel + logo outro
npm run clean       # empties input/, output/ and downscaled/ (deletes all files in them)
```

npm only forwards `--options` after a bare `--`; the direct call doesn't need it. Equivalent direct calls: `.venv/bin/python detect.py` (and `downscale.py`, `rename.py`, `rank.py`, `reel.py`, `clean.py`).

### `detect.py` flags

| Flag | Values | Default | Notes |
|---|---|---|---|
| `--mode` | `set`, `interview` | prompt | Required when stdin isn't a TTY (background runs, scripts); without it the script exits 1. The prompt also accepts `1`/`set`/`music` and `2`/`interview`. |
| `--llm` | `cli`, `api` | `cli` | Interview mode only. See [Reaching Claude](#reaching-claude---llm). |

### `rank.py` arguments

| Argument | Default | Meaning |
|---|---|---|
| `folder` (positional) | — | The set folder, holding the long video and `carrusel/`. |
| `--keep N` | `15` (`KEEP`) | How many clips stay in `carrusel/` as c1..cN. |
| `--gap S` | `45` (`MIN_GAP`) | Minimum seconds between the start times of two kept clips. A discarded clip closer than this to a kept one goes to `repetidos/`. |
| `--video NAME` | `video-audio.mov` | The long video inside the folder (4-alan's is `elsotano2.mov`). |

### `reel.py` arguments

| Argument | Default | Meaning |
|---|---|---|
| `folder` (positional) | — | The set folder, holding the long video and a ranked `carrusel/`. |
| `clips` (positional, 1+) | — | Clip names as they appear in `carrusel/ranking.txt`: `c7`, `c15`, `repetidos/47m52s`, `descartados/59m31s`. One reel per clip; a failing clip doesn't stop the rest. |
| `--start S --duration S` | — | Cut by time (seconds into the long video) instead of by clip name. Doesn't need `ranking.txt`. |
| `--video NAME` | `video-audio.mov` | The long video inside the folder. |
| `--out PATH` | `<folder>/reel-<clip>.mp4` | Output path; single clip only. The approved finals are saved as `reel-promo.mp4`, the name used in 9-kapza. |
| `--zoom Z` | `1.35` (`ZOOM`) | Crop height = source height / Z. Higher = tighter on the DJ. |
| `--follow-x S` | `0.12` (`FOLLOW_X`) | Seconds of smoothing on the horizontal follow. Lower = snappier. |
| `--follow-y S` | `0.3` (`FOLLOW_Y`) | Seconds of smoothing on the vertical follow. |
| `--tilt D` | `3` (`TILT`) | Max camera roll in degrees; `0` disables it. |
| `--no-outro` | off | Skip the flash + logo outro (then the audio isn't faded either). |

## Pipeline architecture

The scripts are independent and chained only by directory convention — each reads one folder and writes the next. `input/`, `output/`, `downscaled/`, `transcripts/` and `assets/` are all gitignored, so the repo ships empty.

```
input/ ──detect.py──▶ output/ ──downscale.py──▶ downscaled/
                         └──rename.py──▶ output/ + downscaled/ (in place)

<set>/carrusel/ (copied from downscaled/) ──rank.py──▶ c1..cN + repetidos/ + descartados/ + ranking.txt
<set>/<long video> + ranking.txt + .times.json ──reel.py──▶ <set>/reel-<clip>.mp4
```

detect/downscale/rename/clean resolve their I/O directories relative to the script file (`__file__`), so they work regardless of the current working directory. `rank.py` and `reel.py` take an arbitrary folder by design: the sets live in Google Drive, not in the repo. `reel.py` finds `assets/` relative to its own file.

`detect.py` splits by mode only in *how it picks the moments* — `plan_set_clips` and `plan_interview_clips` both return the same list of `{at, start, duration}` dicts (interview ones add `title` and `reason`), and everything downstream (cutting, naming, the other scripts) is shared.

### detect.py — sets

For each input video: extract mono 22050 Hz WAV (ffmpeg, to a temp file), load it with librosa, compute RMS energy (`frame_length=4096`, `hop_length=2048`), smooth it with an 11-frame moving average (~1 s), then keep frames above the `ENERGY_THRESHOLD` percentile. Detections are grouped into windows `MIN_DROP_DISTANCE` seconds wide (measured from the first frame of the group) and only the highest-energy frame per group survives. Because a group's peak can sit at its end and the next group's at its start, a second pass then walks the peaks in time order and, whenever two are closer than `MIN_DROP_DISTANCE`, keeps only the louder — without it two near-identical clips came out seconds apart. Each surviving drop is then cut from the **original** video.

Design choice (docstring of `detect_drops`): it targets absolute energy peaks rather than buildup→drop transitions, so chained drops are captured even when the DJ never lowers energy. The flip side is volume: an hour-long set yields ~35–50 drops, which is why `rank.py` exists.

Cut boundaries are **bar-aligned** when `BEAT_ALIGN` is on (the default): `librosa.beat.beat_track` gives the beat grid, seeded with `EXPECTED_BPM` (174 — the sets are drum and bass; without that prior librosa locks onto the half-time pulse at ~87 BPM and the grid comes out half as fine). `align_to_bars` then anchors it on the drop itself — drops land on a downbeat, which sidesteps having to detect downbeats independently. It uses the *median* beat period rather than the reported BPM, which is more robust over a long set. The clip starts `BARS_BEFORE` bars earlier and ends a whole number of `BARS_PER_PHRASE`-bar phrases **after the drop** — counting phrases from the clip start instead would push the end back into the middle of one, which lands on a beat but still hangs.

Phrase quantization, not `TARGET_DURATION`, decides the length: at 172 BPM a 16-bar phrase is 22.3 s, so clips come out ~24.7 s (one phrase plus a 2-bar lead-in) and the next step up is ~47 s. `TARGET_DURATION` only picks which multiple is nearest.

Because stream copy can only cut on keyframes — drifting the start by up to a GOP and undoing the alignment — aligned clips are **re-encoded** with `h264_videotoolbox` at `-q:v 80`, AAC 192k (~7 s per clip). Setting `BEAT_ALIGN = False` reverts to the old fast path: fixed `CLIP_BEFORE`+`CLIP_AFTER` seconds with `-c copy`. That fallback is also used per clip whenever beat tracking is unusable (fewer than 2 bars of beats, or the clip would run past either end of the grid).

| Constant | Value | Effect |
|---|---|---|
| `ENERGY_THRESHOLD` | 85 | Percentile of smoothed RMS above which a frame counts as a drop candidate. Lower → more drops. |
| `MIN_DROP_DISTANCE` | 30 | Seconds; group width and minimum spacing between drops. Clips are ~25 s, so drops never overlap. |
| `BEAT_ALIGN` | `True` | Bar-aligned, re-encoded cuts vs. fixed-length stream-copy cuts. |
| `TARGET_DURATION` | 30 | Picks the nearest whole number of phrases. |
| `BARS_BEFORE` | 2 | Lead-in bars before the drop. |
| `BEATS_PER_BAR` | 4 | 4/4. |
| `BARS_PER_PHRASE` | 16 | Length quantum after the drop. |
| `EXPECTED_BPM` | 174 | Tempo prior for beat tracking. Change it for non-DnB music. |
| `CLIP_BEFORE` / `CLIP_AFTER` | 2 / 28 | Fallback fixed window (seconds). |

### detect.py — interviews

**Interview mode** (`plan_interview_clips`) replaces detection entirely rather than re-tuning it. Absolute energy is meaningless on levelled speech, and an earlier attempt at scoring emphasis against a rolling-median baseline was dropped: it ranks *how* something was said, never *what*, so it can't tell a good story from a loud one. Instead:

1. `transcribe` extracts 16 kHz mono WAV and runs Whisper locally via `mlx-whisper` (`WHISPER_MODEL` = `mlx-community/whisper-large-v3-mlx`, `WHISPER_LANGUAGE = None` autodetects), with `verbose=False` so it draws a progress bar — Whisper's default prints nothing, and an hour of audio in silence reads as a hang. The first run also downloads ~3 GB of weights, whose Xet backend signs off with its own `Reconstruction complete` just before the quiet part starts. Results are cached in `transcripts/<name>.json` (`{"language", "segments": [{start, end, text}]}`) — re-running to try a different prompt should cost an API call, not another pass over the audio. Delete the cache file to force a re-transcribe.
2. The timestamped transcript (`[start - end] text` per line) goes to Claude with `SYSTEM_PROMPT`, which asks for `MOMENTS` (30) self-contained highlights as JSON (`{"moments": [{start, end, title, reason}]}`), **ordered best first**, with `title`/`reason` in the transcript's language. Ranking is on how well a fragment works pulled out of the interview — opening, whether the idea closes on its own, memorability — explicitly *not* on how central it was to the conversation. That order is preserved end to end, so clip `01` is the strongest and `rename.py` makes it `c1`; the clips are not in chronological order (the filename timestamp recovers that). Where two clips overlap, the higher-ranked one wins and the other is dropped.
3. `snap_to_segments` moves each proposed cut onto the nearest Whisper segment boundary — those sit where the speaker actually paused, so this is the speech counterpart of `align_to_bars` — then clamps to the real video duration (`ffprobe`). Clips outside `MIN_MOMENT`..`MAX_MOMENT` (15..90 s) are dropped: the model can return a plausible-looking span that snapping or clamping ruins, and ffmpeg would accept a broken one silently.

Why each moment was picked is written to `transcripts/<name>_moments.txt`, not to `output/` — `rename.py` would otherwise renumber it alongside the clips. Interview clips are always re-encoded, for the same keyframe reason as bar-aligned ones.

#### Reaching Claude (`--llm`)

- **`cli`** (default) shells out to `claude -p --model CLAUDE_MODEL --output-format json --append-system-prompt ...` with the transcript on stdin. It bills to the user's Claude Code subscription (Claude Max) and needs no API key. It runs in a temp directory so Claude Code doesn't load this repo's CLAUDE.md into an unrelated question. Timeout `CLAUDE_TIMEOUT` (600 s). The JSON envelope's `is_error` is checked.
- **`api`** uses the `anthropic` SDK (`max_tokens=16000`) and needs `ANTHROPIC_API_KEY`; billed per token on the Anthropic Console, separately from the subscription (~$0.10 per hour-long interview). The subscription's "extra usage" option is a different thing: it only extends Claude/Claude Code past the plan's limits.

The CLI path can't use structured outputs, so `parse_moments` tolerates a code fence or a stray sentence around the JSON (it takes the fenced block, else the outermost `{...}`).

`CLAUDE_MODEL` is `claude-opus-5`; check it against the current model list when touching this code.

### Cutting and naming

`process_video` prints each clip and cuts it with `cut_clip` (`-ss` before `-i`, `-t duration`). Output names: `<video name>_<drop|moment>_<NN>_<MMmSSs>.mp4`, where `NN` is the 1-based index (chronological for sets, rank for interviews) and `MMmSSs` is the drop time (`at`) — minutes are not wrapped into hours, so 75 minutes is `75m00s`. The input file's name becomes the prefix, so name the copy/symlink in `input/` after the artist.

### downscale.py

Re-encodes every video in `output/` to 1080p height (`scale=-2:1080`, width kept even) with `h264_videotoolbox -q:v 65`, AAC 192k, into `downscaled/` under the same name. Existing files are overwritten. Not portable to non-Mac without changing the codec.

### rename.py

Renames everything in `output/` and `downscaled/` (each folder independently) to `c1`, `c2`, … using a natural sort (`c2` before `c10`, and `_drop_02_` before `_drop_10_`), going through `_temp_c*` names first to avoid mid-rename collisions. The extension is kept. After this the timestamps in the names are gone — which is why `rank.py` has to recover them.

### rank.py

Post-selection for a finished carousel (`<set folder>/carrusel/` next to the long video). The sets are filmed in front of LED walls and the light show, not the drop, is what makes a clip worth posting — reviewing by hand meant scanning ~45 clips for the ones where the lights go off.

1. **Timestamps.** The clips are c1..cN with no time in the name, so their start times are recovered by re-running `detect.plan_set_clips` on the long video (the same deterministic planning that cut them) and matching by index. If the count differs from the number of clips it aborts — the folder was edited or cut with other settings. The result is cached in `carrusel/.times.json` (chronological list of exact start seconds). This step reads the whole long video (a few minutes per set).
2. **Light score.** ffmpeg decodes each clip at `FPS` (12) frames per second, scaled to `WIDTH`×`HEIGHT` (64×36) grayscale; only the top `WALL` (70%) of the frame is scored, since the decks sit at the bottom. The score is the mean absolute frame-to-frame brightness change (0..1): strobes, blackouts and colour hits score high (~0.15–0.22), a lit but static wall scores low (~0.02). 46 clips score in ~25 s. A clip that fails to decode is reported and left in place, unranked.
3. **Selection.** Sorted by score, best first; a clip is kept unless it starts within `--gap` seconds of an already kept clip or `--keep` clips are already kept.
4. **Layout** (inside `carrusel/`, nothing is deleted):
   - kept clips → `c1.mp4`..`cN.mp4`, **by rank** (c1 = most lights), not chronological;
   - discarded within `--gap` of a kept clip → `repetidos/<MMmSSs>.mp4` (same stretch of the set as a kept one);
   - the rest → `descartados/<MMmSSs>.mp4` (fine, just not in the top N).
   All moves go through `_temp_rank<i>.mp4` names first, since on a re-rank a clip's new name can be another clip's current one.
5. **`ranking.txt`** — one line per clip, kept first then discarded, each block best first:

   ```
   score  time    file
   0.214  14m33s  c1.mp4
   0.138  47m52s  repetidos/47m52s.mp4
   0.109  59m31s  descartados/59m31s.mp4
   ```

   `time` is the clip start (2 bars before the drop), so it reads ~3 s earlier than the `MMmSSs` detect.py put in the original filename.

**Re-running** on a folder that has `ranking.txt` re-picks from the stored scores with the given `--keep`/`--gap` and moves clips between `carrusel/`, `repetidos/` and `descartados/` accordingly — no re-scoring, no re-planning, takes seconds, and running it again with the defaults restores the original layout exactly. Clips listed in `ranking.txt` that no longer exist (e.g. deleted by hand while choosing) are reported and left out.

### reel.py

Makes a vertical reel from one carousel clip, modelled on the hand-edited `9-kapza/reel-promo.mp4` (made in CapCut). The output is 1080×1920 at the source frame rate (23.976), ~27.7 s for a standard clip (24.7 s + 2.9 s outro), h264_videotoolbox at 14 Mb/s, AAC 192k. One clip takes ~7 minutes on a busy machine (most of it decoding 4K twice).

1. **Which span.** `clip_span` looks the clip up in `carrusel/ranking.txt` (start rounded to the second), snaps to the nearest exact start in `carrusel/.times.json`, and takes the duration from the carousel file itself. So the reel covers exactly the same moment as the clip, but is **cut again from the 4K long video** — cropping a 9:16 window out of the 1080p clip would leave only 607 px of width to upscale.
2. **Finding the DJ** (`track_dj`). The clip is decoded at `DETECT_FPS` (12) at 640×360 and run through torchvision's `fasterrcnn_mobilenet_v3_large_fpn` (COCO, score ≥ 0.4). The DJ is the biggest person box; its centre x and top y are kept as fractions of the frame. It finds the DJ in ~99% of frames, blackouts included; misses are interpolated. A first attempt that located the DJ as the darkest columns against the bright wall was dropped: the dark gaps between tubes pulled it ~10% to the left.
3. **The camera** (`camera_path`). A 3-frame median filter removes detector jumps, then a Gaussian of `FOLLOW_X` / `FOLLOW_Y` seconds is the camera's inertia. The crop is 9:16, `source height / ZOOM` tall (900×1600 of the 3840×2160 frame at 1.35), centred on the DJ horizontally and placed so the top of their box sits `HEAD_AT` (12%) down the crop. **Tilt:** the camera rolls toward the direction it is panning, proportional to horizontal speed (the fastest 5% of the clip reaches ~80% of `TILT`), smoothed over 0.3 s — what a handheld operator does. To rotate without black corners the crop is cut larger by `cos θ + 16/9·sin θ` (≈1.09 at 3°), rotated, then trimmed back.
4. **Rendering** (`render`). ffmpeg's `sendcmd` feeds per-frame commands (`crop x`, `crop y`, `rotate a`) to a `crop → rotate → crop → scale` chain, so the whole moving camera is one ffmpeg pass over the 4K source. The `.cmd` file lives in a temp dir.
5. **Outro.** The last `FLASH` (0.2) s of the clip get a flash (brightness +0.45, contrast 0.7, 10 px blur, ±24 px red/blue split), then `assets/outro-dnbchile.mp4` (resampled to the clip's frame rate) is concatenated. The audio is the set's own, taken from the long video for clip + outro length, running at full level under the logo and fading out over the last `FADE` (1) s — the same shape as the kapza reel.

How the defaults were tuned, against the kapza reel and the user's feedback on saintcris c15 (v1→v5, since deleted):

| Version | Change | Verdict |
|---|---|---|
| v1 | x smoothing 0.6 s, fixed y | "the movement in kapza's is faster" |
| v2 | x 0.12 s, y follows the head at 0.2 s | better, but "follows the head, goes up and down" |
| v3 | y 0.7 s + outro | "still a bit static" |
| v4 | y 0.4 s | "look at kapza's, it even has some tilt" |
| v5 | y 0.3 s + tilt up to 3° | "goes super well" → the defaults |

Not done yet (seen in the kapza reel): jump cuts every 1–2 s between moments, and segments mirrored horizontally (the "BOX" sign reads backwards).

#### The outro (`assets/outro-dnbchile.mp4`)

The last 2.92 s of `9-kapza/reel-promo.mp4`, from 17.2 s on: the clean half of CapCut's **Signal Glitch 2** transition, the DrumAndBassChile.cl logo on black, and the logo leaving with the **Black Hole** out animation (Mask section). 1080×1920, 23.976 fps, no audio. Everything before 17.2 s still shows the kapza footage, which is why the flash is rebuilt over our own clip instead. `assets/` is gitignored at the user's request (not pushed); rebuild it with:

```bash
ffmpeg -ss 17.2 -i ".../el-sotano/9-kapza/reel-promo.mp4" -an \
  -vf "fps=24000/1001,format=yuv420p" -c:v h264_videotoolbox -b:v 14M assets/outro-dnbchile.mp4
```

reel.py exits with that command when the file is missing (unless `--no-outro`). There is no official CapCut MCP/API; the community MCPs only write CapCut draft files and can't export, so reusing the rendered outro is the practical way to get CapCut's effects.

### clean.py

Deletes every *file* directly inside `input/`, `output/` and `downscaled/` (subdirectories are left alone; a symlink is removed, not its target). It does **not** touch `transcripts/` or `assets/`, nor anything in Google Drive.

## Error handling

Rule for every batch here: one failing item never loses the rest of the batch. The `try` goes inside the loop; the failing item is reported by name with its reason, everything else completes, and the script exits 1 at the end with a list of what failed.

- `detect.py` isolates both levels: a video that fails (bad file, Claude error, unparseable JSON) is skipped, and within a video a clip that fails to cut is skipped. ffmpeg runs through `run_ffmpeg`, which keeps stderr and raises with ffmpeg's last line as the reason (`ffmpeg failed: ...`) instead of a bare `CalledProcessError`. reel.py has the same helper (`ffmpeg`).
- `downscale.py` isolates per clip, same reporting.
- `rank.py` isolates per clip when scoring and when moving.
- `reel.py` isolates per requested clip.
- `rename.py` and `clean.py` don't isolate: they only rename/delete local files.

## The El Sótano workflow

The sets live in Google Drive at:

```
/Users/matiasbn/Google Drive/My Drive/DNBCL (drive)/el-sotano/
  <N>-<artist>/
    video-audio.mov     # the long video exported from DaVinci Resolve with the mixed audio -- THE input
                        # (3840x2160, 23.976 fps, h264 yuv420p bt709, ~1 h, ~20 GB)
    C17xx.MP4           # raw camera recording (10-bit 4:2:2, no colour tags) -- not used
    REC_YYYYMMDD_N.wav  # raw audio recording -- not used
    reel-promo.mp4      # the vertical promo reel (9-kapza's was hand-made in CapCut; newer ones by reel.py)
    carrusel/           # the clips for the Instagram carousel (1920x1080, c1..cN)
      repetidos/  descartados/  ranking.txt  .times.json   # written by rank.py
```

State of the folders (Oct 2026):
- `1-` to `9-` were done by hand before `rank.py` existed; their `carrusel/` holds the hand-picked clips. `5-mute/carrusel` is empty and its clips are in `carrusel-bkp/`.
- `4-alan` has no `video-audio.mov`: its long video is `elsotano2.mov` (pass `--video elsotano2.mov` to rank.py/reel.py). Its carousel was regenerated with the full pipeline in Oct 2026 (41 drops, 15 kept); the old c1–c5 were CapCut exports and were deleted.
- `10-saintcris` to `14-triggered` were produced with the full pipeline (35–52 drops each, 15 kept by `rank.py`). No reels are kept there yet: the c7 reel and the c15 tuning iterations were deleted at the user's request.
- `extras/` and `z-*` are not sets to process.
- Colour: all `video-audio.mov` are DaVinci exports and carry a grade relative to the raw camera file, but 10–14 got a lighter one than the earlier sets (saturation ×1.4–1.7 vs ×1.9–2.3 for 6/9; 11-vagrant ×1.0), and every set has lifted blacks (2nd-percentile luma ~0.12).

Per set, in order:

```bash
cd /Users/matiasbn/drop-detector
SET="/Users/matiasbn/Google Drive/My Drive/DNBCL (drive)/el-sotano/10-saintcris"
npm run clean                                    # input/ output/ downscaled/ must start empty
ln -s "$SET/video-audio.mov" input/saintcris.mov # or cp; name it after the artist: it prefixes the clip names
.venv/bin/python detect.py --mode set
.venv/bin/python downscale.py
.venv/bin/python rename.py
mkdir -p "$SET/carrusel" && cp downscaled/*.mp4 "$SET/carrusel/"   # never over an existing carousel
npm run clean
.venv/bin/python rank.py "$SET"                  # then the user picks from carrusel/
.venv/bin/python reel.py "$SET" c7 --out "$SET/reel-promo.mp4"     # the clip the user chooses
```

Notes:
- Process one set at a time: `detect.py` treats everything in `input/` as one batch, and `rename.py` numbers the whole of `output/` together.
- Run long steps in the background (detect on an hour-long 4K set takes a while; so does each reel).
- `rank.py` must run on the untouched output of the steps above (it recovers timestamps by index). Run it before the user starts deleting clips; afterwards only re-rank (`--keep`/`--gap`), which works from `ranking.txt`. reel.py needs `ranking.txt` and `.times.json` too, unless given `--start/--duration`.
- The user reviews the carousel and the reels in Drive and picks by eye; the selection is theirs, not the scripts'.

## Design history

From the git log, in order — useful when a constant looks arbitrary:

1. Fixed 30 s clips (2 s before + 28 s after the drop), stream copy.
2. `rename.py` extended to both `output/` and `downscaled/`; `clean.py` added.
3. Cuts aligned to the musical grid (re-encode needed for frame-accurate starts).
4. Tempo prior of 174 BPM: librosa was locking onto the half-time pulse.
5. Clip length quantized to whole 16-bar phrases counted from the drop: bar-aligned ends still "hung" mid-phrase.
6. Interview mode: transcript + Claude instead of audio emphasis scoring; 30 moments, ranked best first; Whisper progress bar.
7. Per-video/per-clip failure isolation; second pass removing drops closer than `MIN_DROP_DISTANCE` (adjacent groups produced near-duplicate clips).
8. `rank.py`: light-show ranking, then the `repetidos/` split, then re-ranking with `--keep`/`--gap`, then `--video`.
9. `reel.py`: follow-cam vertical reel with tilt and the DNB Chile outro, tuned over five versions against the kapza reel.
