# Overnight experiments: what matters for multi-camera 3D reconstruction

Run Oct 19, 2026, 02:16–05:23 on ece010 (one Tesla T4). 31 training runs, no failures.

**Setup (same for every run unless stated):** one standing player from the BASKET-Multiview
40-camera dome (cameras ~2 m away, 1920×1080, player ~808 px tall). Starting shape carved from
the silhouettes, 10,000 training steps, then scored on **5 cameras never used for training**.
Score = PSNR inside the player's bounding box. Higher is better; ~30 dB looks clean, below ~25 dB
artifacts are obvious.

**Noise level:** three identical baseline runs gave 30.26 / 30.11 / 30.11 dB. Differences
under ~0.2 dB are noise; everything below is well above that.

---

## 1. Number of cameras (from the evening run)

| Training cameras | 35 | 20 | 10 | 6 | 4 |
|---|---|---|---|---|---|
| PSNR (dB) | 30.5 | 30.3 | 28.9 | 26.8 | 24.0 |

**Takeaway:** a knee at 10–20 cameras seeing the subject. Below ~6, unseen viewpoints break.

## 2. Calibration error

| Error injected into every camera | PSNR | ≈ pixel error at 2 m |
|---|---|---|
| none | 30.2 | 0 |
| aiming 0.05° / 0.1° / 0.25° / 0.5° / 1° | 30.0 / 29.2 / 27.2 / 25.2 / 22.8 | 0.75 / 1.5 / 3.8 / 7.5 / 15 px |
| position 0.5 / 1 / 2 / 5 cm | 28.2 / 26.4 / 24.2 / 20.1 | 2.1 / 4.2 / 8.3 / 21 px |
| focal length 0.5 / 1 / 2 % | 28.9 / 27.3 / 25.4 | 0.9 / 1.8 / 3.6 px |

**Takeaway:** all three kinds of error fall on one curve once expressed in **pixels**:
under ~1 px is free, ~2 px costs ~1–2 dB, ~5 px costs ~4 dB, 15–20 px costs 7–10 dB.
Chart: `3 - Calibration error - chart (all types).png`.

**For lacrosse:** the budget is ~1 px of reprojection error.
- *Aiming* error causes the same pixel shift at any distance. For a 4K camera with a 30° lens
  (fx ≈ 7,200 px), 1 px = **0.008°**. With a 90° lens (fx ≈ 1,900 px), 1 px = 0.03°. Poles
  swaying in the wind will exceed that, so plan on rigid mounts plus per-frame pose refinement.
- *Position* error shrinks with distance. At 60 m, 1 px ≈ 3 cm (90° 4K), so a few cm of
  position error is fine on a field.
- *Focal length* needs ~0.3% accuracy. Fixed-focal (prime) lenses, or zoom locked after
  calibration.

## 3. Pixels on the player (resolution)

| Player height in training images | 808 px | 404 | 202 | 101 | 50 |
|---|---|---|---|---|---|
| PSNR (dB), viewed at full res | 30.2 | 30.1 | 27.9 | 24.3 | 20.7 |

**Takeaway:** no loss down to ~400 px, still usable at ~200 px (−2.3 dB), breaks down at
≤100 px (spiky artifacts, see `4 - Player size in pixels - examples.png`).

**For lacrosse:** a 1.8 m player 60 m away gets ~58 px with a 90° 4K lens (failing zone) and
~215 px with a 30° 4K lens (usable). So wide full-field shots can't carry player detail.
Plan **telephoto cameras assigned to zones**, at ≥200 px per player.

## 4. Camera layout (10 cameras each)

| Layout | Unseen views, front half | Unseen views, behind |
|---|---|---|
| All around (incl. below the player) | 29.9 | 29.7 |
| Upper half only (realistic venue) | 30.1 | 29.1 |
| No overhead (0° and 30° rings only) | 28.2 | 26.4 |
| One side only (like a sideline) | 30.4 | **24.8** |
| One ring at player height (7 cams) | 16.5 | 15.2 (model collapsed) |

**Takeaway:**
- Cameras only on one side look great from that side and lose **5.6 dB** from behind.
- Removing the high/overhead views costs ~2–3 dB.
- All cameras at one height fails completely, because height isn't constrained.

**For lacrosse:** surround the field (both sidelines and both ends), use **at least two
heights**, and include high views. No cameras below ground are needed: upper-half-only was as
good as full surround.

## 5. Time synchronization

Player jogging in place (feet move ~20 cm per frame). Half of 20 training cameras show a later frame.

| Lag of half the cameras | 0 | 33 ms (1 frame) | 67 ms | 133 ms |
|---|---|---|---|---|
| PSNR (dB) | 27.7 | **19.2** | 16.8 | 17.1 |

**Takeaway:** a **single frame** of mismatch costs 8.5 dB, the largest effect of all tests,
with heavy ghosting (`6 - Time sync - examples.png`). Frame-number or network-time sync is not enough.

**For lacrosse:** hardware genlock or PTP-triggered capture is required. This test can't
resolve sub-frame offsets. Physics says the ball (45 m/s) needs < ~0.1–1 ms.

## 6. Compute per frame (training steps)

| Steps | 500 | 2,000 | 5,000 | 10,000 | 30,000 |
|---|---|---|---|---|---|
| PSNR (dB) | 25.3 | 28.1 | 29.5 | 30.2 | 30.8 |
| Time on one T4 | 31 s | 84 s | 3 min | ~6 min | 19 min |

**Takeaway:** most of the quality arrives in the first few thousand steps, and 30k buys
+0.6 dB for 3× the time. The paper's 500 steps/frame relies on starting from a pre-built player
avatar. From scratch, 500 steps gives only 25 dB.

---

## Caveats
- Synthetic data, a single subject, cameras 2 m away, perfect masks. Real lacrosse (far
  players, 20 people, occlusion, sticks, outdoor light) will be harder, so treat these as
  **lower bounds** on what's needed.
- One random seed per setting (the baseline was repeated 3× to measure noise).
- The sync test uses whole-frame offsets only; the ball needs sub-millisecond sync.

## Files (all in this Experiments folder)
- Each experiment has a `chart` and an `examples` image, numbered 1–7 in the order above
  (1 = the 3D player itself, 7 = training time).
- Every number: `All numbers.csv`
- Scripts: `../Code/Experiment scripts/` (copies of the ones on ece010:/scratch/hbommaka/tools);
  raw outputs stay on ece010:/scratch/hbommaka/output/

---

# Update, Nov 24, 2026

The server deleted its scratch disk (28-day cleanup), so I downloaded the data again, rebuilt the
setup, and re-ran all 36 October runs before adding 28 new ones. 64 runs, no failures.

## 0. The October results reproduce

| | 35 cams | 20 | 10 | 6 | 4 |
|---|---|---|---|---|---|
| October (dB) | 30.49 | 30.26 | 28.88 | 26.79 | 23.97 |
| Nov 24 (dB) | 30.47 | 30.28 | 28.76 | 26.74 | 23.92 |

The other 31 repeats are within 0.16 dB of October, except the no-overhead camera layout, which
moved 0.33 dB (25.4 to 25.1).
Both sets are in `All numbers.csv`.

## 7. Occlusion (new)

Simulated on the single player: in the "blocked" cameras, part of the player is cut out of the
mask, as if another player stood in front. 20 training cameras.

| Cameras where half the player is blocked | none | 25% | 50% | 75% | all |
|---|---|---|---|---|---|
| PSNR (dB) | 30.2 | 27.3 | 24.5 | 21.4 | 18.5 |

| How much of the player is hidden (in half the cameras) | none | 25% | 50% | 75% |
|---|---|---|---|---|
| PSNR (dB) | 30.2 | 26.7 | 24.5 | 21.9 |

- The released code crashes on this. Its last step deletes every Gaussian that falls outside any
  camera's mask, and with occlusion that deleted the whole player (4,283 of 4,283). The numbers
  above are with that step turned off. Turning it off costs nothing on clean masks (30.2 vs 30.2).
- Even then it loses about 3 dB for every extra quarter of the cameras that are blocked. The
  training treats "hidden" as "not there", so the cameras that can see a body part fight with the
  ones that can't.

**For lacrosse:** players block each other all the time, so this has to be fixed in the method, not
with more cameras. The loss needs to ignore pixels where another player is in front (which needs
depth ordering between players), and the delete step has to go.

## 8. Mask errors (new)

The dataset masks are perfect. Here every training mask is made wrong on purpose. The player is
about 808 px tall.

| Mask edge off by | 2 px | 5 px | 10 px |
|---|---|---|---|
| Too loose (dB) | 29.1 | 26.7 | 24.7 |
| Too tight (dB) | 26.5 | 22.6 | 19.9 |

Wobbly edge that moves in and out by about 2.5 px / 5 px: 26.8 / 23.3 dB.
Three 40 px holes in each mask: 22.0 dB (most of the model gets deleted, 1,192 Gaussians left).

- Too tight is two to three times worse than too loose. A 2 px shrink already costs 3.7 dB.
- The image background is black here, so a loose mask only lets in black pixels. With a real
  background a loose mask would probably cost more than this.

**For lacrosse:** masks need to be right to about 1 to 2 px at this player size, and if they are
off they should be off on the loose side. Holes (missed sticks, gaps in the segmentation) are very
bad with the current delete step.

## 9. Time sync follow-up (new)

Frames are 33 ms apart and neither clip has slow motion, so I could not test less than one frame
directly. Two other things instead.

One frame late in half the cameras, at moments where the player moves less or more:

| Outline movement in one frame (px) | 5.7 | 7.3 | 10.7 | 12.8 | 14.1 | 15.8 |
|---|---|---|---|---|---|---|
| In sync (dB) | 27.4 | 27.8 | 27.8 | 26.5 | 27.8 | 27.5 |
| Half the cameras 1 frame late (dB) | 19.7 | 20.9 | 19.2 | 18.2 | 19.2 | 17.6 |

How many of the 20 cameras are one frame late:

| Late cameras | none | 1 | 5 | 10 |
|---|---|---|---|---|
| PSNR (dB) | 27.8 | 25.8 | 22.0 | 19.2 |

- The slowest moment I could find (5.7 px of movement) still loses 7.7 dB, so there is no safe
  amount of motion in this range. The loss grows only slowly with more motion (about 7 to 10 dB).
- A single late camera out of 20 already costs 2 dB.

**For lacrosse:** same conclusion as before but stronger. Every camera has to be on the hardware
sync, not most of them. The sub-frame tolerance is still not measured. It needs data rendered at a
higher frame rate.

## Caveats for the new runs
- Occlusion is simulated with masks on one player. It is not a real scene with two players.
- One random seed per setting.
- Charts: `8 - Occlusion`, `9 - Mask errors`, `10 - Time sync follow-up` in this folder.
- Inputs for every run: `../Experiment inputs.zip`. Raw outputs: `../Code/Server copy/`
  (copied off the server, which deletes files after 28 days).
