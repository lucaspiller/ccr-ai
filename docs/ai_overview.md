# &#x20;AI Overview

## 1 · Goal & Scope

Build a lightweight RL agent that can **solve and compete** in the tile‑placement puzzle game *ChuChu Rocket!* (Dreamcast/GBA ruleset) while training and running on a single hobbyist GPU.

---

## 2 · Game Primer (core mechanics only)

| Element | Description                                                                                              |
| ------- | -------------------------------------------------------------------------------------------------------- |
| Board   | 10 × 14 tile grid (140 tiles) (configurable)                                                             |
| Actors  | *ChuChu* (mice) seek straight ahead; *KapuKapu* (cats) hunt mice.                                        |
| Goal    | Place **arrow tiles** (←↑↓→) so mice enter your rocket; avoid cats and guide them to opponents’ rockets. |
| Tick    | Every 0.5 s: entities move one tile, wrap around edges.                                                  |
| Actions | At most one tile placement/removal per 0.5 s tick.                                                       |

Single‑player puzzle mode is deterministic → ideal for curriculum RL.

---

| Layer What it does                    |                                                                                                                                                                                                         |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **1. Perception**                     | Turns raw board data into tensors the network can read. *Includes the CNN grid‑encoder for walls/arrows/flows and the cat set‑encoder for exact cat positions.*                                         |
| **2. State‑Fusion (embedding)**       | Concatenates the two perception outputs with the 16‑dim global‑feature vector, then passes them through a small MLP to create a single 128‑d latent that summarises “what’s going on this tick.”        |
| **3. Policy Head**                    | Reads the latent and produces a 700‑way soft‑max over *place‑arrow* and *erase* actions—i.e., it decides **where and what to place right now**.                                                         |
| **4. Value Head**                     | Uses the same latent to predict the expected future score differential; supplies a bootstrapping target for RL.                                                                                         |
| **5. Planner (optional)**             | A tiny one‑step look‑ahead that checks “would this action immediately send a cat toward an enemy rocket?” and masks obviously bad moves; keeps the action space sane before the policy is well‑trained. |
| **6. Training Loop**                  | **Behaviour‑Cloning → PPO self‑play** curriculum that updates the policy/value heads; handles replay buffer, advantage calculation, and parameter updates.                                              |
| **7. World‑Model (future extension)** | If we later want multi‑tick look‑ahead, we can add a lightweight predictive model of mouse/cat trajectories and embed it in an MCTS roll‑out—similar in spirit to the Transport‑Tycoon world‑model.     |

## 3 · State Representation (v3)

ChuChu Rocket! requires **sub‑tile precision for barriers and ownership‑aware arrows**. The representation below reconciles your feedback while keeping the tensor small.

### 3.1 Grid Tensor

| Ch.    | Name                              | Range / Type                          | Details & Encoding                                                                   | Why it matters                           |
| ------ | --------------------------------- | ------------------------------------- | ------------------------------------------------------------------------------------ | ---------------------------------------- |
|  0     | Wall Vertical                     | {0, 1}                                | 1 = wall exists **between this tile and its right neighbour**.                       | Blocks E ↔ W movement.                   |
|  1     | Wall Horizontal                   | {0, 1}                                | 1 = wall exists **between this tile and the tile below**.                            | Blocks N ↔ S movement.                   |
|  2–9   | Rocket masks P0–P7                | {0, 1}                                | One channel per player (8 max). 1 = rocket occupies tile.                            | Score targets (own vs. opponents).       |
|  10    | Spawner mask                      | {0, 1}                                | Tile spawns mice or cats periodically.                                               | Anticipate traffic.                      |
|  11–14 | Arrow ↑↓←→ (geometry)             | {0, 1}                                | Direction occupied by **any** arrow.                                                 | Routing field.                           |
|  15    | Arrow owner ID                    | 0 (no arrow) .. 8                     | Encodes which player owns the arrow (0 = none, 1–8 = player id+1).                   | Ownership for cat routing.               |
|  16    | Arrow health                      | 0 (no arrow), 1 = healthy, 2 = shrunk | After one cat hit the arrow shrinks; second hit removes it.                          | Temporal arrow decay.                    |
|  17–20 | **Mouse‑flow Up/Down/Left/Right** | 0–255 → [0, 1]                        | Majority‑direction flow over next 4 ticks.                                           | Guides placement without per‑mouse cost. |
|  21    | Mouse flow confidence             | 0–255                                 | Sum of absolute flow counts across dirs—high ⇒ many mice present.                    | Confidence weight.                       |
|  22–25 | Cat orientation ↑↓←→              | {0, 1}                                | Individual cat positions; direction placed into one of four channels. Up to 16 cats. | Cat avoidance & attack planning.         |
|  26    | **Gold Mouse mask**               | {0, 1}                                | 1 = gold mouse occupies tile (unique).                                               | High‑reward opportunity.                 |
|  27    | **Bonus Mouse mask**              | {0, 1}                                | 1 = bonus mouse occupies tile (unique).                                              | Triggers bonus round.                    |

Total channels **28**. Board size 9 × 12 → tensor 28 × 9 × 12 ≈ 3.0 kB.

### 3.2 Global Feature Vector  (v4)

| Feature                        | Dim / Type       | Purpose                                                                                          |
| ------------------------------ | ---------------- | ------------------------------------------------------------------------------------------------ |
| Remaining time ticks           | 1 (int, /10 000) | Encourages urgency as the puzzle clock winds down.                                               |
| Arrow budget remaining (self)  | 1 (int 0‑3)      | Capacity awareness; duplicated here because tensor only shows *spatial* arrow usage, not budget. |
| Cat count (live)               | 1 (int 0‑16)     | Global threat level; complements per‑tile cat channels.                                          |
| Bonus state one‑hot (None + 4) | 5 dims           | Indicates active bonus: Mouse Mania, Cat Mania, Speed Up, Slow Down.                             |
| Player scores P0–P7            | 8 ints / 100     | Current scoreboard; lets policy decide whether to help or hinder others.                         |

Total global feature dims: **1 + 1 + 1 + 5 + 8 = 16**.  After normalisation and one‑hot encoding they are concatenated with the CNN + cat‑encoder embeddings.

### 3.3 Pre‑processing Notes

- **Wall channels**: walls come from edge list; flag vertical walls in the *left* tile, horizontal walls in the *upper* tile.
- **Arrow owner‑ID**: 0 if no arrow; else owner\_id + 1.  Embedded to 8‑d vector inside the network.
- **Mouse‑flow majority**: compute flow counts per tile; route the dominant direction into its channel and total magnitude into confidence.
- **Gold / Bonus mice**: because only one of each may exist, set a single 1‑bit mask in channels 26 & 27.
- **Cat channels**: mark each cat in its facing‑dir channel; positions also streamed to the cat set‑encoder.
- **Global features**: normalise numeric scalars (divide ticks by max, scores by 100) and one‑hot categorical bonus state.

### 3.3 Pre‑processing Notes

- **Wall channels**: walls come from edge list; for each edge set the bit in the *prefix* tile (vertical: left tile, horizontal: upper tile).
- **Arrow owner‑ID channel**: 0 if channel 11–14 have no arrow; else owner\_id (1‑8).  Use an embedding lookup to convert to 8‑d vector inside the net.
- **Mouse‑flow majority**: for each tile compute flow counts along four dirs; keep the largest in its directional channel, zero others.  Confidence channel holds the sum.
- **Cat channels**: mark the tile of each cat in the channel that matches its facing dir.

---

## 4 · Model Architecture (v3)

```
                 ┌────────────┐      global feats
Grid Tensor ─► CNN Encoder ─┐ │
                 │           ▼ ▼  (cat set encoder)
                 │        MLP‑pool  ◄─ List of cats {x,y,dir}
                 │           │
                 └── concat ─┴──► 128‑d fused embedding
                                   │
                      ┌────────────┴───────────┐
                      ▼                        ▼
               Policy Head               Value Head
              (2‑layer MLP)            (2‑layer MLP)
```

### 4.1 Components

| Block               | Detail                                                                              | Params      |
| ------------------- | ----------------------------------------------------------------------------------- | ----------- |
| **CNN Encoder**     | 4 × Conv‑BN‑ReLU (26→32→64) + flatten                                               | \~80 K      |
| **Cat Set Encoder** | For each cat: [x/12, y/9, dir‑one‑hot(4)] → 2‑layer MLP (32 → 32) → max‑pool over N | \~4 K       |
| **Fusion MLP**      | Concatenate [CNN embed + Cat pool + global feats] → Linear 128                      | 10 K        |
| **Policy Head**     | Hidden 128 → ReLU → 256 → ReLU → softmax(   #actions )                              | 40 K        |
| **Value Head**      | Hidden 128 → ReLU → 64 → ReLU → scalar                                              | 8 K         |
| **Total**           |                                                                                     | **≈ 142 K** |

### 4.2 Action Space

- **Tile‑plus‑direction (10 × 14 × 4 = 560)** → place arrow of that dir at tile.\* **Erase‑tile (10 × 14 = 140)** → remove arrow owner‑agnostically.\* Total = **700 discrete actions**; output softmax of size 700.

### 4.3 Why this works

- **CNN** captures local arrow‑wall interactions (L‑junctions, loops).
- **Cat set‑encoder** keeps **exact positions** and allows reasoning about up to 16 cats without adding dense channels.
- **Majority mouse‑flow** compresses traffic so the CNN sees directional pressure instead of flickering individual sprites.
- Parameter count still tiny → inference < 0.3 ms in FP16.

### 4.4 Cat Encoder Details

The cat encoder converts a *variable‑length list* of live cats into a **fixed 32‑d embedding**:

1. **Per‑cat feature vector (8 dims)**\
   • `(x_norm, y_norm)` – board‑normalised coordinates in [0,1]²\
   • `dir_onehot(4)` – facing ↑↓←→\
   • `shrunk_arrow_ahead` – 1 if the tile one step ahead contains a low‑health arrow (cats will remove it)\
   • `dist_to_enemy_rocket` – L1 distance to nearest opponent rocket divided by 20.
2. **Shared MLP (8 → 32 → 32)** with ReLU maps each cat to a 32‑d latent representation (weights shared across cats).
3. **Symmetric Pooling** – element‑wise **max** over all cat latents (pad with −∞ for fewer than 16 cats).  Max emphasises the most threatening cat on each dimension.
4. **LayerNorm** on the pooled 32‑d vector stabilises scale before fusion.

Total extra parameters ≈ 4 K; negligible latency (< 0.01 ms).  This set‑encoder is permutation‑invariant and scales gracefully from 0 to the 16‑cat hard limit.

---

## 5 · Logical Layer Stack

| #  | Layer                         | Role in the agent                                                                                                                                 |
| -- | ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
|  1 | **Perception**                | *CNN grid‑encoder* (board tensor) + *Cat set‑encoder* convert raw game objects into dense feature maps and a 32‑d cat embedding.                  |
|  2 | **State‑Fusion**              | Concatenates perception outputs with the 16‑d global feature vector; an MLP compresses this into a single 128‑d latent that summarises the tick.  |
|  3 | **Policy Head**               | Reads the latent and outputs a 700‑way soft‑max over arrow‑placement / erase actions — the agent’s immediate decision.                            |
|  4 | **Value Head**                | Predicts expected future score differential, providing the baseline for PPO advantage calculation.                                                |
|  5 | **Planner (light rule mask)** | Optional pre‑filter that masks suicidal moves (e.g., placing an arrow that instantly feeds your rocket to a cat) before sampling from the policy. |
|  6 | **Training Loop**             | Orchestrates Behaviour‑Cloning and PPO updates; maintains replay buffer, computes advantages, applies gradients.                                  |
|  7 | **World‑Model (future)**      | A small predictive module that could simulate mouse/cat motion for multi‑tick look‑ahead and search (MCTS) once basic agent is solid.             |

---

## 6 · Training Pipeline

1. **Synthetic puzzle generator**
   - Random walls + rockets + 3–10 mice.  Label with BFS‑optimal solutions for imitation.
2. **Stage A · Behaviour Cloning** from generator labels until 95 % of simple puzzles solved.
3. **Stage B · Self‑Play RL** (PPO): agent plays timed puzzles where arrow budget limited; reward = mice‑in‑rocket – cats‑in‑rocket – arrow cost.
4. **Curriculum**: gradually increase mice count, walls, cat presence, then add multiplayer chaos (optional later).

---

## 7 · Milestone Timeline 

| Week | Deliverable                                    | Success Criterion                         |
| ---- | ---------------------------------------------- | ----------------------------------------- |
|   1  | Python wrapper around open‑source ChuChu clone | Headless board step/reset works.          |
|   2  | Grid encoder + random agent baseline           | Runs 60 fps without drop.                 |
|   3  | BFS puzzle generator + BC training loop        | Agent ≥ 70 % solve rate on easy set.      |
|   4  | PPO fine‑tune with sparse reward               | Surpasses BC by +10 pp on medium puzzles. |
|   5  | Arrow‑budget & time‑limit hard puzzles         | 80 % solve rate within 100 moves.         |
|   6  | Public web demo via WebAssembly or streamed    | Clears randomly seeded puzzle live.       |

---

