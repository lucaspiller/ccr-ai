# Puzzle‑Generator & BFS PRD

## 1 · Purpose

Produce a large, reproducible corpus of **static Puzzle‑mode boards** for behaviour‑cloning.  Boards have **no spawners and no time limit**; the player may place up to `arrow_budget` arrows before pressing *Start*.  A BFS solver finds the *shortest* winning arrow sequence, and a difficulty label is assigned for curriculum scheduling.

In Puzzle mode, the player can place a number of arrows before starting the map. They can place all or none. The game then begins, and further modification of arrows is not allowed. In Puzzle mode there are no spawners, it's purely routing that is important. The game is won when all mice have entered a rocket. The game is lost if a mice falls down a hole, or a cat eats a mouse.

---

## 2 · CSV Schema

| Column             | Type / range                            | Meaning                                       |
| ------------------ | --------------------------------------- | --------------------------------------------- |
| `seed`             | `uint32`                                | RNG seed for deterministic placement.         |
| `board_w`          | `int` 5–14                              | Grid width.                                   |
| `board_h`          | `int` 5–10                              | Grid height.                                  |
| `num_walls`        | `int` 0–50                              | Count of wall segments.                       |
| `num_mice`         | `int` 1–10                              | Number of mice to route.                      |
| `num_rockets`      | `int` 1–5                               | Player‑owned rockets.                         |
| `num_cats`         | `int` 0–2                               | Number of cats.                               |
| `num_holes`        | `int` 0–8                               | Trap tiles that lose the level.               |
| `arrow_budget`     | `int` 1–5                               | Arrows allowed before press Start.            |
| `bfs_solution`     | `list[Tuple[Tuple[int,int],Direction]]` | Optimal arrow placements (x,y) and direction. |
| `difficulty_label` | enum {Easy, Medium, Hard, Brutal}       | Based on solution length, cats, holes.        |

---

## 3 · Generation Workflow

```
Seed → BoardBuilder(settings) →
    solution = BFSSolver(board, depth_cap)
    if solution found:
        label = DifficultyScorer(board, solution)
        emit CSV row
    else:
        discard board
```

*Unsolvable boards are dropped; generation continues with the next seed.*

---

## 4 · BFS Solver

| Parameter     | Default                              | Notes                                        |
| ------------- | ------------------------------------ | -------------------------------------------- |
| `depth_cap`   | 40                                   | Hard stop depth.                             |
| `timeout_ms`  | 50                                   | CPU guard.                                   |
| `branch_hint` | "nearest‑mouse first"                | Orders arrow placements toward closest mice. |
| `hash`        | sorted tile positions and directions | Duplicate pruning.                           |

Returns `solution_actions (list[int])`, `len(solution)`.

---

## 5 · Difficulty Heuristic

```
score = 1.0 * len(solution)
      + 4.0 * num_cats
      + 2.0 * num_holes
```

| Score | Label  |
| ----- | ------ |
| ≤ 10  | Easy   |
| 11–20 | Medium |
| 21–35 | Hard   |
| > 35  | Brutal |

---

## 6 · Implementation Phases

| Phase | Deliverable                       | Target                              |
| ----- | --------------------------------- | ----------------------------------- |
| A     | `BoardBuilder` with seed intake   | Generates valid walls / holes.      |
| B     | `BFSSolver` + transposition table | Solves ≤ 7×7 puzzles < 10 ms.       |
| C     | CSV writer + difficulty scorer    | Produce 100 k puzzle rows in < 1 h. |

---

## 7 · Throughput Target

≥ 50 rows/sec on 8‑core CPU for 5 × 5 boards (including BFS solve time).

---

## 8 · Risks & Mitigations

| Risk                              | Impact               | Mitigation                                    |
| --------------------------------- | -------------------- | --------------------------------------------- |
| BFS depth blow‑up on large boards | Generation stalls    | `depth_cap` + `timeout_ms`; discard unsolved. |
| Difficulty skew                   | Curriculum imbalance | Periodic manual audit; tune weights.          |
| Duplicate boards                  | Wasted samples       | Hash board layout; skip if seen.              |
