# ChuChu Rocket! – Complete Game Mechanics

## 1. Board Anatomy

- **Hatches** – Infinite spawners for ChuChus (mice) or KapuKapus (cats); each has its own spawn timer.
- **Rocket (one per team)** – Goal tile. +1 per ChuChu, +50 per Gold ChuChu; a cat deducts one‑third of current cargo and then exits.
- **Walls** – Solid blocks that force a right turn when struck.
- **Holes / Void Tiles** – Any sprite that enters is removed from play.
- **Wrap Edges** – Optional. Boards may wrap on one or both axes so a sprite that walks off one side appears on the opposite side.

---

## 2. Sprites & Movement Algorithm

| Sprite       | Speed                                    | Notes                                                                                         |   |
| ------------ | ---------------------------------------- | --------------------------------------------------------------------------------------------- | - |
| **ChuChu**   | 1 tile / **8 ticks** (≈7.5 tiles ⁄ sec)  | Follows current direction; on obstruction turns **right**, then left, then U‑turn as fallback |   |
| **KapuKapu** | 1 tile / **16 ticks** (≈3.8 tiles ⁄ sec) | Same turning hierarchy; devours ChuChus on contact and damages rockets                        |   |

*Speed modifiers*: Global roulette events or custom rules can double or halve both species.

---

## 3. Arrow Panel System

- **Directions** – Up, Right, Down, Left.
- **Placement Limit** – Each player may have **3** active arrows; dropping a fourth removes their oldest.
- **Lifetime** – \~600 frames (≈10 s @60 fps); flashes before expiring.
- **Cat Interaction** – A cat that collides with an arrow head‑on shrinks the arrow; a second hit removes it instantly ("cat‑breaker").
- **Overwrite Rules** – By default you cannot place on another player’s arrow (toggleable in custom settings).

Strategic insight: Because cats move half‑speed, precisely timed cat‑breakers can sabotage an opponent’s junction without disrupting your own mouse stream.

---

## 4. Scoring & Victory (Battle / Team Battle)

| Event                     | Score Change                             |
| ------------------------- | ---------------------------------------- |
| ChuChu enters rocket      | **+1**                                   |
| Gold ChuChu enters rocket | **+50**                                  |
| Cat enters rocket         | **−⅓** of current mice (rounded down)    |
| Timer expires             | Highest score wins (or best‑of‑X rounds) |

*(Cats never give points to their launcher—purely offensive.)*

---

## 5. Roulette Events (triggered by “? ChuChu”)

| In‑Game English Name    | Effect (≈7 s)                                              |
| ----------------------- | ---------------------------------------------------------- |
| **Mouse Mania!**        | Rapid ChuChu (and Gold) spawns from all hatches            |
| **Cat Mania!**          | Only cats spawn from hatches                               |
| **Speed Up!**           | All sprites move twice as fast                             |
| **Slow Down!**          | All sprites move half‑speed                                |
| **Place Arrows Again!** | Global freeze: sprites stop, players can reposition arrows |
| **Mouse Monopoly!**     | All new mice warp straight into the roller’s rocket        |
| **Cat Attack!**         | One cat parachutes into every opponent’s rocket            |
| **Everybody Move!**     | Each rocket swaps ownership colours (massive point swings) |

### Spawning of “? ChuChu”

- “?” mice are **random replacements** for a regular ChuChu in a hatch’s spawn burst during arcade modes (Battle, Team Battle, Stage Challenge).
- Their frequency is controlled by the *% of ? Mice* slider in the Options → 4‑Player Battle settings (manual label **“Change the frequency of “? Mice”**). Default *Medium* ≈ 1 in 24 ChuChus; *Low* ≈ 1 / 48; *High* ≈ 1 / 12
- The engine throttles them so **only one “?” mouse can be on the board at once**; another will not spawn until the current one is rescued or lost
- When a hatch spawns a “?” mouse, it keeps the same facing direction as a normal ChuChu from that hatch.

---

## 6. Advanced Physics & Edge Cases

- **Entity Resolution Order** – Cat bite resolves before mouse scoring if they enter the same tile simultaneously.
- **Corner Priority** – Natural right‑turn happens before arrow redirection; two arrows may be required to override a T‑junction.
- **Infinite Loops** – A closed circuit keeps sprites cycling until a rule/event breaks the loop.
- **Wrap vs. Hole** – Level designers can combine wrap edges with interior holes for Pac‑Man‑style tunnels or death pits.

---

## 7. Timing & Performance Constants (Dreamcast default)

| Constant            | Default                                 | Notes                                   |
| ------------------- | --------------------------------------- | --------------------------------------- |
| Tick rate           | 60 ticks per second (1 tick = 1 ⁄ 60 s) | Core simulation clock                   |
| Arrow TTL           | \~600 ticks                             | ≈10 s                                   |
| Hatch interval      | 120 ticks                               | One sprite every 2 s (varies per board) |
| Mouse step interval | 8 ticks                                 | 1 tile every 0.133 s                    |
| Cat step interval   | 16 ticks                                | 1 tile every 0.267 s                    |
| Cat speed divisor   | 2                                       | Cat interval ÷ Mouse interval           |
