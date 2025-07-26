Issues:

- [ ] Path finding needs to be updated for walls. Walls were changed from tiles, to being between tiles.
- [ ] Holes do not function, mice and cats treat them as walls.
- [ ] Rockets do not function reliably, sometimes mice walk over them.

Missing features:

- [ ] We need to implement mouse spawners.

Improvements:

- [ ] Show fractional movement in visualizer (e.g. mouse and cats jump from one tile to the next)
- [ ] Add a game timer next to the steps (e.g. <mins>:<secs>), it should count down to 0:00 (max_steps).
- [ ] We don't need to show 'dead' mice. Remove that functionality.
- [ ] There is a 'win' condition, where there shouldn't be. The game simply continues until the timer reaches 0:00.