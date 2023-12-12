# Efficient and Safe Multi-Stage Closed-Loop Planning for Snake Game

[ [Planning Paper](./manuscript.pdf) ]

[![Video](./assets/snake.mp4)]

## Implementation Details

**Note: The implementation is in desperate need of a refactor, do not look at it too closely (or at all) right now**

The game is implemented and rendered using PyGame and the algorithm is implented using mostly stock python with the exception of a couple of numpy arrays.

Game settings including which planner to use, grid size, game speed, are set in the bottom of the `snake.py` script.


Installation is as simple as setting up your Python Virtual Environment however you prefer and running: `pip3 install -r requirements.txt`.


To run the main script: `python3 snake.py`, optionally `python3 snake.py > /dev/null` if the STDOUT is annoying.

There are no CLI arguments yet.


The implementation includes the first attempts at #1 and #2 detailed below, but has a couple of logical bugs that sometimes cause infeasible paths to be generated. The plan is to fix these bugs and implement the optional #3 step.
  

*Note: pygame initialization is not reliable, a display environment variable might have to be set like:*

`export DISPLAY=:0`

## General Paper To-Dos:

- [X] Fill in sections that are outlined or empty
- [x] Add citations and supporting details - first pass
- [ ] Check literature for any conflicting naming conventions
- [X] Spell check, grammar, and flow
- [X] Check that equations and notations are accurate and consistent.
- [x] Implement the first iteration of the proposed 2-Stage planner
- [X] Finalize Abstract
- [X] Finalize Title
- [X] Replace Hamilton Cycle -> Hamiltonian Cycle

  
## Optional Enhancements

- [ ] Implementation of the exact approaches.
- [x] Better rendering to show motion and plans more clearly.
- [ ] CLI arguments for implementation
