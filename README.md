# Working Title: Efficient and Safe Two-Stage Closed-Loop Planning for Snake Game

  

# Implementation Details

The game is implemented and rendered using PyGame and the algorithm is implented using mostly stock python with the exception of a couple of numpy arrays.

  

Installation is as simple as setting up your Python Virtual Environment however you prefer and running: `pip3 install -r requirements.txt`.

  

To run the main script: `python3 snake.py`

There are no CLI arguments yet.

  

The implementation includes the first attempts at #1 and #2 detailed below, but has a couple of logical bugs that sometimes cause infeasible paths to be generated. The plan is to fix these bugs and implement the optional #3 step.

  

*Note: pygame initialization is not reliable, a display environment variable might have to be set like:*

`export DISPLAY=:0`

  

# State of the Paper

Paper is writtten in markdown, which I find easier to draft in then LaTeX and uses the same equation syntax. As a result, the citations are not yet incorporated correctly.

  

As of 11/14, this paper is mostly a brain-dump from the ideas I have been kicking around for the past month and a half and recently have tried. Citations and supporting work are not included yet.

  

11/17: Filling in relevant related work and theory, and adding more details to the Multi-Stage Planner Implementation proposed to match implementations

# General Paper To-Dos:

- [ ] Fill in sections that are outlined or empty
- [x] Add citations and supporting details - first pass
- [ ] Check literature for any conflicting naming conventions
- [ ] Spell check, grammar, and flow
- [ ] Check that equations and notations are accurate and consistent.
- [x] Implement the first iteration of the proposed 2-Stage planner
- [ ] Finalize Abstract
- [ ] Finalize Title
- [ ] Replace Hamilton Cycle -> Hamiltonian Cycle

  

# Optional Enhancements

- [ ] Implementation of the exact approaches.
- [x] Better rendering to show motion and plans more clearly.

  

# Abstract

In this work, the potentially impractical and time-wasting motion planning problem of the popular Snake Game is approached in a thorough manner to give it the pedantic treatment it deserves. This planning problem is formulated rigorously, and definitions of optimality and snake safety are presented and discussed. Two exact safe and optimal solutions are described for complexity analysis. The A* discrete optimal planner is implemented to find head-to-goal paths to discuss the shortcomings of not considering future safety in this dynamic problem. A computationally efficient and safe multi-stage path planning implementation is detailed to advance the field of snake path planning and enhance the general merriment of the reader.

  

# Introduction

*TODO: complete section, currently just an outline*

  

*TODO: History of the game, and thoughts other works*

  

*TODO: add applications and related for this work, i.e. coverage planning etc.*

  

## Context and Related Work

  

*TODO: talk about people using learning approaches to control the snake*

  

Much of the inspiration of this work comes from the following two videos:

  

- [Viral Video of snake planning algorithms](https://www.youtube.com/watch?v=tjQIO1rqTBE) **_(Caution: Explicit Language)_**
	- This initial video by a YouTube creator, Code Bullet, reached 10M views.
	- This video illustrates various implementations of the A* path planning algorithm and various path-rejection metrics.
	- The planning implementation of this algorithm considers the Dynamic C-space.
	- This implementation has two planning modes:
		1. When the snake length is below a certain threshold, it pursues A* directly to the goal, with a path rejection criteria of that a certain proportion of the free nodes on the board must be goal-reachable.

		2. When the snake length is above or equal to the threshold, it uses the opposite of A*, which is not directly given but it is assumed that the costs and heuristics are negated or inverted. The result is the paths are as long as they can be while still obeying the path rejection metric.

  

- [Video of Hamiltonian Paths with cycle repair](https://youtu.be/TOpBcfbAgPg?si=rARS0FTdCOZeze-u) by Brian Haidet, PhD.
	- Contrary to Tapsell's Hamiltonian cycle, this approach focuses on applying cuts and splices to the existing Hamiltonian cycle.
	- The base of this implementation searches for potential cuts and splices to a Hamiltonian Cycle in order to reach the goal more quickly. This approach is a highly simplified to avoid the NP-hardness of finding Hamiltonian Cycles in grid graphs.
	- Various augmentations and special cases are added to account for shortcomings to the base, such as hugging a wall when available.
	- Haidet's implementation serves as the primary inspiration of this work.

  

Self research online:

- [Neel Gajjar's A* and Augmented A*](https://github.com/neelgajjar/Snake-game-AI-Solver#:~:text=1.-,Best%20First%20Search,head%20is%20to%20the%20apple)
- 	Simply implements A* and A* with forward checking, implementation isn't very clear, but as this work uses A* with a style of forward checking, Gajjar's work is included.

- [John Tapsell](https://johnflux.com/2015/05/02/nokia-6110-part-3-algorithms/)
	- The objective of this work is to implement a snake AI on the actual screen of a Nokia 6110.
	- The planning implementation assumes that the screen size is fixed and subsequently the grid size for the game is always the same. An initial Hamiltonian Curve is given that entirely fills the grid size. The Hamiltonian Cycle is represented as a 1-D array and is used to determine when a "shortcut" can be made. A shortcut in this sense is skipping a portion of the cycle when it can do so safely.
	- A drawback of this implementation is that when a shortcut is made, it potentially blocks off a portion of the Hamiltonian cycle which the next goal can assume resulting in the snake having to traverse most of the cycle again regardless.

  

## Graph Theory and Hamiltonian Cycles

  

*Note: need source on connectivity*

https://arxiv.org/pdf/1805.03192.pdf


- Hamilton Paths in Grid Graphs (1982)
- 	From this work, "Hamilton circuit problem for planar bipartite graphs with maximum degree 3 is NP-Complete"

- The planar Hamilton circuit problem is NP-complete (1979)
	- Could not access this work from library.tamu.edu yet

## Coverage planning

*TODO*

## Contributions

In this paper:
- Definitions such as **Plan Safety**, **Plan Optimality**, Reachability, and Connectivity Constraints for the Snake Game planning problem are defined, including a simple representation for the dynamic obstacle region is shown.
- Impractical but Exact planning algorithms for the problem formulation are shown: Exact Tree Search and Hamilton Cycle Planning.
- The short comings of the static environment optimal discrete planner, A*, is shown for this problem formulation.
- Lastly, a sub-optimal but safe and computationally efficient mixed planning implmentation is given.

# Problem Formulation

The Snake Game Problem ("problem"), is a seemingly simple but deceptively complex planning problem where the player is responsible for navigating the serpent through a two-dimensional (2D) fixed-size grid world towards the current goal.

  

## Rules of the Game

Navigation happens in discrete game steps, where the snake head will move in a commanded direction of the player if the commanded direction is not directly opposite of the previous direction moved (except if the snake is only one node long). If no movement or an invalid movement is selected, the snake will advance forward by one node in the previously selected direction. Each node of the snake is shifted forward to the previous position of the preceding node up to the snake's head. At the achievement of each goal, the snake's size is extended by one node by adding a node where the tail was on the previous cycle. Each goal is assumed to be sampled uniformly random from the set of unoccupied nodes.

  

This is implemented using a list where the front node is the tail at index 0 and the last node is the head at the maximum index. When the snake advances one node, the new node is appended to the end of the list. If there is no goal reached that movement, the first node in the list is removed. The list is reindexed to start at 0 at the new tail.

  

The game is won when the entirety of the grid world is filled with the snake's nodes. The game is lost by the snake's head exiting the bounds of of the grid world or the snake's head new node position is occupied by any other component of the new snake body. In practice, this means for any given valid game configuration, the safe actions are limited to navigating to a free space, goal space, or the space that is currently occupied by the tail of the snake which will be freed when the head moves to it. There also must be a non-standard failure case where there is an upper bound to the number of moves that are allowed between goals such that the plan cannot cycle forever without achieving a goal.

  

## C-Space Representation

For a formal definition, we will have a heavy focus on the accumulation of game steps for the construction of our various configuration space (C-Space) sets where the superscript $k$ means "at the $k$-th game step" enumerated from 0 which is the starting configuration. The tail node and head node at step $k$ is $T^k$ and $H^k$. We will use the subscript $i$ to denote the index within the buffer which a general node $N_i^k$ occupies at the $k$-th step. The snake's length in nodes is $n$. From this, it is apparent that $T^k=N_0^k$, $H^k=N_{n-1}^k$, and the snake propagation forward is $$\{T^{k+1},...,H^{k+1}\} = \{N_1^k, ..., H^k\}\cup\{H^{k+1}\}$$ if the goal is not reached at $H^{k+1}$ and the order is respected and $$\{T^{k+1}, ..., H^{k+1}\} =\{T^{k+1}, ..., H^k\}\cup \{H^{k+1}\}$$ if the goal node $G^k$ is reached at $H^{k+1}$. Additionally, the length at the next step $n^{k+1}$ is incremented by one $n^k+1$ if a goal is reached or remains the same $n^k$. For simplicity, the set of nodes that comprise the snake at step $k$ will be referenced as $S^k= \{T^k, N_1^k, ..., N_{n-2}^k, H^k\}$.

  

The world is represented as a bipartite grid graph with a fixed size, $G(V,E)$. The vertices in $V$ are defined by the cartesian product of every row and column index in the grid $V=\{(x, y) | \forall (x, y) \in X \times Y\}$ where $X=\{0, 1, 2, ..., n_x-1\}$ and $Y=\{0,1,2, ..., n_y-1\}$ where $n_x$ and $n_y$ are the number of rows and columns, respectively. The edges are defined as pairs in the node's adjacency set $(x,y)\rightarrow (x',y')$ where $(x',y') \in Adj((x,y))$; formally $$E=\{(x,y) \rightarrow (x',y')|\forall (x',y') \in Adj((x,y)), \forall (x,y)\in X\times Y\}$$. The adjacency set $Adj(\bullet )$ of a vertex $(x,y)$ is equivalent to the set $$Adj(x,y)=\{(x+1, y), (x-1, y), (x, y+1), (x, y-1)\}$$ which corresponds to the directions right, left, down, and up, respectively, as rendered.

  

It is critical to accurately and efficiently represent the C-space regions as the following sections require some of the following to make their plans effectively. There are two subsets of the total C-Space at step $k$ ($C^k$), the free region which can be used at $k$ ($C_{free}^k$) and the obstacle region which would result in a failure or "game over" state ($C^k_{obs}$). The union of these sets is equivalent to the total C-space at each step $k$ $$C^k = C^k_{free} \cup C^k_{obs}$$ with no intersections $$C^k_{free} \cap C^k_{obs} = \emptyset$$ and subsequently $$C^k_{free}=C^k \setminus C^k_{obs}$$ which is what will the primary tool for rejecting head-adjacent nodes in the obstacle region. The obstacle region has two disjoint contributors, the out-of-bounds (OOB) region $C^k_{OOB}$ and the set of the nodes occupied by the snake at step $k$ ($S^k$). It is important to note that both of these are trivial to check and propagate forward for steps $k+j$. This allows that the nodes within $S^k$ are not in $C^{k+j}_{obs}$ if their index $i < j$. Assuming that all of the paths planned do not self-intersect, this allows for precise calculation of the length of path required to overcome nodes in $S^k$ when planning forward from $H^k$, simply by referring to their index $i$.

  

Valid paths will be denoted as $$\pi_{[k, k+j]}=\{(x_k,y_k), (x_{k+1}, y_{k+1}), ..., (x_{k+j}, y_{k+j})\}$$ where $$(x_{p+1}, y_{p+1}) \in Adj((x_p, y_p)) \setminus C^{p+1}_{obs}, \forall p\in [k, k+j-1]$$ and $$(x_p, y_p) \neq (x_q, y_p), \forall p,q\in[k, k+j], p\ne q$$

An entire plan (or full valid game) is the ordered union of paths from the start to finish of the game $$\Pi=\cup_{g=0}^{n_{goals}-1}\pi_{g} \supset \cup_{g=0}^{n_{goals}-1}\{G^g\}$$ where the goal index $g$ is the game step range $[k, k+j]$ for the $g$-th goal. In other words, goal node $G^g$ is the singular goal node during the game steps $k, k+1, ..., k+j$.

  

## Optimality

In this planning problem, optimality of paths is narrowly focused on their length, which will be the cost measure for this work. The cost of a path is simply calculated by determined by counting the quantity of nodes in the path. An optimal path from $H^k$ to $G^k$ is a path $\pi^\star_{[k, k+j]}$ where there are no other paths between $H^k$ and $G^k$ which have lesser costs. Path $\pi^\star_{[k, k+j]}$ is not necessarily unique. It is also important to note that the minimum total game path $\Pi^\star$ is not necessarily the union of individual optimal paths, and since the goals are randomly sampled from $C^k_{free}$, $\Pi^\star$ will receive no treatment in our planning approaches, but consideration will be taken for future goals. It should also be noted that when constructing optimal paths, the index $i$ of a node $N^k_i \in S^k$ is equivalent to the cost of a potential head node $H^{p} \in S^{p}, p\ge k+i$ to use that node for a plan $H^p = N_i^k \subset \pi_{[k, k+j]}$. In other words, the cost of a node in a plan is equivalent to the maximum index $i$ in the current snake set which it can use. *TODO: Improve the wording here*

  

This is plain to see for the tail of the snake at index 0. If a path node adjacent to the current snake tail has cost greater than or equal to 0, the the current tail of the snake can be occupied by the head of the snake in the next turn from that node.

  

## Safety

The definition of safety is a broad one, and specific targeted subsets of this definition will be used heavily in the upcoming sections. A game has a safe total game path if and only if each $H^k$ is not in the obstacle set except the head $C^{k-1}_{obs} \setminus H^k$ for all $k$ game steps in the game.

  

There are many, many paths that allow for safe planning, but there is one important subset of safe plans that are guaranteed safe which will be the cornerstone of this work, cyclic paths. Cyclic paths $\pi^\circ_{[k,k+j]}$ are paths which create a closed loop by enforcing that the path begins at a node in $Adj(H^k)$, optionally navigates the graph returning to $Adj(T^k)$, and traverses $S^k$ in order. A cyclic path can be repeated indefinitely and the $H^k$ will never enter $C^k_{obs}$. It should be noted that individually optimal paths $\pi^\star$ and cyclical paths $\pi^\circ$ are usually mutually exclusive. However, a considerable amount of the [Mixed Implementation] will use goal optimal paths that are subsets of a cyclic path back to the tail position $\pi^\star \subset \pi^\circ$.

  

## Node-to-Node Reachability

*TODO: Elaborate on this section, it's okay if it's short but make sure it's supported*

  

- Weak $N-N'-$reachable: $\exists \pi \Rightarrow N, N' \in \pi$ -- There exists a valid path from N to N'.
- Strong $N-N'-$reachable: $\exists \pi \Rightarrow N, N' \in \pi \subseteq C^k_{free}$ -- There exists a valid path from N to N' and the entire path is within $C^k_{free}$.

  

Strong $N$-$N'$-reachable implies weak $N-N'-$reachable and is easier to calculate.

  

Cyclic plans enforce weak $H^k-T^k-$reachability.

## Connectivity Constraints

Another important consideration for optimality and safety is the connectivity of $C^k_{free}$. If $C^k_{free}$ is not fully connected, then the planning algorithm risks wasting game steps waiting for a viable path into the goal connected region of $C^k_{free}$. In this respect, it would be worth choosing a path with higher cost that guarantees that all of $C^{k+j}_{free}$ is connected after finishing the path. This constraint will be referred to as the **connectivity constraint**, and the objective of this is to reduce the overall cost of $\Pi$ and provide safety guarantees.

  

We have two extensions of the reachability measures above:

- Weak connectivity constraint: A set $A \subseteq C$ is weakly connected if $N_i, N_j$ is weak $N_i$-$N_j$-reachable for $i \ne j$.
- Strong connectivity constraint: A set $A \subseteq C$ is strongly connected if $N_i, N_j$ is strong $N_i$-$N_j$ reachable for $i \ne j$.

  

Similarly, strong connectivity constraint satisfies the weak connectivity constraint and is similarly easy to calculate. For the implementations in this paper, we will rely exclusively on the strong connectivity constraint, **referred to generally throughout this work as the connectivity constraint**.

  

This leads tangentially into an interesting set of results, which is $\exists N \in Adj(T^k) \Rightarrow N \in H^k \cup C^k_{free}$. This comes naturally as as the tail increments forward with game steps, either $T^{k-1}$ is converted to a node in $C^k_{free}$ or $H^k$ and $T^{k-1}\in Adj(T^k)$. We use this result to assert that for a valid full game path $\Pi$ requires that $C^k_{free}$ is weakly connected for all $k$. Additionally, $\exists N \in Adj(H^k) \Rightarrow N \in T^k \cup C^k_{free} \forall \Pi$, meaning that for a full valid game, the head is always able to move into the C-free or to the space the tail was previously occupied. It is clear from the above that every $\Pi$ requires that $C^k_{free}$ always satisfies the weak connectivity constraint for all $k$ and safety requires this constraint. Strong connectivity of $C^k_{free}$ at $k$ when a goal is achieved allows satisfaction of this constraint with less computation and guarantees strong $H^k$-$G^k$-reachability for all $k$.

# Optimal and Safe Snake Planning Implementations Under the Connectivity Constraint

There are two optimal and safe planning formulations which do not violate the connectivity constraint that will be considered in this section, an exact tree search and a Hamilton cycle approach. Both of which will have safety guarantees and are optimal under connectivity constraint; however, the formulations described here are not practical, especially for large grid worlds. This section serves to demonstrate that the complexity of exact planning in this problem formulation is comparable to similar games such as chess.

  

## Exact Search

*TODO: Update to involve more of the definitions above* so that this section can stay around this length except for figures.

  

*TODO: Update*

  

The exact search involves the enumeration of a directed Tree-Graph $G(V,E)$ in which every walk along the graph either ends at the goal or at a dead end. The Vertices and Edges sets are enumerated propagating forward from the starting node, the snake head $H^k$. Each edge is from a source vertex representing a path $\pi = \{H^k, N^{k+1}, ..., N\} \in E$ up to node $N=(x,y)$ and directs to a node in its adjacency set $N'=(x',y')$ such that $N' \notin \pi$ which gives us the set of Vertices set is $V=\{(\pi \rightarrow \pi \cup N')| N' \in Adj(N) \cup C_{free}^{N}\setminus \pi\}$ where $C_{free}^{N}$ is the $C_{free}$ when $N$ would be added. The graph $G$ is enumerated until the goal is reached $G^k \in Adj(N) \cup C_{free}^{N} \notin \pi$ or a dead end is encountered $Adj(N) \cup C_{free}^{N} \setminus \pi = \emptyset$ for that branch. Once the graph is enumerated, the lengths of the paths can be counted by the number of nodes in the leaf vertices, and subsequently, all paths that meet the following criteria can be pruned in this order of priority:

1. The path hits a dead end
2. The path bisects $C^k_{free}$
3. The path has a cost that is higher than the minimum after #1 and #2.

If this approach is adopted for every planning cycle, it is clear that $C^k_{free}$ will be connected when each goal is reached, such that when a new goal is sampled randomly from $C^{k+j}_{free}$, it is guaranteed to be reachable by the snake head. Enforcing that $C^{k+j}_{free}$ is fully connected at the end of every path guarantees that Safety is maintained, and the exact tree search pruning ensures that Optimality under Connectivity Constraints is maintained.

  

## Hamilton Cycle Planning
*TODO: This is basically the implementation by Haidet above*

*TODO: Determine if to keep (since the above is in video form and not written media) or just simply cite, determine complexity*

Hamilton Cycle Planning is an interesting method to solve this problem; a closed-loop path is considered which passes once through all of $C^k_{free}$ and $S^k$, in order, without violating connectivity within $S^k$. This path is defined as $\pi^{H\circ,k} = \pi^{\circ,k} \cup S^k$ such that $|\pi^{H\circ, k}| = |C^k|$. This assumes that a Hamilton Cycle is possible in the bipartite grid graph $G$. Any $\pi^{H\circ,k}$ is guaranteed safe and to complete the game, but it would be extremely uninteresting.

  

However, it is possible to convert one Hamilton Cycle in our graph $G$ to another using cuts and splices between sets of nodes. If the Hamilton Cycle is transformed such that it includes an optimal cost path under the connectivity constraint, then a safe plan is maintained with this qualified optimality.

*TODO: Make figures to visually show this*

*TODO: Use citations with respect to the complexity of these approaches, especially Hamilton Cycle planning.*

# Discrete Optimal Planning Algorithm A* Discussion

A* is an extremely popular which is guaranteed to be optimal if the heuristic used is admissible, and is a intuitive first consideration for this problem as it involves graph search from a start and goal vertices. In fact, many of the above (*TODO: Citations*) use A* as a first attempt or as a component of their broader algorithm. However, A* does not guarantee safety and frequently causes self-entrapment and a game over state. Applying A* from $H^k$ to $G^k$ does not guarantee weak $C^k_{free}$ connectivity; however it does return $\pi^{\star,[k,k+j]}$, the optimal path from $H^k$ to $G^k$ over $j$ steps. This result will be used in the next section.

  

*TODO: Make figure showing A\* trapping itself*

  

# Multi-Stage Closed-Loop Planner

In the previous sections, it is shown that a complete game $\Pi$ requires weak $C^k_{free}$ connectivity for all $k$ and that cyclic paths enforce strong $H^k$-$G^k$-reachability, yet a simple implementation using A* to plan from $H^k$ to $G^k$ is not sufficient to guarantee the weak connectivity constraint of $C^k_{free}$. However, the optimality of the paths generated from A* is a property which would be beneficial when $|C^k_{free}| \gg |S^k|$ particularly in the early and middle game. Another useful property of A* is that it provides an infeasibility result when it cannot find it's goal from the starting node given the obstacle space. Lastly, A* is computationally fast, especially when a feasible path exists.

  

This leads us neatly into the proposed implementation of this work, which has implementations to address each phase of the game, beginning, middle, and end. During the beginning, there is relatively little within the $C^k_{obs}$ set granting high, if not complete, connectivity of $C^k_{free}$. This motivates the heavy usage of the A* algorithm detailed in Stage #1 below to quickly find $\pi^\star$ and a candidate $\pi^\circ$. During the middle and end portions, it becomes important to create closed loop paths which has far greater coverage of $C^k_{free}$ to both block potentially inconvenient future goal positions and maintain safe traversal and expand stage #1 options by increasing the distance between the head and tail on the closed-loop path by *wasting time*, which is the motivation for the cheeky first sentence of the abstract. This motivates the greedy $\pi^\circ$ path changes in stage #2. However, pursuing these greedy changes inevitably leads to a path which has stable nodes within $C^k_{free}$ which are not able to be reached without an exact solution. This leads to the optimization step in optional stage #3 which attempts to convert the path into a Hamiltonian Cycle by searching for changes between the head and the tail. After such a cycle is implemented, the Snake will follow the path generated in stage #3 until the game is won. This is referred to as the **Multi-Stage Closed-Loop Planner** or the **planner** and contains two distinct modes when planning:

1. From $H^k$ to $G^k$, A* is used to plan a $\pi^{\star,[k, k+\alpha]}$, if one exists. The snake buffer $S^k$ is then projected forward using $\pi^{\star,[k, k+\alpha]}$ to a candidate snake of ${S'}^{k+\alpha}$. A* is again used to plan a $\pi^{\star, [k+\alpha, k+\alpha +\beta]}$ from ${H'}^{k +\alpha}$ to ${T'}^{k+\alpha}$, if one exists and this time considering the candidate snake as a static obstacle. If all of this is feasible, then it has quickly been determined that $\exists \pi^{\circ,[k, k+\alpha+\beta+|{S'}^{k+\alpha}|]} \Leftarrow \pi^{\star, [k+\alpha, k+\alpha +\beta]} \cup {S'}^{k+\alpha}$ which is a safe path to take. The plan $\pi^{\star, [k, k+\alpha]}$ can be executed without re-planning. If any of the above are infeasible, proceed to 2 immediately.

2. Adopt the previously determined safe path created in the previous planning iteration $\pi^{\circ,[k-\Delta, k]}$. Perform a walk along $\pi^{\circ,[k-\Delta, k]}$ starting from the snake head and identify any simple cost-increasing, $C^k_{free}$ connectivity increasing, goal capture, or general space filling changes available make to the path while maintaining a closed-loop path, implement the first valid favorable change in a greedy and adopt the new closed-loop plan $\pi^{\circ, k+}$ and try to re-plan with #1 again on the next game step. If no changes are found, lock the current path while still attempting #1 until #1 is successful, optionally go to #3 when no changes can be made using #2 when the Snake completes an entire loop of the path.

3. At this stage, a space filling path has been devised which covers most of the grid; however, since #2 is implemented in a greedy manner, there will likely be opportunities to increase the cost further using a Hamiltonian cycle. Once this stage is triggered, it is assumed that the snake is long enough that it is no longer valuable to pursue #1 or #2 and disrupt the near-Hamiltonian Cycle. For this construction, the segment of the closed loop between the head and current tail position is taken. If there are $C^k_{free}$ nodes adjacent to this path segment, a backtracking depth-first search is used to identify if a series of moves could be taken to either (i) increase the length of the path segment safely or (ii) increase the connectivity of $C^k_{free}$ safely without increasing the cost. If such a segment is found, it is adopted. #3 is repeated until the entire game board is filled and the snake will simply follow the cycle until game is won, assuming a cycle exists.

  

*Note: #1 has been implemented and is very promising, #2 completed and also promising but needs some further iterations, 3 is yet to be implemented*

  

*TODO: be more exact with #2*

  

*TODO: give more clarity on favorable proposed changes*


*TODO: add algorithm blocks*


This planner construction has some clear benefits; first, it takes full advantage of A* to provide optimal safe paths and infeasibility report if it cannot in the early and middle game. As $|S^k|$ grows, it naturally transitions into a space-filling closed-loop path planning. The first stage allows the efficient and fast accumulation of score, the second stage exerts a safe but time-wasting effort to wait until the goal can be safely achieved, and the third-stage attempts to win the game in a safe fashion using Hamiltonian Cycles.

# Conclusions

*Todo*