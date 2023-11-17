# importing libraries
import pygame
import time
import random
import copy
import numpy as np
import multiprocessing as mp

# Global values so I don't have to retype strings
EPS = 1e-3
RIGHT = 'RIGHT'
LEFT = 'LEFT'
UP = 'UP'
DOWN = 'DOWN'

DIRECTIONS = [LEFT, RIGHT, UP, DOWN]

ANTI_DIRECTIONS = {
    LEFT: RIGHT,
    RIGHT: LEFT,
    UP: DOWN,
    DOWN: UP
}

# Color
BLACK = pygame.Color(0, 0, 0)
GRAY = pygame.Color(255//2, 255//2, 255//2)
LGRAY = pygame.Color(3*255//4, 3*255//4, 3*255//4)
WHITE = pygame.Color(255, 255, 255)
RED = pygame.Color(255, 0, 0)
GREEN = pygame.Color(0, 255, 0)
GREENL = pygame.Color(0, 255//2, 0)
GREENL2 = pygame.Color(0, 255//4, 0)
BLUE = pygame.Color(0, 0, 255)
BLUEL = pygame.Color(0, 0, 255//2)
REDL = pygame.Color(255 // 2, 0, 0)

SCORE_COLOR = WHITE

GAMEOVER_COLOR = RED

BACKGROUND_COLOR = BLACK
# SNAKE_COLORS = [GREEN, GREENL, GREENL2]
SNAKE_COLOR = GREEN
GOAL_COLOR = RED
PATH_COLOR = WHITE

# Custom algorithm color
GOAL_QUEUE_COLOR = REDL
GOAL_EXPANDED_COLOR = pygame.Color(247, 0, 255, 0)

TAIL_QUEUE_COLOR = pygame.Color(169, 255, 221, 0)
TAIL_EXPANDED_COLOR = pygame.Color(0, 255, 154, 0)

HEAD_QUEUE_COLOR = pygame.Color(254, 185, 255)
HEAD_EXPANDED_COLOR = GRAY

VIRT_GOAL_COLOR = BLUEL

# Helper functions


def manhattan_distance(pos0, pos1):

    assert len(pos0) == len(
        pos1), f'Length of pos0 {len(pos0)} has to be equal to the length of pos1 {len(pos1)}'

    dist = 0

    for p0, p1 in zip(pos0, pos1):
        dist += abs(p0 - p1)

    return dist


def euclidean_distance_squared(pos0, pos1):

    assert len(pos0) == len(
        pos1), f'Length of pos0 {len(pos0)} has to be equal to the length of pos1 {len(pos1)}'

    dist = 0

    for p0, p1 in zip(pos0, pos1):
        dp = p0 - p1
        dist += dp * dp

    return dist  # Not worrying about sqrt because this is a 1-1 mapping for distance regardless


def get_direction_from_nodes(src, dst):
    dcol = dst[0] - src[0]
    drow = dst[1] - src[1]

    # Left
    if dcol == -1 and drow == 0:
        return LEFT

    # Right
    if dcol == 1 and drow == 0:
        return RIGHT

    # Up
    if dcol == 0 and drow == -1:
        return UP

    # Down
    if dcol == 0 and drow == 1:
        return DOWN

    raise ValueError(f'Could not decipher direction from {src} to {dst}')


def get_neighbor_at_direction(src, direction):

    assert direction in DIRECTIONS, f'Cannot parse {direction}'
    assert len(src) >= 2

    if direction == LEFT:
        return [src[0] - 1, src[1]]

    if direction == RIGHT:
        return [src[0] + 1, src[1]]

    if direction == UP:
        return [src[0], src[1] - 1]

    if direction == DOWN:
        return [src[0], src[1] + 1]


def get_neighbors(src, exclusions, grid_shape):
    # Expand the node
    candidate_surrounding = [
        get_neighbor_at_direction(src, LEFT),
        get_neighbor_at_direction(src, RIGHT),  # Right
        get_neighbor_at_direction(src, UP),  # Up
        get_neighbor_at_direction(src, DOWN)  # Down
    ]

    surrounding = []
    for n in candidate_surrounding:
        if not (n in exclusions or min(n) < 0 or n[0] >= grid_shape[0] or n[1] >= grid_shape[1]):
            surrounding.append(n)

    return surrounding


def is_adjacent(src, node, grid_shape):
    return node in get_neighbors(src, [], grid_shape)


def get_min_cost_idx(nodes, costs):
    minimum = np.inf
    minimum_idx = 0
    for idx, pos in zip(range(len(nodes)), nodes):
        val = costs[pos[0], pos[1]]
        if val < minimum:
            minimum_idx = idx
            minimum = val

    return minimum_idx

# Classes


class GameRenderer:
    def __init__(self, grid_shape: tuple, pixel_density: int, window_title: str = None, font_size: int = None):
        self.pixel_density = pixel_density
        assert self.pixel_density >= 10, f'Cannot have {self.pixel_density} < 10'
        self.grid_shape = grid_shape
        self.font_size = font_size if font_size is not None else 10

        title = window_title if window_title is not None else 'Snake Game'

        # Set up the game:
        self.window_x = self.grid_shape[0] * self.pixel_density
        self.window_y = self.grid_shape[1] * self.pixel_density

        pygame.display.set_caption(title)
        self.game_window = pygame.display.set_mode(
            (self.window_x, self.window_y))

        # Fonts
        self.SCORE_FONT = pygame.font.SysFont(
            'times new roman', self.font_size)
        self.GAMEOVER_FONT = pygame.font.SysFont(
            'times new roman', self.font_size)

    def blank(self, update: bool = None):
        self.game_window.fill(BACKGROUND_COLOR)

        if update is not None and update:
            self.update()

    def convert_to_pixel_space(self, coords: list, add: int = None):
        pixels = copy.deepcopy(coords)
        # for pix in pixels:
        #    pix *= self.pixel_density
        pixels[0] *= self.pixel_density
        pixels[1] *= self.pixel_density

        if add is not None:
            pixels[0] += add
            pixels[1] += add
        return pixels

    def draw_rectangle(self, coords: list, color: pygame.Color, update: bool = None, shrinkage: int = None):
        if shrinkage is None:
            shrinkage = 0
        pygame.draw.rect(self.game_window, color, pygame.Rect(
            *self.convert_to_pixel_space(coords, add=shrinkage), self.pixel_density-2*shrinkage, self.pixel_density-2*shrinkage))

        if update is not None and update:
            self.update()

    def fill_in_shrinkage(self, coords0: list, coords1: list, color: pygame.Color, shrinkage: int):
        pixels0 = self.convert_to_pixel_space(coords0)

        direction = get_direction_from_nodes(coords0, coords1)

        if direction == UP:
            pygame.draw.rect(self.game_window, color, pygame.Rect(
                pixels0[0] + shrinkage, pixels0[1]-shrinkage, self.pixel_density - 2*shrinkage, 2*shrinkage))

        elif direction == DOWN:
            pygame.draw.rect(self.game_window, color, pygame.Rect(
                pixels0[0] + shrinkage, pixels0[1]+self.pixel_density-shrinkage, self.pixel_density - 2 * shrinkage, 2*shrinkage))

        elif direction == RIGHT:
            pygame.draw.rect(self.game_window, color, pygame.Rect(
                pixels0[0] + self.pixel_density - shrinkage, pixels0[1] + shrinkage, 2*shrinkage, self.pixel_density - 2*shrinkage))

        else:
            pygame.draw.rect(self.game_window, color, pygame.Rect(
                pixels0[0] - shrinkage, pixels0[1]+shrinkage, 2*shrinkage, self.pixel_density-2*shrinkage))

    def draw_snake(self, snake, shrinkage: int = None):

        if shrinkage is None:
            shrinkage = 1
        for node in snake:
            self.draw_rectangle(node, SNAKE_COLOR, shrinkage=shrinkage)

        if len(snake) == 1:
            return

        for node, next_node in zip(snake[:-1], snake[1:]):
            self.fill_in_shrinkage(node, next_node, SNAKE_COLOR, shrinkage)

    def draw_path(self, path, color: pygame.Color = None, shrinkage: int = None, connect_loop: bool = None):

        if color is None:
            color = PATH_COLOR

        if shrinkage is None:
            shrinkage = 1

        if connect_loop is None:
            connect_loop = len(path) > 3 and is_adjacent(
                path[0], path[-1], self.grid_shape)

        for node in path:
            self.draw_rectangle(node, color, shrinkage=shrinkage)

        if len(path) == 1:
            return

        for node, next_node in zip(path[:-1], path[1:]):
            self.fill_in_shrinkage(node, next_node, color, shrinkage)

        if connect_loop:
            self.fill_in_shrinkage(path[0], path[-1], color, shrinkage)

    def draw_snake_head(self, snake, direction):
        head = snake[-1]
        pixels = self.convert_to_pixel_space(head)
        if direction == UP:
            pygame.draw.circle(self.game_window, BLACK, [
                               pixels[0] + self.pixel_density // 3, pixels[1] + self.pixel_density//3], self.pixel_density//10)
            pygame.draw.circle(self.game_window, BLACK, [
                               pixels[0] + 2 * self.pixel_density // 3, pixels[1] + self.pixel_density//3], self.pixel_density//10)
            pygame.draw.rect(self.game_window, BLACK, pygame.Rect(
                pixels[0]+self.pixel_density//6, pixels[1], 2*self.pixel_density//3, self.pixel_density//6))
        if direction == DOWN:
            pygame.draw.circle(self.game_window, BLACK, [
                               pixels[0] + self.pixel_density // 3, pixels[1] + 2*self.pixel_density//3], self.pixel_density//10)
            pygame.draw.circle(self.game_window, BLACK, [
                               pixels[0] + 2 * self.pixel_density // 3, pixels[1] + 2*self.pixel_density//3], self.pixel_density//10)
            pygame.draw.rect(self.game_window, BLACK, pygame.Rect(
                pixels[0]+self.pixel_density//6, pixels[1] + 5*self.pixel_density//6, 2*self.pixel_density//3, self.pixel_density//6))
        if direction == LEFT:
            pygame.draw.circle(self.game_window, BLACK, [
                               pixels[0] + self.pixel_density // 3, pixels[1] + self.pixel_density//3], self.pixel_density//10)
            pygame.draw.circle(self.game_window, BLACK, [
                               pixels[0] + self.pixel_density // 3, pixels[1] + 2*self.pixel_density//3], self.pixel_density//10)
            pygame.draw.rect(self.game_window, BLACK, pygame.Rect(
                pixels[0], pixels[1] + self.pixel_density//6, self.pixel_density//6, 2*self.pixel_density//3))
        if direction == RIGHT:
            pygame.draw.circle(self.game_window, BLACK, [
                               pixels[0] + 2*self.pixel_density // 3, pixels[1] + self.pixel_density//3], self.pixel_density//10)
            pygame.draw.circle(self.game_window, BLACK, [
                               pixels[0] + 2*self.pixel_density // 3, pixels[1] + 2*self.pixel_density//3], self.pixel_density//10)
            pygame.draw.rect(self.game_window, BLACK, pygame.Rect(
                pixels[0]+5*self.pixel_density//6, pixels[1] + self.pixel_density//6, self.pixel_density//6, 2*self.pixel_density//3))

    def show_score(self, score):
        score_surface = self.SCORE_FONT.render(
            f'Score : {score}', True, SCORE_COLOR)
        score_rect = score_surface.get_rect()
        self.game_window.blit(score_surface, score_rect)

    def game_over(self, score):
        go_surface = self.GAMEOVER_FONT.render(
            f'Game Over, Score: {score}', True, GAMEOVER_COLOR)
        go_rect = go_surface.get_rect()
        go_rect.midtop = (self.window_x // 2, self.window_y // 4)
        self.game_window.blit(go_surface, go_rect)
        pygame.display.flip()

    def update(self):
        pygame.display.update()

    def close(self):
        pygame.display.quit()


class KBController:
    """
    Passes keyboard inputs to snake game
    """

    def __init__(self):
        pass

    def get(self, *args, **kwargs):
        # Just return the keypress equivalent direction
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    return UP

                if event.key == pygame.K_DOWN:
                    return DOWN

                if event.key == pygame.K_LEFT:
                    return LEFT

                if event.key == pygame.K_RIGHT:
                    return RIGHT

    def get_future_path(self):
        return None


class NaiveController:
    """
    Very simple controller that tries to navigate towards the goal, but will turn if about to self collide if able. Does not look ahead at all
    """

    def __init__(self):
        pass

    def get(self, snake, direction, goal, *args):
        # Create direction candidate sets
        left_candidate = copy.deepcopy(snake[-1])
        right_candidate = copy.deepcopy(snake[-1])
        up_candidate = copy.deepcopy(snake[-1])
        down_candidate = copy.deepcopy(snake[-1])

        left_candidate[0] -= 1
        right_candidate[0] += 1
        up_candidate[1] -= 1
        down_candidate[1] += 1

        if goal[0] > snake[-1][0] and right_candidate not in snake and direction != LEFT:
            # To increase x is to go RIGHT
            return RIGHT

        if goal[0] < snake[-1][0] and left_candidate not in snake and direction != RIGHT:
            return LEFT

        if goal[1] < snake[-1][1] and up_candidate not in snake and direction != DOWN:
            return UP

        if goal[1] > snake[-1][1] and down_candidate not in snake and direction != UP:
            return DOWN

        if left_candidate in snake and right_candidate not in snake and direction != LEFT:
            # To increase x is to go RIGHT
            return RIGHT

        if right_candidate in snake and left_candidate not in snake and direction != RIGHT:
            return LEFT

        if down_candidate in snake and up_candidate not in snake and direction != DOWN:
            return UP

        if up_candidate in snake and down_candidate not in snake and direction != UP:
            return DOWN

        return None


class AStar:  # Converted to standalone, not a controller
    def __init__(self, heuristic, grid_shape):
        self.h = heuristic
        self.grid_shape = grid_shape

    def plan(self, start: list, goal: list, snake: list, static_obstacles: list = None):
        if static_obstacles is None:
            static_obstacles = []

        # We will be using a grid graph approach here
        costs = np.ones(self.grid_shape) * np.inf

        costs[start[0], start[1]] = 0.
        fscore = copy.deepcopy(costs)
        fscore[start[0], start[1]
               ] += self.h(start, goal)

        fromgraph = {}
        # Keys for from graph will defined as x,y = divmod(key, self.grid_shape[0])
        # Where key = x*self.grid_shape[0] + y

        queue = [start]

        node = None

        expanded = []

        # Since we only care about the immediate first step from the starting, we will return the best point from the starting
        # Assuming that the goal is in surrounding and immediately replaced
        reached_goal = False

        while len(queue) > 0:
            # Pick the node from the queue that has the minimum cost
            minimum_idx = get_min_cost_idx(queue, fscore)

            # Remove and return the node at the minimum fscore
            node = queue.pop(minimum_idx)
            expanded.append(node)

            if node == goal:
                reached_goal = True
                break

            # Get surrounding

            # Maximum snake index to bypass is equivalent to the minimum cost
            node_cost = int(costs[node[0], node[1]])

            leading_index = max(node_cost-1, 0)

            if leading_index < len(snake):
                snake_obstacles = snake[leading_index:]
            else:
                snake_obstacles = []

            surrounding = get_neighbors(
                node, snake_obstacles+static_obstacles, self.grid_shape)

            # Add the valid nodes into the queue

            cand_val = costs[node[0], node[1]] + 1
            for n in surrounding:
                # If valid append to queue, calculate cost and heuristics

                if cand_val < costs[n[0], n[1]]:

                    costs[n[0], n[1]] = cand_val
                    fromgraph[self.grid_shape[0]*n[0] + n[1]] = node
                    fscore[n[0], n[1]] = cand_val + self.h(n, goal)

                    if n not in queue:
                        queue.append(n)

        # Build a path using the cost
        current_path = [goal]

        while current_path[-1] != start and reached_goal:
            n = current_path[-1]
            current_path.append(
                fromgraph[n[0]*self.grid_shape[0] + n[1]]
            )

        current_path.reverse()

        # Remove the first index
        _ = current_path.pop(0)
        return reached_goal, current_path, queue, expanded, costs


class AStarPlanner:
    def __init__(self, planner):
        self.sub = planner
        self.current_path = []
        self.expanded = []
        self.queue = []
        self.costs = None

    def get_future_path(self):
        return self.current_path[1:]

    def draw(self, renderer, exclusions):
        for poses, color in zip(
            [self.queue, self.expanded, self.current_path],
            [HEAD_QUEUE_COLOR, HEAD_EXPANDED_COLOR, PATH_COLOR]
        ):
            for draw_pos in poses:
                if draw_pos not in exclusions:
                    renderer.draw_rectangle(draw_pos, color)

        renderer.update()

    def get(self, snake, direction, goal, renderer):

        # Exit early if we have a plan already
        if len(self.current_path) > 0:
            self.seq += 1
            self.draw(renderer, snake+[goal])
            next_pos = self.current_path.pop(0)

            return get_direction_from_nodes(snake[-1], next_pos)

        self.seq = 0

        start = snake[-1]

        # First, feasibility test from the head to the goal
        plan_valid, path, self.queue, self.expanded, self.costs = self.sub.plan(
            start, goal, snake)

        if plan_valid:
            self.draw(renderer, snake+[goal])

            self.current_path = copy.deepcopy(path)

            next_pos = self.current_path.pop(0)

            return get_direction_from_nodes(start, next_pos)
        print('Failed to find path!')
        return None


class TwoStagePlanner:
    def __init__(self, subplanner, grid_shape):
        self.sub = subplanner
        self.grid_shape = grid_shape

        # Current path: path starting from the adjacency of the head of the snake to the goal
        self.current_path = []

        # Tail path: path starting from the adjacency of the goal to the propagated tail of the snake
        self.tail_path = []

        # Closed-loop path: path union of current path tail path and body of the snake upon successful planning of first stage
        # To be altered in second stage to consume more cost
        self.closed_loop_path = []

        # Variable to track when replanning can be triggered or old plan can be used
        self.replan = True
        self.no_closed_loop_changes = False

        # Value to check if any cost increases need to occur
        self.max_length = self.grid_shape[0]*self.grid_shape[1]
        self.node_space = []
        for x in range(self.grid_shape[0]):
            for y in range(self.grid_shape[1]):
                self.node_space.append([x, y])

    def get_future_path(self):
        if len(self.current_path) > 0:
            return self.current_path[1:]
        else:
            return []

    def get(self, snake, direction, goal, renderer):
        head = snake[-1]
        tail = snake[0]

        # First check if we can reference the current plan
        if len(self.current_path) > 0 and not self.replan and self.current_path[0] not in snake:
            # print('Running old plan')
            renderer.draw_path(self.tail_path, color=LGRAY,
                               shrinkage=3, connect_loop=False)
            renderer.draw_path(self.current_path,
                               shrinkage=1, connect_loop=False)
            # renderer.update()
            next_node = self.current_path.pop(0)

            # Final node in current path is goal
            if len(self.current_path) == 0:
                self.replan = True

            return get_direction_from_nodes(head, next_node)

        self.replan = True

        print('Stage 1')

        # Stage 1: A* to goal from head, A* to projected tail
        valid, head2goal_path, head2goal_queue, head2goal_expanded, head2goal_costs = self.sub.plan(
            head, goal, snake)

        if valid:
            # Propagate snake forward past the goal
            propagated_snake = (snake + head2goal_path)[-(len(snake)+1):]
            propagated_head = propagated_snake[-1]
            propagated_tail = propagated_snake[0]
            valid, goal2tail_path, goal2tail_queue, goal2tail_expanded, goal2tail_costs = self.sub.plan(
                propagated_head, propagated_tail, [], propagated_snake[1:])

            # Valid double check
            valid = valid and len(head2goal_path) + len(goal2tail_path) > 1

        if valid:
            self.replan = False
            self.no_closed_loop_changes = False
            self.current_path = copy.deepcopy(head2goal_path)
            self.tail_path = copy.deepcopy(goal2tail_path)
            self.closed_loop_path = copy.deepcopy(
                self.tail_path) + propagated_snake[1:]

            # First entry in self.current_path is the next step
            next_pos = self.current_path.pop(0)

            # Render:
            renderer.draw_path(self.tail_path, color=LGRAY,
                               shrinkage=3, connect_loop=False)
            renderer.draw_path(self.current_path,
                               shrinkage=1, connect_loop=False)
            # renderer.update()
            print('Stage 1 Success')
            print('Closed Cycle plan result')
            print(self.closed_loop_path)

            print('Current path')
            print(self.current_path)

            print('Tail Path')
            print(self.tail_path)

            print('Propagated snake')
            print(propagated_snake[1:])
            return get_direction_from_nodes(head, next_pos)

        print('Stage 1 failed \nStage 2:')
        # If the above was successful, the methhod has return a direction for the next phase, else move on to stage 2:

        # Stage 2: Find cost increasing closed loop plan modifications to take and execute
        path_length = len(self.closed_loop_path)
        empty_nodes = [
            n for n in self.node_space if n not in self.closed_loop_path]

        head_idx = None
        for idx, node in zip(range(path_length), self.closed_loop_path):
            if node == head:
                head_idx = idx

        assert head_idx is not None, f'Could not find head node {head} in closed loop plan:\n{self.closed_loop_path}'

        # print('Checking if next pos is goal')
        # Simple check phase
        next_idx = (head_idx + 1) % path_length
        next_pos = self.closed_loop_path[next_idx]
        if next_pos == goal or self.no_closed_loop_changes:
            # print('Goal is next node, not changing anything')
            renderer.draw_path(self.closed_loop_path, GRAY, shrinkage=1)
            return get_direction_from_nodes(head, next_pos)

        # Create walk of indeces starting from the head and looping back around to the front
        indeces = [x for x in range(
            head_idx, path_length)] + [x for x in range(head_idx)]

        # Count margin that is available between the head and the tail on this current path
        # This is to keep the planner from adding a goal that would result in a game over
        margin = 0
        for idx in indeces[1:]:
            if self.closed_loop_path[idx] in snake:
                break
            margin += 1

        print('Margin: ', margin)

        # TODO: combine the two following items into a single loop that alternates, doing one loop for each is missing opportunities to increase cost early after changing the trajectory

        # Cost increase phase
        # Each planning cycle when this is active, attempt to increase the cost of the path by making simple changes
        if not valid and len(snake) <= path_length < self.max_length:
            # print('Checking for path length increases...')
            for i in indeces[2:-1]:
                # Find range respecting indeces for the next two nodes
                idx_a = i
                idx_b = (i+1) % path_length

                node_a = self.closed_loop_path[idx_a]
                node_b = self.closed_loop_path[idx_b]

                # Check neighbors of nodes_a and b to see if there's any pairs that are both adjacent and not in the path currently
                neighbors_a = get_neighbors(
                    node_a, snake+self.closed_loop_path, self.grid_shape)
                neighbors_b = get_neighbors(
                    node_b, snake+self.closed_loop_path + neighbors_a, self.grid_shape)

                # Find any adjacencies
                # TODO: add change to keep it from adding a goal if there's not enough room infront of the snake head
                adjacent_neighbor_sets = []
                for na in neighbors_a:
                    if na == goal and margin == 0:
                        continue
                    for nb in neighbors_b:
                        if nb == goal and margin == 0:
                            continue
                        if is_adjacent(na, nb, self.grid_shape):
                            adjacent_neighbor_sets.append((na, nb))

                if len(adjacent_neighbor_sets) > 0:
                    print(
                        f'Found candidate to increase path length [{node_a}, {node_b}] -> [{node_a}, {na}, {nb}, {node_b}]')
                    na, nb = adjacent_neighbor_sets[0]
                    # print(na, nb)
                    candidate_path = self.closed_loop_path[:idx_a+1]
                    candidate_path.append(na)
                    candidate_path.append(nb)
                    candidate_path += self.closed_loop_path[idx_b:]
                    valid = True
                    break

        # Discontinuity Elimination phase
        # Consider triples of points, if there is a discontinuity detected, see if it can be shifted closer to the goal.
        if not valid and len(snake) < path_length < self.max_length:
            # print('Checking for potential swaps...')
            for i in indeces[2:-2]:
                # Find range respecting indeces for the next 3 nodes
                idx_a = i
                idx_b = (i+1) % path_length
                idx_c = (i+2) % path_length

                node_a = self.closed_loop_path[idx_a]
                node_b = self.closed_loop_path[idx_b]
                node_c = self.closed_loop_path[idx_c]

                # Ignore this index if we are safe to capture a node
                if margin > 0 and goal == node_b:
                    continue

                neighbors_a = get_neighbors(
                    node_a, self.closed_loop_path+snake, self.grid_shape)

                neighbors_c = get_neighbors(
                    node_c, self.closed_loop_path+snake, self.grid_shape)

                shared = [
                    na for na in neighbors_a if na in neighbors_c and na != node_b]

                # Add goal to path if low on margin
                if goal in shared and margin > 0:
                    candidate_path = copy.deepcopy(self.closed_loop_path)
                    candidate_path[idx_b] = goal
                    valid = True
                    break

                # Try to remove a goal if no margin
                if margin == 0 and goal == node_b and len(shared) > 0:
                    candidate_path = copy.deepcopy(self.closed_loop_path)
                    candidate_path[idx_b] = shared[0]
                    valid = True
                    break

                # Try to combine dislocations or move dislocations closer to goal
                if len(shared) > 0:
                    # TODO: Clean this up
                    shared.append(node_b)
                    dists_squared = []
                    for node in shared:
                        dists_squared.append(
                            euclidean_distance_squared(node, goal))

                    farthest_idx = np.argmax(dists_squared)

                    if shared[farthest_idx] != node_b and dists_squared[farthest_idx] != dists_squared[-1]:
                        print(
                            f'Found replacement! {node_b} -> {shared[farthest_idx]}')
                        candidate_path = copy.deepcopy(self.closed_loop_path)
                        candidate_path[idx_b] = shared[farthest_idx]
                        valid = True
                        break

                    # TODO: clean the below up

                    # Look to see if we increase the connectivity of regions:
                    neighbors = get_neighbors(
                        shared[0], snake+self.closed_loop_path+[node_a, node_b, node_c], self.grid_shape)
                    neighbors_b = get_neighbors(
                        node_b, snake+self.closed_loop_path+[node_a, node_b, node_c, shared[0]], self.grid_shape)
                    empty_neighbors = [
                        n for n in neighbors if n not in empty_nodes]
                    empty_neighbors_b = [
                        n for n in neighbors if n not in empty_nodes]
                    if len(empty_neighbors) > len(empty_neighbors_b):
                        print(
                            f'Node to shift to make more empty {node_b} -> {shared[0]}')

                        candidate_path = copy.deepcopy(self.closed_loop_path)
                        candidate_path[idx_b] = shared[0]
                        valid = True
                        break

        # TODO: Add optimization pass where the path through the current head-reachable nodes is potentially optimized using backtracking to find a total plan that consumes more space

        if valid:
            self.closed_loop_path = copy.deepcopy(candidate_path)
            # print('Adopting path:')
            # print(self.closed_loop_path)

        else:
            print('No changes found!')
            self.no_closed_loop_changes = True
        path_length = len(self.closed_loop_path)

        renderer.draw_path(self.closed_loop_path, GRAY, shrinkage=1)
        # renderer.update()

        # Get next step and return
        for idx in range(path_length):
            if self.closed_loop_path[idx] == head:
                head_idx = idx
                break
        next_idx = (head_idx + 1) % path_length
        # print(head_idx, next_idx)
        # print('Head:')
        # print(head)
        next_pos = self.closed_loop_path[next_idx]
        # print('Next pos:')
        # print(next_pos)
        print(f'Running closed loop, plan size {path_length}')
        for idx in range(path_length):
            node_back = self.closed_loop_path[idx]
            node_forw = self.closed_loop_path[(idx+1) % path_length]
            if not is_adjacent(node_back, node_forw, self.grid_shape):
                print('Non adjacent nodes in path: ', node_back, node_forw)

        if head not in self.closed_loop_path:
            print('Head not in path!!!!')

        if next_pos in snake[1:]:
            print('Going to colide!')
        return get_direction_from_nodes(head, next_pos)


class SnakeGame:
    def __init__(self, grid_size: tuple, pixels_per_square: int, speed: int = None, window_title: str = None, font_size: int = None, controller=None, render_update=None, render_pre_blank=None):
        # Capture parameters
        self.grid_size = grid_size
        self.speed = speed
        self.controller = controller if controller is not None else KBController()
        self.fps = pygame.time.Clock()
        self.render_update = True if render_update is None else render_update
        self.render_pre_blank = True if render_pre_blank is None else render_pre_blank

        assert min(
            self.grid_size) > 1, f'Grid size must be above 1 in both dimensions! {self.grid_size}'

        self.renderer = GameRenderer(
            self.grid_size, pixels_per_square, window_title, font_size)

        # initialize the snake
        self.snake = [[  # Lowest index is the tail, highest index is the head
            random.randrange(1, self.grid_size[0]-1),
            random.randrange(1, self.grid_size[1]-1)
        ]]
        self.snake_head = copy.deepcopy(
            self.snake[-1])  # Copy the end of the snake

        self.snake_length = len(self.snake)

        # Initialize the snake direction
        self.snake_dir = random.choice(DIRECTIONS)
        self.next_snake_dir = self.snake_dir

        # Initialize Score
        self.score = 0
        self.win_score = self.grid_size[0] * self.grid_size[1] - 1

        # Initialize game sequence counter
        self.seq = 0
        self.moves_since_goal = 0
        self.not_gameover = False

        # initiailze the goal
        self.goal = copy.copy(self.snake_head)
        self.sample_goal()
        self.render()
        self.start_time = time.perf_counter()

    def sample_goal(self):
        goal_cand = copy.copy(self.snake_head)

        while self.score < self.win_score and goal_cand in self.snake or goal_cand == self.goal:
            goal_cand = [
                random.randrange(0, self.grid_size[0]),
                random.randrange(0, self.grid_size[1])
            ]

        self.goal = goal_cand

    def step(self):

        if self.render_pre_blank:
            self.renderer.blank()
        st = time.perf_counter()
        self.next_snake_dir = self.controller.get(
            self.snake, self.snake_dir, self.goal, self.renderer)

        self.render()
        #
        #
        # print(time.perf_counter() - st)

        if self.next_snake_dir is not None and (ANTI_DIRECTIONS[self.next_snake_dir] != self.snake_dir or self.snake_length <= 1):
            self.snake_dir = self.next_snake_dir

        if self.snake_dir != self.next_snake_dir:
            print(
                f'could not use requested direction {self.next_snake_dir}, used {self.snake_dir}')

        if self.snake_dir == UP:
            self.snake_head[1] -= 1
        elif self.snake_dir == DOWN:
            self.snake_head[1] += 1
        elif self.snake_dir == LEFT:
            self.snake_head[0] -= 1
        else:
            self.snake_head[0] += 1

        self.moves_since_goal += 1

        if self.snake_head == self.goal:
            self.score += 1
            self.snake_length += 1

            if self.score == self.win_score:
                print('You won')
                return
            self.sample_goal()
            go_cond = False
            self.moves_since_goal = 0
            print('SCORED, score: ', self.score)
            # print('Score per step: ', self.score / self.seq)
            # print('Avg Step per Sec ', self.seq /
            #       (time.perf_counter() - self.start_time))
        else:
            self.snake = self.snake[1:]
            # Game Over conditions
            cycle_detected = self.moves_since_goal > 10*self.win_score

            go_cond = min(self.snake_head) < 0 or self.snake_head[0] >= self.grid_size[
                0] or self.snake_head[1] >= self.grid_size[1] or self.snake_head in self.snake or cycle_detected

            if cycle_detected:
                print('Ending game since a cycle is detected')

        # Add the head to the snake
        self.snake.append(copy.deepcopy(self.snake_head))

        # Increment sequence
        self.seq += 1

        if go_cond:
            if self.snake_head in self.snake:
                for idx, node in zip(range(self.snake_length), self.snake):
                    if node == self.snake_head:
                        print('Intersected at idx: ', idx)
                        if isinstance(self.controller, AStarPlanner):
                            print('cost at point: ',
                                  self.controller.costs[self.snake[idx][0], self.snake[idx][1]])
                            print('Sequence number: ', self.controller.seq)
                        break
            self.game_over()

        # time.sleep(1)

    def game_over(self):
        self.not_gameover = False
        self.render()
        self.renderer.game_over(self.score)
        self.renderer.update()
        print(self.snake_dir)
        print(self.snake_head)
        print(self.seq)
        print(time.perf_counter() - self.start_time)
        if self.seq != 0:
            print((time.perf_counter()-self.start_time)/self.seq)
        _ = input('Enter to exit')

    def render(self):
        # Color everything in for the snake
        if not self.render_pre_blank:
            self.renderer.blank()
        self.renderer.draw_rectangle(self.goal, GOAL_COLOR)

        self.renderer.draw_snake(self.snake, shrinkage=2)
        self.renderer.draw_snake_head(self.snake, self.snake_dir)
        if self.score > 0 and self.seq > 0:
            self.renderer.show_score(self.score)

        if self.render_update:
            self.renderer.update()

    def loop(self):
        self.not_gameover = True
        try:
            while self.not_gameover and self.score < self.win_score:
                self.step()
                if self.speed is not None:
                    self.fps.tick(self.speed)

        except Exception as e:
            print(e)
            self.game_over()

        if self.score == self.win_score:
            print('You won snake!')


if __name__ == '__main__':
    # TODO: Accept arguments of the below options
    pygame.init()
    grid_shape = (30, 30)
    render_size = 25
    # rate = 200
    # grid_shape = (100, 4)
    # render_size = 50
    # rate = 10
    # grid_shape = (250, 100)
    # render_size = 10
    rate = 200

    subplanner = AStar(euclidean_distance_squared, grid_shape)
    planner = TwoStagePlanner(subplanner, grid_shape)
    # planner = AStarPlanner(subplanner)
    # planner = KBController()

    g = SnakeGame(grid_shape, render_size, rate, 'Snek', 24,
                  planner, render_update=True, render_pre_blank=True)
    g.loop()
    print(g.score)

    pygame.quit()
