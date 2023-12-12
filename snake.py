# NOTE to READER: the implementation is in desperate need of a refactor, if you want to change settings for running,
# flick your scroll wheel as fast as possible to the bottom.

# importing libraries
import pygame
import time
import random
import copy
import numpy as np
import multiprocessing as mp
import pathlib

# Global values so I don't have to retype strings
EPS = 1e-3
RIGHT = 'RIGHT'
LEFT = 'LEFT'
UP = 'UP'
DOWN = 'DOWN'

DIRECTIONS = [LEFT, RIGHT, UP, DOWN]

DIRECTION_INDECES = {direction: idx for idx,
                     direction in enumerate(DIRECTIONS)}

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

    dist_sq = 0.

    for p0, p1 in zip(pos0, pos1):
        dp = p0 - p1
        dist_sq += dp * dp

    return dist_sq  # Not worrying about sqrt because this is a 1-1 mapping for distance regardless


def get_direction_from_nodes(src, dst):
    assert src != dst, f'Cannot have {src=} be the same as {dst=} for direction acertation.'

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
        get_neighbor_at_direction(src, RIGHT),
        get_neighbor_at_direction(src, UP),
        get_neighbor_at_direction(src, DOWN)
    ]

    surrounding = []
    for n in candidate_surrounding:
        if not (n in exclusions or min(n) < 0 or n[0] >= grid_shape[0] or n[1] >= grid_shape[1]):
            surrounding.append(n)

    return surrounding


def is_adjacent(src, node, grid_shape):
    return node in get_neighbors(src, [], grid_shape)


def get_min_cost_idx(nodes: list, costs: np.array):
    minimum = np.inf
    minimum_idx = 0
    for idx, pos in enumerate(nodes):
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

        square_side_length = self.pixel_density-2*shrinkage

        pygame.draw.rect(self.game_window, color, pygame.Rect(
            *self.convert_to_pixel_space(coords, add=shrinkage), square_side_length, square_side_length))

        if update is not None and update:
            self.update()

    def fill_in_shrinkage(self, coords0: list, coords1: list, color: pygame.Color, shrinkage: int):
        pixels0 = self.convert_to_pixel_space(coords0)

        direction = get_direction_from_nodes(coords0, coords1)

        square_side_length = self.pixel_density - 2*shrinkage
        gap_length = 2*shrinkage

        if direction == UP:
            pygame.draw.rect(self.game_window, color, pygame.Rect(
                pixels0[0] + shrinkage, pixels0[1]-shrinkage, square_side_length, gap_length))

        elif direction == DOWN:
            pygame.draw.rect(self.game_window, color, pygame.Rect(
                pixels0[0] + shrinkage, pixels0[1]+self.pixel_density-shrinkage, square_side_length, gap_length))

        elif direction == RIGHT:
            pygame.draw.rect(self.game_window, color, pygame.Rect(
                pixels0[0] + self.pixel_density - shrinkage, pixels0[1] + shrinkage, gap_length, square_side_length))

        else:
            pygame.draw.rect(self.game_window, color, pygame.Rect(
                pixels0[0] - shrinkage, pixels0[1]+shrinkage, gap_length, square_side_length))

    def draw_snake(self, snake, shrinkage: int = None):

        if shrinkage is None:
            shrinkage = 1

        for node in snake:
            self.draw_rectangle(node, SNAKE_COLOR, shrinkage=shrinkage)

        if len(snake) == 1:
            # Do not try to do more rendering if the snake is only 1 unit long
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

        # There are many offsets in the following, the goal is to simply draw the eyes and mouth using circles and rectangles using
        # simple geometries
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

    def save(self, filename):
        pygame.image.save(self.game_window, filename)

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


class PriorityQueue:
    def __init__(self):
        # This will hold values in the form (value, cost) as a tuple where the minimum cost is at idx=0 and the maximum cost is at the end
        self.queue = []

    def add(self, value, cost):
        for idx, loc in enumerate(self.queue):
            if cost < loc[1]:
                self.queue.insert(idx, (value, cost))
                return

        # Cost exceeds anything else, append
        self.queue.append((value, cost))

    def remove(self, value):
        # Find index of value
        for idx, loc in enumerate(self.queue):
            if loc[0] == value:
                self.pop(idx)
                return

    def reorder(self, value, cost):
        assert value in self, f'{value=} is not in queue at all, cannot reorder'
        self.remove(value)
        self.add(value, cost)

    def pop(self, idx: int = None):
        if idx is None:
            idx = 0

        return self.queue.pop(idx)

    def __len__(self):
        return len(self.queue)

    def __bool__(self):
        return len(self.queue) > 0

    def __iter__(self):
        return [loc[0] for loc in self.queue]

    def to_list(self):
        return self.__iter__()

    def __contains__(self, value):
        return value in self.__iter__()


class AStar:  # Converted to standalone, not a controller
    def __init__(self, heuristic_function, grid_shape):
        self.h = heuristic_function
        self.grid_shape = grid_shape

    def plan(self, start: list, goal: list, snake: list, static_obstacles: list = None):
        if static_obstacles is None:
            static_obstacles = []

        # We will be using a grid graph with costs stored in an array
        costs = np.ones(self.grid_shape) * np.inf

        costs[start[0], start[1]] = 0.
        fscore = copy.deepcopy(costs)
        fscore[start[0], start[1]
               ] += self.h(start, goal)

        fromgraph = {}
        # Keys for from graph will defined as x,y = divmod(key, self.grid_shape[0])
        # Where key = x*self.grid_shape[0] + y

        queue = PriorityQueue()
        queue.add(start, fscore[start[0], start[1]])

        node = None

        expanded = []

        # Since we only care about the immediate first step from the starting, we will return the best point from the starting
        # Assuming that the goal is in surrounding and immediately replaced
        reached_goal = False

        while queue:
            # Remove and return the node at the minimum fscore
            node, node_fcost = queue.pop()
            expanded.append(node)

            if node == goal:
                reached_goal = True
                break

            # Get surrounding

            # Maximum snake index to bypass is equivalent to the minimum cost
            node_cost = int(costs[node[0], node[1]])

            leading_index = max(node_cost-3, 0)

            if leading_index < len(snake):
                snake_obstacles = snake[leading_index:]
            else:
                snake_obstacles = []

            surrounding = get_neighbors(
                node, snake_obstacles+static_obstacles, self.grid_shape)

            # Add the valid nodes into the queue

            cand_val = costs[node[0], node[1]] + 1

            # for idx, snake_node in enumerate(snake):
            #     if snake_node in surrounding:
            #         print(
            #             f'Snake node {snake_node} found in surrounding at {idx=} where theres a {cand_val=}')
            for n in surrounding:
                # If valid append to queue, calculate cost and heuristic_functions

                if cand_val < costs[n[0], n[1]]:

                    costs[n[0], n[1]] = cand_val
                    fromgraph[self.grid_shape[0]*n[0] + n[1]] = node
                    fscore[n[0], n[1]] = cand_val + self.h(n, goal)

                    if n not in queue:
                        queue.add(n, fscore[n[0], n[1]])

                    else:
                        queue.reorder(n, fscore[n[0], n[1]])

        if not reached_goal:
            # Do not return a bad path if the goal was not reached
            return reached_goal, [], queue.to_list(), expanded, costs

        # Build a path using the cost
        current_path = [goal]

        while current_path[-1] != start and reached_goal:
            n = current_path[-1]
            current_path.append(
                fromgraph[n[0]*self.grid_shape[0] + n[1]]
            )

        current_path.reverse()

        # check path
        for idx, node in enumerate(current_path[1:]):
            if node in snake[idx+1:]:
                # A node is in a conflicting part of the snake, unsure what went wrong but fail out and replan later
                return False, [], queue.to_list(), expanded, costs

        return reached_goal, current_path[1:], queue.to_list(), expanded, costs


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


class MultiStagePlanner:
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
        self.final_stage_lock = False
        self.final_path_lock = False
        self.closed_loop_counter = 0
        self.closed_loop_length = np.inf

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

        # Check if we must replan
        # if not self.replan and self.current_path and self.current_path[0] in snake[1:]:
        #    print('Issue with current plan, aborting')
        #    self.replan = True
        #    print(self.current_path)

        # First check if we can reference the current plan
        # and self.current_path[0] not in snake:
        if not self.final_stage_lock and self.current_path and not self.replan:
            # print('Running old plan')
            renderer.draw_path(self.tail_path, color=LGRAY,
                               shrinkage=3, connect_loop=False)
            renderer.draw_path(self.current_path,
                               shrinkage=1, connect_loop=False)
            # renderer.update()
            next_node = self.current_path.pop(0)

            if next_node == goal:
                self.replan = True

            return get_direction_from_nodes(head, next_node)

        if not self.final_stage_lock:

            # If get to this point, force replan for stage 1
            self.replan = True

            print('Stage 1')

            # Stage 1: A* to goal from head, A* to projected tail
        valid = False

        if not self.final_stage_lock:
            valid, head2goal_path, head2goal_queue, head2goal_expanded, head2goal_costs = self.sub.plan(
                head, goal, snake)

        if not self.final_stage_lock and valid:
            # Propagate snake forward past the goal
            propagated_snake = (snake + head2goal_path)[-(len(snake)+1):]
            propagated_head = propagated_snake[-1]
            propagated_tail = propagated_snake[0]
            valid, goal2tail_path, goal2tail_queue, goal2tail_expanded, goal2tail_costs = self.sub.plan(
                propagated_head, propagated_tail, propagated_snake, propagated_snake[1:])

            # Valid double check, check for margin of atleast 2 (goal2 tail path = head so len(goal2ttail_path)-1 = margin) after valid path is found
            valid = valid and (len(goal2tail_path) > 1 or len(snake) == 1)
            # Need at least 2 margin after planning as the snake's length will increase once by eating, then it is possible that the goal will be in the next location on the path and the snake will eat it's own tail

        if not self.final_stage_lock and valid:
            self.replan = False
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

        if not self.final_stage_lock:
            print('Stage 1 failed \nStage 2:')
        # If the above was successful, the method has return a direction for the next phase, else move on to stage 2:

        # Stage 2: Find cost increasing closed loop plan modifications to take and execute
        path_length = len(self.closed_loop_path)

        head_idx = None
        for idx, node in zip(range(path_length), self.closed_loop_path):
            if node == head:
                head_idx = idx

        assert head_idx is not None, f'Could not find head node {head} in closed loop plan:\n{self.closed_loop_path}'

        # Simple check phase
        next_idx = (head_idx + 1) % path_length
        next_pos = self.closed_loop_path[next_idx]

        if self.final_path_lock:
            renderer.draw_path(self.closed_loop_path, GRAY, shrinkage=1)
            return get_direction_from_nodes(head, next_pos)

        if not self.final_stage_lock and next_pos == goal:
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

        # Cost increase phase + Dislocation moving phase
        # Each planning cycle when this is active, attempt to increase the cost of the path by making simple changes or increase the connectivity of the dislocations
        if not self.final_stage_lock and not valid and len(indeces) > 10:
            for i in indeces[2:-2]:
                idx_a = i
                idx_b = (i+1) % path_length
                idx_c = (i+2) % path_length

                node_a = self.closed_loop_path[idx_a]
                node_b = self.closed_loop_path[idx_b]
                node_c = self.closed_loop_path[idx_c]

                # Get options for discontinuity increases
                neighbors_a = get_neighbors(
                    node_a, snake+self.closed_loop_path, self.grid_shape)

                neighbors_c = get_neighbors(
                    node_c, snake+self.closed_loop_path, self.grid_shape)

                ac_shared_nodes = [
                    node for node in neighbors_c if node in neighbors_a and node != node_b]

                if ac_shared_nodes:
                    # There should only be one node in this set
                    node = ac_shared_nodes[0]

                    dist_condition = euclidean_distance_squared(
                        goal, node) > euclidean_distance_squared(goal, node_b)
                    goal_accept_margin_condition = node == goal and margin > 0
                    goal_reject_margin_condition = node_b == goal and margin == 0

                    if (dist_condition and not goal_reject_margin_condition) or goal_accept_margin_condition or goal_reject_margin_condition:
                        # We have an option to change things
                        candidate_path = copy.deepcopy(self.closed_loop_path)
                        candidate_path[idx_b] = ac_shared_nodes[0]
                        valid = True

                        print(
                            f'Swapping node {node} in for {node_b} because {dist_condition=} or {goal_accept_margin_condition=} or {goal_reject_margin_condition=}')
                        break

                # check for a cost increasing opportunity
                neighbors_b = get_neighbors(
                    node_b, snake+self.closed_loop_path+neighbors_a, self.grid_shape)

                # Get options for cost increase
                adjoint_neighbor_pairs = []
                for node_a_neighbor in neighbors_a:
                    for node_b_neighbor in [n for n in neighbors_b if is_adjacent(node_a_neighbor, n, self.grid_shape)]:
                        adjoint_neighbor_pairs.append(
                            (node_a_neighbor, node_b_neighbor))

                if adjoint_neighbor_pairs:
                    # We have more than 0 adjoint neighbors, choose one and append
                    node_a_prime, node_b_prime = random.choice(
                        adjoint_neighbor_pairs)
                    candidate_path = copy.deepcopy(self.closed_loop_path)

                    candidate_path = candidate_path[:idx_a+1] + [
                        node_a_prime, node_b_prime] + candidate_path[idx_b:]
                    valid = True

                    print(
                        f'Found opportunity to increase cost, changing segment [{node_a},{node_b}] -> [{node_a}, {node_a_prime}, {node_b_prime}, {node_b}]')
                    break

        # TODO: Add optimization pass where the path through the current head-reachable nodes is potentially optimized using backtracking to find a total plan that consumes more space
        if not self.final_path_lock and (not valid and 0 <= margin <= 10 and self.closed_loop_counter > self.closed_loop_length or margin == 0 and goal in [self.closed_loop_path[idx] for idx in indeces[1: margin+2]]):
            print('Stage 2 failed \nLock in stage 3')
            # This is the final step to consider avenues for further optimizations
            # Using the margin previously calculated, a segment of the closed loop path will be taken

            self.final_stage_lock = True

            # First get the segment of the path in the closed path
            # Reorder the closed path
            reordered_closed_loop_path = [
                self.closed_loop_path[idx] for idx in indeces]

            path_segment = reordered_closed_loop_path[1:margin+2]

            path_segment_empties = []
            for node in path_segment:
                for empty_node in get_neighbors(node, self.closed_loop_path + snake, self.grid_shape):
                    path_segment_empties.append(empty_node)

            if not path_segment_empties:
                return get_direction_from_nodes(head, next_pos)

            cand_segment = []

            # using backtracking, build a series of moves which incrementally builds a new path segment that does not touch the snake,
            # if a candidate is found that is higher cost than the original (which is equal to the margin), adopt it as the new path segment

            # We want to prefer continuing in the same direction
            directions_stack = []
            # Initialize with the head's previous direction as reference
            potential_directions = []
            for d in DIRECTIONS:
                if d == ANTI_DIRECTIONS[direction]:
                    continue

                dnode = get_neighbor_at_direction(head, d)
                if min(dnode) < 0 or dnode[0] >= self.grid_shape[0] or dnode[1] >= self.grid_shape[1] or dnode in snake + cand_segment:
                    continue

                if d == direction and potential_directions:
                    potential_directions.insert(0, d)
                    continue

                potential_directions.append(d)

            if potential_directions:
                directions_stack.append(potential_directions)

            # Use the direction the current head index is using. This is treated as a stack
            while directions_stack:
                print(len(directions_stack))

                # Rebuild the path every time
                cand_segment = [get_neighbor_at_direction(
                    head, directions_stack[0][0])]

                for dirs in directions_stack[1:]:
                    cand_segment.append(get_neighbor_at_direction(
                        cand_segment[-1], dirs[0]))

                print(directions_stack)
                print(cand_segment)

                if is_adjacent(cand_segment[-1], tail, self.grid_shape) and len(cand_segment) > len(path_segment):
                    valid = True
                    break

                # Attempt to expand the current selection
                potential_directions = []
                for d in DIRECTIONS:
                    if d == ANTI_DIRECTIONS[directions_stack[-1][0]]:
                        continue

                    dnode = get_neighbor_at_direction(cand_segment[-1], d)

                    if min(dnode) < 0 or dnode[0] >= self.grid_shape[0] or dnode[1] >= self.grid_shape[1] or dnode in snake + cand_segment:
                        continue

                    if d == directions_stack[-1][0] and potential_directions:
                        potential_directions.insert(0, d)
                        continue

                    potential_directions.append(d)

                bad_path = len(cand_segment) > 0 and len(potential_directions) > 0 and is_adjacent(get_neighbor_at_direction(
                    cand_segment[-1], potential_directions[0]), tail, self.grid_shape) and len(cand_segment) + 1 <= len(path_segment) and goal in path_segment and goal not in cand_segment and margin > 1

                if bad_path:
                    print('Bad path')

                if potential_directions and not bad_path:
                    directions_stack.append(potential_directions)
                    continue

                # Failed to find any options, backtrack
                # Delete the previous selection
                directions_stack[-1].pop(0)

                # If there's more than one on the previous one, continue to the next loop iteration
                if directions_stack and len(directions_stack[-1]) > 1:
                    continue

                # If there's a single element in the iterations that got us here, delete them until we find an iteration we can use
                while directions_stack and len(directions_stack[-1]) <= 1:
                    directions_stack.pop(-1)

                # Pop
                if directions_stack:
                    directions_stack[-1].pop(0)

            if valid:
                # Insert the values into the new ordered path
                print('Found good alternative!:')
                # for idx, node in enumerate(cand_segment):
                #     # Plus one on the index because head_idx is at 0 and the segment starts at 1
                #     if idx+1 < margin + 2:
                #         reordered_closed_loop_path[idx+1] = node

                #     else:
                #         reordered_closed_loop_path.insert(idx+1, node)

                candidate_path = copy.deepcopy(reordered_closed_loop_path)

                candidate_path = candidate_path[:1] + \
                    cand_segment + candidate_path[1+len(cand_segment):]
                print(candidate_path)

                if len(candidate_path) == self.grid_shape[0] * self.grid_shape[1]:
                    self.final_path_lock = True
                    print('Path fills entire space now, lock in path')

        if valid:
            self.closed_loop_path = copy.deepcopy(candidate_path)
            self.closed_loop_length = len(self.closed_loop_path)
            self.closed_loop_counter = 0
            print('Adopting path:')
            print(self.closed_loop_path)

        else:
            self.closed_loop_counter += 1
            print(
                f'No changes found! Closed loop step counter {self.closed_loop_counter} / {self.closed_loop_length}')

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
    def __init__(self, grid_size: tuple, pixels_per_square: int, speed: int = None, window_title: str = None, font_size: int = None, controller=None, render_update=None, render_pre_blank=None, render_save_dir=None):
        # Capture parameters
        self.grid_size = grid_size
        self.speed = speed
        self.controller = controller if controller is not None else KBController()
        self.fps = pygame.time.Clock()
        self.render_update = True if render_update is None else render_update
        self.render_pre_blank = True if render_pre_blank is None else render_pre_blank
        self.render_save_dir = render_save_dir

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
        print(f'Starting Step: {self.seq}')

        if self.render_pre_blank:
            self.renderer.blank()

        st = time.perf_counter()

        self.next_snake_dir = self.controller.get(
            self.snake, self.snake_dir, self.goal, self.renderer)

        self.render()

        if self.next_snake_dir is not None and (ANTI_DIRECTIONS[self.next_snake_dir] != self.snake_dir or self.snake_length <= 1):
            self.snake_dir = self.next_snake_dir

        if self.snake_dir != self.next_snake_dir:
            print(
                f'could not use requested direction {self.next_snake_dir}, used {self.snake_dir}')

        # propagate snake
        if self.snake_dir == UP:
            self.snake_head[1] -= 1
        elif self.snake_dir == DOWN:
            self.snake_head[1] += 1
        elif self.snake_dir == LEFT:
            self.snake_head[0] -= 1
        else:
            self.snake_head[0] += 1

        print(f'Head pos: {self.snake_head}, using direction {self.snake_dir}')

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

        else:
            self.snake = self.snake[1:]
            # Game Over conditions
            cycle_detected = self.moves_since_goal > 20*self.win_score

            go_cond = min(self.snake_head) < 0 or self.snake_head[0] >= self.grid_size[
                0] or self.snake_head[1] >= self.grid_size[1] or self.snake_head in self.snake or cycle_detected

            if cycle_detected:
                print('Ending game since a cycle is detected')

        # Add the head to the snake
        # needs deep copy as updating the snake head will spread to all of the items in the snake otherwise
        self.snake.append(copy.deepcopy(self.snake_head))

        # Increment sequence
        print(f'Ending Step: {self.seq}')
        self.seq += 1

        if go_cond:
            # Find intersection idx and associated cost
            if self.snake_head in self.snake:
                for idx, node in enumerate(self.snake):
                    if node == self.snake_head:
                        print('Intersected at idx: ', idx)
                        if isinstance(self.controller, AStarPlanner):
                            print('cost at point: ',
                                  self.controller.costs[self.snake[idx][0], self.snake[idx][1]])
                            print('Sequence number: ', self.controller.seq)
                        break

            self.game_over()

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
        if not self.render_pre_blank:
            self.renderer.blank()
        self.renderer.draw_rectangle(self.goal, GOAL_COLOR)

        self.renderer.draw_snake(self.snake, shrinkage=2)
        self.renderer.draw_snake_head(self.snake, self.snake_dir)
        if self.score > 0 and self.seq > 0:
            self.renderer.show_score(self.score)

        if self.render_update:
            self.renderer.update()

        if self.render_save_dir is not None:
            filename = self.render_save_dir / \
                f'move_{time.perf_counter_ns()}_{self.seq}.png'
            print(f'Saving {filename}')
            self.renderer.save(str(filename))

    def loop(self):
        self.not_gameover = True
        # try:
        while self.not_gameover and self.score < self.win_score:
            self.step()
            if self.speed is not None:
                self.fps.tick(self.speed)

        # except Exception as e:
        #    print(e)
        #    self.game_over()

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
    planner = MultiStagePlanner(subplanner, grid_shape)
    # planner = AStarPlanner(subplanner)
    # planner = KBController()

    g = SnakeGame(grid_shape, render_size, rate, 'Snek', 24,
                  planner, render_update=True, render_pre_blank=True,
                  # render_save_dir=pathlib.Path('images/')
                  )
    g.loop()
    print(g.score)

    pygame.quit()
