import numpy as np
import heapq

def heuristic(a, b):
    return np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)


def array_pathfinding(array, start, goal, diagonal=False):
    """Path Finding Function
        input must be array of type np.array of 0's and 1's followed by tuples for start and goal in the array
        0 represents a walkable field
        1 represents a blocked field
        return an array with steps of 1 from start to goal
        the starting position will bot be included hence the first item in the list will be the first checkpoint
    """
    array = np.swapaxes(array, 0, 1)
    if diagonal:
        neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    else:
        neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    close_set = set()
    came_from = {}
    gscore = {start: 0}
    fscore = {start: heuristic(start, goal)}
    oheap = []
    heapq.heappush(oheap, (fscore[start], start))
    while oheap:
        current = heapq.heappop(oheap)[1]
        if current[0] == goal[0] and current[1] == goal[1]:
            data = []
            while current in came_from:
                data.append(current)
                current = came_from[current]
            return list(reversed(data))
        close_set.add(current)

        for i, j in neighbors:
            neighbor = current[0] + i, current[1] + j
            tentative_g_score = gscore[current] + heuristic(current, neighbor)
            if 0 <= neighbor[0] < array.shape[0]:
                if 0 <= neighbor[1] < array.shape[1]:
                    if array[neighbor[0]][neighbor[1]] == 1:
                        continue
                else:
                    # array bound y walls
                    continue
            else:
                # array bound x walls
                continue
            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                continue
            if tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1] for i in oheap]:
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(oheap, (fscore[neighbor], neighbor))


def to_features(self_data, target_pos, field):
    if target_pos is None:
        target_pos = [-1, -1]
    return np.concatenate([[self_data[3][0], self_data[3][1], self_data[2]], target_pos, field.reshape(-1)])

def generate_field(game_state):
    field = np.copy(game_state['field'])
    field[field == -1] = -10
    for coin_position in game_state['coins']:
        field[coin_position[0], coin_position[1]] = 2
    for bomb in game_state['bombs']:
        bomb_position = bomb[0]
        field[bomb_position[0], bomb_position[1]] = -5 + bomb[1]
    field[game_state['explosion_map'] == 1] = -10

    # Cuts the outer edges.
    field = field[1:field.shape[0]-1, 1:field.shape[0]-1]
    return field


def move_towards(self_position, goal_position, field):
    field[field!=0] = 1
    path = array_pathfinding(np.swapaxes(field, 0, 1), self_position, goal_position)
    if path is None or len(path) == 0:
        return None, True

    dist = [path[0][0] - self_position[0], path[0][1] - self_position[1]]
    action = None
    if dist[0] == 0:
        if dist[1] == 1:
            action = "DOWN"
        elif dist[1] == -1:
            action = "UP"
        else:
            raise ArithmeticError
    elif dist[1] == 0:
        if dist[0] == 1:
            action = "RIGHT"
        elif dist[0] == -1:
            action = "LEFT"
        else:
            raise ArithmeticError
    return action, len(path) == 1

def seek_coin(model, game_state, already_chosen_position=None):
    self_data = game_state['self']
    self_pos = self_data[3]
    field = generate_field(game_state)
    max_q_s = -100000
    selected_coin_pos = already_chosen_position
    if not selected_coin_pos:
        for coin_pos in game_state['coins']:
            obstacles_field = get_obstacles_field(game_state)
            if move_towards(self_pos, coin_pos, obstacles_field)[0] is not None:
                pos_in_field_without_corners = [coin_pos[0] - 1, coin_pos[1] - 1]
                features = to_features(self_data, pos_in_field_without_corners, field)
                q_s = model.predict(features.reshape(1, -1))
                if max_q_s < q_s:
                    max_q_s = q_s
                    selected_coin_pos = coin_pos
    if selected_coin_pos is None:
        return None, 0, True, None

    obstacles_field = get_obstacles_field(game_state)
    action, is_done = move_towards(self_pos, selected_coin_pos, obstacles_field)
    return action, max_q_s, is_done, selected_coin_pos


def destroy_crate(model, game_state, already_chosen_crate_pos=None):
    crate_positions = np.argwhere(game_state['field'] == 1)
    self_data = game_state['self']
    self_pos = self_data[3]
    field = generate_field(game_state)
    max_q_s = -100000
    selected_crate_pos = already_chosen_crate_pos
    if selected_crate_pos is None:
        for crate_pos in crate_positions:
            obstacles_field = get_obstacles_field(game_state)
            obstacles_field[crate_pos[0], crate_pos[1]] = 0
            if move_towards(self_pos, crate_pos, obstacles_field)[0] is not None:
                pos_in_field_without_corners = [crate_pos[0]-1, crate_pos[1]-1]
                features = to_features(self_data, pos_in_field_without_corners, field)
                q_s = model.predict(features.reshape(1, -1))
                if max_q_s < q_s:
                    max_q_s = q_s
                    selected_crate_pos = crate_pos
    if selected_crate_pos is None:
        return None, 0, True, None

    dist = (selected_crate_pos[0] - self_pos[0], selected_crate_pos[1] - self_pos[1])
    if dist[0] == 0 and abs(dist[1]) < 3 or dist[1] == 0 and abs(dist[0]) < 3:
        if self_data[2]:
            return "BOMB", max_q_s, True, selected_crate_pos
        else:
            return None, max_q_s, True, None
    obstacles_field = get_obstacles_field(game_state)
    obstacles_field[selected_crate_pos[0], selected_crate_pos[1]] = 0

    action, _ = move_towards(self_pos, selected_crate_pos, obstacles_field)
    return action, max_q_s, False, selected_crate_pos


def dodge_bomb(model, game_state, already_chosen_position=None):
    self_data = game_state['self']
    self_pos = self_data[3]
    field = generate_field(game_state)
    predicted_explosions_in_map = get_obstacles_field(game_state, True)
    if predicted_explosions_in_map[self_pos[0], self_pos[1]]==0:
        return None, 0, True, None

    predicted_explosions_in_map[game_state['field'] == 1] = 1
    # Finds all safe positions.
    search_size = 4
    safe_positions = []
    for _y in range(2*search_size + 1):
        y = _y - search_size + self_pos[1]
        for _x in range(2 * search_size + 1):
            x = _x - search_size + self_pos[0]
            if x < 0 or y < 0 or x >= predicted_explosions_in_map.shape[0] or y >= predicted_explosions_in_map.shape[1]:
                continue
            if predicted_explosions_in_map[x, y] == 0:
                safe_positions.append((x, y))

    max_q_s = -100000
    selected = already_chosen_position
    if not selected:
        for safe_position in safe_positions:
            obstacles_field = get_obstacles_field(game_state)
            if move_towards(self_pos, safe_position, obstacles_field)[0] is not None:
                pos_in_field_without_corners = [safe_position[0]-1, safe_position[1]-1]
                features = to_features(self_data, pos_in_field_without_corners, field)
                q_s = model.predict(features.reshape(1, -1))
                if max_q_s < q_s:
                    max_q_s = q_s
                    selected = safe_position
    if selected is None:
        return None, 0, True, None

    obstacles_field = get_obstacles_field(game_state)
    action, done = move_towards(self_pos, selected, np.copy(obstacles_field))
    return action, max_q_s, done, selected


def destroy_enemy(model, game_state):
    other_positions = [other[3] for other in game_state['others']]
    self_data = game_state['self']
    self_pos = self_data[3]
    field = generate_field(game_state)
    max_q_s = -100000
    selected = None
    for other_pos in other_positions:
        dist = (other_pos[0] - self_pos[0], other_pos[1] - self_pos[1])
        if dist[0] == 0 and abs(dist[1]) < 3 or dist[1] == 0 and abs(dist[0]) < 3:
            features = to_features(self_data, other_pos, field)
            q_s = model.predict(features.reshape(1, -1))
            if max_q_s < q_s:
                max_q_s = q_s
                selected = other_pos
    if selected is not None and self_data[2]:
        return "BOMB", max_q_s, True, None
    return None, 0, True, None

def get_obstacles_field(game_state, predictive=False):
    field = np.copy(game_state['field'])
    field[game_state['explosion_map'] == 1] = 1
    predicted_explosions_in_map = np.copy(game_state['field'])
    predicted_explosions_in_map[predicted_explosions_in_map==1] = 0
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    explosion_power = 3
    for bomb in game_state['bombs']:
        bomb_pos = bomb[0]
        if predictive or bomb[1] <= 1:
            predicted_explosions_in_map[bomb_pos[0], bomb_pos[1]] = 1
            for direction in directions:
                for i in range(explosion_power):
                    j = i + 1
                    new_pos = (bomb_pos[0] + j * direction[0], bomb_pos[1] + j * direction[1])
                    if new_pos[0] < 0 or new_pos[1] < 0 or new_pos[0] >= predicted_explosions_in_map.shape[0] or new_pos[1] >= predicted_explosions_in_map.shape[1] or predicted_explosions_in_map[new_pos[0], new_pos[1]] != 0:
                        break
                    predicted_explosions_in_map[new_pos[0], new_pos[1]] = 1
    field[predicted_explosions_in_map==1] = 1
    return field