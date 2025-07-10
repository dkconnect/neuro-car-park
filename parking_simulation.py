import numpy as np
import random
import math
import time
import threading
from flask import Flask, render_template, jsonify, request

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'

SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 800
FPS = 120
CAR_LENGTH = 40
CAR_WIDTH = 20
MAX_SPEED = 10  # Increased for faster movement, only for now lets see might edit later
MAX_STEERING_ANGLE = math.radians(30)
ACCELERATION_RATE = 0.5  # Increased for faster acceleration, same i m makring so i can edit later
BRAKING_RATE = 0.5
TURN_RATE = math.radians(5)  # smooth steer
POPULATION_SIZE = 10
MUTATION_RATE = 0.2
MAX_GENERATIONS = 100
MAX_STEPS_PER_CAR = 200
SIMULATION_TICK_RATE = 0.005 
UPDATE_FREQUENCY = 5

COLORS = {
    "WHITE": [255, 255, 255], "BLACK": [0, 0, 0], "GRAY": [150, 150, 150],
    "ROAD_COLOR": [50, 50, 50], "PARKING_COLOR": [255, 200, 0],
    "RED": [255, 0, 0], "GREEN": [0, 255, 0], "BLUE": [0, 0, 255]
}

ROAD_SEGMENTS = [
    {'x': 100, 'y': 0, 'width': 80, 'height': 800},
    {'x': 180, 'y': 100, 'width': 200, 'height': 80},
    {'x': 180, 'y': 450, 'width': 200, 'height': 80},
    {'x': 380, 'y': 450, 'width': 80, 'height': 350}
]
PARKING_SPOTS = [
    {'x': 400, 'y': 50, 'width': 100, 'height': 50}, {'x': 510, 'y': 50, 'width': 100, 'height': 50},
    {'x': 620, 'y': 50, 'width': 100, 'height': 50}, {'x': 470, 'y': 450, 'width': 50, 'height': 100},
    {'x': 470, 'y': 560, 'width': 50, 'height': 100}, {'x': 470, 'y': 670, 'width': 50, 'height': 100},
    {'x': 470, 'y': 780, 'width': 50, 'height': 100}
]

class Car:
    def __init__(self, x, y, angle=0):
        self.x = x
        self.y = y
        self.angle = angle
        self.speed = 0
        self.steering_angle = 0
        self.length = CAR_LENGTH
        self.width = CAR_WIDTH
        self.color = COLORS["BLUE"]
        self.is_parked = False
        self.collided = False

    def move(self, acceleration, steer_input):
        if self.collided:
            return
        self.speed += acceleration * ACCELERATION_RATE
        self.speed = np.clip(self.speed, -MAX_SPEED, MAX_SPEED)
        self.steering_angle += steer_input * TURN_RATE
        self.steering_angle = np.clip(self.steering_angle, -MAX_STEERING_ANGLE, MAX_STEERING_ANGLE)
        if abs(self.speed) > 0.1:
            rear_x = self.x - (self.length / 2) * math.cos(self.angle)
            rear_y = self.y - (self.length / 2) * math.sin(self.angle)
            rear_x += self.speed * math.cos(self.angle)
            rear_y += self.speed * math.sin(self.angle)
            self.angle += (self.speed / self.length) * math.tan(self.steering_angle)
            self.x = rear_x + (self.length / 2) * math.cos(self.angle)
            self.y = rear_y + (self.length / 2) * math.sin(self.angle)

    def get_corners(self):
        half_length = self.length / 2
        half_width = self.width / 2
        corners_local = [
            (-half_length, -half_width), (half_length, -half_width),
            (half_length, half_width), (-half_length, half_width)
        ]
        rotated_corners = []
        for lx, ly in corners_local:
            gx = self.x + lx * math.cos(self.angle) - ly * math.sin(self.angle)
            gy = self.y + lx * math.sin(self.angle) + ly * math.cos(self.angle)
            rotated_corners.append((gx, gy))
        return rotated_corners

    def get_rect(self):
        corners = self.get_corners()
        min_x = min(c[0] for c in corners)
        max_x = max(c[0] for c in corners)
        min_y = min(c[1] for c in corners)
        max_y = max(c[1] for c in corners)
        return {'x': min_x, 'y': min_y, 'width': max_x - min_x, 'height': max_y - min_y}

    def to_dict(self):
        return {
            'x': self.x, 'y': self.y, 'angle': self.angle, 'speed': self.speed,
            'steering_angle': self.steering_angle, 'length': self.length, 'width': self.width,
            'color': self.color, 'is_parked': self.is_parked, 'collided': self.collided,
            'corners': self.get_corners()
        }

def point_in_polygon(point, polygon):
    x, y = point
    num_vertices = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(num_vertices + 1):
        p2x, p2y = polygon[i % num_vertices]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

def polygons_intersect(poly1, poly2):
    for p in poly1:
        if point_in_polygon(p, poly2):
            return True
    for p in poly2:
        if point_in_polygon(p, poly1):
            return True
    return False

def rect_collide(rect1, rect2):
    return (rect1['x'] < rect2['x'] + rect2['width'] and
            rect1['x'] + rect1['width'] > rect2['x'] and
            rect1['y'] < rect2['y'] + rect2['height'] and
            rect1['y'] + rect1['height'] > rect2['y'])

def is_on_road_or_parking(car):
    car_rect = car.get_rect()
    buffer = 5
    buffered_rect = {
        'x': car_rect['x'] - buffer,
        'y': car_rect['y'] - buffer,
        'width': car_rect['width'] + 2 * buffer,
        'height': car_rect['height'] + 2 * buffer
    }
    for road in ROAD_SEGMENTS:
        if rect_collide(buffered_rect, road):
            return True
    for parking in PARKING_SPOTS:
        if rect_collide(buffered_rect, parking):
            return True
    return False

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.weights1 = np.random.randn(self.input_size, self.hidden_size) * np.sqrt(2 / self.input_size)
        self.bias1 = np.zeros((1, self.hidden_size))
        self.weights2 = np.random.randn(self.hidden_size, self.output_size) * np.sqrt(2 / self.hidden_size)
        self.bias2 = np.zeros((1, self.output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        return np.tanh(x)

    def forward(self, inputs):
        hidden_layer_input = np.dot(inputs, self.weights1) + self.bias1
        hidden_layer_output = self.tanh(hidden_layer_input)
        output_layer_input = np.dot(hidden_layer_output, self.weights2) + self.bias2
        output = self.tanh(output_layer_input)
        print(f"NN Inputs: {inputs.tolist()}, Outputs: {output.tolist()}")
        return output

    def get_weights(self):
        return {'w1': self.weights1, 'b1': self.bias1, 'w2': self.weights2, 'b2': self.bias2}

    def set_weights(self, weights):
        self.weights1 = weights['w1']
        self.bias1 = weights['b1']
        self.weights2 = weights['w2']
        self.bias2 = weights['b2']

class GeneticAlgorithm:
    def __init__(self, input_size, hidden_size, output_size, population_size=POPULATION_SIZE,
                 mutation_rate=MUTATION_RATE):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.population = self.create_initial_population()

    def create_initial_population(self):
        return [{'nn': NeuralNetwork(self.input_size, self.hidden_size, self.output_size), 'fitness': 0}
                for _ in range(self.population_size)]

    def calculate_fitness(self, car, target_parking_spot):
        fitness = 0
        if car.collided:
            fitness -= 500
            return fitness
        target_center_x = target_parking_spot['x'] + target_parking_spot['width'] / 2
        target_center_y = target_parking_spot['y'] + target_parking_spot['height'] / 2
        distance = math.sqrt((car.x - target_center_x)**2 + (car.y - target_center_y)**2)
        fitness += max(0, 1000 - distance)
        if is_on_road_or_parking(car):
            fitness += 100
        car_rect = car.get_rect()
        if (car_rect['x'] >= target_parking_spot['x'] and
            car_rect['x'] + car_rect['width'] <= target_parking_spot['x'] + target_parking_spot['width'] and
            car_rect['y'] >= target_parking_spot['y'] and
            car_rect['y'] + car_rect['height'] <= target_parking_spot['y'] + target_parking_spot['height']):
            fitness += 500
            car.is_parked = True
            fitness += 1000
            target_angle = 0 if target_parking_spot['width'] > target_parking_spot['height'] else math.pi / 2
            angle_diff = abs(car.angle - target_angle) % (2 * math.pi)
            if angle_diff > math.pi:
                angle_diff = 2 * math.pi - angle_diff
            fitness += max(0, 500 - (angle_diff * 100))
            print(f"Car parked in {car_simulation_steps} steps, Fitness={fitness}")
        return fitness

    def selection(self):
        self.population.sort(key=lambda x: x['fitness'], reverse=True)
        num_elite = max(1, int(self.population_size * 0.1))
        elite = [ind['nn'] for ind in self.population[:num_elite]]
        total_fitness = sum(ind['fitness'] for ind in self.population if ind['fitness'] > 0)
        selected_nns = []
        if total_fitness == 0:
            selected_nns = [random.choice(self.population)['nn'] for _ in range(self.population_size - num_elite)]
        else:
            for _ in range(self.population_size - num_elite):
                pick = random.uniform(0, total_fitness)
                current = 0
                for individual in self.population:
                    if individual['fitness'] <= 0:
                        continue
                    current += individual['fitness']
                    if current >= pick:
                        selected_nns.append(individual['nn'])
                        break
        return elite + selected_nns

    def crossover(self, parent1_nn, parent2_nn):
        child_nn = NeuralNetwork(self.input_size, self.hidden_size, self.output_size)
        parent1_weights = parent1_nn.get_weights()
        parent2_weights = parent2_nn.get_weights()
        child_weights = {}
        for key in parent1_weights:
            shape = parent1_weights[key].shape
            child_weights[key] = np.zeros(shape)
            if parent1_weights[key].ndim == 1:
                for i in range(shape[0]):
                    child_weights[key][i] = parent1_weights[key][i] if random.random() < 0.5 else parent2_weights[key][i]
            else:
                for i in range(shape[0]):
                    for j in range(shape[1]):
                        child_weights[key][i, j] = parent1_weights[key][i, j] if random.random() < 0.5 else parent2_weights[key][i, j]
        child_nn.set_weights(child_weights)
        return child_nn

    def mutate(self, nn):
        weights = nn.get_weights()
        for key in weights:
            weights[key] += np.random.randn(*weights[key].shape) * self.mutation_rate
        nn.set_weights(weights)
        return nn

    def evolve(self):
        selected_nns = self.selection()
        next_population = []
        num_elite = max(1, int(self.population_size * 0.1))
        elite_nns = [ind['nn'] for ind in sorted(self.population, key=lambda x: x['fitness'], reverse=True)][:num_elite]
        for nn_elite in elite_nns:
            next_population.append({'nn': nn_elite, 'fitness': 0})
        for _ in range(self.population_size - num_elite):
            parent1 = random.choice(selected_nns)
            parent2 = random.choice(selected_nns)
            child = self.crossover(parent1, parent2)
            child = self.mutate(child)
            next_population.append({'nn': child, 'fitness': 0})
        self.population = next_population

simulation_running = False
simulation_thread = None
ga_instance = None
current_generation_num = 0
car_simulation_steps = 0
simulation_state = {
    'car': None,
    'target_spot_index': -1,
    'current_generation': 0,
    'current_car_in_gen': 0,
    'max_cars_in_gen': POPULATION_SIZE,
    'steps': 0,
    'max_steps': MAX_STEPS_PER_CAR,
    'status': 'ready'
}
INPUT_SIZE = 9
HIDDEN_SIZE = 16
OUTPUT_SIZE = 2

def run_simulation_logic():
    global simulation_running, ga_instance, current_generation_num, car_simulation_steps, simulation_state
    try:
        if ga_instance is None:
            ga_instance = GeneticAlgorithm(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
            current_generation_num = 0

        target_parking_spots = PARKING_SPOTS
        while simulation_running and current_generation_num < MAX_GENERATIONS:
            print(f"\n--- Generation {current_generation_num + 1} ---")
            generation_successful = True
            total_fitness_this_generation = 0

            for i, individual in enumerate(ga_instance.population):
                if not simulation_running:
                    break
                nn = individual['nn']
                car_index = i % len(target_parking_spots)
                target_parking_spot = target_parking_spots[car_index]
                start_x = ROAD_SEGMENTS[0]['x'] + ROAD_SEGMENTS[0]['width'] / 2
                start_y = SCREEN_HEIGHT - CAR_LENGTH / 2
                car = Car(start_x, start_y, angle=math.radians(-90))
                if not is_on_road_or_parking(car):
                    print(f"Warning: Car {i+1} starts off-road at ({start_x}, {start_y})!")
                print(f"Simulating Car {i+1} (NN {i+1}) for Spot {car_index+1}")
                car_simulation_steps = 0
                car_parked_successfully = False

                simulation_state.update({
                    'car': car.to_dict(),
                    'target_spot_index': car_index,
                    'current_generation': current_generation_num + 1,
                    'current_car_in_gen': i + 1,
                    'max_cars_in_gen': POPULATION_SIZE,
                    'steps': car_simulation_steps,
                    'max_steps': MAX_STEPS_PER_CAR,
                    'status': 'moving'
                })
                time.sleep(SIMULATION_TICK_RATE)

                while car_simulation_steps < MAX_STEPS_PER_CAR:
                    if not simulation_running:
                        break
                    inputs = []
                    sensor_angles = [-math.pi/2, -math.pi/4, 0, math.pi/4, math.pi/2]
                    sensor_distances = []
                    for s_angle in sensor_angles:
                        ray_angle = car.angle + s_angle
                        max_ray_dist = 300
                        hit_dist = max_ray_dist
                        for step in range(1, max_ray_dist, 5):
                            test_x = car.x + step * math.cos(ray_angle)
                            test_y = car.y + step * math.sin(ray_angle)
                            test_point_rect = {'x': test_x - 1, 'y': test_y - 1, 'width': 2, 'height': 2}
                            is_valid_path = False
                            for road_seg in ROAD_SEGMENTS:
                                if rect_collide(test_point_rect, road_seg):
                                    is_valid_path = True
                                    break
                            if not is_valid_path:
                                for parking_spot in PARKING_SPOTS:
                                    if rect_collide(test_point_rect, parking_spot):
                                        is_valid_path = True
                                        break
                            if not is_valid_path:
                                hit_dist = step
                                break
                        sensor_distances.append(hit_dist / max_ray_dist)
                    inputs.extend(sensor_distances)
                    inputs.append(car.speed / MAX_SPEED)
                    inputs.append(car.steering_angle / MAX_STEERING_ANGLE)
                    target_center_x = target_parking_spot['x'] + target_parking_spot['width'] / 2
                    target_center_y = target_parking_spot['y'] + target_parking_spot['height'] / 2
                    target_vec_x = target_center_x - car.x
                    target_vec_y = target_center_y - car.y
                    angle_to_target = math.atan2(target_vec_y, target_vec_x)
                    angle_diff = (angle_to_target - car.angle + math.pi) % (2 * math.pi) - math.pi
                    inputs.append(angle_diff / math.pi)
                    dist_to_target = math.sqrt(target_vec_x**2 + target_vec_y**2)
                    inputs.append(dist_to_target / SCREEN_WIDTH)
                    nn_inputs = np.array([inputs])
                    nn_output = nn.forward(nn_inputs)[0]
                    acceleration_output = nn_output[0]
                    steer_output = nn_output[1]
                    car.move(acceleration_output, steer_output)

                    if not (0 <= car.x <= SCREEN_WIDTH and 0 <= car.y <= SCREEN_HEIGHT):
                        car.collided = True
                        print(f"Car {i+1} hit screen boundary at ({car.x}, {car.y})")
                    if not is_on_road_or_parking(car):
                        car.collided = True
                        print(f"Car {i+1} collided off-road at ({car.x}, {car.y}), angle {math.degrees(car.angle)}")
                    if car.collided:
                        generation_successful = False
                        print(f"Car {i+1} collided. Generation failed!")
                        break
                    car_rect = car.get_rect()
                    if (car_rect['x'] >= target_parking_spot['x'] and
                        car_rect['x'] + car_rect['width'] <= target_parking_spot['x'] + target_parking_spot['width'] and
                        car_rect['y'] >= target_parking_spot['y'] and
                        car_rect['y'] + car_rect['height'] <= target_parking_spot['y'] + target_parking_spot['height'] and
                        abs(car.speed) < 0.5):
                        car_parked_successfully = True
                        car.is_parked = True
                        print(f"Car {i+1} parked in Spot {car_index+1}!")
                        break
                    car_simulation_steps += 1
                    if car_simulation_steps % UPDATE_FREQUENCY == 0:
                        simulation_state.update({
                            'car': car.to_dict(),
                            'target_spot_index': car_index,
                            'current_generation': current_generation_num + 1,
                            'current_car_in_gen': i + 1,
                            'max_cars_in_gen': POPULATION_SIZE,
                            'steps': car_simulation_steps,
                            'max_steps': MAX_STEPS_PER_CAR,
                            'status': 'collided' if car.collided else ('parked' if car.is_parked else 'moving')
                        })
                    time.sleep(SIMULATION_TICK_RATE)
                fitness = ga_instance.calculate_fitness(car, target_parking_spot)
                individual['fitness'] = fitness
                total_fitness_this_generation += fitness
                print(f"Car {i+1}: Steps={car_simulation_steps}, Position=({car.x:.2f}, {car.y:.2f}), "
                      f"Angle={math.degrees(car.angle):.2f}, Fitness={fitness}")
                if not car_parked_successfully and not car.collided:
                    individual['fitness'] -= 500
                    generation_successful = False
                    print(f"Car {i+1} failed to park within {MAX_STEPS_PER_CAR} steps!")
                    simulation_state.update({
                        'car': car.to_dict(),
                        'target_spot_index': car_index,
                        'current_generation': current_generation_num + 1,
                        'current_car_in_gen': i + 1,
                        'max_cars_in_gen': POPULATION_SIZE,
                        'steps': car_simulation_steps,
                        'max_steps': MAX_STEPS_PER_CAR,
                        'status': 'failed_to_park'
                    })
                    time.sleep(1)
                time.sleep(0.5)
            if simulation_running:
                print(f"Generation {current_generation_num + 1} Fitness: Avg={total_fitness_this_generation/POPULATION_SIZE:.2f}, "
                      f"Max={max(ind['fitness'] for ind in ga_instance.population):.2f}")
                if generation_successful:
                    print(f"Generation {current_generation_num + 1} SUCCESS!")
                    simulation_state.update({'status': 'success'})
                    simulation_running = False
                else:
                    print(f"Generation {current_generation_num + 1} FAILED. Evolving...")
                    simulation_state.update({'status': 'evolving'})
                    ga_instance.evolve()
                    current_generation_num += 1
                    if current_generation_num >= MAX_GENERATIONS:
                        print(f"Reached max generations ({MAX_GENERATIONS}). Resetting GA.")
                        simulation_state.update({'status': 'max_generations'})
                        ga_instance = GeneticAlgorithm(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
                        current_generation_num = 0
    except Exception as e:
        print(f"Simulation thread error: {e}")
        simulation_state.update({'status': f'error: {str(e)}'})
        simulation_running = False
    finally:
        simulation_running = False
        print("Simulation thread terminated.")
        simulation_state.update({'status': 'done'})

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_sim', methods=['POST'])
def start_simulation_endpoint():
    global simulation_running, simulation_thread
    if not simulation_running:
        if simulation_thread and simulation_thread.is_alive():
            return jsonify({'message': 'Simulation thread already running.', 'type': 'warning'})
        simulation_running = True
        simulation_thread = threading.Thread(target=run_simulation_logic)
        simulation_thread.daemon = True
        simulation_thread.start()
        print("Simulation thread started successfully.")
        return jsonify({'message': 'Simulation started.', 'type': 'info'})
    return jsonify({'message': 'Simulation already running.', 'type': 'warning'})

@app.route('/stop_sim', methods=['POST'])
def stop_simulation_endpoint():
    global simulation_running
    if simulation_running:
        simulation_running = False
        print("Simulation stop requested.")
        return jsonify({'message': 'Simulation stopped.', 'type': 'done'})
    return jsonify({'message': 'Simulation not running.', 'type': 'warning'})

@app.route('/get_simulation_state')
def get_simulation_state():
    return jsonify(simulation_state)

@app.route('/get_environment_data')
def get_environment_data():
    return jsonify({
        'road_segments': ROAD_SEGMENTS,
        'parking_spots': PARKING_SPOTS,
        'screen_width': SCREEN_WIDTH,
        'screen_height': SCREEN_HEIGHT,
        'car_length': CAR_LENGTH,
        'car_width': CAR_WIDTH,
        'colors': COLORS
    })

if __name__ == '__main__':
    import os
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
