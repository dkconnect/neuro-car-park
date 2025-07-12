import math
import random
import numpy as np
import time
from flask import Flask, render_template, jsonify, request
from threading import Thread, Event
import threading
import json

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 800
CAR_LENGTH = 40
CAR_WIDTH = 20
SIMULATION_TICK_RATE = 0.005 
UPDATE_FREQUENCY = 5      
MAX_SPEED = 5            
ACCELERATION_RATE = 0.1  
TURN_RATE = math.radians(2)

MAX_RAY_DIST = 300 

POPULATION_SIZE = 10
MUTATION_RATE = 0.1      
MAX_GENERATIONS = 100
MAX_STEPS_PER_CAR = 500   

ROAD_SEGMENTS = [
    {'x': 100, 'y': 0, 'width': 80, 'height': 800},  
]
PARKING_SPOTS = [
    {'x': 200, 'y': 600, 'width': 40, 'height': 60, 'angle': 0}, 
    {'x': 260, 'y': 600, 'width': 40, 'height': 60, 'angle': 0},
    {'x': 320, 'y': 600, 'width': 40, 'height': 60, 'angle': 0},
    {'x': 380, 'y': 600, 'width': 40, 'height': 60, 'angle': 0},
]

START_X = ROAD_SEGMENTS[0]['x'] + ROAD_SEGMENTS[0]['width'] / 2
START_Y = SCREEN_HEIGHT - CAR_LENGTH / 2
START_ANGLE = math.radians(-90) 

def rect_collide(rect1, rect2):
    """Checks for AABB collision between two rectangles."""
    return (rect1['x'] < rect2['x'] + rect2['width'] and
            rect1['x'] + rect1['width'] > rect2['x'] and
            rect1['y'] < rect2['y'] + rect2['height'] and
            rect1['y'] + rect1['height'] > rect2['y'])

def rotate_point(x, y, cx, cy, angle):
    """Rotates a point (x,y) around a center (cx,cy) by 'angle'."""
    temp_x = x - cx
    temp_y = y - cy
    rotated_x = temp_x * math.cos(angle) - temp_y * math.sin(angle)
    rotated_y = temp_x * math.sin(angle) + temp_y * math.cos(angle)
    return rotated_x + cx, rotated_y + cy

def get_rotated_rectangle_corners(x, y, width, height, angle):
    """Returns the four corners of a rotated rectangle."""
    half_width = width / 2
    half_height = height / 2

    corners_local = [
        (-half_width, -half_height),  
        (half_width, -half_height),   
        (half_width, half_height),   
        (-half_width, half_height)   
    ]

    rotated_corners = []
    for lx, ly in corners_local:
        gx, gy = rotate_point(lx, ly, 0, 0, angle) 
        rotated_corners.append((gx + x, gy + y))

    return rotated_corners

class Car:
    def __init__(self, x, y, angle, length, width):
        self.x = x
        self.y = y
        self.angle = angle 
        self.length = length
        self.width = width
        self.speed = 0
        self.steering_angle = 0 
        self.collided = False
        self.is_parked = False
        self.steps_taken = 0
        self.corners = [] 
        self.update_corners()

    def update_corners(self):
        """Calculates and updates the car's current corner coordinates."""
        self.corners = get_rotated_rectangle_corners(self.x, self.y, self.width, self.length, self.angle)

    def get_rect(self):
        """Returns a non-rotated bounding box for rough collision checks."""

        min_x = min(c[0] for c in self.corners)
        max_x = max(c[0] for c in self.corners)
        min_y = min(c[1] for c in self.corners)
        max_y = max(c[1] for c in self.corners)
        return {'x': min_x, 'y': min_y, 'width': max_x - min_x, 'height': max_y - min_y}

    def move(self, acceleration_input, steering_input):
        if self.collided or self.is_parked:
            return

        self.speed += acceleration_input * ACCELERATION_RATE
        self.speed = max(-MAX_SPEED, min(self.speed, MAX_SPEED)) 

        self.steering_angle += steering_input * TURN_RATE
        self.steering_angle = max(-math.radians(45), min(self.steering_angle, math.radians(45))) 

        rear_x = self.x - (self.length / 2) * math.cos(self.angle)
        rear_y = self.y - (self.length / 2) * math.sin(self.angle)

        rear_x += self.speed * math.cos(self.angle + self.steering_angle)
        rear_y += self.speed * math.sin(self.angle + self.steering_angle)

        self.x = rear_x + (self.length / 2) * math.cos(self.angle)
        self.y = rear_y + (self.length / 2) * math.sin(self.angle)

        self.angle += self.speed * math.sin(self.steering_angle) / self.length
        self.angle = math.atan2(math.sin(self.angle), math.cos(self.angle))

        self.steps_taken += 1
        self.update_corners()

    def get_sensors(self, road_segments, parking_spots):
        sensors = []
        sensor_angles = [-math.pi / 2, -math.pi / 4, 0, math.pi / 4, math.pi / 2]

        for s_angle in sensor_angles:
            ray_angle = self.angle + s_angle
            hit_dist = MAX_RAY_DIST

            for step in range(1, MAX_RAY_DIST, 5): 
                test_x = self.x + step * math.cos(ray_angle)
                test_y = self.y + step * math.sin(ray_angle)

                if not (0 <= test_x <= SCREEN_WIDTH and 0 <= test_y <= SCREEN_HEIGHT):
                    hit_dist = step
                    break

                is_on_valid_path = False
                test_point_rect = {'x': test_x - 1, 'y': test_y - 1, 'width': 2, 'height': 2} 
                for road in road_segments:
                    if rect_collide(test_point_rect, road):
                        is_on_valid_path = True
                        break
                if not is_on_valid_path:
                    for parking in parking_spots:
                        if rect_collide(test_point_rect, parking):
                            is_on_valid_path = True
                            break

                if not is_on_valid_path: 
                    hit_dist = step
                    break
            sensors.append(hit_dist / MAX_RAY_DIST) 
        return sensors

    def to_dict(self):
        return {
            'x': self.x,
            'y': self.y,
            'angle': self.angle,
            'length': self.length,
            'width': self.width,
            'speed': self.speed,
            'steering_angle': self.steering_angle,
            'collided': self.collided,
            'is_parked': self.is_parked,
            'steps_taken': self.steps_taken,
            'corners': [[c[0], c[1]] for c in self.corners]
        }

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights1 = np.random.randn(input_size, hidden_size) * 0.01
        self.bias1 = np.zeros((1, hidden_size))
        self.weights2 = np.random.randn(hidden_size, output_size) * 0.01
        self.bias2 = np.zeros((1, output_size))

    def forward(self, inputs):
        self.hidden_layer_input = np.dot(inputs, self.weights1) + self.bias1
        self.hidden_layer_output = np.tanh(self.hidden_layer_input) 

        self.output_layer_input = np.dot(self.hidden_layer_output, self.weights2) + self.bias2

        output = 2 * (1 / (1 + np.exp(-self.output_layer_input))) - 1 
        return output

class GeneticAlgorithm:
    def __init__(self, population_size, input_size, hidden_size, output_size):
        self.population_size = population_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.population = self.create_initial_population()
        self.best_individual = None

    def create_initial_population(self):
        population = []
        for _ in range(self.population_size):
            nn = NeuralNetwork(self.input_size, self.hidden_size, self.output_size)
            population.append({
                'nn': nn,
                'fitness': 0
            })
        return population

    def calculate_fitness(self, car, target_spot):
        distance_to_target = math.sqrt((car.x - target_spot['x'])**2 + (car.y - target_spot['y'])**2)
        
        fitness = 0

        fitness += max(0, 1000 - distance_to_target)

        if is_on_road_or_parking(car):
            fitness += 100
      
        car_rect_corners = get_rotated_rectangle_corners(car.x, car.y, car.width, car.length, car.angle)

        car_center_in_spot = target_spot['x'] <= car.x <= target_spot['x'] + target_spot['width'] and \
                             target_spot['y'] <= car.y <= target_spot['y'] + target_spot['height']
        
        if car_center_in_spot:
            fitness += 500 
        if car.is_parked:
            fitness += 1000 
            print("Car is parked!")

        if car.collided:
            fitness -= 500

        if car.steps_taken >= MAX_STEPS_PER_CAR and not car.is_parked:
            fitness -= 500 

        target_angle = math.radians(target_spot['angle'])
        angle_diff = abs(car.angle - target_angle)

        angle_diff = math.atan2(math.sin(angle_diff), math.cos(angle_diff))
        
        fitness += max(0, 500 - (abs(angle_diff) * 100)) 
        return max(0, fitness) 

    def select(self):

        elite = max(self.population, key=lambda x: x['fitness'])

        total_fitness = sum(ind['fitness'] for ind in self.population)
        if total_fitness == 0:
            return [random.choice(self.population)['nn'] for _ in range(self.population_size - 1)] + [elite['nn']]

        selected = [elite['nn']] 
        for _ in range(self.population_size - 1):
            pick = random.uniform(0, total_fitness)
            current = 0
            for individual in self.population:
                current += individual['fitness']
                if current > pick:
                    selected.append(individual['nn'])
                    break
        return selected

    def crossover(self, parent1_nn, parent2_nn):
        child_nn = NeuralNetwork(self.input_size, self.hidden_size, self.output_size)

        child_nn.weights1 = (parent1_nn.weights1 + parent2_nn.weights1) / 2
        child_nn.bias1 = (parent1_nn.bias1 + parent2_nn.bias1) / 2
        child_nn.weights2 = (parent1_nn.weights2 + parent2_nn.weights2) / 2
        child_nn.bias2 = (parent1_nn.bias2 + parent2_nn.bias2) / 2
        return child_nn

    def mutate(self, nn):

        nn.weights1 += np.random.randn(*nn.weights1.shape) * MUTATION_RATE
        nn.bias1 += np.random.randn(*nn.bias1.shape) * MUTATION_RATE
        nn.weights2 += np.random.randn(*nn.weights2.shape) * MUTATION_RATE
        nn.bias2 += np.random.randn(*nn.bias2.shape) * MUTATION_RATE

    def evolve(self):
        selected_nns = self.select()
        new_population = []

        best_individual_nn = max(self.population, key=lambda x: x['fitness'])['nn']
        new_population.append({'nn': best_individual_nn, 'fitness': 0}) 

        for _ in range(self.population_size - 1): 
            parent1 = random.choice(selected_nns)
            parent2 = random.choice(selected_nns)
            child_nn = self.crossover(parent1, parent2)
            if random.random() < MUTATION_RATE:
                self.mutate(child_nn)
            new_population.append({'nn': child_nn, 'fitness': 0})

        self.population = new_population

        self.best_individual = max(self.population, key=lambda x: x['fitness'])
        if self.best_individual['fitness'] > 0: 
             print(f"Best fitness of last generation: {self.best_individual['fitness']:.2f}")


def is_on_road_or_parking(car):
    """
    Checks if any part of the car's bounding box (AABB) overlaps
    with any road segment or parking spot.
    """
    car_rect = car.get_rect()
    for road in ROAD_SEGMENTS:
        if rect_collide(car_rect, road):
            return True
    for parking in PARKING_SPOTS:
        if rect_collide(car_rect, parking):
            return True
    return False

app = Flask(__name__)

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
sim_thread = None
sim_running = Event() 
sim_thread_lock = threading.Lock() 
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_environment_data')
def get_environment_data():
    return jsonify({
        'screen_width': SCREEN_WIDTH,
        'screen_height': SCREEN_HEIGHT,
        'car_length': CAR_LENGTH,
        'car_width': CAR_WIDTH,
        'road_segments': ROAD_SEGMENTS,
        'parking_spots': PARKING_SPOTS,
        'start_x': START_X,
        'start_y': START_Y,
        'start_angle': START_ANGLE,
        'max_generations': MAX_GENERATIONS, 
        'max_steps_per_car': MAX_STEPS_PER_CAR 
    })

@app.route('/get_simulation_state')
def get_simulation_state():
    with sim_thread_lock:
        return jsonify(simulation_state)

@app.route('/start_sim', methods=['POST'])
def start_simulation():
    global sim_thread
    with sim_thread_lock:
        if sim_thread and sim_thread.is_alive():
            return jsonify({'message': 'Simulation is already running.', 'type': 'warning'})
        
        sim_running.set() 
        sim_thread = Thread(target=run_simulation_logic)
        sim_thread.daemon = True
        sim_thread.start()
        print("Simulation started.")
        return jsonify({'message': 'Simulation started successfully.', 'type': 'info'})

@app.route('/stop_sim', methods=['POST'])
def stop_simulation():
    global sim_thread
    with sim_thread_lock:
        if sim_thread and sim_thread.is_alive():
            sim_running.clear() 
            sim_thread.join(timeout=2) 
            if sim_thread.is_alive():
                print("Simulation thread did not stop gracefully.")
            else:
                print("Simulation stopped.")
            
            simulation_state['status'] = 'stopped' 
            return jsonify({'message': 'Simulation stopped.', 'type': 'done'})
        else:
            return jsonify({'message': 'Simulation is not running.', 'type': 'info'})

def run_simulation_logic():
    global simulation_state

    ga = GeneticAlgorithm(POPULATION_SIZE, 7, 10, 2)
    
    current_generation_num = 0

    while sim_running.is_set() and current_generation_num < MAX_GENERATIONS:
        current_generation_num += 1
        print(f"--- Generation {current_generation_num} ---")
        
        with sim_thread_lock:
            simulation_state['current_generation'] = current_generation_num
            simulation_state['status'] = 'evolving' 

        for i, individual in enumerate(ga.population):
            if not sim_running.is_set(): 
                print("Stopping simulation mid-generation.")
                break

            car = Car(START_X, START_Y, START_ANGLE, CAR_LENGTH, CAR_WIDTH)
            nn = individual['nn']
            car_index = random.randint(0, len(PARKING_SPOTS) - 1)
            target_spot = PARKING_SPOTS[car_index]

            print(f"  Car {i+1}/{POPULATION_SIZE} aiming for spot {car_index}")

            car_simulation_steps = 0
            while not car.collided and not car.is_parked and car_simulation_steps < MAX_STEPS_PER_CAR:
                if not sim_running.is_set(): 
                    print("Stopping current car simulation.")
                    break

                sensors = car.get_sensors(ROAD_SEGMENTS, PARKING_SPOTS)

                dist_to_target = math.sqrt((car.x - target_spot['x'])**2 + (car.y - target_spot['y'])**2)
                dist_to_target_norm = dist_to_target / math.sqrt(SCREEN_WIDTH**2 + SCREEN_HEIGHT**2) 

                angle_to_target = math.atan2(target_spot['y'] - car.y, target_spot['x'] - car.x)
                angle_diff = angle_to_target - car.angle
                angle_diff = math.atan2(math.sin(angle_diff), math.cos(angle_diff))
                angle_diff_norm = angle_diff / math.pi 

                inputs = np.array([sensors + [dist_to_target_norm, angle_diff_norm]])
       
                output = nn.forward(inputs)[0]
                
                acceleration_input = output[0] 
                steering_input = output[1] 
                car.move(acceleration_input, steering_input)

                car_center_x = car.x
                car_center_y = car.y
         
                car_bbox_for_parking = {
                    'x': car_center_x - CAR_WIDTH/2,
                    'y': car_center_y - CAR_LENGTH/2,
                    'width': CAR_WIDTH,
                    'height': CAR_LENGTH
                }


                is_overlapping_spot = rect_collide(car_bbox_for_parking, target_spot)

 
                target_angle_rad = math.radians(target_spot['angle'])
                angle_deviation = abs(car.angle - target_angle_rad)
                angle_deviation = math.atan2(math.sin(angle_deviation), math.cos(angle_deviation))
                is_angle_aligned = abs(angle_deviation) < math.radians(10)

                is_stopped = abs(car.speed) < 0.1

                if is_overlapping_spot and is_angle_aligned and is_stopped:
                    car.is_parked = True

                    time.sleep(0.5) 

                if not (0 <= car.x <= SCREEN_WIDTH and 0 <= car.y <= SCREEN_HEIGHT):
                    car.collided = True
                    print(f"  Car {i+1} hit screen boundary at ({car.x:.2f}, {car.y:.2f})")
                    
                if not car.collided and not is_on_road_or_parking(car):
                    car.collided = True
                    print(f"  Car {i+1} collided off-road at ({car.x:.2f}, {car.y:.2f}), angle {math.degrees(car.angle):.2f}")

                car_simulation_steps += 1

                if car_simulation_steps % UPDATE_FREQUENCY == 0 or car.collided or car.is_parked:
                    with sim_thread_lock:
                        simulation_state.update({
                            'car': car.to_dict(),
                            'target_spot_index': car_index,
                            'current_car_in_gen': i + 1,
                            'steps': car_simulation_steps,
                            'status': 'collided' if car.collided else ('parked' if car.is_parked else 'moving')
                        })
                time.sleep(SIMULATION_TICK_RATE) 

            individual['fitness'] = ga.calculate_fitness(car, target_spot)
            
            if not car.is_parked and not car.collided and car.steps_taken >= MAX_STEPS_PER_CAR:
                 with sim_thread_lock:
                    simulation_state['status'] = 'failed_to_park'

                    simulation_state['car'] = car.to_dict()
                 time.sleep(0.5) 

            print(f"  Car {i+1} finished. Fitness: {individual['fitness']:.2f}, Status: {'Parked' if car.is_parked else 'Collided' if car.collided else 'Timed Out'}")
            
        if not sim_running.is_set():
            break 

        print(f"Generation {current_generation_num} complete. Evolving population...")
        ga.evolve()

    print("Simulation finished.")
    with sim_thread_lock:
        if current_generation_num >= MAX_GENERATIONS:
            simulation_state['status'] = 'max_generations'
        else:
            simulation_state['status'] = 'stopped'

        if ga.best_individual:
            final_best_car_nn = ga.best_individual['nn']
            car = Car(START_X, START_Y, START_ANGLE, CAR_LENGTH, CAR_WIDTH)
            car_index = random.randint(0, len(PARKING_SPOTS) - 1)
            target_spot = PARKING_SPOTS[car_index]
            
            final_car_steps = 0
            print("Replaying best car's performance...")
            while not car.collided and not car.is_parked and final_car_steps < MAX_STEPS_PER_CAR * 1.5: 
                if not sim_running.is_set(): break
                sensors = car.get_sensors(ROAD_SEGMENTS, PARKING_SPOTS)
                dist_to_target = math.sqrt((car.x - target_spot['x'])**2 + (car.y - target_spot['y'])**2)
                dist_to_target_norm = dist_to_target / math.sqrt(SCREEN_WIDTH**2 + SCREEN_HEIGHT**2)
                angle_to_target = math.atan2(target_spot['y'] - car.y, target_spot['x'] - car.x)
                angle_diff = math.atan2(math.sin(angle_to_target - car.angle), math.cos(angle_to_target - car.angle))
                angle_diff_norm = angle_diff / math.pi
                inputs = np.array([sensors + [dist_to_target_norm, angle_diff_norm]])
                output = final_best_car_nn.forward(inputs)[0]
                car.move(output[0], output[1])

                car_center_x = car.x
                car_center_y = car.y
                car_bbox_for_parking = {
                    'x': car_center_x - CAR_WIDTH/2, 'y': car_center_y - CAR_LENGTH/2,
                    'width': CAR_WIDTH, 'height': CAR_LENGTH
                }
                is_overlapping_spot = rect_collide(car_bbox_for_parking, target_spot)
                target_angle_rad = math.radians(target_spot['angle'])
                angle_deviation = abs(car.angle - target_angle_rad)
                angle_deviation = math.atan2(math.sin(angle_deviation), math.cos(angle_deviation))
                is_angle_aligned = abs(angle_deviation) < math.radians(10)
                is_stopped = abs(car.speed) < 0.1

                if is_overlapping_spot and is_angle_aligned and is_stopped:
                    car.is_parked = True

                if not (0 <= car.x <= SCREEN_WIDTH and 0 <= car.y <= SCREEN_HEIGHT): car.collided = True
                if not car.collided and not is_on_road_or_parking(car): car.collided = True

                final_car_steps += 1
                with sim_thread_lock:
                    simulation_state.update({
                        'car': car.to_dict(),
                        'target_spot_index': car_index,
                        'current_generation': current_generation_num,
                        'current_car_in_gen': 'BEST',
                        'steps': final_car_steps,
                        'status': 'success' if car.is_parked else ('collided' if car.collided else 'replaying_best')
                    })
                time.sleep(SIMULATION_TICK_RATE)
                if car.is_parked or car.collided:
                    time.sleep(1) 
                    break
        
        simulation_state['status'] = 'ready' 


if __name__ == '__main__':

    MAX_GENERATIONS = 200 

    app.run(debug=True, use_reloader=False) 
