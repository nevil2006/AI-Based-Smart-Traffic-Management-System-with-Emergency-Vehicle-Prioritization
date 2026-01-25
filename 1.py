import random
import time
import pygame
import sys

# This class holds all the data and logic for one simulation instance.
class SimulationInstance:
    def __init__(self, scenario_name, scenario_dist):
        self.scenario_name = scenario_name
        self.scenario_dist = scenario_dist
        
        # Default values of signal timers
        self.defaultGreen = {0: 10, 1: 10, 2: 10, 3: 10}
        self.defaultRed = 150
        self.defaultYellow = 5

        self.signals = []
        self.noOfSignals = 4
        self.currentGreen = 0
        self.nextGreen = (self.currentGreen + 1) % self.noOfSignals
        self.currentYellow = 0

        self.speeds = {'car': 2.25, 'bus': 1.8, 'truck': 1.8, 'bike': 2.5, 'ev': 3.0} # Added EV speed

        # Coordinates
        self.x = {'right': [0, 0, 0], 'down': [755, 727, 697], 'left': [1400, 1400, 1400], 'up': [602, 627, 657]}
        self.y = {'right': [348, 370, 398], 'down': [0, 0, 0], 'left': [498, 466, 436], 'up': [800, 800, 800]}

        self.vehicles = {'right': {0: [], 1: [], 2: []}, 'down': {0: [], 1: [], 2: []},
                         'left': {0: [], 1: [], 2: []}, 'up': {0: [], 1: [], 2: []}}
        
        self.vehicles_passed = 0
        self.vehicleTypes = {0: 'car', 1: 'bus', 2: 'truck', 3: 'bike'}
        self.directionNumbers = {0: 'right', 1: 'down', 2: 'left', 3: 'up'}

        # Drawing coordinates
        self.signalCoods = [(530, 230), (810, 230), (810, 570), (530, 570)]
        self.signalTimerCoods = [(530, 210), (810, 210), (810, 550), (530, 550)]
        self.vehicleCountCoods = [(480,210),(880,210),(880,550),(480,550)]
        self.checkpoints = {'right': 450, 'left': 950} # Checkpoint locations

        self.stopLines = {'right': 590, 'down': 330, 'left': 800, 'up': 535}
        self.defaultStop = {'right': 580, 'down': 320, 'left': 810, 'up': 545}

        self.stoppingGap = 25
        self.movingGap = 25
        
        self.allowedVehicleTypes = {'car': True, 'bus': True, 'truck': True, 'bike': True}
        self.allowedVehicleTypesList = [i for i, v in enumerate(self.allowedVehicleTypes.values()) if v]
        
        self.randomGreenSignalTimer = True
        self.randomGreenSignalTimerRange = [10, 20]

        self.timeElapsed = 0
        self.simulation_sprites = pygame.sprite.Group()

        self.last_signal_update_time = time.time()
        self.last_vehicle_spawn_time = time.time()

        # --- NEW Emergency Vehicle State Variables ---
        self.ev_spawned = False
        self.emergency_mode_active = False
        self.emergency_yellow_timer = -1
        self.emergency_target_green = -1
        self.original_current_green = -1
        
        self.initialize_signals()

    def initialize_signals(self):
        minTime = self.randomGreenSignalTimerRange[0]; maxTime = self.randomGreenSignalTimerRange[1]
        self.signals.append(TrafficSignal(0, self.defaultYellow, random.randint(minTime, maxTime)))
        self.signals.append(TrafficSignal(self.signals[0].red + self.signals[0].yellow + self.signals[0].green, self.defaultYellow, random.randint(minTime, maxTime)))
        self.signals.append(TrafficSignal(self.defaultRed, self.defaultYellow, random.randint(minTime, maxTime)))
        self.signals.append(TrafficSignal(self.defaultRed, self.defaultYellow, random.randint(minTime, maxTime)))

    def update(self):
        current_time = time.time()
        
        if current_time - self.last_signal_update_time >= 1.0:
            self.update_signal_timers()
            self.last_signal_update_time = current_time
            self.timeElapsed += 1

        # --- NEW EV Spawning Logic ---
        if self.scenario_name == "heavy_horizontal" and not self.ev_spawned and self.timeElapsed >= 20:
            Vehicle(self, 1, 'ev', 2, 'left') # Spawns in lane 3 (direction 'left')
            self.ev_spawned = True
        elif self.scenario_name == "default" and not self.ev_spawned and self.timeElapsed >= 35:
            Vehicle(self, 1, 'ev', 0, 'right') # Spawns in lane 1 (direction 'right')
            self.ev_spawned = True

        if current_time - self.last_vehicle_spawn_time >= 1.0 + random.uniform(-0.5, 0.5):
            if not self.emergency_mode_active: # Don't spawn normal cars during an emergency
                self.generateVehicles()
            self.last_vehicle_spawn_time = current_time

        self.simulation_sprites.update()

    def trigger_emergency_mode(self, ev_direction_number):
        if not self.emergency_mode_active:
            print(f"SIM {self.scenario_name}: EMERGENCY TRIGGERED for lane {ev_direction_number+1}")
            self.emergency_mode_active = True
            self.emergency_yellow_timer = 5 # 5 second yellow phase
            self.emergency_target_green = ev_direction_number
            self.original_current_green = self.currentGreen

    def update_signal_timers(self):
        if self.emergency_mode_active:
            if self.emergency_yellow_timer > 0:
                self.emergency_yellow_timer -= 1
                self.signals[self.original_current_green].signalText = self.emergency_yellow_timer
                self.signals[self.emergency_target_green].signalText = self.emergency_yellow_timer
            elif self.emergency_yellow_timer == 0:
                print(f"SIM {self.scenario_name}: Giving GREEN to EV lane {self.emergency_target_green+1}")
                self.currentGreen = self.emergency_target_green
                self.currentYellow = 0
                self.signals[self.currentGreen].green = 99
                self.signals[self.currentGreen].yellow = self.defaultYellow
                self.signals[self.currentGreen].red = self.defaultRed
                self.nextGreen = (self.currentGreen + 1) % self.noOfSignals
                for i in range(self.noOfSignals):
                    if i != self.currentGreen:
                        self.signals[i].red = self.signals[self.currentGreen].green + self.signals[self.currentGreen].yellow
                self.emergency_mode_active = False
                self.emergency_yellow_timer = -1
            return

        if self.signals[self.currentGreen].green > 0 and self.currentYellow == 0:
            self.signals[self.currentGreen].green -= 1
        elif self.signals[self.currentGreen].green == 0 and self.currentYellow == 0:
            self.currentYellow = 1
            self.signals[self.currentGreen].yellow = self.defaultYellow
        
        if self.signals[self.currentGreen].yellow > 0 and self.currentYellow == 1:
            self.signals[self.currentGreen].yellow -= 1
        elif self.signals[self.currentGreen].yellow == 0 and self.currentYellow == 1:
            self.currentYellow = 0
            self.signals[self.currentGreen].green = random.randint(self.randomGreenSignalTimerRange[0], self.randomGreenSignalTimerRange[1])
            self.signals[self.currentGreen].red = self.defaultRed
            
            self.currentGreen = self.nextGreen
            self.nextGreen = (self.currentGreen + 1) % self.noOfSignals
            self.signals[self.nextGreen].red = self.signals[self.currentGreen].yellow + self.signals[self.currentGreen].green

        for i in range(self.noOfSignals):
            if i != self.currentGreen and self.signals[i].red > 0:
                self.signals[i].red -= 1
    
    def generateVehicles(self):
        vehicle_type = self.vehicleTypes[random.choice(self.allowedVehicleTypesList)]
        lane_number = random.randint(1, 2)
        temp = random.randint(0, 99)
        direction_number = 0
        dist = self.scenario_dist
        if temp < dist[0]: direction_number = 0
        elif temp < dist[1]: direction_number = 1
        elif temp < dist[2]: direction_number = 2
        else: direction_number = 3
        Vehicle(self, lane_number, vehicle_type, direction_number, self.directionNumbers[direction_number])

    def draw_checkpoints(self, surface):
        RED = (255, 0, 0)
        pygame.draw.line(surface, RED, (self.checkpoints['right'], 340), (self.checkpoints['right'], 435), 3)
        pygame.draw.line(surface, RED, (self.checkpoints['left'], 440), (self.checkpoints['left'], 535), 3)

    def draw(self, surface, assets):
        surface.blit(assets['background'], (0, 0))
        self.draw_checkpoints(surface)
        
        for i in range(self.noOfSignals):
            if self.emergency_mode_active and self.emergency_yellow_timer > 0 and (i == self.original_current_green or i == self.emergency_target_green):
                surface.blit(assets['yellowSignal'], self.signalCoods[i])
                text_surf = assets['font'].render(str(self.emergency_yellow_timer), True, (255,255,255), (0,0,0))
                surface.blit(text_surf, self.signalTimerCoods[i])
                continue
            
            if i == self.currentGreen:
                if self.currentYellow == 1:
                    self.signals[i].signalText = self.signals[i].yellow
                    surface.blit(assets['yellowSignal'], self.signalCoods[i])
                else:
                    self.signals[i].signalText = self.signals[i].green
                    surface.blit(assets['greenSignal'], self.signalCoods[i])
            else:
                self.signals[i].signalText = self.signals[i].red if self.signals[i].red <= 10 else "---"
                surface.blit(assets['redSignal'], self.signalCoods[i])
            
            text_surf = assets['font'].render(str(self.signals[i].signalText), True, (255, 255, 255), (0,0,0))
            surface.blit(text_surf, self.signalTimerCoods[i])

        for i in range(self.noOfSignals):
            displayText = sum(len(self.vehicles[self.directionNumbers[i]][lane]) for lane in range(3))
            vehicleCountTexts = assets['font'].render(str(displayText), True, (0,0,0), (255,255,255))
            surface.blit(vehicleCountTexts, self.vehicleCountCoods[i])

        time_text = assets['font'].render(f"Time: {self.timeElapsed}", True, (0,0,0), (255,255,255))
        surface.blit(time_text, (1100, 50))
        self.simulation_sprites.draw(surface)

class TrafficSignal:
    def __init__(self, red, yellow, green):
        self.red = red; self.yellow = yellow; self.green = green; self.signalText = ""

class Vehicle(pygame.sprite.Sprite):
    def __init__(self, simulation, lane, vehicleClass, direction_number, direction):
        pygame.sprite.Sprite.__init__(self)
        self.simulation = simulation
        self.lane = lane
        self.vehicleClass = vehicleClass
        self.speed = self.simulation.speeds[vehicleClass]
        self.direction_number = direction_number
        self.direction = direction
        self.crossed = 0
        
        # --- UNIFIED AND CORRECTED IMAGE LOADING LOGIC ---
        path = f"images/{direction}/{vehicleClass}.png"
        try:
            self.image = pygame.image.load(path)
        except pygame.error:
            print(f"Error: Image not found at '{path}'. Using a default car image instead.")
            fallback_path = f"images/{direction}/car.png"
            self.image = pygame.image.load(fallback_path)
        
        self.rect = self.image.get_rect()
        vehicles_in_lane = self.simulation.vehicles[direction][lane]

        # Bug fix for vehicle spawning
        if len(vehicles_in_lane) > 0:
            last_vehicle = vehicles_in_lane[-1]
            if direction == 'right':
                self.x = last_vehicle.rect.left - self.rect.width - self.simulation.stoppingGap
                self.y = last_vehicle.rect.y
            elif direction == 'left':
                self.x = last_vehicle.rect.right + self.simulation.stoppingGap
                self.y = last_vehicle.rect.y
            elif direction == 'down':
                self.y = last_vehicle.rect.top - self.rect.height - self.simulation.stoppingGap
                self.x = last_vehicle.rect.x
            elif direction == 'up':
                self.y = last_vehicle.rect.bottom + self.simulation.stoppingGap
                self.x = last_vehicle.rect.x
        else:
            self.x = self.simulation.x[direction][lane]
            self.y = self.simulation.y[direction][lane]
            
        self.rect.topleft = (self.x, self.y)

        self.simulation.vehicles[direction][lane].append(self)
        self.simulation.simulation_sprites.add(self)

    def update(self):
        sim = self.simulation
        
        if self.vehicleClass == 'ev' and not sim.emergency_mode_active:
            crossed = False
            if self.direction == 'right' and self.rect.right > sim.checkpoints['right']: crossed = True
            elif self.direction == 'left' and self.rect.left < sim.checkpoints['left']: crossed = True
            if crossed:
                sim.trigger_emergency_mode(self.direction_number)

        vehicles_in_lane = sim.vehicles[self.direction][self.lane]
        try:
            current_index = vehicles_in_lane.index(self)
        except ValueError:
            return

        is_light_green = (sim.currentGreen == self.direction_number and sim.currentYellow == 0)
        if self.vehicleClass == 'ev':
            is_light_green = True

        can_move_to_stopline = (
            (self.direction == 'right' and self.rect.right < sim.stopLines['right']) or
            (self.direction == 'left' and self.rect.left > sim.stopLines['left']) or
            (self.direction == 'down' and self.rect.bottom < sim.stopLines['down']) or
            (self.direction == 'up' and self.rect.top > sim.stopLines['up'])
        )

        should_move = is_light_green or can_move_to_stopline or self.crossed == 1
        
        if current_index > 0:
            prev_vehicle = vehicles_in_lane[current_index - 1]
            if (self.direction == 'right' and self.rect.right >= prev_vehicle.rect.left - self.simulation.movingGap) or \
               (self.direction == 'left' and self.rect.left <= prev_vehicle.rect.right + self.simulation.movingGap) or \
               (self.direction == 'down' and self.rect.bottom >= prev_vehicle.rect.top - self.simulation.movingGap) or \
               (self.direction == 'up' and self.rect.top <= prev_vehicle.rect.bottom + self.simulation.movingGap):
                should_move = False
        
        if should_move:
            if self.direction == 'right': self.rect.x += self.speed
            elif self.direction == 'left': self.rect.x -= self.speed
            elif self.direction == 'down': self.rect.y += self.speed
            elif self.direction == 'up': self.rect.y -= self.speed

        if self.crossed == 0 and (
            (self.direction == 'right' and self.rect.left > sim.stopLines[self.direction]) or
            (self.direction == 'left' and self.rect.right < sim.stopLines[self.direction]) or
            (self.direction == 'down' and self.rect.top > sim.stopLines[self.direction]) or
            (self.direction == 'up' and self.rect.bottom < sim.stopLines[self.direction])
        ):
            self.crossed = 1
            sim.vehicles_passed += 1
        
        if self.rect.left > 1400 or self.rect.right < 0 or self.rect.top > 800 or self.rect.bottom < 0:
            vehicles_in_lane.remove(self)
            self.kill()

def draw_dashboard(surface, simulations, assets):
    y_offset = 20
    GREEN, RED, YELLOW, WHITE, GRAY = (0,200,0), (200,0,0), (220,220,0), (255,255,255), (150,150,150)
    for i, sim in enumerate(simulations):
        title_text = f"Signal {i+1} ({sim.scenario_name.replace('_', ' ').title()})"
        title_surf = assets['dash_font_bold'].render(title_text, True, WHITE)
        surface.blit(title_surf, (10, y_offset)); y_offset += 30
        for lane_id in range(sim.noOfSignals):
            direction = sim.directionNumbers[lane_id]; lane_name = direction.title()
            status_text, status_color = ("RED", RED)
            if lane_id == sim.currentGreen: status_text, status_color = ("YELLOW", YELLOW) if sim.currentYellow == 1 else ("GREEN", GREEN)
            lane_info_text = f"  Lane {lane_id+1} ({lane_name}):"
            lane_info_surf = assets['dash_font'].render(lane_info_text, True, GRAY)
            surface.blit(lane_info_surf, (15, y_offset))
            status_surf = assets['dash_font_bold'].render(status_text, True, status_color)
            surface.blit(status_surf, (200, y_offset)); y_offset += 22
            if status_text == "RED":
                vehicle_count = sum(len(sub_lane) for sub_lane in sim.vehicles[direction].values())
                total_text = f"    - Total Vehicles: {vehicle_count}"
                total_surf = assets['dash_font'].render(total_text, True, WHITE)
                surface.blit(total_surf, (20, y_offset)); y_offset += 20
                standing_text = f"    - Vehicles Standing: {vehicle_count}"
                standing_surf = assets['dash_font'].render(standing_text, True, WHITE)
                surface.blit(standing_surf, (20, y_offset)); y_offset += 20
        passed_text = f"Total Vehicles Passed: {sim.vehicles_passed}"
        passed_surf = assets['dash_font_bold'].render(passed_text, True, GREEN)
        surface.blit(passed_surf, (15, y_offset)); y_offset += 35

def main():
    pygame.init()
    
    SCENARIOS = { "heavy_horizontal": [40, 50, 90, 100], "heavy_vertical": [10, 50, 60, 100], "heavy_right": [60, 70, 85, 100], "default": [25, 50, 75, 100] }
    scenarios_to_run = list(SCENARIOS.keys())

    SIM_WIDTH, SIM_HEIGHT = 1400, 800; QUADRANT_WIDTH, QUADRANT_HEIGHT = 700, 400
    DASHBOARD_WIDTH = 350; SCREEN_WIDTH, SCREEN_HEIGHT = (QUADRANT_WIDTH*2) + DASHBOARD_WIDTH, QUADRANT_HEIGHT*2
    
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Multi-Intersection Simulation with Live Dashboard")
    sub_surface = pygame.Surface((SIM_WIDTH, SIM_HEIGHT))

    assets = {
        'background': pygame.image.load('images/intersection.png').convert(),
        'redSignal': pygame.image.load('images/signals/red.png'), 'yellowSignal': pygame.image.load('images/signals/yellow.png'),
        'greenSignal': pygame.image.load('images/signals/green.png'), 'font': pygame.font.Font(None, 40),
        'dash_font': pygame.font.SysFont('Arial', 16), 'dash_font_bold': pygame.font.SysFont('Arial', 16, bold=True)
    }

    simulations = [SimulationInstance(name, SCENARIOS[name]) for name in scenarios_to_run]
    quadrant_positions = [(0, 0), (QUADRANT_WIDTH, 0), (0, QUADRANT_HEIGHT), (QUADRANT_WIDTH, QUADRANT_HEIGHT)]
    dashboard_rect = pygame.Rect(QUADRANT_WIDTH*2, 0, DASHBOARD_WIDTH, SCREEN_HEIGHT)

    clock = pygame.time.Clock()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False

        for i, sim in enumerate(simulations):
            sim.update()
            sim.draw(sub_surface, assets) 
            scaled_surface = pygame.transform.scale(sub_surface, (QUADRANT_WIDTH, QUADRANT_HEIGHT))
            screen.blit(scaled_surface, quadrant_positions[i])

        screen.fill((20, 20, 30), dashboard_rect)
        draw_dashboard(screen.subsurface(dashboard_rect), simulations, assets)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()
 
if __name__ == '__main__':
    main()
