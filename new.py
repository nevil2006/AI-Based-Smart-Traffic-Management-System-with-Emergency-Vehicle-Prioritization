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

        self.speeds = {'car': 2.25, 'bus': 1.8, 'truck': 1.8, 'bike': 2.5}

        # Original coordinates
        self.x = {'right': [0, 0, 0], 'down': [755, 727, 697], 'left': [1400, 1400, 1400], 'up': [602, 627, 657]}
        # FIX: Y-coordinates adjusted for better horizontal alignment
        self.y = {'right': [348, 375, 405], 'down': [0, 0, 0], 'left': [498, 470, 440], 'up': [800, 800, 800]}

        self.vehicles = {'right': {0: [], 1: [], 2: []}, 'down': {0: [], 1: [], 2: []},
                         'left': {0: [], 1: [], 2: []}, 'up': {0: [], 1: [], 2: []}}
        
        self.vehicles_passed = 0

        self.vehicleTypes = {0: 'car', 1: 'bus', 2: 'truck', 3: 'bike'}
        self.directionNumbers = {0: 'right', 1: 'down', 2: 'left', 3: 'up'}

        # Coordinates for drawing elements
        self.signalCoods = [(530, 230), (810, 230), (810, 570), (530, 570)]
        self.signalTimerCoods = [(530, 210), (810, 210), (810, 550), (530, 550)]
        self.vehicleCountCoods = [(480,210),(880,210),(880,550),(480,550)]

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
        
        self.initialize_signals()

    def initialize_signals(self):
        minTime = self.randomGreenSignalTimerRange[0]
        maxTime = self.randomGreenSignalTimerRange[1]
        if self.randomGreenSignalTimer:
            ts1 = TrafficSignal(0, self.defaultYellow, random.randint(minTime, maxTime))
            self.signals.append(ts1)
            ts2 = TrafficSignal(ts1.red + ts1.yellow + ts1.green, self.defaultYellow, random.randint(minTime, maxTime))
            self.signals.append(ts2)
            ts3 = TrafficSignal(self.defaultRed, self.defaultYellow, random.randint(minTime, maxTime))
            self.signals.append(ts3)
            ts4 = TrafficSignal(self.defaultRed, self.defaultYellow, random.randint(minTime, maxTime))
            self.signals.append(ts4)

    def update(self):
        current_time = time.time()
        
        if current_time - self.last_signal_update_time >= 1.0:
            self.update_signal_timers()
            self.last_signal_update_time = current_time
            self.timeElapsed += 1

        if current_time - self.last_vehicle_spawn_time >= 1.0 + random.uniform(-0.5, 0.5):
            self.generateVehicles()
            self.last_vehicle_spawn_time = current_time

        self.simulation_sprites.update()

    def update_signal_timers(self):
        if self.signals[self.currentGreen].green > 0 and self.currentYellow == 0:
            self.signals[self.currentGreen].green -= 1
        elif self.signals[self.currentGreen].green == 0 and self.currentYellow == 0:
            self.currentYellow = 1
        
        if self.signals[self.currentGreen].yellow > 0 and self.currentYellow == 1:
            self.signals[self.currentGreen].yellow -= 1
        elif self.signals[self.currentGreen].yellow == 0 and self.currentYellow == 1:
            self.currentYellow = 0
            if self.randomGreenSignalTimer:
                self.signals[self.currentGreen].green = random.randint(self.randomGreenSignalTimerRange[0], self.randomGreenSignalTimerRange[1])
            self.signals[self.currentGreen].yellow = self.defaultYellow
            self.signals[self.currentGreen].red = self.defaultRed
            
            self.currentGreen = self.nextGreen
            self.nextGreen = (self.currentGreen + 1) % self.noOfSignals
            self.signals[self.nextGreen].red = self.signals[self.currentGreen].yellow + self.signals[self.currentGreen].green

        for i in range(self.noOfSignals):
            if i != self.currentGreen:
                self.signals[i].red -= 1
    
    def generateVehicles(self):
        vehicle_type_num = random.choice(self.allowedVehicleTypesList)
        vehicle_type = self.vehicleTypes[vehicle_type_num]
        lane_number = random.randint(1, 2)
        
        temp = random.randint(0, 99)
        direction_number = 0
        dist = self.scenario_dist
        if temp < dist[0]: direction_number = 0
        elif temp < dist[1]: direction_number = 1
        elif temp < dist[2]: direction_number = 2
        else: direction_number = 3
        
        Vehicle(self, lane_number, vehicle_type, direction_number, self.directionNumbers[direction_number])
    
    def draw(self, surface, assets):
        surface.blit(assets['background'], (0, 0))
        
        for i in range(self.noOfSignals):
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
        
        path = f"images/{direction}/{vehicleClass}.png"
        self.image = pygame.image.load(path)
        self.rect = self.image.get_rect()
        
        vehicles_in_lane = self.simulation.vehicles[direction][lane]

        # FIX: Set spawn position behind the last car to prevent pile-ups
        if len(vehicles_in_lane) > 0:
            last_vehicle = vehicles_in_lane[-1]
            if direction == 'right':
                self.x = last_vehicle.x - last_vehicle.rect.width - self.simulation.stoppingGap
                self.y = last_vehicle.y
            elif direction == 'left':
                self.x = last_vehicle.x + last_vehicle.rect.width + self.simulation.stoppingGap
                self.y = last_vehicle.y
            elif direction == 'down':
                self.y = last_vehicle.y - last_vehicle.rect.height - self.simulation.stoppingGap
                self.x = last_vehicle.x
            elif direction == 'up':
                self.y = last_vehicle.y + last_vehicle.rect.height + self.simulation.stoppingGap
                self.x = last_vehicle.x
        else:
            # First car in the lane spawns at the default edge position
            self.x = self.simulation.x[direction][lane]
            self.y = self.simulation.y[direction][lane]

        self.simulation.vehicles[direction][lane].append(self)
        self.rect.topleft = (self.x, self.y)
        self.stop = self.simulation.defaultStop[direction]
        self.simulation.simulation_sprites.add(self)

    def update(self):
        sim = self.simulation
        vehicles_in_lane = sim.vehicles[self.direction][self.lane]
        
        try:
            current_index = vehicles_in_lane.index(self)
        except ValueError:
            return 

        is_light_green = (sim.currentGreen == self.direction_number and sim.currentYellow == 0)

        gap_is_clear = True
        if current_index > 0:
            prev_vehicle = vehicles_in_lane[current_index - 1]
            if self.direction == 'right' and self.x + self.rect.width >= prev_vehicle.x - sim.movingGap: gap_is_clear = False
            elif self.direction == 'left' and self.x <= prev_vehicle.x + prev_vehicle.rect.width + sim.movingGap: gap_is_clear = False
            elif self.direction == 'down' and self.y + self.rect.height >= prev_vehicle.y - sim.movingGap: gap_is_clear = False
            elif self.direction == 'up' and self.y <= prev_vehicle.y + prev_vehicle.rect.height + sim.movingGap: gap_is_clear = False

        should_move = False
        if self.crossed == 1:
            should_move = True
        elif is_light_green:
            should_move = True
        else:
            if (self.direction == 'right' and self.x + self.rect.width < self.stop) or \
               (self.direction == 'left' and self.x > self.stop) or \
               (self.direction == 'down' and self.y + self.rect.height < self.stop) or \
               (self.direction == 'up' and self.y > self.stop):
                should_move = True
        
        if should_move and gap_is_clear:
            if self.direction == 'right': self.x += self.speed
            elif self.direction == 'left': self.x -= self.speed
            elif self.direction == 'down': self.y += self.speed
            elif self.direction == 'up': self.y -= self.speed

        if self.crossed == 0:
            if (self.direction == 'right' and self.x + self.rect.width > sim.stopLines[self.direction]) or \
               (self.direction == 'left' and self.x < sim.stopLines[self.direction]) or \
               (self.direction == 'down' and self.y + self.rect.height > sim.stopLines[self.direction]) or \
               (self.direction == 'up' and self.y < sim.stopLines[self.direction]):
                self.crossed = 1
                sim.vehicles_passed += 1
        
        if (self.x > 1400 or self.x < -400 or self.y > 800 or self.y < -400):
            vehicles_in_lane.remove(self)
            self.kill()

        self.rect.topleft = (self.x, self.y)


def draw_dashboard(surface, simulations, assets):
    y_offset = 20
    GREEN, RED, YELLOW, WHITE, GRAY = (0, 200, 0), (200, 0, 0), (220, 220, 0), (255, 255, 255), (150, 150, 150)

    for i, sim in enumerate(simulations):
        # --- Draw Title ---
        title_text = f"Signal {i+1} ({sim.scenario_name.replace('_', ' ').title()})"
        title_surf = assets['dash_font_bold'].render(title_text, True, WHITE)
        surface.blit(title_surf, (10, y_offset))
        y_offset += 30

        # --- Draw Lane Status and Vehicle Counts ---
        for lane_id in range(sim.noOfSignals):
            direction = sim.directionNumbers[lane_id]
            lane_name = direction.title()
            
            # Determine lane color
            status_text, status_color = ("RED", RED)
            if lane_id == sim.currentGreen:
                status_text, status_color = ("YELLOW", YELLOW) if sim.currentYellow == 1 else ("GREEN", GREEN)
            
            # Render and draw basic lane info
            lane_info_text = f"  Lane {lane_id+1} ({lane_name}):"
            lane_info_surf = assets['dash_font'].render(lane_info_text, True, GRAY)
            surface.blit(lane_info_surf, (15, y_offset))
            
            status_surf = assets['dash_font_bold'].render(status_text, True, status_color)
            surface.blit(status_surf, (200, y_offset))
            y_offset += 22

            # If the light is RED, show vehicle counts
            if status_text == "RED":
                # Calculate counts for this direction
                vehicle_count = sum(len(sub_lane) for sub_lane in sim.vehicles[direction].values())
                
                # Render and draw total vehicles in lane
                total_text = f"    - Total Vehicles: {vehicle_count}"
                total_surf = assets['dash_font'].render(total_text, True, WHITE)
                surface.blit(total_surf, (20, y_offset))
                y_offset += 20
                
                # Render and draw standing vehicles
                standing_text = f"    - Vehicles Standing: {vehicle_count}"
                standing_surf = assets['dash_font'].render(standing_text, True, WHITE)
                surface.blit(standing_surf, (20, y_offset))
                y_offset += 20

        # --- Draw Total Passed Vehicles ---
        passed_text = f"Total Vehicles Passed: {sim.vehicles_passed}"
        passed_surf = assets['dash_font_bold'].render(passed_text, True, GREEN)
        surface.blit(passed_surf, (15, y_offset))
        y_offset += 35


def main():
    pygame.init()
    
    SCENARIOS = {
        "heavy_horizontal": [40, 50, 90, 100], "heavy_vertical": [10, 50, 60, 100],
        "heavy_right": [60, 70, 85, 100], "default": [25, 50, 75, 100]
    }
    scenarios_to_run = list(SCENARIOS.keys())

    SIM_WIDTH, SIM_HEIGHT = 1400, 800
    QUADRANT_WIDTH, QUADRANT_HEIGHT = 700, 400
    DASHBOARD_WIDTH = 350
    SCREEN_WIDTH, SCREEN_HEIGHT = (QUADRANT_WIDTH * 2) + DASHBOARD_WIDTH, QUADRANT_HEIGHT * 2
    
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Multi-Intersection Simulation with Live Dashboard")

    sub_surface = pygame.Surface((SIM_WIDTH, SIM_HEIGHT))

    assets = {
        'background': pygame.image.load('images/intersection.png').convert(),
        'redSignal': pygame.image.load('images/signals/red.png'),
        'yellowSignal': pygame.image.load('images/signals/yellow.png'),
        'greenSignal': pygame.image.load('images/signals/green.png'),
        'font': pygame.font.Font(None, 40),
        'dash_font': pygame.font.SysFont('Arial', 16),
        'dash_font_bold': pygame.font.SysFont('Arial', 16, bold=True)
    }

    simulations = [SimulationInstance(name, SCENARIOS[name]) for name in scenarios_to_run]
    quadrant_positions = [(0, 0), (QUADRANT_WIDTH, 0), (0, QUADRANT_HEIGHT), (QUADRANT_WIDTH, QUADRANT_HEIGHT)]
    dashboard_rect = pygame.Rect(QUADRANT_WIDTH * 2, 0, DASHBOARD_WIDTH, SCREEN_HEIGHT)

    clock = pygame.time.Clock()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

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