import pandas as pd
import io
import sys
import numpy as np
import matplotlib.pyplot as plt
import random
import copy
import time
import os
import math
from collections import defaultdict
from contextlib import redirect_stdout

# Set matplotlib to use non-interactive backend
import matplotlib
matplotlib.use('Agg')  # This ensures plots can be saved without display

class AmbulanceRoutingSystem:
    def __init__(self):
        # Data structures
        self.hospitals = None  # Hospital locations and capacities
        self.gathering_points = None  # Gathering points with patient counts
        self.distances = None  # Distance matrix between all points
        self.ambulances = None  # Available ambulances
        
        # Mapping dictionaries to store names
        self.hospital_names = {}  # Maps hospital IDs to names
        self.gathering_point_names = {}  # Maps gathering point IDs to names
        
        # Solution data
        self.solution = {}  # Will hold final routes for each ambulance
        self.hospital_utilization = {}  # Track hospital capacity utilization
        self.response_times = []  # Track response times for patients
        self.unserved_patients = []  # Patients who couldn't be served
        
        # Constants
        self.GREEN_TREATMENT_TIME = 5  # Fixed treatment time for green patients in minutes
        self.CANDIDATE_LIST_SIZE = 1  # Number of closest patients to consider

        # Fitness calculation weights
        self.RED_WEIGHT = 5
        self.GREEN_WEIGHT = 1

        # Response time tracking for fitness
        self.red_response_times = []
        self.green_response_times = []

        #terminal output
        self.terminal_output = []

    def get_unique_filename(self, base_name):
        """
        base_name: örn. 'ambulance_routing_report.txt'
        Eğer aynı isimde bir dosya varsa, 'ambulance_routing_report1.txt', '...2.txt' şeklinde
        bir sonraki boş numaralı adı döndürür.
        """
        current_dir = os.getcwd()
        path = os.path.join(current_dir, base_name)
        if not os.path.exists(path):
            return path

        name, ext = os.path.splitext(base_name)
        i = 1
        while True:
            new_name = f"{name}{i}{ext}"
            new_path = os.path.join(current_dir, new_name)
            if not os.path.exists(new_path):
                return new_path
            i += 1

    def log_output(self, message):
        """Terminal çıktısını hem ekrana hem de listeye yazdır"""
        print(message)
        self.terminal_output.append(message)

    def capture_terminal_output(self, func, *args, **kwargs):
        """Bir fonksiyonun terminal çıktısını yakala"""
        # StringIO buffer oluştur
        output_buffer = io.StringIO()
        
        # stdout'u geçici olarak buffer'a yönlendir
        with redirect_stdout(output_buffer):
            result = func(*args, **kwargs)
        
        # Yakalanan çıktıyı al
        captured_output = output_buffer.getvalue()
        
        # Çıktıyı hem ekrana bas hem de sakla
        print(captured_output, end='')
        self.terminal_output.append(captured_output)
        
        return result
        
    def load_data(self, hospitals_file, gathering_points_file, ambulances_file):
        """Load data from CSV files"""
        # Load data with semicolon delimiter
        self.hospitals = pd.read_csv(hospitals_file, delimiter=';')
        self.gathering_points = pd.read_csv(gathering_points_file, delimiter=';')
        self.ambulances = pd.read_csv(ambulances_file, delimiter=';')
        
        # Create name mappings
        for _, hospital in self.hospitals.iterrows():
            self.hospital_names[hospital['id']] = hospital['name']
            
        for _, gp in self.gathering_points.iterrows():
            self.gathering_point_names[gp['id']] = gp['name']
        
        # Initialize hospital utilization
        for _, hospital in self.hospitals.iterrows():
            self.hospital_utilization[hospital['id']] = 0
            
        # Generate distances using Haversine formula
        self.generate_distances()
            
        print(f"Loaded data: {len(self.hospitals)} hospitals, {len(self.gathering_points)} gathering points, {len(self.ambulances)} ambulances")
        
    def haversine_distance(self, lat1, lon1, lat2, lon2):
        """Calculate the great-circle distance between two points using the Haversine formula"""
        # Convert latitude and longitude from degrees to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [float(lat1), float(lon1), float(lat2), float(lon2)])
        
        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        r = 6371  # Earth radius in kilometers
        
        return c * r
    
    def generate_distances(self):
        """Generate distance matrix between all locations using Haversine formula"""
        # Create empty list for distances
        distances_data = []
        
        # Calculate distances between hospitals and gathering points
        for _, hospital in self.hospitals.iterrows():
            h_id = hospital['id']
            h_lat = hospital['lat_hospital']
            h_lon = hospital['lon_hospital']
            
            # Hospital to hospital distances
            for _, other_hospital in self.hospitals.iterrows():
                if h_id != other_hospital['id']:
                    oh_id = other_hospital['id']
                    oh_lat = other_hospital['lat_hospital']
                    oh_lon = other_hospital['lon_hospital']
                    
                    dist = self.haversine_distance(h_lat, h_lon, oh_lat, oh_lon)
                    
                    distances_data.append({
                        'from_id': h_id,
                        'to_id': oh_id,
                        'from_type': 'hospital',
                        'to_type': 'hospital',
                        'distance': dist
                    })
            
            # Hospital to gathering point distances
            for _, gp in self.gathering_points.iterrows():
                gp_id = gp['id']
                gp_lat = gp['lat_gp']
                gp_lon = gp['lon_gp']
                
                dist = self.haversine_distance(h_lat, h_lon, gp_lat, gp_lon)
                
                distances_data.append({
                    'from_id': h_id,
                    'to_id': gp_id,
                    'from_type': 'hospital',
                    'to_type': 'gathering_point',
                    'distance': dist
                })
        
        # Calculate gathering point to gathering point distances
        for _, gp1 in self.gathering_points.iterrows():
            gp1_id = gp1['id']
            gp1_lat = gp1['lat_gp']
            gp1_lon = gp1['lon_gp']
            
            for _, gp2 in self.gathering_points.iterrows():
                if gp1_id != gp2['id']:
                    gp2_id = gp2['id']
                    gp2_lat = gp2['lat_gp']
                    gp2_lon = gp2['lon_gp']
                    
                    dist = self.haversine_distance(gp1_lat, gp1_lon, gp2_lat, gp2_lon)
                    
                    distances_data.append({
                        'from_id': gp1_id,
                        'to_id': gp2_id,
                        'from_type': 'gathering_point',
                        'to_type': 'gathering_point',
                        'distance': dist
                    })
        
        # Create DataFrame from distances data
        self.distances = pd.DataFrame(distances_data)
        print(f"Generated {len(self.distances)} distance pairs using Haversine formula")
        
    def get_distance(self, point1, point2):
        """Get distance between two points"""
        # If points are the same, distance is 0
        if point1 == point2:
            return 0
            
        # Find distance from distances dataframe
        distance = self.distances[(self.distances['from_id'] == point1) & 
                                 (self.distances['to_id'] == point2)]['distance'].values
        
        if len(distance) > 0:
            return distance[0]
        else:
            # If not found, check reverse direction (assuming symmetry)
            distance = self.distances[(self.distances['from_id'] == point2) & 
                                     (self.distances['to_id'] == point1)]['distance'].values
            if len(distance) > 0:
                return distance[0]
            else:
                print(f"Warning: No distance found between {point1} and {point2}")
                return 9999  # Large value to indicate no path
    
    def get_point_name(self, point_id):
        """Get name of a point (hospital or gathering point) by ID"""
        # Try hospital names first
        if point_id in self.hospital_names:
            return f"{point_id} ({self.hospital_names[point_id]})"
        # Then try gathering point names
        elif point_id in self.gathering_point_names:
            return f"{point_id} ({self.gathering_point_names[point_id]})"
        else:
            return f"{point_id}"  # Return just the ID if name not found
    
    def get_closest_hospital(self, point_id):
        """Find closest hospital to a gathering point"""
        closest_hospital = None
        min_distance = float('inf')
        
        for _, hospital in self.hospitals.iterrows():
            if self.hospital_utilization[hospital['id']] < int(hospital['capacity']):
                distance = self.get_distance(point_id, hospital['id'])
                if distance < min_distance:
                    min_distance = distance
                    closest_hospital = hospital['id']
        
        return closest_hospital, min_distance
    
    def get_closest_patients(self, point_id):
        """Get list of unserved patients closest to a point
        Returns a list of candidates (size defined by CANDIDATE_LIST_SIZE) 
        sorted by distance"""
        unserved_patients = []
        
        for _, gp in self.gathering_points.iterrows():
            if int(gp['red_patients']) > 0 or int(gp['green_patients']) > 0:
                distance = self.get_distance(point_id, gp['id'])
                unserved_patients.append({
                    'id': gp['id'],
                    'distance': distance,
                    'red': int(gp['red_patients']),
                    'green': int(gp['green_patients'])
                })
        
        # Sort by distance
        unserved_patients.sort(key=lambda x: x['distance'])
        
        # Return up to CANDIDATE_LIST_SIZE closest patients
        return unserved_patients[:min(self.CANDIDATE_LIST_SIZE, len(unserved_patients))]
    
    def run_heuristic_algorithm(self):
        """Implement the heuristic algorithm with modified patient selection"""
        start_time = time.time()
        
        # Initialize solution
        self.solution = {amb['id']: [] for _, amb in self.ambulances.iterrows()}
        
        # Track ambulance status
        ambulance_status = {amb['id']: {'location': amb['initial_hospital'], 
                                       'available_time': 0,
                                       'is_hospital': True} 
                           for _, amb in self.ambulances.iterrows()}
        
        # Continue until all patients are visited or no feasible solution exists
        all_patients_visited = False
        iteration = 0
        max_iterations = 10000  # Safety to prevent infinite loops
        
        while not all_patients_visited and iteration < max_iterations:
            iteration += 1
            
            # Find ambulance that becomes available earliest
            available_ambulance = None
            earliest_time = float('inf')
            
            for amb_id, status in ambulance_status.items():
                if status['available_time'] < earliest_time:
                    earliest_time = status['available_time']
                    available_ambulance = amb_id
            
            if available_ambulance is None:
                break  # No ambulances available
                
            # Get ambulance current location and time
            current_location = ambulance_status[available_ambulance]['location']
            current_time = ambulance_status[available_ambulance]['available_time']
            is_at_hospital = ambulance_status[available_ambulance]['is_hospital']
            
            # Get list of candidate unserved patients closest to current location
            closest_patients = self.get_closest_patients(current_location)
            
            # If no patients left, mark this ambulance as done
            if not closest_patients:
                # If not at hospital, return to the closest hospital
                if not is_at_hospital:
                    closest_hosp, dist = self.get_closest_hospital(current_location)
                    if closest_hosp is not None:
                        travel_time = dist / 50 * 60  # Assuming 50 km/h average speed, convert to minutes
                        self.solution[available_ambulance].append({
                            'from': current_location,
                            'from_name': self.get_point_name(current_location),
                            'to': closest_hosp,
                            'to_name': self.get_point_name(closest_hosp),
                            'time': current_time,
                            'action': 'return_to_hospital',
                            'distance': dist,
                            'duration': travel_time  # Record travel duration
                        })
                
                ambulance_status[available_ambulance]['available_time'] = float('inf')
                continue
            
            # Randomly select a patient from the candidate list
            patient = random.choice(closest_patients)
            patient_location = patient['id']
            
            # Initialize new route
            new_route = []
            
            # Travel to patient
            dist_to_patient = self.get_distance(current_location, patient_location)
            travel_time = dist_to_patient / 50 * 60  # Assuming 50 km/h, converting to minutes
            
            # Determine patient type
            patient_type = 'red' if patient['red'] > 0 else 'green'

            if patient_type == 'red':
                # Kırmızı hasta için pickup
                new_route.append({
                    'from': current_location,
                    'from_name': self.get_point_name(current_location),
                    'to': patient_location,
                    'to_name': self.get_point_name(patient_location),
                    'time': current_time,
                    'action': 'pickup',
                    'distance': dist_to_patient,
                    'patient_type': patient_type,
                    'duration': travel_time
                })
                current_time += travel_time
            else:
                # Yeşil hasta için visit_and_treat (tek seferde - yol + tedavi)
                treatment_time = self.GREEN_TREATMENT_TIME
                total_duration = travel_time + treatment_time
                
                new_route.append({
                    'from': current_location,
                    'from_name': self.get_point_name(current_location),
                    'to': patient_location,
                    'to_name': self.get_point_name(patient_location),
                    'time': current_time,
                    'action': 'visit_and_treat',  # Tek aksiyon
                    'distance': dist_to_patient,
                    'patient_type': patient_type,
                    'duration': total_duration  # Yol + tedavi süresi
                })
                current_time += total_duration

            # Update gathering point (decrement patient count)
            for i, gp in self.gathering_points.iterrows():
                if gp['id'] == patient_location:
                    if patient_type == 'red':
                        self.gathering_points.at[i, 'red_patients'] = int(gp['red_patients']) - 1
                    else:
                        self.gathering_points.at[i, 'green_patients'] = int(gp['green_patients']) - 1
                    break

            # If red code patient, must go directly to nearest hospital
            if patient_type == 'red':
                closest_hosp, dist_to_hospital = self.get_closest_hospital(patient_location)
                
                if closest_hosp is None:
                    # All hospitals at capacity, revert changes
                    for i, gp in self.gathering_points.iterrows():
                        if gp['id'] == patient_location:
                            self.gathering_points.at[i, 'red_patients'] = int(gp['red_patients']) + 1
                            break
                    
                    self.unserved_patients.append({
                        'location': patient_location,
                        'location_name': self.get_point_name(patient_location),
                        'type': 'red',
                        'reason': 'no_hospital_capacity'
                    })
                    
                    # Mark ambulance as unavailable for a while
                    ambulance_status[available_ambulance]['available_time'] = current_time + 30  # 30 minute penalty
                    continue
                
                travel_time = dist_to_hospital / 50 * 60  # Convert to minutes
                
                new_route.append({
                    'from': patient_location,
                    'from_name': self.get_point_name(patient_location),
                    'to': closest_hosp,
                    'to_name': self.get_point_name(closest_hosp),
                    'time': current_time,
                    'action': 'dropoff',
                    'distance': dist_to_hospital,
                    'patient_type': patient_type,
                    'duration': travel_time  # Record travel duration
                })
                
                current_time += travel_time
                
                # Update hospital utilization
                self.hospital_utilization[closest_hosp] += 1
                
                # Record response time
                self.response_times.append(current_time)
                self.red_response_times.append(current_time)
                
                # Update ambulance status
                ambulance_status[available_ambulance]['location'] = closest_hosp
                ambulance_status[available_ambulance]['available_time'] = current_time
                ambulance_status[available_ambulance]['is_hospital'] = True
                
            else:  # Green patient - ambulance stays at gathering point
                # Record response time for green patient
                self.response_times.append(current_time)
                self.green_response_times.append(current_time)
                
                # After treating green patient, ambulance stays at the gathering point
                ambulance_status[available_ambulance]['location'] = patient_location
                ambulance_status[available_ambulance]['available_time'] = current_time
                ambulance_status[available_ambulance]['is_hospital'] = False
                            
            # Add route to solution
            self.solution[available_ambulance].extend(new_route)
            
            # Check if all patients have been visited
            total_patients = self.gathering_points['red_patients'].astype(int).sum() + self.gathering_points['green_patients'].astype(int).sum()
            if total_patients == 0:
                all_patients_visited = True
        
        # When all patients are served, return all ambulances to hospitals
        for amb_id, status in ambulance_status.items():
            # Skip ambulances that are already marked as unavailable (float('inf'))
            if status['available_time'] == float('inf'):
                continue
            
            # If not at hospital, return to closest hospital
            if not status['is_hospital']:
                current_location = status['location']
                current_time = status['available_time']
                
                closest_hosp, dist = self.get_closest_hospital(current_location)
                if closest_hosp is not None:
                    travel_time = dist / 50 * 60  # Convert to minutes
                    self.solution[amb_id].append({
                        'from': current_location,
                        'from_name': self.get_point_name(current_location),
                        'to': closest_hosp,
                        'to_name': self.get_point_name(closest_hosp),
                        'time': current_time,
                        'action': 'return_to_hospital',
                        'distance': dist,
                        'duration': travel_time
                    })
        
        # Calculate execution time
        execution_time = time.time() - start_time
        print(f"Algorithm execution time: {execution_time:.2f} seconds")
        print(f"Total iterations: {iteration}")
        
        if iteration == max_iterations:
            print("Warning: Reached maximum iterations without finding complete solution")
        
        return self.solution
    
    def calculate_fitness(self):
        """Calculate fitness value based on last served patients"""
        last_red_time = max(self.red_response_times) if self.red_response_times else 0
        last_green_time = max(self.green_response_times) if self.green_response_times else 0

        fitness = (self.RED_WEIGHT * last_red_time) + (self.GREEN_WEIGHT * last_green_time)
        return fitness
    
    def calculate_metrics(self):
        """Calculate performance metrics"""
        metrics = {}
        
        # Total travel distance
        total_distance = 0
        for amb_id, routes in self.solution.items():
            for route in routes:
                total_distance += route['distance']
        metrics['total_distance'] = total_distance
        
        # Response times
        if self.response_times:
            metrics['avg_response_time'] = sum(self.response_times) / len(self.response_times)
            metrics['max_response_time'] = max(self.response_times)
        else:
            metrics['avg_response_time'] = 0
            metrics['max_response_time'] = 0
        
        # Ambulance utilization
        total_route_count = sum(len(routes) for routes in self.solution.values())
        metrics['ambulance_utilization'] = total_route_count / len(self.ambulances) if len(self.ambulances) > 0 else 0
        
        # Hospital capacity utilization
        total_capacity = sum(int(hospital['capacity']) for _, hospital in self.hospitals.iterrows())
        total_utilization = sum(self.hospital_utilization.values())
        metrics['hospital_utilization'] = total_utilization / total_capacity if total_capacity > 0 else 0
        
        # Calculate separate metrics for red and green patients
        red_pickup_routes = 0
        green_treatment_routes = 0
        
        for amb_id, routes in self.solution.items():
            for route in routes:
                if route.get('action') == 'pickup' and route.get('patient_type') == 'red':
                    red_pickup_routes += 1
                elif route.get('action') == 'visit_and_treat' and route.get('patient_type') == 'green':

                    green_treatment_routes += 1
        
        metrics['red_patients_served'] = red_pickup_routes  # Each red patient needs pickup
        metrics['green_patients_served'] = green_treatment_routes  # Each green patient needs one visit
        
        # Unserved patients
        metrics['unserved_patients'] = len(self.unserved_patients)
        
        # Calculate treatment statistics
        total_treatment_time = 0
        treatment_count = 0
        
        for amb_id, routes in self.solution.items():
            for route in routes:
                if route.get('action') == 'treatment':
                    total_treatment_time += route.get('duration', 0)
                    treatment_count += 1
        
        metrics['avg_treatment_time'] = total_treatment_time / treatment_count if treatment_count > 0 else 0
        metrics['total_treatments'] = treatment_count

        # Fitness value
        metrics['fitness_value'] = self.calculate_fitness()
        
        return metrics
    
    def visualize_solution(self):
        """Generate visualizations for the solution with improved clarity"""
        # Use real coordinates for visualization with larger figure size
        plt.figure(figsize=(20, 16))  # Increased size for better visibility
        
        # Get coordinate bounds to focus on the relevant area
        lon_coords = []
        lat_coords = []
        
        # Collect all coordinates
        for _, hospital in self.hospitals.iterrows():
            lon_coords.append(float(hospital['lon_hospital']))
            lat_coords.append(float(hospital['lat_hospital']))
        
        for _, gp in self.gathering_points.iterrows():
            if 'lon_gp' in gp and 'lat_gp' in gp:
                lon_coords.append(float(gp['lon_gp']))
                lat_coords.append(float(gp['lat_gp']))
        
        # Calculate bounds with some padding
        min_lon, max_lon = min(lon_coords), max(lon_coords)
        min_lat, max_lat = min(lat_coords), max(lat_coords)
        
        # Add padding (5% of range)
        lon_padding = (max_lon - min_lon) * 0.05
        lat_padding = (max_lat - min_lat) * 0.05
        
        # Set the axis limits with padding
        plt.xlim(min_lon - lon_padding, max_lon + lon_padding)
        plt.ylim(min_lat - lat_padding, max_lat + lat_padding)
        
        # Plot hospitals with larger markers
        for _, hospital in self.hospitals.iterrows():
            x = float(hospital['lon_hospital'])
            y = float(hospital['lat_hospital'])
            plt.scatter(x, y, c='blue', marker='s', s=200, label='Hospital' if _ == 0 else "")
            plt.text(x+0.001, y+0.001, f"H{hospital['id']}", fontsize=10, fontweight='bold')
        
        # Plot gathering points
        for _, gp in self.gathering_points.iterrows():
            if 'lon_gp' in gp and 'lat_gp' in gp:
                x = float(gp['lon_gp'])
                y = float(gp['lat_gp'])
                plt.scatter(x, y, c='red', marker='^', s=100, label='Gathering Point' if _ == 0 else "")
                
                # Only show labels for gathering points that are part of a route to reduce clutter
                should_label = False
                for amb_id, routes in self.solution.items():
                    for route in routes:
                        if route['from'] == gp['id'] or route['to'] == gp['id']:
                            should_label = True
                            break
                    if should_label:
                        break
                
                if should_label:
                    plt.text(x+0.001, y+0.001, f"G{gp['id']}", fontsize=9)
        
        # Define distinct colors for ambulance routes - using vibrant colors like in the original code
        colors = [
            'green', 'red', 'blue', 'purple', 'orange', 'brown', 'pink', 'cyan', 
            'magenta', 'lime', 'teal', 'lavender', 'yellow', 'darkblue', 'gold',
            'salmon', 'darkgreen', 'indigo', 'darkred', 'olive', 'navy', 'maroon',
            'crimson', 'darkviolet', 'mediumseagreen', 'chocolate', 'steelblue',
            'tomato', 'darkorange', 'cornflowerblue', 'yellowgreen', 'hotpink',
            'turquoise', 'goldenrod', 'darkgoldenrod', 'mediumorchid', 'slategrey'
        ]
        
        # For more ambulances than colors, cycle through colors but vary the linestyle
        linestyles = ['-', '--', '-.', ':']
        
        # Plot routes with different colors for each ambulance
        amb_ids = sorted(self.solution.keys())
        for i, amb_id in enumerate(amb_ids):
            routes = self.solution[amb_id]
            color_idx = i % len(colors)
            linestyle_idx = i // len(colors) % len(linestyles)
            
            color = colors[color_idx]
            linestyle = linestyles[linestyle_idx]
            label_used = False
            
            for route in routes:
                # Skip treatment actions as they don't involve travel
                if route.get('action') == 'treatment' and route.get('from') == route.get('to'):
                    continue
                
                # Get coordinates for from location
                from_x, from_y = self.get_coordinates(route['from'])
                
                # Get coordinates for to location
                to_x, to_y = self.get_coordinates(route['to'])
                
                if from_x is not None and to_x is not None:
                    if not label_used:
                        plt.plot([from_x, to_x], [from_y, to_y], c=color, linestyle=linestyle, 
                                linewidth=1.5, label=f"Ambulance {amb_id}")
                        label_used = True
                    else:
                        plt.plot([from_x, to_x], [from_y, to_y], c=color, linestyle=linestyle, 
                                linewidth=1.5)
                    
                    # Calculate arrow position (place it at 70% of the path)
                    arrow_x = from_x + 0.7*(to_x-from_x)
                    arrow_y = from_y + 0.7*(to_y-from_y)
                    dx = 0.3*(to_x-from_x)
                    dy = 0.3*(to_y-from_y)
                    
                    # Only add arrows for longer paths to reduce clutter
                    path_length = self.haversine_distance(from_y, from_x, to_y, to_x)
                    if path_length > 0.3:  # Only show arrows for paths longer than 300m
                        # Use the same color as the line for the arrow
                        plt.arrow(arrow_x, arrow_y, dx, dy, head_width=0.001, 
                                head_length=0.002, fc=color, ec=color)
        
        # Add grid with lighter color and thinner lines
        plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        # Customize the plot
        plt.title('Ambulance Routing Solution', fontsize=16)
        plt.xlabel('Longitude', fontsize=12)
        plt.ylabel('Latitude', fontsize=12)
        
        # Put the legend outside the plot to avoid covering data
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
        
        # Adjust layout
        plt.tight_layout()
        
        # Get current directory to save output
        current_dir = os.getcwd()
        output_path = self.get_unique_filename('ambulance_routing_solution.png')
        
        try:
            # Save with higher DPI for better quality
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Successfully saved solution visualization to: {output_path}")
        except Exception as e:
            print(f"Error saving solution visualization: {e}")
        
        plt.close()
        
        # Also create a simplified version showing only hospital and gathering points
        self.create_simplified_map()
        
        # Create bar chart for metrics
        metrics = self.calculate_metrics()
        
        try:
            fig, ax = plt.subplots(2, 2, figsize=(15, 12))
            
            # Hospital utilization
            hosp_util = []
            hosp_names = []
            for hosp_id, util in self.hospital_utilization.items():
                hosp_util.append(util)
                hosp_names.append(f"{hosp_id}")
            
            ax[0, 0].bar(hosp_names, hosp_util)
            ax[0, 0].set_title('Hospital Utilization', fontsize=14)
            ax[0, 0].set_ylabel('Patients', fontsize=12)
            ax[0, 0].set_xlabel('Hospital ID', fontsize=12)
            
            # Ambulance route counts
            amb_routes = []
            amb_names = []
            for amb_id, routes in self.solution.items():
                amb_routes.append(len(routes))
                amb_names.append(f"{amb_id}")
            
            ax[0, 1].bar(amb_names, amb_routes)
            ax[0, 1].set_title('Ambulance Route Counts', fontsize=14)
            ax[0, 1].set_ylabel('Routes', fontsize=12)
            ax[0, 1].set_xlabel('Ambulance ID', fontsize=12)
            
            # Sample a subset of ambulances for better visibility if there are many
            if len(amb_names) > 20:
                sample_size = 20
                sample_step = len(amb_names) // sample_size
                ax[0, 1].set_xticks(range(0, len(amb_names), sample_step))
                ax[0, 1].set_xticklabels([amb_names[i] for i in range(0, len(amb_names), sample_step)])
                ax[0, 1].tick_params(axis='x', rotation=45)
            
            # Response time distribution
            if self.response_times:
                ax[1, 0].hist(self.response_times, bins=15, color='skyblue', edgecolor='black')
                ax[1, 0].set_title('Response Time Distribution', fontsize=14)
                ax[1, 0].set_xlabel('Response Time (minutes)', fontsize=12)
                ax[1, 0].set_ylabel('Frequency', fontsize=12)
            
            # Summary metrics
            metrics_display = [
                f"Total Distance: {metrics['total_distance']:.1f} km",
                f"Avg Response Time: {metrics['avg_response_time']:.1f} min",
                f"Max Response Time: {metrics['max_response_time']:.1f} min",
                f"Hospital Utilization: {metrics['hospital_utilization']*100:.1f}%",
                f"Red Patients Served: {metrics.get('red_patients_served', 0)}",
                f"Green Patients Served: {metrics.get('green_patients_served', 0)}",
                f"Total Treatments: {metrics.get('total_treatments', 0)}",
                f"Avg Treatment Time: {metrics.get('avg_treatment_time', 0):.1f} min",
                f"Unserved Patients: {metrics['unserved_patients']}"
            ]
            
            ax[1, 1].axis('off')
            y_pos = 0.9
            for metric in metrics_display:
                ax[1, 1].text(0.1, y_pos, metric, fontsize=14, fontweight='bold' if 'Patients Served' in metric else 'normal')
                y_pos -= 0.11
            
            plt.tight_layout()
            
            metrics_output_path = self.get_unique_filename('ambulance_routing_metrics.png')
            plt.savefig(metrics_output_path, dpi=300, bbox_inches='tight')
            print(f"Successfully saved metrics visualization to: {metrics_output_path}")
            plt.close()
            
        except Exception as e:
            print(f"Error creating or saving metrics visualization: {e}")

    def create_simplified_map(self):
        """Create a simplified map showing only hospitals and gathering points without routes"""
        plt.figure(figsize=(18, 15))
        
        # Get coordinate bounds
        lon_coords = []
        lat_coords = []
        
        # Collect all coordinates
        for _, hospital in self.hospitals.iterrows():
            lon_coords.append(float(hospital['lon_hospital']))
            lat_coords.append(float(hospital['lat_hospital']))
        
        for _, gp in self.gathering_points.iterrows():
            if 'lon_gp' in gp and 'lat_gp' in gp:
                lon_coords.append(float(gp['lon_gp']))
                lat_coords.append(float(gp['lat_gp']))
        
        # Calculate bounds with some padding
        min_lon, max_lon = min(lon_coords), max(lon_coords)
        min_lat, max_lat = min(lat_coords), max(lat_coords)
        
        # Add padding (5% of range)
        lon_padding = (max_lon - min_lon) * 0.05
        lat_padding = (max_lat - min_lat) * 0.05
        
        # Set the axis limits with padding
        plt.xlim(min_lon - lon_padding, max_lon + lon_padding)
        plt.ylim(min_lat - lat_padding, max_lat + lat_padding)
        
        # Plot hospitals with larger markers
        for _, hospital in self.hospitals.iterrows():
            x = float(hospital['lon_hospital'])
            y = float(hospital['lat_hospital'])
            plt.scatter(x, y, c='blue', marker='s', s=200, label='Hospital' if _ == 0 else "")
            plt.text(x+0.001, y+0.001, f"H{hospital['id']}", fontsize=12, fontweight='bold')
        
        # Plot gathering points
        for _, gp in self.gathering_points.iterrows():
            if 'lon_gp' in gp and 'lat_gp' in gp:
                x = float(gp['lon_gp'])
                y = float(gp['lat_gp'])
                # Vary size based on patient count
                size = 100 + (int(gp['red_patients']) * 30) + (int(gp['green_patients']) * 10)
                plt.scatter(x, y, c='red', marker='^', s=size, 
                            label='Gathering Point' if _ == 0 else "",
                            alpha=0.7)
                plt.text(x+0.001, y+0.001, f"G{gp['id']}", fontsize=10)
        
        # Add grid with lighter color
        plt.grid(True, alpha=0.3, linestyle='--')
        
        # Customize the plot
        plt.title('Hospitals and Gathering Points Map', fontsize=16)
        plt.xlabel('Longitude', fontsize=12)
        plt.ylabel('Latitude', fontsize=12)
        
        # Put the legend in a good position
        plt.legend(loc='best', fontsize=12)
        
        # Adjust layout
        plt.tight_layout()
        
        # Get current directory to save output
        current_dir = os.getcwd()
        output_path = self.get_unique_filename('hospitals_and_gathering_points_map.png')
        
        try:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Successfully saved simplified map to: {output_path}")
        except Exception as e:
            print(f"Error saving simplified map: {e}")
        
        plt.close()
    
    def get_coordinates(self, location_id):
        """Get coordinates (longitude, latitude) for a location by ID"""
        # Check if it's a hospital
        for _, hospital in self.hospitals.iterrows():
            if hospital['id'] == location_id:
                return float(hospital['lon_hospital']), float(hospital['lat_hospital'])
        
        # Check if it's a gathering point
        for _, gp in self.gathering_points.iterrows():
            if gp['id'] == location_id and 'lon_gp' in gp and 'lat_gp' in gp:
                return float(gp['lon_gp']), float(gp['lat_gp'])
        
        print(f"Warning: No coordinates found for location {location_id}")
        return None, None
    
    def generate_report(self):
        """Generate detailed report of the solution"""
        metrics = self.calculate_metrics()
        
        report = "="*80 + "\n"
        report += "POST-DISASTER AMBULANCE ROUTING SYSTEM - SOLUTION REPORT\n"
        report += "="*80 + "\n\n"
        
        # Terminal çıktılarını ekle (eğer varsa)
        if hasattr(self, 'terminal_output') and self.terminal_output:
            report += "TERMINAL OUTPUT LOG\n" + "-"*80 + "\n"
            report += ''.join(self.terminal_output)
            report += "\n" + "="*80 + "\n"
            report += "SOLUTION ANALYSIS\n"
            report += "="*80 + "\n\n"
        
        # Summary metrics
        report += "PERFORMANCE METRICS\n" + "-"*80 + "\n"
        report += f"Fitness Value: {metrics.get('fitness_value', 0):.2f}\n"
        report += f"Total Travel Distance: {metrics['total_distance']:.2f} km\n"
        report += f"Average Response Time: {metrics['avg_response_time']:.2f} minutes\n"
        report += f"Maximum Response Time: {metrics['max_response_time']:.2f} minutes\n"
        report += f"Hospital Capacity Utilization: {metrics['hospital_utilization']*100:.2f}%\n"
        report += f"Ambulance Utilization Rate: {metrics['ambulance_utilization']:.2f} routes per ambulance\n"
        report += f"Red Patients Served: {metrics.get('red_patients_served', 0)}\n"
        report += f"Green Patients Served: {metrics.get('green_patients_served', 0)}\n"
        report += f"Average Treatment Time: {metrics.get('avg_treatment_time', 0):.2f} minutes\n"
        report += f"Total Treatments: {metrics.get('total_treatments', 0)}\n"
        report += f"Unserved Patients: {metrics['unserved_patients']}\n\n"
        
        # Detailed route information for each ambulance
        report += "AMBULANCE ROUTES\n" + "-"*80 + "\n"
        
        for amb_id, routes in self.solution.items():
            report += f"Ambulance {amb_id}:\n"
            
            if not routes:
                report += "  No routes assigned\n\n"
                continue
                
            total_distance = sum(route['distance'] for route in routes)
            report += f"  Total routes: {len(routes)}, Total distance: {total_distance:.2f} km\n"
            
            for i, route in enumerate(routes):
                report += f"  Route {i+1}: From {route['from']} to {route['to']} ({route['distance']:.2f} km)\n"
                report += f"    Action: {route['action']}, Patient Type: {route.get('patient_type', 'N/A')}\n"
                report += f"    Start Time: {route['time']:.2f} minutes\n"
                report += f"    Duration: {route.get('duration', 0):.2f} minutes\n"
                
                if route.get('action') == 'treatment':
                    report += f"    Treatment Duration: {route.get('duration', 0):.2f} minutes\n"
            
            report += "\n"
        
        # Hospital utilization
        report += "HOSPITAL UTILIZATION\n" + "-"*80 + "\n"
        for hosp_id, util in self.hospital_utilization.items():
            capacity = 0
            for _, hospital in self.hospitals.iterrows():
                if hospital['id'] == hosp_id:
                    capacity = hospital['capacity']
                    break
                    
            report += f"Hospital {hosp_id}: {util}/{capacity} patients ({util/capacity*100 if capacity > 0 else 0:.2f}% full)\n"
        
        report += "\n"
        
        # Unserved patients details
        if self.unserved_patients:
            report += "UNSERVED PATIENTS\n" + "-"*80 + "\n"
            for i, patient in enumerate(self.unserved_patients):
                report += f"Patient {i+1}: Location {patient['location']}, Type: {patient['type']}, Reason: {patient['reason']}\n"
        
        # Write report to file with error handling
        current_dir = os.getcwd()
        report_path = self.get_unique_filename("ambulance_routing_report.txt")
        
        try:
            with open(report_path, "w") as f:
                f.write(report)
            print(f"Successfully saved report to: {report_path}")
        except Exception as e:
            print(f"Error saving report: {e}")
            print("Printing report to console instead:")
            print(report)
        
        return report

    def extract_patient_sequences(self, solution):
        """Solution'dan ambulans başına hasta sıralarını çıkar"""
        patient_sequences = {}
        
        for ambulance_id, routes in solution.items():
            patients = []
            for route in routes:
                if route.get('action') in ['pickup', 'visit', 'visit_and_treat']:
                    patient_info = {
                        'location': route['to'],
                        'type': route['patient_type'],
                        'action': route['action']
                    }
                    patients.append(patient_info)
            patient_sequences[ambulance_id] = patients
        
        return patient_sequences

    def rebuild_solution_with_sequences(self, patient_sequences):
        """Hasta sıralarından tamamen yeni solution oluştur - tam implementasyon"""
        
        # Orijinal durumları kaydet
        original_gathering_points = self.gathering_points.copy()
        original_hospital_utilization = self.hospital_utilization.copy()
        original_response_times = self.response_times.copy()
        original_unserved_patients = self.unserved_patients.copy()
        
        # Durumları resetle
        self.reset_state_for_rebuild()
        
        new_solution = {}
        
        # Her ambulans için ambulans durumu
        ambulance_status = {}
        for _, amb in self.ambulances.iterrows():
            ambulance_status[amb['id']] = {
                'location': amb['initial_hospital'], 
                'available_time': 0,
                'is_hospital': True
            }
            new_solution[amb['id']] = []
        
        try:
            # Hasta sıralarını zaman sırası ile birleştir
            all_assignments = []
            for ambulance_id, patients in patient_sequences.items():
                for order, patient in enumerate(patients):
                    all_assignments.append({
                        'ambulance_id': ambulance_id,
                        'patient': patient,
                        'order': order
                    })
            
            # Her ambulans için sırayla işle
            for ambulance_id, patients in patient_sequences.items():
                if not patients:
                    continue
                    
                current_time = ambulance_status[ambulance_id]['available_time']
                current_location = ambulance_status[ambulance_id]['location']
                
                for patient in patients:
                    patient_location = patient['location']
                    patient_type = patient['type']
                    
                    # Hastaya seyahat
                    dist_to_patient = self.get_distance(current_location, patient_location)
                    travel_time = dist_to_patient / 50 * 60
                    
                    if patient_type == 'red':
                        # Kırmızı hasta - pickup
                        new_solution[ambulance_id].append({
                            'from': current_location,
                            'from_name': self.get_point_name(current_location),
                            'to': patient_location,
                            'to_name': self.get_point_name(patient_location),
                            'time': current_time,
                            'action': 'pickup',
                            'distance': dist_to_patient,
                            'patient_type': patient_type,
                            'duration': travel_time
                        })
                        
                        current_time += travel_time
                        current_location = patient_location
                        
                        # En yakın hastaneyi bul
                        closest_hosp, dist_to_hospital = self.get_closest_hospital(patient_location)
                        
                        if closest_hosp is None:
                            # Hastane kapasitesi doluysa unserved olarak işaretle
                            self.unserved_patients.append({
                                'location': patient_location,
                                'location_name': self.get_point_name(patient_location),
                                'type': 'red',
                                'reason': 'no_hospital_capacity'
                            })
                            continue
                        
                        # Hastaneye git - dropoff
                        hospital_travel_time = dist_to_hospital / 50 * 60
                        
                        new_solution[ambulance_id].append({
                            'from': patient_location,
                            'from_name': self.get_point_name(patient_location),
                            'to': closest_hosp,
                            'to_name': self.get_point_name(closest_hosp),
                            'time': current_time,
                            'action': 'dropoff',
                            'distance': dist_to_hospital,
                            'patient_type': patient_type,
                            'duration': hospital_travel_time
                        })
                        
                        current_time += hospital_travel_time
                        current_location = closest_hosp
                        
                        # Hospital utilization güncelle
                        self.hospital_utilization[closest_hosp] += 1
                        
                        # Response time kaydet
                        self.red_response_times.append(current_time)
                        self.response_times.append(current_time)
                        
                        # Ambulans durumunu güncelle
                        ambulance_status[ambulance_id]['location'] = closest_hosp
                        ambulance_status[ambulance_id]['available_time'] = current_time
                        ambulance_status[ambulance_id]['is_hospital'] = True
                    
                    else:  # Yeşil hasta
                        # Yeşil hasta - visit and treat
                        treatment_time = self.GREEN_TREATMENT_TIME
                        total_duration = travel_time + treatment_time
                        
                        new_solution[ambulance_id].append({
                            'from': current_location,
                            'from_name': self.get_point_name(current_location),
                            'to': patient_location,
                            'to_name': self.get_point_name(patient_location),
                            'time': current_time,
                            'action': 'visit_and_treat',
                            'distance': dist_to_patient,
                            'patient_type': patient_type,
                            'duration': total_duration
                        })
                        
                        current_time += total_duration
                        current_location = patient_location
                        
                        # Response time kaydet
                        self.green_response_times.append(current_time)
                        self.response_times.append(current_time)
                        
                        # Ambulans durumunu güncelle
                        ambulance_status[ambulance_id]['location'] = patient_location
                        ambulance_status[ambulance_id]['available_time'] = current_time
                        ambulance_status[ambulance_id]['is_hospital'] = False
            
            # Tüm ambulansları hastanelere döndür
            for ambulance_id, status in ambulance_status.items():
                if not status['is_hospital']:
                    current_location = status['location']
                    current_time = status['available_time']
                    
                    closest_hosp, dist = self.get_closest_hospital(current_location)
                    if closest_hosp is not None:
                        travel_time = dist / 50 * 60
                        new_solution[ambulance_id].append({
                            'from': current_location,
                            'from_name': self.get_point_name(current_location),
                            'to': closest_hosp,
                            'to_name': self.get_point_name(closest_hosp),
                            'time': current_time,
                            'action': 'return_to_hospital',
                            'distance': dist,
                            'duration': travel_time
                        })
            
            return new_solution
            
        except Exception as e:
            # Hata durumunda orijinal durumları geri yükle
            self.gathering_points = original_gathering_points
            self.hospital_utilization = original_hospital_utilization
            self.response_times = original_response_times
            self.unserved_patients = original_unserved_patients
            
            print(f"Error in rebuild_solution_with_sequences: {e}")
            return {}

    def reset_state_for_rebuild(self):
        """Rebuild için durumları resetle"""
        # Hospital utilization sıfırla
        for hospital_id in self.hospital_utilization:
            self.hospital_utilization[hospital_id] = 0
        
        # Response times temizle
        self.response_times = []
        self.red_response_times = []
        self.green_response_times = []
        
        # Unserved patients temizle
        self.unserved_patients = []

    def get_ambulance_initial_location(self, ambulance_id):
        """Ambulansın başlangıç lokasyonunu getir"""
        for _, amb in self.ambulances.iterrows():
            if amb['id'] == ambulance_id:
                return amb['initial_hospital']
        return 1  # Default hospital
    
    def apply_random_neighbor_operation_optimized(self, solution):
        """Optimize edilmiş neighbor operation"""
        operations = [
            #self.internal_patients_relocate,    #1. neighbor
            #self.internal_patients_swap,    #2. neighbor
            #self.external_patients_relocate,    #3. neighbor
            self.external_patients_swap,    #4. neighbor
            #self.nearest_x_relocate #5. neighbor
        ]
        
        operation = random.choice(operations)
        new_solution, affected_ambulances = operation(solution)
        
        # Response time'ları güncelle (opsiyonel)
        # self.update_response_times_for_ambulances(new_solution, affected_ambulances)
        
        return new_solution

    def update_response_times_for_ambulances(self, solution, affected_ambulances):
        """Etkilenen ambulanslar için response time'ları güncelle"""
        # Önce etkilenen ambulansların response time'larını temizle
        temp_red_times = []
        temp_green_times = []
        temp_response_times = []
        
        # Tüm ambulansların rotalarını kontrol et
        for ambulance_id, routes in solution.items():
            if ambulance_id in affected_ambulances:
                # Sadece etkilenen ambulanslar için yeniden hesapla
                for route in routes:
                    if route.get('action') == 'dropoff' and route.get('patient_type') == 'red':
                        end_time = route['time'] + route.get('duration', 0)
                        temp_red_times.append(end_time)
                        temp_response_times.append(end_time)
                    elif route.get('action') == 'visit_and_treat' and route.get('patient_type') == 'green':
                        end_time = route['time'] + route.get('duration', 0)
                        temp_green_times.append(end_time)
                        temp_response_times.append(end_time)
        
        # Güncellenmiş değerleri kaydet
        self.red_response_times.extend(temp_red_times)
        self.green_response_times.extend(temp_green_times)
        self.response_times.extend(temp_response_times)

    def internal_patients_relocate(self, solution):
        """1. Aynı rota içinde yeşil hastayı farklı pozisyona taşı - OPTİMİZE EDİLMİŞ"""
        new_solution = copy.deepcopy(solution)
        patient_sequences = self.extract_patient_sequences(new_solution)
        
        # Rastgele bir ambulans seç
        ambulance_ids = [amb_id for amb_id, patients in patient_sequences.items() 
                        if len([p for p in patients if p['type'] == 'green']) >= 2]
        
        if not ambulance_ids:
            return new_solution, []  # Değişiklik yapılamaz, boş liste döndür
        
        selected_ambulance = random.choice(ambulance_ids)
        patients = patient_sequences[selected_ambulance]
        
        # Yeşil hastaları bul
        green_indices = [i for i, p in enumerate(patients) if p['type'] == 'green']
        
        if len(green_indices) < 2:
            return new_solution, []
        
        # Rastgele bir yeşil hastayı seç
        old_index = random.choice(green_indices)
        patient_to_move = patients[old_index]
        
        # Geçerli yeni pozisyonları bul (kırmızı hastadan sonra olamaz)
        valid_positions = []
        for i in range(len(patients)):
            if i != old_index and self.is_valid_green_position_in_sequence(patients, i):
                valid_positions.append(i)
        
        if not valid_positions:
            return new_solution, []
        
        new_index = random.choice(valid_positions)
        
        # Hastayı taşı
        patients.pop(old_index)
        patients.insert(new_index, patient_to_move)
        
        # SADECE bu ambulansın rotasını yeniden oluştur
        new_solution[selected_ambulance] = self.rebuild_single_ambulance_route(
            selected_ambulance, 
            patients
        )
        
        # Etkilenen ambulans listesini döndür
        return new_solution, [selected_ambulance]

    def internal_patients_swap(self, solution):
        """2. Aynı rotadaki iki yeşil hastanın yerini değiştir - OPTİMİZE EDİLMİŞ"""
        new_solution = copy.deepcopy(solution)
        patient_sequences = self.extract_patient_sequences(new_solution)
        
        # En az 2 yeşil hastası olan ambulansları bul
        ambulance_ids = [amb_id for amb_id, patients in patient_sequences.items() 
                        if len([p for p in patients if p['type'] == 'green']) >= 2]
        
        if not ambulance_ids:
            return new_solution, []
        
        selected_ambulance = random.choice(ambulance_ids)
        patients = patient_sequences[selected_ambulance]
        
        # Yeşil hastaları bul
        green_indices = [i for i, p in enumerate(patients) if p['type'] == 'green']
        
        if len(green_indices) < 2:
            return new_solution, []
        
        # İki farklı yeşil hasta seç
        idx1, idx2 = random.sample(green_indices, 2)
        
        # Yerlerini değiştir
        patients[idx1], patients[idx2] = patients[idx2], patients[idx1]
        
        # SADECE bu ambulansın rotasını yeniden oluştur
        new_solution[selected_ambulance] = self.rebuild_single_ambulance_route(
            selected_ambulance, 
            patients
        )
        
        # Etkilenen ambulans listesini döndür
        return new_solution, [selected_ambulance]

    def external_patients_relocate(self, solution):
        """3. Bir hastayı farklı ambulansa taşı - OPTİMİZE EDİLMİŞ"""
        new_solution = copy.deepcopy(solution)
        patient_sequences = self.extract_patient_sequences(new_solution)
        
        ambulance_ids = list(patient_sequences.keys())
        if len(ambulance_ids) < 2:
            return new_solution, []
        
        # Kaynak ve hedef ambulans seç
        source_amb, target_amb = random.sample(ambulance_ids, 2)
        source_patients = patient_sequences[source_amb]
        target_patients = patient_sequences[target_amb]
        
        if not source_patients:
            return new_solution, []
        
        # Kaynak ambulanstan rastgele hasta seç
        patient_index = random.randint(0, len(source_patients) - 1)
        patient_to_move = source_patients[patient_index]
        
        # Hedef ambulansta geçerli pozisyon bul
        if patient_to_move['type'] == 'green':
            valid_positions = []
            for i in range(len(target_patients) + 1):
                if self.is_valid_green_position_in_sequence(target_patients, i):
                    valid_positions.append(i)
            
            if not valid_positions:
                return new_solution, []
            
            insert_position = random.choice(valid_positions)
        else:  # Kırmızı hasta - sadece sona eklenebilir
            insert_position = len(target_patients)
        
        # Hastayı taşı
        moved_patient = source_patients.pop(patient_index)
        target_patients.insert(insert_position, moved_patient)
        
        # SADECE etkilenen 2 ambulansın rotalarını yeniden oluştur
        new_solution[source_amb] = self.rebuild_single_ambulance_route(
            source_amb, 
            source_patients
        )
        new_solution[target_amb] = self.rebuild_single_ambulance_route(
            target_amb, 
            target_patients
        )
        
        # Etkilenen ambulans listesini döndür
        return new_solution, [source_amb, target_amb]

    def external_patients_swap(self, solution):
        """4. İki farklı ambulanstaki hastaları değiştir - OPTİMİZE EDİLMİŞ"""
        new_solution = copy.deepcopy(solution)
        patient_sequences = self.extract_patient_sequences(new_solution)
        
        ambulance_ids = list(patient_sequences.keys())
        if len(ambulance_ids) < 2:
            return new_solution, []
        
        # İki farklı ambulans seç
        amb1, amb2 = random.sample(ambulance_ids, 2)
        patients1 = patient_sequences[amb1]
        patients2 = patient_sequences[amb2]
        
        if not patients1 or not patients2:
            return new_solution, []
        
        # Her ambulanstan rastgele hasta seç
        idx1 = random.randint(0, len(patients1) - 1)
        idx2 = random.randint(0, len(patients2) - 1)
        
        patient1 = patients1[idx1]
        patient2 = patients2[idx2]
        
        # Geçerlilik kontrolü
        if not self.is_valid_patient_swap(patients1, patients2, idx1, idx2, patient1, patient2):
            return new_solution, []
        
        # Hastaları değiştir
        patients1[idx1] = patient2
        patients2[idx2] = patient1
        
        # SADECE etkilenen 2 ambulansın rotalarını yeniden oluştur
        new_solution[amb1] = self.rebuild_single_ambulance_route(
            amb1, 
            patients1
        )
        new_solution[amb2] = self.rebuild_single_ambulance_route(
            amb2, 
            patients2
        )
        
        # Etkilenen ambulans listesini döndür
        return new_solution, [amb1, amb2]
    
    def nearest_x_relocate(self, solution):
        """5. En yakın X hastaya gitme stratejisi - bir ambulansın tüm rotasını yeniden düzenle"""
        new_solution = copy.deepcopy(solution)
        patient_sequences = self.extract_patient_sequences(new_solution)
        
        # Rastgele bir ambulans seç
        ambulance_ids = list(patient_sequences.keys())
        if not ambulance_ids:
            return new_solution, []
        
        selected_ambulance = random.choice(ambulance_ids)
        current_patients = patient_sequences[selected_ambulance]
        
        if len(current_patients) < 2:
            return new_solution, []  # En az 2 hasta olmalı
        
        # X değerini belirle (2 ile 5 arası)
        x = random.randint(2, min(5, len(current_patients)))
        
        # Yeni strateji: Her adımda en yakın X hastadan birini seç
        new_sequence = []
        visited = set()
        current_location = self.get_ambulance_initial_location(selected_ambulance)
        remaining_patients = current_patients.copy()
        
        while remaining_patients:
            # Mevcut konumdan en yakın X hastayı bul
            candidates = []
            for patient in remaining_patients:
                if patient['location'] not in visited:
                    dist = self.get_distance(current_location, patient['location'])
                    candidates.append((dist, patient))
            
            # En yakın X tanesini al
            candidates.sort(key=lambda x: x[0])
            top_x_candidates = candidates[:x]
            
            if not top_x_candidates:
                break
            
            # Bu X hastadan rastgele birini seç
            chosen = random.choice(top_x_candidates)[1]
            new_sequence.append(chosen)
            visited.add(chosen['location'])
            remaining_patients.remove(chosen)
            
            # Kırmızı hastaysa hastaneye git
            if chosen['type'] == 'red':
                closest_hosp, _ = self.get_closest_hospital(chosen['location'])
                current_location = closest_hosp if closest_hosp else chosen['location']
            else:
                current_location = chosen['location']
        
        # Yeni sırayla rotayı oluştur
        new_solution[selected_ambulance] = self.rebuild_single_ambulance_route(
            selected_ambulance, 
            new_sequence
        )
        
        return new_solution, [selected_ambulance]

    def is_valid_green_position_in_sequence(self, patients, position):
        """Yeşil hasta için geçerli pozisyon mu kontrol et"""
        # Kırmızı hastadan sonra yeşil hasta olamaz
        for i in range(position, len(patients)):
            if patients[i]['type'] == 'red':
                return False
        return True

    def is_valid_patient_swap(self, patients1, patients2, idx1, idx2, patient1, patient2):
        """Hasta değişiminin geçerli olup olmadığını kontrol et"""
        # Yeşil hasta kontrolü
        if patient1['type'] == 'green':
            if not self.is_valid_green_position_in_sequence(patients2, idx2):
                return False
        
        if patient2['type'] == 'green':
            if not self.is_valid_green_position_in_sequence(patients1, idx1):
                return False
        
        return True

    def run_with_simulated_annealing(self, init_from_current=True):
        """
        init_from_current=True  → içeride generate_initial_solution() çağrılır
        init_from_current=False → dışarıda önceden atanmış self.solution kullanılır
        """
        """Simulated Annealing ile çözüm iyileştirmesi"""
        print("Starting Simulated Annealing optimization...")
        
        # SA parametreleri
        initial_temperature = 100
        final_temperature = 1
        cooling_rate = 0.95
        max_iterations = 9999         
        max_iterations_per_temp = 10 
        
        # Başlangıç çözümünü oluştur
        if init_from_current:
            print("Generating initial solution...")
            current_solution = self.run_heuristic_algorithm()
        else:
            # comparison’dan gelen self.solution’ı kullan
            current_solution = self.solution

        current_fitness = self.calculate_fitness()
        initial_fitness_value = current_fitness
        
        # En iyi çözümü kaydet
        best_solution = copy.deepcopy(current_solution)
        best_fitness = current_fitness
        
        # SA istatistikleri
        iteration_history = []
        temperature_history = []
        fitness_history = []
        acceptance_history = []
        
        temperature = initial_temperature
        iteration = 0
        total_neighbors_tried = 0
        total_neighbors_accepted = 0
        
        print(f"Initial fitness: {current_fitness:.2f}")
        print("Starting SA iterations...")
        
        while temperature > final_temperature and iteration < max_iterations:
            iteration += 1
            temp_iteration = 0
            temp_acceptances = 0
            
            # Her sıcaklık seviyesinde birden fazla komşu dene
            while temp_iteration < max_iterations_per_temp:
                temp_iteration += 1
                total_neighbors_tried += 1
                
                # Geçici değişkenleri başlat
                temp_solution = None
                temp_red_times = None
                temp_green_times = None
                temp_response_times = None
                
                # Rastgele bir neighbor operation uygula
                try:
                    # Mevcut durumu kaydet
                    temp_solution = copy.deepcopy(self.solution)
                    temp_red_times = self.red_response_times.copy()
                    temp_green_times = self.green_response_times.copy()
                    temp_response_times = self.response_times.copy()
                    
                    # Yeni komşu oluştur
                    neighbor_solution = self.apply_random_neighbor_operation_optimized(current_solution)
                    
                    # Yeni solution'ı set et ve fitness hesapla
                    self.solution = neighbor_solution
                    self.extract_response_times_from_solution()
                    neighbor_fitness = self.calculate_fitness()
                    
                    # Kabul etme kriterini kontrol et
                    accept = self.accept_solution(current_fitness, neighbor_fitness, temperature)
                    
                    if accept:
                        # Yeni çözümü kabul et
                        current_solution = neighbor_solution
                        current_fitness = neighbor_fitness
                        temp_acceptances += 1
                        total_neighbors_accepted += 1
                        
                        # En iyi çözümü güncelle
                        if neighbor_fitness < best_fitness:
                            best_solution = copy.deepcopy(neighbor_solution)
                            best_fitness = neighbor_fitness
                            print(f"  New best fitness found: {best_fitness:.2f}")
                    else:
                        # Eski çözümü geri yükle
                        self.solution = temp_solution
                        self.red_response_times = temp_red_times
                        self.green_response_times = temp_green_times
                        self.response_times = temp_response_times
                    
                except Exception as e:
                    print(f"  Error in neighbor generation: {e}")
                    # Hata durumunda eski çözümü geri yükle
                    if temp_solution is not None:
                        self.solution = temp_solution
                        self.red_response_times = temp_red_times if temp_red_times is not None else []
                        self.green_response_times = temp_green_times if temp_green_times is not None else []
                        self.response_times = temp_response_times if temp_response_times is not None else []
            
            # İstatistikleri kaydet
            iteration_history.append(iteration)
            temperature_history.append(temperature)
            fitness_history.append(current_fitness)
            acceptance_rate = temp_acceptances / max_iterations_per_temp if max_iterations_per_temp > 0 else 0
            acceptance_history.append(acceptance_rate)
            
            # İlerleme raporu
            if True:
                print(f"Iteration {iteration}: Temp={temperature:.2f}, "
                    f"Current Fitness={current_fitness:.2f}, "
                    f"Best Fitness={best_fitness:.2f}, "
                    f"Acceptance Rate={acceptance_rate:.2%}")
            
            # Sıcaklığı düşür
            temperature *= cooling_rate
        
        # En iyi çözümü geri yükle
        self.solution = best_solution
        self.extract_response_times_from_solution()
        
        # Final rapor
        initial_fitness = fitness_history[0] if fitness_history else current_fitness
        improvement = ((initial_fitness_value - best_fitness) / initial_fitness_value * 100) if initial_fitness_value > 0 else 0
        acceptance_rate_total = total_neighbors_accepted / total_neighbors_tried if total_neighbors_tried > 0 else 0
        
        print(f"\nSimulated Annealing completed!")
        print(f"Initial fitness: {initial_fitness_value:.2f}")
        print(f"Final best fitness: {best_fitness:.2f}")
        print(f"Improvement: {improvement:.2f}%")
        print(f"Total iterations: {iteration}")
        print(f"Total neighbors tried: {total_neighbors_tried}")
        print(f"Total neighbors accepted: {total_neighbors_accepted}")
        print(f"Overall acceptance rate: {acceptance_rate_total:.2%}")
        
        # SA performans grafiği oluştur
        self.plot_sa_performance(iteration_history, fitness_history, temperature_history, acceptance_history)
        
        return self.solution

    def accept_solution(self, current_fitness, new_fitness, temperature):
        """Çözümü kabul etme kriterini belirle"""
        if new_fitness < current_fitness:  # Daha iyi çözüm
            return True
        else:  # Daha kötü çözüm
            if temperature <= 0:
                return False
            try:
                probability = math.exp(-(new_fitness - current_fitness) / temperature)
                return random.random() < probability
            except OverflowError:
                return False

    def extract_response_times_from_solution(self):
        """Mevcut solution'dan response time'ları çıkar"""
        self.red_response_times = []
        self.green_response_times = []
        self.response_times = []
        
        for ambulance_id, routes in self.solution.items():
            for route in routes:
                if route.get('action') == 'dropoff' and route.get('patient_type') == 'red':
                    end_time = route['time'] + route.get('duration', 0)
                    self.red_response_times.append(end_time)
                    self.response_times.append(end_time)
                elif route.get('action') == 'visit_and_treat' and route.get('patient_type') == 'green':
                    end_time = route['time'] + route.get('duration', 0)
                    self.green_response_times.append(end_time)
                    self.response_times.append(end_time)

    def plot_sa_performance(self, iterations, fitness_values, temperatures, acceptance_rates):
        """SA performansını görselleştir"""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # Fitness değişimi
            ax1.plot(iterations, fitness_values, 'b-', linewidth=2)
            ax1.set_title('Fitness Evolution', fontsize=12)
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('Fitness Value')
            ax1.grid(True, alpha=0.3)
            
            # Sıcaklık değişimi
            ax2.plot(iterations, temperatures, 'r-', linewidth=2)
            ax2.set_title('Temperature Cooling', fontsize=12)
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('Temperature')
            ax2.grid(True, alpha=0.3)
            
            # Kabul oranı
            ax3.plot(iterations, acceptance_rates, 'g-', linewidth=2)
            ax3.set_title('Acceptance Rate', fontsize=12)
            ax3.set_xlabel('Iteration')
            ax3.set_ylabel('Acceptance Rate')
            ax3.grid(True, alpha=0.3)
            
            # Fitness histogram
            ax4.hist(fitness_values, bins=20, alpha=0.7, color='purple')
            ax4.set_title('Fitness Distribution', fontsize=12)
            ax4.set_xlabel('Fitness Value')
            ax4.set_ylabel('Frequency')
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Kaydet
            current_dir = os.getcwd()
            output_path = self.get_unique_filename('simulated_annealing_performance.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"SA performance chart saved to: {output_path}")
            plt.close()
            
        except Exception as e:
            print(f"Error creating SA performance chart: {e}")

    def rebuild_single_ambulance_route(self, ambulance_id, patient_sequence):
        """Tek bir ambulansın rotasını yeniden oluştur"""
        route = []
        
        # Ambulansın başlangıç konumu ve zamanı
        current_location = self.get_ambulance_initial_location(ambulance_id)
        current_time = 0
        
        for patient in patient_sequence:
            patient_location = patient['location']
            patient_type = patient['type']
            
            # Hastaya git
            dist_to_patient = self.get_distance(current_location, patient_location)
            travel_time = dist_to_patient / 50 * 60
            
            if patient_type == 'red':
                # Kırmızı hasta pickup
                route.append({
                    'from': current_location,
                    'from_name': self.get_point_name(current_location),
                    'to': patient_location,
                    'to_name': self.get_point_name(patient_location),
                    'time': current_time,
                    'action': 'pickup',
                    'distance': dist_to_patient,
                    'patient_type': patient_type,
                    'duration': travel_time
                })
                
                current_time += travel_time
                current_location = patient_location
                
                # Hastaneye git
                closest_hosp, dist_to_hospital = self.get_closest_hospital(patient_location)
                if closest_hosp:
                    hospital_travel_time = dist_to_hospital / 50 * 60
                    route.append({
                        'from': patient_location,
                        'from_name': self.get_point_name(patient_location),
                        'to': closest_hosp,
                        'to_name': self.get_point_name(closest_hosp),
                        'time': current_time,
                        'action': 'dropoff',
                        'distance': dist_to_hospital,
                        'patient_type': patient_type,
                        'duration': hospital_travel_time
                    })
                    current_time += hospital_travel_time
                    current_location = closest_hosp
                    
            else:  # Yeşil hasta
                total_duration = travel_time + self.GREEN_TREATMENT_TIME
                route.append({
                    'from': current_location,
                    'from_name': self.get_point_name(current_location),
                    'to': patient_location,
                    'to_name': self.get_point_name(patient_location),
                    'time': current_time,
                    'action': 'visit_and_treat',
                    'distance': dist_to_patient,
                    'patient_type': patient_type,
                    'duration': total_duration
                })
                current_time += total_duration
                current_location = patient_location
        
        # Son konumda hastane değilse dön
        if current_location not in self.hospital_names:
            closest_hosp, dist = self.get_closest_hospital(current_location)
            if closest_hosp:
                travel_time = dist / 50 * 60
                route.append({
                    'from': current_location,
                    'from_name': self.get_point_name(current_location),
                    'to': closest_hosp,
                    'to_name': self.get_point_name(closest_hosp),
                    'time': current_time,
                    'action': 'return_to_hospital',
                    'distance': dist,
                    'duration': travel_time
                })
        
        return route


# Main simulation function
def run_simulation():
    """Run a complete simulation with the uploaded CSV data"""
    try:
        # Initialize the system
        ars = AmbulanceRoutingSystem()
        
        # Terminal çıktılarını yakalamaya başla
        import io
        output_capture = io.StringIO()
        
        # Tee yapısı - hem ekrana hem buffer'a yaz
        class TeeOutput:
            def __init__(self, *outputs):
                self.outputs = outputs
            
            def write(self, data):
                for output in self.outputs:
                    output.write(data)
            
            def flush(self):
                for output in self.outputs:
                    if hasattr(output, 'flush'):
                        output.flush()
        
        # stdout'u kaydet ve yeni tee stdout oluştur
        old_stdout = sys.stdout
        sys.stdout = TeeOutput(old_stdout, output_capture)
        
        # Load data from the uploaded CSV files
        print("Loading data from CSV files...")
        ars.load_data('hospitals.csv', 'gathering_points.csv', 'ambulances.csv')
        
        # Run the algorithm
        print("Running heuristic algorithm...")
        solution = ars.run_heuristic_algorithm()
        
        # Calculate metrics
        metrics = ars.calculate_metrics()
        print("\nPerformance Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
        
        # stdout'u geri al ve yakalanan çıktıyı kaydet
        sys.stdout = old_stdout
        ars.terminal_output.append(output_capture.getvalue())
        
        # Visualize solution
        print("\nGenerating visualizations...")
        ars.visualize_solution()
        
        # Generate report
        print("\nGenerating detailed report...")
        ars.generate_report()
        
        # Verify files were created
        #current_dir = os.getcwd()
        #expected_files = [
        #    os.path.join(current_dir, "ambulance_routing_solution.png"),
        #    os.path.join(current_dir, "ambulance_routing_metrics.png"),
        #    os.path.join(current_dir, "ambulance_routing_report.txt")
        #]
        
        print("\nChecking for output files:")
        for file_path in expected_files:
            if os.path.exists(file_path):
                print(f"  - {file_path} ✓ (File size: {os.path.getsize(file_path)/1024:.1f} KB)")
            else:
                print(f"  - {file_path} ✗ (Not found)")
                
    except Exception as e:
        sys.stdout = old_stdout  # Hata durumunda stdout'u geri al
        print(f"Error in simulation: {e}")
        import traceback
        traceback.print_exc()


def run_comparison():
    """
    Normal Heuristic ile Simulated Annealing (SA)’yi
    aynı başlangıç çözümünden başlatarak karşılaştırır.
    """
    import copy, io, sys

    print("\n" + "="*60)
    print("AMBULANCE ROUTING: NORMAL vs SIMULATED ANNEALING")
    print("="*60)

    # --- 1) Normal Heuristic ---
    print("\n1) Running Normal Heuristic...")
    ars_normal = AmbulanceRoutingSystem()
    ars_normal.load_data('hospitals.csv', 'gathering_points.csv', 'ambulances.csv')
    # Bu metot self.solution'u da ayarlar
    ars_normal.run_heuristic_algorithm()
    # Metric’leri al
    normal_metrics = ars_normal.calculate_metrics()  # :contentReference[oaicite:0]{index=0}

    # --- 2) Simulated Annealing from that solution ---
    print("\n2) Running SA from the heuristic’s solution...")
    ars_sa = AmbulanceRoutingSystem()
    ars_sa.load_data('hospitals.csv', 'gathering_points.csv', 'ambulances.csv')

    # Başlangıç çözümünü aynen ver:
    ars_sa.solution = copy.deepcopy(ars_normal.solution)
    # Response-timer listeleri de güncelleyelim ki SA içindeki metrikler doğru çalışsın
    ars_sa.extract_response_times_from_solution()     # :contentReference[oaicite:1]{index=1}
    # (isteğe bağlı: best_fitness görmek için)
    print(f"Initial fitness: {ars_sa.calculate_fitness():.2f}")  # :contentReference[oaicite:2]{index=2}

    # Terminal çıktıyı yakalamak için
    class TeeOutput:
        def __init__(self, *outs): self.outs = outs
        def write(self, d):    [o.write(d) for o in self.outs]
        def flush(self):       [o.flush() for o in self.outs if hasattr(o, 'flush')]

    capture = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout   = TeeOutput(old_stdout, capture)

    # Burada SA, init_from_current=False sayesinde kendi içinde
    # run_heuristic_algorithm() çağırmayacak:
    ars_sa.run_with_simulated_annealing(init_from_current=False)

    sys.stdout = old_stdout
    sa_metrics = ars_sa.calculate_metrics()  # :contentReference[oaicite:3]{index=3}

    # --- 3) Karşılaştırma tablosu ---
    print("\n" + "-"*60)
    print(f"{'Metric':<25}{'Normal':>10}{'SA':>10}{'Improvement':>15}")
    print("-"*60)
    rows = [
        ("Fitness Value",         normal_metrics.get('fitness_value', 0), sa_metrics.get('fitness_value', 0)),
        ("Total Distance (km)",   normal_metrics['total_distance'],      sa_metrics['total_distance']),
        ("Avg Response Time",     normal_metrics['avg_response_time'],  sa_metrics['avg_response_time']),
        ("Max Response Time",     normal_metrics['max_response_time'],  sa_metrics['max_response_time']),
        ("Unserved Patients",     normal_metrics['unserved_patients'],    sa_metrics['unserved_patients']),
    ]
    for name, n, s in rows:
        imp = f"{((n-s)/n*100):+.2f}%" if n else " N/A"
        print(f"{name:<25}{n:>10.2f}{s:>10.2f}{imp:>15}")
    print("-"*60)

    # --- 4) SA’nın kendi rapor ve görselleştirmeleri ---
    ars_sa.terminal_output.append(capture.getvalue())
    ars_sa.visualize_solution()
    ars_sa.generate_report()



# Ana program
if __name__ == "__main__":
    # Kullanıcıya seçenekler sun
    print("Select an option:")
    print("1. Run normal simulation")
    print("2. Run Simulated Annealing")
    print("3. Run comparison (Normal vs SA)")

    runs_input = input("Kaç kez tekrar etsin? [Default 20]: ").strip()
    try:
        runs = int(runs_input) if runs_input else 20
    except ValueError:
        runs = 20
    
    choice = input("Enter your choice (1/2/3): ").strip()
    
    for i in range (1, runs+1):
        print(f"\n=== Run {i}/{runs} ===")
        start_time = time.time()

        if choice == "1":
            run_simulation()
        elif choice == "2":
            ars = AmbulanceRoutingSystem()
            ars.load_data('hospitals.csv', 'gathering_points.csv', 'ambulances.csv')
            
            # Terminal çıktılarını yakala
            import io
            output_capture = io.StringIO()
            
            class TeeOutput:
                def __init__(self, *outputs):
                    self.outputs = outputs
                
                def write(self, data):
                    for output in self.outputs:
                        output.write(data)
                
                def flush(self):
                    for output in self.outputs:
                        if hasattr(output, 'flush'):
                            output.flush()
            
            old_stdout = sys.stdout
            sys.stdout = TeeOutput(old_stdout, output_capture)
            
            ars.run_with_simulated_annealing()
            
            sys.stdout = old_stdout
            ars.terminal_output.append(output_capture.getvalue())
            
            ars.visualize_solution()
            ars.generate_report()
        elif choice == "3":
            run_comparison()
        else:
            print("Invalid choice. Running normal simulation by default.")
            run_simulation()

        elapsed = time.time() - start_time
        print(f"-> Iteration {i} tamamlandı: {elapsed:.2f} saniye")