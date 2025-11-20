import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class FieldPosition:
    # Represents a car's position on track at a given lap
    lap: int
    position: int
    car_number: int
    elapsed_time: float
    gap_to_leader: float
    gap_to_ahead: float

@dataclass
class TrafficOpportunity:
    # Represents a traffic (undercut or overcut) opportunity for pit strategy
    opportunity_type: str # 'undercut' or 'overcut'
    target_car_number: int
    target_position: int
    current_gap: float
    pit_now_advantage: float # Expected time advantage (positive = good)
    confidence: str # 'high', 'medium', 'low'
    description: str

class TrafficModel:
    # Constants for traffic modeling (all measured in seconds)
    TRAFFIC_LOSS_PER_LAP = 0.4 # Seconds lost per lap following another car
    UNDERCUT_ADVANTAGE = 1.5 # Typical advantage from undercutting
    PIT_DELTA_UNCERTAINTY = 0.5 # Uncertainty in pit stop time
    
    def __init__(self, endurance_data: pd.DataFrame):
        # Initialize the traffic model with endurance analysis data
        # Clean column names (some of the CSVs have leading/trailing spaces)
        endurance_data.columns = endurance_data.columns.str.strip()
        
        # Make sure that 'NUMBER', 'LAP_NUMBER', 'ELAPSED' columns exist in each of the endurance data files
        required_cols = ['NUMBER', 'LAP_NUMBER', 'ELAPSED']
        missing_cols = [col for col in required_cols if col not in endurance_data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}") # This is a must!
        
        # Parse ELAPSED as timedelta
        self.data = endurance_data.copy()
        self.data['elapsed_seconds'] = self._parse_elapsed_time(self.data['ELAPSED'])
        
        # Build a position lookup table
        self._build_position_table()
    
    def _parse_elapsed_time(self, elapsed_col: pd.Series) -> pd.Series:
        # Parse elapsed time strings into total seconds
        def parse_time(time_str: str) -> float:
            if pd.isna(time_str):
                return np.nan
            
            parts = str(time_str).split(':')
            
            # Two possible formats: MM:SS.mmm or HH:MM:SS.mmm
            if len(parts) == 2: # MM:SS.mmm
                minutes, seconds = parts
                return float(minutes) * 60 + float(seconds)
            elif len(parts) == 3: # HH:MM:SS.mmm
                hours, minutes, seconds = parts
                return float(hours) * 3600 + float(minutes) * 60 + float(seconds)
            else:
                raise ValueError(f"Unknown time format: {time_str}")
        
        return elapsed_col.apply(parse_time)
    
    def _build_position_table(self):
        # Build a lookup table of field positions for each and every lap
        self.position_table: Dict[int, List[FieldPosition]] = {}
        
        for lap_num in self.data['LAP_NUMBER'].unique():
            lap_data = self.data[self.data['LAP_NUMBER'] == lap_num].copy()
            
            # Sort by elapsed time to get running order
            lap_data = lap_data.sort_values('elapsed_seconds')
            
            # Calculate gaps
            leader_time = lap_data['elapsed_seconds'].iloc[0]
            
            positions = []
            for idx, (pos, row) in enumerate(lap_data.iterrows(), start=1):
                gap_to_leader = row['elapsed_seconds'] - leader_time
                
                # Gap to car ahead
                if idx == 1:
                    gap_to_ahead = 0.0 # The leader doesn't have anybody ahead
                else:
                    prev_time = lap_data['elapsed_seconds'].iloc[idx - 2]
                    gap_to_ahead = row['elapsed_seconds'] - prev_time
                
                positions.append(FieldPosition(
                    lap = int(row['LAP_NUMBER']),
                    position = idx,
                    car_number = int(row['NUMBER']),
                    elapsed_time = row['elapsed_seconds'],
                    gap_to_leader = gap_to_leader,
                    gap_to_ahead = gap_to_ahead
                ))
            
            self.position_table[lap_num] = positions
    
    def get_field_position(self, car_number: int, lap: int) -> Optional[FieldPosition]:
        # Get a specific car's position at a specific lap
        # Convert to int to handle float/string inputs
        try:
            lap = int(lap)
            car_number = int(car_number)
        except (ValueError, TypeError):
            return None
        
        # Look up in the position table
        if lap not in self.position_table:
            return None
        
        for pos in self.position_table[lap]:
            if pos.car_number == car_number:
                return pos
        
        return None
    
    def get_running_order(self, lap: int) -> List[FieldPosition]:
        # Get the full running order for a specific lap
        return self.position_table.get(lap, [])
    
    def estimate_position_after_pit(self, car_number: int, current_lap: int, pit_time_loss: float) -> Tuple[Optional[int], Optional[float]]:
        # Estimate the car's new position and gap to the leader after pitting
        current_pos = self.get_field_position(car_number, current_lap)
        if not current_pos:
            return None, None
        
        # Calculate expected elapsed time after pit
        time_after_pit = current_pos.elapsed_time + pit_time_loss
        
        # Find where this car would slot into the field
        running_order = self.get_running_order(current_lap)
        
        new_position = len(running_order) # Start at back
        for pos in running_order:
            if time_after_pit < pos.elapsed_time:
                new_position = pos.position
                break
        
        # Calculate new gap to leader
        leader_time = running_order[0].elapsed_time if running_order else 0
        new_gap = time_after_pit - leader_time
        
        return new_position, new_gap
    
    def detect_undercut_opportunities(self, car_number: int, current_lap: int, pit_time_loss: float, degradation_rate: float, laps_since_pit_ahead: Dict[int, int]) -> List[TrafficOpportunity]:
        # Find cars that can be undercut by pitting now
        opportunities = []
        
        # Get current position
        current_pos = self.get_field_position(car_number, current_lap)
        if not current_pos:
            return opportunities
        
        running_order = self.get_running_order(current_lap)
        
        # Look at cars within 5 positions ahead
        for ahead_pos in running_order:
            if ahead_pos.position >= current_pos.position:
                continue # Skip cars behind us
            
            if ahead_pos.position < current_pos.position - 5:
                continue # Too far ahead
            
            # Check if they're on older tires
            laps_on_tires = laps_since_pit_ahead.get(ahead_pos.car_number, 0)
            if laps_on_tires < 5:
                continue # Their tires still good
            
            # Calculate undercut potential
            gap = ahead_pos.gap_to_ahead if ahead_pos.position < current_pos.position else current_pos.gap_to_ahead
            
            # Estimate advantage
            tire_advantage = degradation_rate * laps_on_tires

            # Net adavantage calculation (the value can be negative, meaning the undercut will not work since gap and pit loss are too large)
            net_advantage = tire_advantage + self.UNDERCUT_ADVANTAGE - pit_time_loss - gap
            
            if net_advantage > 0:
                # Determine confidence based on gap and tire delta
                if net_advantage > 2.0 and laps_on_tires > 10:
                    confidence = 'high'
                elif net_advantage > 1.0 and laps_on_tires > 7:
                    confidence = 'medium'
                else:
                    confidence = 'low'
                
                opportunities.append(TrafficOpportunity(
                    opportunity_type = 'undercut',
                    target_car_number = ahead_pos.car_number,
                    target_position = ahead_pos.position,
                    current_gap = gap,
                    pit_now_advantage = net_advantage,
                    confidence = confidence,
                    description = f"Undercut #{ahead_pos.car_number} in P{ahead_pos.position} "f"({laps_on_tires} laps on tires, {gap:.1f}s gap)"
                ))
        
        return opportunities
    
    def detect_overcut_opportunities(self, car_number: int, current_lap: int, pit_time_loss: float, laps_since_own_pit: int, cars_pitting_soon: List[int]) -> List[TrafficOpportunity]:
        # Find cars ahead that can be overcut by staying out
        opportunities = []
        
        # Get current position
        current_pos = self.get_field_position(car_number, current_lap)
        if not current_pos:
            return opportunities
        
        # Don't overcut if tires are too old (>15 laps)
        if laps_since_own_pit > 15:
            return opportunities
        
        running_order = self.get_running_order(current_lap)
        
        # Look at cars ahead that might pit
        for ahead_pos in running_order:
            if ahead_pos.position >= current_pos.position:
                continue
            
            if ahead_pos.car_number not in cars_pitting_soon:
                continue
            
            # Calculate potential gain from staying out
            gap = ahead_pos.gap_to_ahead if ahead_pos.position < current_pos.position else current_pos.gap_to_ahead
            
            # Calculate the position gain
            position_gain = pit_time_loss - gap
            
            if position_gain > 1.0: # Meaningful gain
                confidence = 'high' if position_gain > 3.0 else 'medium'
                
                opportunities.append(TrafficOpportunity(
                    opportunity_type = 'overcut',
                    target_car_number = ahead_pos.car_number,
                    target_position =ahead_pos.position,
                    current_gap = gap,
                    pit_now_advantage = -position_gain, # Staying out would be better, so it's negative
                    confidence = confidence,
                    description = f"Overcut #{ahead_pos.car_number} in P{ahead_pos.position} "f"by staying out ({gap:.1f}s gap, gain {position_gain:.1f}s)"
                ))
        
        return opportunities
    
    def calculate_traffic_impact(self, car_number: int, current_lap: int, laps_ahead: int) -> float:
        # First calculate expected time loss due to traffic over the next laps ahead
        total_traffic_loss = 0.0
        
        # Iterate over the next laps ahead
        for lap_offset in range(laps_ahead):
            lap = current_lap + lap_offset
            current_pos = self.get_field_position(car_number, lap)
            
            if not current_pos:
                break
            
            # Assume traffic loss if within 2 seconds of car ahead
            if current_pos.gap_to_ahead > 0 and current_pos.gap_to_ahead < 2.0:
                total_traffic_loss += self.TRAFFIC_LOSS_PER_LAP
        
        return total_traffic_loss
    
    def get_cars_within_window(self, car_number: int, current_lap: int, time_window: float) -> List[FieldPosition]:
        # Get cars within a certain time window (seconds) of the specified car at the current lap
        current_pos = self.get_field_position(car_number, current_lap)
        if not current_pos:
            return []
        
        # Get running order for the current lap
        running_order = self.get_running_order(current_lap)
        nearby_cars = []
        
        for pos in running_order:
            if pos.car_number == car_number:
                continue
            
            # Calculate time difference to the current car
            time_delta = abs(pos.elapsed_time - current_pos.elapsed_time)
            if time_delta <= time_window:
                nearby_cars.append(pos)
        
        return nearby_cars