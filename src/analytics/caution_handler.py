from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass

@dataclass
class CautionScenario:
    # For representing a potential caution scenario
    laps_until_caution: int
    probability: float
    expected_time_saved: float
    confidence: str # 'high', 'medium', 'low'
    description: str

def recommend_under_caution(current_recommendation: Dict, pit_time_cost: float = 20.0, caution_pit_factor: float = 0.5) -> Dict:
    # Calculate effective pit cost under caution
    effective_cost = pit_time_cost * caution_pit_factor # we get 10 seconds cost under caution

    # Create a simple check if there was previously no recommended pit
    if current_recommendation.get('recommended_lap') is None:

        # Pit now if the effective pit cost is very low (below 10s)
        if effective_cost < 10:
            return {"action": "pit_now", "reason": "caution_reduces_cost", "effective_cost": effective_cost}

        return {"action": "stay", "reason": "caution_not_beneficial", "effective_cost": effective_cost}

    # Best to pit now if there was a recommended future lap
    return {"action": "pit_now", "reason": "existing_recommendation_preempted", "effective_cost": effective_cost}

def estimate_caution_probability(current_lap: int, total_laps: int, cautions_per_race: float = 2.0, early_race_factor: float = 0.5, late_race_factor: float = 1.5) -> List[Tuple[int, float]]:
    remaining_laps = total_laps - current_lap
    if remaining_laps <= 0:
        return []

    # Calculate the base probability per lap
    base_prob_per_lap = cautions_per_race / total_laps
    
    # Build non-uniform distribution
    lap_probabilities = []
    total_weight = 0.0
    
    # Apply weighting factors within phases of the race
    for lap in range(current_lap + 1, total_laps + 1):
        race_progress = lap / total_laps
        
        # Apply factors based on race progress
        if race_progress < 0.25: # Early part of the race (first 25%)
            weight = base_prob_per_lap * early_race_factor
        elif race_progress > 0.75: # Late part of the race (last 25%)
            weight = base_prob_per_lap * late_race_factor
        else: # Middle of the race
            weight = base_prob_per_lap
        
        lap_probabilities.append((lap, weight))
        total_weight += weight
    
    # Ensure probabilities sum to expected cautions per race
    if total_weight > 0:
        lap_probabilities = [(lap, prob / total_weight * cautions_per_race) 
                            for lap, prob in lap_probabilities]
    
    return lap_probabilities

def calculate_expected_value_with_caution(current_lap: int, pit_recommendation: Dict, pit_time_cost: float, caution_pit_cost: float, baseline_lap_time: float, degradation_per_lap: float, total_laps: int, laps_since_pit: int, caution_probabilities: Optional[List[Tuple[int, float]]] = None) -> Dict:
    # Calculate the expected values of different pit strategies considering the cautions
    if caution_probabilities is None:
        caution_probabilities = estimate_caution_probability(current_lap, total_laps)
    
    remaining_laps = total_laps - current_lap
    if remaining_laps <= 0:
        return {
            "recommended_strategy": "stay",
            "reason": "no_laps_remaining",
            "strategies": []
        }
    
    # Strategy 1: Pit now (no caution benefit)
    pit_now_time = pit_time_cost
    for i in range(remaining_laps):
        tire_age = i # Fresh tires
        pit_now_time += baseline_lap_time + (tire_age * degradation_per_lap)
    
    # Strategy 2: Optimal timing
    optimal_pit_lap = pit_recommendation.get('recommended_lap')
    if optimal_pit_lap and optimal_pit_lap > current_lap:
        laps_before_pit = optimal_pit_lap - current_lap
        laps_after_pit = remaining_laps - laps_before_pit
        
        optimal_time = 0
        # Time before pit (on current tires)
        for i in range(laps_before_pit):
            tire_age = laps_since_pit + i
            optimal_time += baseline_lap_time + (tire_age * degradation_per_lap)
        
        # Pit stop (change tires)
        optimal_time += pit_time_cost
        
        # Time after pit (fresh tires)
        for i in range(laps_after_pit):
            tire_age = i
            optimal_time += baseline_lap_time + (tire_age * degradation_per_lap)
    else:
        # No pit recommended, just run to end
        optimal_time = 0
        for i in range(remaining_laps):
            tire_age = laps_since_pit + i
            optimal_time += baseline_lap_time + (tire_age * degradation_per_lap)
    
    # Strategy 3: Wait for caution (expected value across scenarios)
    wait_expected_time = 0
    for caution_lap, probability in caution_probabilities:
        if caution_lap <= current_lap:
            continue
        
        laps_until_caution = caution_lap - current_lap
        
        if laps_until_caution <= remaining_laps:
            # Scenario: Caution comes at this lap
            scenario_time = 0
            
            # Time before caution (on current tires)
            for i in range(min(laps_until_caution, remaining_laps)):
                tire_age = laps_since_pit + i
                scenario_time += baseline_lap_time + (tire_age * degradation_per_lap)
            
            # Pit under caution (change tires)
            scenario_time += caution_pit_cost
            
            # Time after caution pit (fresh tires)
            laps_after_caution = remaining_laps - laps_until_caution
            for i in range(laps_after_caution):
                tire_age = i
                scenario_time += baseline_lap_time + (tire_age * degradation_per_lap)
            
            wait_expected_time += probability * scenario_time
    
    # Take in to account the scenario where no caution occurs
    no_caution_prob = max(0, 1.0 - sum(p for _, p in caution_probabilities))
    if no_caution_prob > 0:
        no_caution_time = optimal_time # This is the same as Strategy 2 
        wait_expected_time += no_caution_prob * no_caution_time
    
    # Compare strategies
    strategies = [
        {
            "name": "pit_now",
            "expected_time": round(pit_now_time, 2),
            "variance": 0.0, # No uncertainty
            "confidence": "high"
        },
        {
            "name": "optimal_timing",
            "expected_time": round(optimal_time, 2),
            "variance": 0.0,
            "confidence": "high"
        },
        {
            "name": "wait_for_caution",
            "expected_time": round(wait_expected_time, 2),
            "variance": round(abs(wait_expected_time - optimal_time), 2), # Simplified variance
            "confidence": "medium" if no_caution_prob > 0.3 else "high"
        }
    ]
    
    # Determine the best strategy
    best_strategy = min(strategies, key=lambda s: s["expected_time"])
    time_saved_vs_optimal = optimal_time - best_strategy["expected_time"]
    
    # Determine confidence in recommendation (high confidence is the best)
    time_diff = abs(strategies[0]["expected_time"] - strategies[2]["expected_time"])
    if time_diff < 1.0:
        confidence = "low"  # Strategies are very close
    elif time_diff < 3.0:
        confidence = "medium"
    else:
        confidence = "high"
    
    return {
        "recommended_strategy": best_strategy["name"],
        "expected_time_saved": round(time_saved_vs_optimal, 2),
        "confidence": confidence,
        "reason": f"Best expected value with {confidence} confidence",
        "strategies": strategies,
        "caution_probability_next_10_laps": round(
            sum(p for lap, p in caution_probabilities if lap <= current_lap + 10), 3
        )
    }

def analyze_caution_scenarios(current_lap: int, pit_recommendation: Dict, pit_time_cost: float, total_laps: int, laps_since_pit: int, baseline_lap_time: float = 100.0, degradation_per_lap: float = 0.15, caution_pit_factor: float = 0.5, cautions_per_race: float = 2.0) -> Dict:
    # Calculate the reduced pit cost
    caution_pit_cost = pit_time_cost * caution_pit_factor # This will give us 10s
    
    # Get the caution probability distribution
    caution_probs = estimate_caution_probability(
        current_lap, total_laps, cautions_per_race
    )
    
    # Calculate expected values
    ev_analysis = calculate_expected_value_with_caution(
        current_lap = current_lap,
        pit_recommendation = pit_recommendation,
        pit_time_cost = pit_time_cost,
        caution_pit_cost = caution_pit_cost,
        baseline_lap_time = baseline_lap_time,
        degradation_per_lap = degradation_per_lap,
        total_laps = total_laps,
        laps_since_pit = laps_since_pit,
        caution_probabilities = caution_probs
    )
    
    # Generate specific scenarios
    scenarios = []
    for laps_ahead in [5, 10, 15, 20]:
        caution_lap = current_lap + laps_ahead
        if caution_lap > total_laps:
            continue
        
        # Find the probability for this lap
        prob = next((p for lap, p in caution_probs if lap == caution_lap), 0.0)
        
        # Calculate time saved if caution comes at this lap
        savings = (pit_time_cost - caution_pit_cost) # 10s saved
        tire_wear_loss = degradation_per_lap * laps_ahead # 0.15 x 10 = 1.5s
        net_benefit = savings - tire_wear_loss # 10 - 1.5 = 8.5s
        
        confidence = "high" if prob > 0.1 else "medium" if prob > 0.05 else "low"
        
        scenarios.append(CautionScenario(
            laps_until_caution = laps_ahead,
            probability = round(prob, 3),
            expected_time_saved = round(net_benefit, 2),
            confidence = confidence,
            description = f"Caution in {laps_ahead} laps: {round(prob*100, 1)}% chance, "f"save {round(net_benefit, 1)}s if it happens"
        ))
    
    return {
        **ev_analysis,
        "scenarios": [
            {
                "laps_until": s.laps_until_caution,
                "probability": s.probability,
                "time_saved": s.expected_time_saved,
                "confidence": s.confidence,
                "description": s.description
            }
            for s in scenarios
        ],
        "caution_pit_cost": round(caution_pit_cost, 2),
        "normal_pit_cost": pit_time_cost
    }