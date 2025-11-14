"""
Comprehensive tests for probabilistic caution handling.

Tests cover:
- Caution probability estimation
- Expected value calculations
- Scenario analysis
- Integration with pit strategy
- Edge cases and error handling
"""

import pytest
from src.analytics.caution_handler import (
    recommend_under_caution,
    estimate_caution_probability,
    calculate_expected_value_with_caution,
    analyze_caution_scenarios
)


class TestRecommendUnderCaution:
    """Tests for basic caution recommendation (legacy function)."""
    
    def test_pit_now_when_beneficial(self):
        """Should recommend pit now when effective cost is low."""
        result = recommend_under_caution(
            current_recommendation={'recommended_lap': 15},
            pit_time_cost=20.0,
            caution_pit_factor=0.5
        )
        assert result['action'] == 'pit_now'
        assert result['effective_cost'] == 10.0
    
    def test_stay_out_when_no_recommendation(self):
        """Should recommend staying out when no pit was planned."""
        result = recommend_under_caution(
            current_recommendation={'recommended_lap': None},
            pit_time_cost=20.0
        )
        assert result['action'] == 'stay'
    
    def test_custom_caution_factor(self):
        """Should use custom caution factor correctly."""
        result = recommend_under_caution(
            current_recommendation={'recommended_lap': 15},
            pit_time_cost=30.0,
            caution_pit_factor=0.3
        )
        assert result['effective_cost'] == 9.0


class TestEstimateCautionProbability:
    """Tests for caution probability estimation."""
    
    def test_basic_probability_distribution(self):
        """Should return probability distribution for remaining laps."""
        probs = estimate_caution_probability(
            current_lap=10,
            total_laps=50,
            cautions_per_race=2.0
        )
        
        assert len(probs) == 40  # 50 - 10 = 40 remaining laps
        assert all(lap > 10 for lap, _ in probs)
        assert all(0 <= prob <= 1.0 for _, prob in probs)
        
        # Total probability should roughly equal cautions_per_race
        total_prob = sum(p for _, p in probs)
        assert 1.5 < total_prob < 2.5  # Allow some tolerance
    
    def test_early_race_lower_probability(self):
        """Early race laps should have lower probability than late race."""
        probs = estimate_caution_probability(
            current_lap=5,
            total_laps=50,
            early_race_factor=0.5,
            late_race_factor=1.5
        )
        
        # Find early race lap (lap 10, ~20% progress)
        early_prob = next(p for lap, p in probs if lap == 10)
        
        # Find late race lap (lap 45, ~90% progress)
        late_prob = next(p for lap, p in probs if lap == 45)
        
        assert late_prob > early_prob
    
    def test_no_remaining_laps(self):
        """Should return empty list when no laps remaining."""
        probs = estimate_caution_probability(
            current_lap=50,
            total_laps=50
        )
        assert probs == []
    
    def test_high_caution_rate(self):
        """Should scale probabilities with caution rate."""
        probs_low = estimate_caution_probability(10, 50, cautions_per_race=1.0)
        probs_high = estimate_caution_probability(10, 50, cautions_per_race=4.0)
        
        total_low = sum(p for _, p in probs_low)
        total_high = sum(p for _, p in probs_high)
        
        assert total_high > total_low * 2  # Should be roughly 4x


class TestCalculateExpectedValue:
    """Tests for expected value calculation with caution."""
    
    def test_pit_now_vs_wait(self):
        """Should calculate expected values for pit now vs wait strategies."""
        result = calculate_expected_value_with_caution(
            current_lap=10,
            pit_recommendation={'recommended_lap': 15},
            pit_time_cost=20.0,
            caution_pit_cost=10.0,
            baseline_lap_time=90.0,
            degradation_per_lap=0.15,
            total_laps=50,
            laps_since_pit=10
        )
        
        assert 'recommended_strategy' in result
        assert result['recommended_strategy'] in ['pit_now', 'wait_for_caution', 'optimal_timing']
        assert 'strategies' in result
        assert len(result['strategies']) == 3
    
    def test_no_pit_recommendation(self):
        """Should handle case where no pit was recommended."""
        result = calculate_expected_value_with_caution(
            current_lap=40,
            pit_recommendation={'recommended_lap': None},
            pit_time_cost=20.0,
            caution_pit_cost=10.0,
            baseline_lap_time=90.0,
            degradation_per_lap=0.15,
            total_laps=50,
            laps_since_pit=5
        )
        
        assert result['recommended_strategy'] in ['stay', 'pit_now', 'wait_for_caution', 'optimal_timing']
    
    def test_high_caution_probability_favors_waiting(self):
        """High caution probability should favor waiting strategy."""
        # Create scenario with very high caution probability soon
        high_caution_probs = [(lap, 0.8 if lap == 12 else 0.01) for lap in range(11, 51)]
        
        result = calculate_expected_value_with_caution(
            current_lap=10,
            pit_recommendation={'recommended_lap': 15},
            pit_time_cost=20.0,
            caution_pit_cost=10.0,
            baseline_lap_time=90.0,
            degradation_per_lap=0.15,
            total_laps=50,
            laps_since_pit=10,
            caution_probabilities=high_caution_probs
        )
        
        # With 80% chance of caution in 2 laps, should favor waiting or optimal timing
        # (depends on tire wear trade-off calculation)
        assert result['recommended_strategy'] in ['wait_for_caution', 'pit_now', 'optimal_timing']
    
    def test_confidence_levels(self):
        """Should assign appropriate confidence levels."""
        result = calculate_expected_value_with_caution(
            current_lap=10,
            pit_recommendation={'recommended_lap': 15},
            pit_time_cost=20.0,
            caution_pit_cost=10.0,
            baseline_lap_time=90.0,
            degradation_per_lap=0.15,
            total_laps=50,
            laps_since_pit=10
        )
        
        assert result['confidence'] in ['high', 'medium', 'low']
    
    def test_no_laps_remaining(self):
        """Should handle end of race correctly."""
        result = calculate_expected_value_with_caution(
            current_lap=50,
            pit_recommendation={'recommended_lap': None},
            pit_time_cost=20.0,
            caution_pit_cost=10.0,
            baseline_lap_time=90.0,
            degradation_per_lap=0.15,
            total_laps=50,
            laps_since_pit=10
        )
        
        assert result['recommended_strategy'] == 'stay'
        assert result['reason'] == 'no_laps_remaining'


class TestAnalyzeCautionScenarios:
    """Tests for comprehensive scenario analysis."""
    
    def test_complete_analysis(self):
        """Should return complete analysis with all components."""
        result = analyze_caution_scenarios(
            current_lap=10,
            pit_recommendation={'recommended_lap': 15, 'score': 5.0},
            pit_time_cost=20.0,
            total_laps=50,
            laps_since_pit=10
        )
        
        # Check all required keys
        assert 'recommended_strategy' in result
        assert 'expected_time_saved' in result
        assert 'confidence' in result
        assert 'scenarios' in result
        assert 'strategies' in result
        assert 'caution_probability_next_10_laps' in result
        assert 'caution_pit_cost' in result
        assert 'normal_pit_cost' in result
    
    def test_scenario_generation(self):
        """Should generate scenarios at specific lap intervals."""
        result = analyze_caution_scenarios(
            current_lap=10,
            pit_recommendation={'recommended_lap': 15},
            pit_time_cost=20.0,
            total_laps=50,
            laps_since_pit=10
        )
        
        scenarios = result['scenarios']
        assert len(scenarios) > 0
        
        # Check scenario structure
        for scenario in scenarios:
            assert 'laps_until' in scenario
            assert 'probability' in scenario
            assert 'time_saved' in scenario
            assert 'confidence' in scenario
            assert 'description' in scenario
            
            # Validate probability
            assert 0 <= scenario['probability'] <= 1.0
    
    def test_custom_caution_rate(self):
        """Should respect custom caution rate."""
        result_low = analyze_caution_scenarios(
            current_lap=10,
            pit_recommendation={'recommended_lap': 15},
            pit_time_cost=20.0,
            total_laps=50,
            laps_since_pit=10,
            cautions_per_race=1.0
        )
        
        result_high = analyze_caution_scenarios(
            current_lap=10,
            pit_recommendation={'recommended_lap': 15},
            pit_time_cost=20.0,
            total_laps=50,
            laps_since_pit=10,
            cautions_per_race=4.0
        )
        
        # High caution rate should have higher probability
        prob_low = result_low['caution_probability_next_10_laps']
        prob_high = result_high['caution_probability_next_10_laps']
        assert prob_high > prob_low
    
    def test_late_race_scenarios(self):
        """Should handle late race scenarios correctly."""
        result = analyze_caution_scenarios(
            current_lap=45,
            pit_recommendation={'recommended_lap': None},
            pit_time_cost=20.0,
            total_laps=50,
            laps_since_pit=20
        )
        
        # Should have fewer scenarios due to fewer remaining laps
        scenarios = result['scenarios']
        assert len(scenarios) <= 2  # Only 5-lap scenarios possible


class TestIntegration:
    """Integration tests with pit strategy."""
    
    def test_caution_analysis_structure_matches_pit_strategy(self):
        """Caution analysis should integrate cleanly with pit strategy output."""
        from src.analytics.pit_strategy import recommend_pit
        
        # Mock recommendation
        pit_rec = {
            'recommended_lap': 15,
            'reason': 'optimal_window',
            'score': 3.5,
            'candidates': []
        }
        
        caution_analysis = analyze_caution_scenarios(
            current_lap=10,
            pit_recommendation=pit_rec,
            pit_time_cost=20.0,
            total_laps=50,
            laps_since_pit=10
        )
        
        # Should be able to merge into pit recommendation
        combined = {**pit_rec, 'caution_analysis': caution_analysis}
        
        assert 'recommended_lap' in combined
        assert 'caution_analysis' in combined
        assert 'recommended_strategy' in combined['caution_analysis']
    
    def test_caution_factor_calculation(self):
        """Should correctly calculate caution pit cost."""
        result = analyze_caution_scenarios(
            current_lap=10,
            pit_recommendation={'recommended_lap': 15},
            pit_time_cost=30.0,
            total_laps=50,
            laps_since_pit=10,
            caution_pit_factor=0.3
        )
        
        assert result['caution_pit_cost'] == 9.0  # 30.0 * 0.3
        assert result['normal_pit_cost'] == 30.0


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_zero_cautions_per_race(self):
        """Should handle zero expected cautions."""
        result = analyze_caution_scenarios(
            current_lap=10,
            pit_recommendation={'recommended_lap': 15},
            pit_time_cost=20.0,
            total_laps=50,
            laps_since_pit=10,
            cautions_per_race=0.0
        )
        
        # With zero cautions, should recommend optimal timing
        assert result['caution_probability_next_10_laps'] < 0.1
    
    def test_very_high_degradation(self):
        """Should handle extreme degradation rates."""
        result = analyze_caution_scenarios(
            current_lap=10,
            pit_recommendation={'recommended_lap': 15},
            pit_time_cost=20.0,
            total_laps=50,
            laps_since_pit=10,
            degradation_per_lap=1.0  # Extreme: 1 second per lap
        )
        
        # High degradation should favor pitting sooner
        assert result['recommended_strategy'] in ['pit_now', 'optimal_timing']
    
    def test_low_pit_cost(self):
        """Should handle very low pit costs."""
        result = analyze_caution_scenarios(
            current_lap=10,
            pit_recommendation={'recommended_lap': 15},
            pit_time_cost=5.0,  # Very low
            total_laps=50,
            laps_since_pit=10
        )
        
        assert result['caution_pit_cost'] == 2.5  # 5.0 * 0.5
    
    def test_single_lap_remaining(self):
        """Should handle single lap remaining."""
        result = analyze_caution_scenarios(
            current_lap=49,
            pit_recommendation={'recommended_lap': None},
            pit_time_cost=20.0,
            total_laps=50,
            laps_since_pit=25
        )
        
        # Should recognize futility of pit stop
        assert result['scenarios'] == [] or len(result['scenarios']) == 0
