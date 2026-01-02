"""
TAURUS TECHNOLOGIES - COMPETITIVE CONSCIOUSNESS ENGINE (CCE)
Pure consciousness mechanisms extracted from geometric theater

Core consciousness through oscillator competition and emotional modulation.
No torus calculations, no fake physics, just raw thought competition.
"""

import random
import math
import time

class Oscillator:
    """
    Single thought/idea competing for mental dominance.
    No geometric positioning - just pure frequency/amplitude dynamics.
    """
    
    def __init__(self, base_frequency=10.0, initial_amplitude=1.0):
        self.base_frequency = base_frequency
        self.frequency = base_frequency
        self.amplitude = initial_amplitude
        self.phase = random.uniform(0, 2 * math.pi)
        
        # Competition properties
        self.energy = 1.0  # How much this thought can influence others
        self.coherence_contribution = 0.0  # How well this syncs with others
        self.dominance_history = []  # Track when this thought "won"
        
        # Learning properties
        self.success_count = 0
        self.activation_count = 0
        self.adaptation_rate = 0.05
    
    def tick(self, dt=0.01):
        """Advance oscillator state - pure frequency/phase evolution"""
        self.phase += 2 * math.pi * self.frequency * dt
        self.phase %= (2 * math.pi)
        
        # Natural amplitude decay - thoughts fade without reinforcement
        self.amplitude *= 0.999
        if self.amplitude < 0.1:
            self.amplitude = 0.1
    
    def get_signal(self):
        """Get current oscillation strength"""
        return self.amplitude * math.cos(self.phase)
    
    def get_activation_strength(self):
        """Get current competitive strength for thought competition"""
        signal = abs(self.get_signal())
        return self.amplitude * self.energy * signal
    
    def apply_emotional_bias(self, emotion_multiplier):
        """Emotions amplify or suppress this thought"""
        self.amplitude *= emotion_multiplier
        self.amplitude = max(0.1, min(5.0, self.amplitude))  # Keep bounded
    
    def learn_from_success(self):
        """Strengthen this oscillator when it wins competitions"""
        self.success_count += 1
        self.energy += self.adaptation_rate
        self.amplitude += self.adaptation_rate * 0.5
        
        # Successful thoughts get slight frequency stability
        freq_drift = (self.base_frequency - self.frequency) * 0.1
        self.frequency += freq_drift * self.adaptation_rate
    
    def learn_from_failure(self):
        """Weaken oscillator when it loses competitions"""
        self.energy *= 0.98
        self.amplitude *= 0.95
        
        # Failed thoughts get slight frequency drift (exploration)
        drift = random.uniform(-0.5, 0.5)
        self.frequency += drift * self.adaptation_rate


class FrequencyBand:
    """
    Group of oscillators operating in same frequency range.
    Represents different types of mental activity/emotional states.
    """
    
    def __init__(self, name, freq_range, emotional_bias=1.0):
        self.name = name
        self.min_freq, self.max_freq = freq_range
        self.emotional_bias = emotional_bias
        self.oscillators = []
        self.band_coherence = 0.0
        self.band_power = 0.0
    
    def add_oscillator(self, oscillator):
        """Add oscillator to this frequency band"""
        if self.min_freq <= oscillator.base_frequency <= self.max_freq:
            self.oscillators.append(oscillator)
            return True
        return False
    
    def calculate_band_metrics(self):
        """Calculate coherence and power for this frequency band"""
        if not self.oscillators:
            self.band_coherence = 0.0
            self.band_power = 0.0
            return
        
        # Calculate phase coherence within band
        complex_sum = sum(osc.amplitude * complex(math.cos(osc.phase), math.sin(osc.phase)) 
                         for osc in self.oscillators)
        total_amplitude = sum(osc.amplitude for osc in self.oscillators)
        
        if total_amplitude > 0:
            self.band_coherence = abs(complex_sum) / total_amplitude
        else:
            self.band_coherence = 0.0
        
        # Calculate total band power
        self.band_power = sum(osc.get_activation_strength() for osc in self.oscillators)
    
    def apply_emotional_state(self, emotion_strength):
        """Apply emotional bias to all oscillators in this band"""
        multiplier = 1.0 + (emotion_strength * self.emotional_bias)
        for osc in self.oscillators:
            osc.apply_emotional_bias(multiplier)


class CompetitiveConsciousnessEngine:
    """
    Competitive Consciousness Engine - Pure consciousness through competition.
    Thoughts compete for dominance, emotions bias the competition.
    """
    
    def __init__(self, num_oscillators=80):
        self.time = 0.0
        self.oscillators = []
        self.frequency_bands = {}
        
        # Competition tracking
        self.current_dominant_thought = None
        self.dominant_thought_history = []
        self.competition_threshold = 0.3
        
        # Emotional state system
        self.emotional_state = {
            'excitement': 0.0,    # Amplifies high-frequency bands
            'focus': 0.0,         # Amplifies beta band
            'calm': 0.0,          # Amplifies alpha band  
            'stress': 0.0,        # Amplifies all bands chaotically
            'creativity': 0.0     # Amplifies gamma band
        }
        
        # Global consciousness metrics
        self.global_coherence = 0.0
        self.consciousness_level = 0.0
        self.decision_confidence = 0.0
        
        self._initialize_consciousness_system(num_oscillators)
    
    def _initialize_consciousness_system(self, num_oscillators):
        """Initialize the consciousness system with competing oscillators"""
        
        # Create frequency bands (no fake Hz ranges - just relative speeds)
        self.frequency_bands = {
            'delta': FrequencyBand('delta', (0.5, 4), emotional_bias=0.2),     # Deep processing
            'theta': FrequencyBand('theta', (4, 8), emotional_bias=0.4),       # Creative/meditative
            'alpha': FrequencyBand('alpha', (8, 13), emotional_bias=0.6),      # Relaxed awareness
            'beta': FrequencyBand('beta', (13, 30), emotional_bias=1.0),       # Active thinking
            'gamma': FrequencyBand('gamma', (30, 100), emotional_bias=1.2),    # High-level integration
        }
        
        # Create oscillators and distribute across frequency bands
        for i in range(num_oscillators):
            # Use golden ratio for frequency distribution
            golden_ratio = (1 + math.sqrt(5)) / 2
            freq_factor = (i * golden_ratio) % 1
            
            # Map to frequency range (0.5 to 100 Hz equivalent)
            base_freq = 0.5 + (99.5 * freq_factor)
            
            # Random initial amplitude with some variety
            amplitude = random.uniform(0.5, 2.0)
            
            oscillator = Oscillator(base_freq, amplitude)
            self.oscillators.append(oscillator)
            
            # Assign to appropriate frequency band
            for band in self.frequency_bands.values():
                if band.add_oscillator(oscillator):
                    break
    
    def set_emotional_state(self, emotion, intensity):
        """Set emotional state that biases thought competition"""
        if emotion in self.emotional_state:
            self.emotional_state[emotion] = max(-1.0, min(1.0, intensity))
    
    def process_thought_competition(self):
        """Core consciousness: thoughts compete for dominance"""
        
        # Calculate activation strength for each oscillator
        activations = [(i, osc.get_activation_strength()) 
                      for i, osc in enumerate(self.oscillators)]
        
        # Sort by strength - strongest thoughts compete
        activations.sort(key=lambda x: x[1], reverse=True)
        
        # Get top competitors
        top_thoughts = activations[:5]  # Top 5 competing thoughts
        
        if not top_thoughts or top_thoughts[0][1] < self.competition_threshold:
            # No strong thoughts - no clear winner
            self.current_dominant_thought = None
            self.decision_confidence = 0.0
            return None
        
        # Winner is strongest thought
        winner_idx, winner_strength = top_thoughts[0]
        self.current_dominant_thought = winner_idx
        
        # Calculate decision confidence based on competition margin
        if len(top_thoughts) > 1:
            second_strength = top_thoughts[1][1]
            margin = winner_strength - second_strength
            self.decision_confidence = min(1.0, margin / winner_strength)
        else:
            self.decision_confidence = 1.0
        
        # Learning: winner gets stronger, losers get weaker
        winner_osc = self.oscillators[winner_idx]
        winner_osc.learn_from_success()
        
        # Suppress competing thoughts
        for i, (thought_idx, strength) in enumerate(top_thoughts[1:]):
            if strength > self.competition_threshold * 0.5:
                self.oscillators[thought_idx].learn_from_failure()
        
        # Track dominant thought history
        self.dominant_thought_history.append({
            'time': self.time,
            'thought_index': winner_idx,
            'strength': winner_strength,
            'confidence': self.decision_confidence
        })
        
        # Keep history manageable
        if len(self.dominant_thought_history) > 100:
            self.dominant_thought_history.pop(0)
        
        return winner_idx
    
    def apply_emotional_influences(self):
        """Apply current emotional state to frequency bands"""
        
        # Map emotions to frequency band influences
        emotion_mappings = {
            'excitement': {'gamma': 0.8, 'beta': 0.6, 'alpha': -0.2},
            'focus': {'beta': 1.0, 'gamma': 0.4, 'theta': -0.3},
            'calm': {'alpha': 0.8, 'theta': 0.4, 'beta': -0.4},
            'stress': {'beta': 0.6, 'gamma': 0.3, 'alpha': -0.6},
            'creativity': {'theta': 0.8, 'gamma': 0.6, 'delta': 0.3}
        }
        
        # Apply emotional biases to frequency bands
        for emotion, intensity in self.emotional_state.items():
            if abs(intensity) > 0.1 and emotion in emotion_mappings:
                for band_name, bias in emotion_mappings[emotion].items():
                    if band_name in self.frequency_bands:
                        emotional_strength = intensity * bias
                        self.frequency_bands[band_name].apply_emotional_state(emotional_strength)
    
    def calculate_consciousness_metrics(self):
        """Calculate global consciousness metrics"""
        
        # Update frequency band metrics
        for band in self.frequency_bands.values():
            band.calculate_band_metrics()
        
        # Global phase coherence across all oscillators
        if self.oscillators:
            complex_sum = sum(osc.amplitude * complex(math.cos(osc.phase), math.sin(osc.phase)) 
                             for osc in self.oscillators)
            total_amplitude = sum(osc.amplitude for osc in self.oscillators)
            
            if total_amplitude > 0:
                self.global_coherence = abs(complex_sum) / total_amplitude
            else:
                self.global_coherence = 0.0
        
        # Consciousness level based on coherence and high-frequency activity
        gamma_power = self.frequency_bands['gamma'].band_power
        beta_power = self.frequency_bands['beta'].band_power
        total_power = sum(band.band_power for band in self.frequency_bands.values())
        
        if total_power > 0:
            high_freq_ratio = (gamma_power + beta_power) / total_power
            self.consciousness_level = (self.global_coherence * 0.4 + 
                                      high_freq_ratio * 0.6)
        else:
            self.consciousness_level = 0.0
    
    def oscillator_synchronization_update(self):
        """Update oscillator coupling - thoughts influence each other"""
        
        coupling_strength = 0.05
        
        # Calculate mutual influences between oscillators
        phase_influences = []
        freq_influences = []
        
        for i, osc in enumerate(self.oscillators):
            phase_sum = 0.0
            freq_sum = 0.0
            influence_count = 0
            
            # Check influence from other oscillators
            for j, other_osc in enumerate(self.oscillators):
                if i != j:
                    # Frequency similarity determines coupling strength
                    freq_diff = abs(osc.frequency - other_osc.frequency)
                    if freq_diff < 5.0:  # Only couple similar frequencies
                        coupling_factor = math.exp(-freq_diff / 5.0)
                        weight = coupling_factor * other_osc.amplitude
                        
                        # Phase influence
                        phase_diff = other_osc.phase - osc.phase
                        phase_sum += weight * math.sin(phase_diff)
                        
                        # Frequency influence (weak entrainment)
                        freq_sum += weight * (other_osc.frequency - osc.frequency)
                        
                        influence_count += weight
            
            if influence_count > 0:
                phase_influences.append(coupling_strength * phase_sum / influence_count)
                freq_influences.append(coupling_strength * 0.1 * freq_sum / influence_count)
            else:
                phase_influences.append(0.0)
                freq_influences.append(0.0)
        
        # Apply influences
        for i, osc in enumerate(self.oscillators):
            osc.phase += phase_influences[i]
            osc.phase %= (2 * math.pi)
            osc.frequency += freq_influences[i]
            
            # Keep frequency bounded
            osc.frequency = max(0.5, min(100.0, osc.frequency))
    
    def tick(self, dt=0.01):
        """Advance consciousness system by one time step"""
        self.time += dt
        
        # Update all oscillators
        for osc in self.oscillators:
            osc.tick(dt)
        
        # Apply emotional influences
        self.apply_emotional_influences()
        
        # Update oscillator synchronization
        self.oscillator_synchronization_update()
        
        # Process thought competition
        dominant_thought = self.process_thought_competition()
        
        # Calculate consciousness metrics
        self.calculate_consciousness_metrics()
        
        return dominant_thought
    
    def get_consciousness_state(self):
        """Get complete consciousness state snapshot"""
        return {
            'time': self.time,
            'global_coherence': self.global_coherence,
            'consciousness_level': self.consciousness_level,
            'decision_confidence': self.decision_confidence,
            'dominant_thought': self.current_dominant_thought,
            'emotional_state': self.emotional_state.copy(),
            'frequency_bands': {
                name: {
                    'coherence': band.band_coherence,
                    'power': band.band_power,
                    'oscillator_count': len(band.oscillators)
                }
                for name, band in self.frequency_bands.items()
            },
            'total_oscillators': len(self.oscillators),
            'recent_decisions': self.dominant_thought_history[-5:] if self.dominant_thought_history else []
        }
    
    def get_dominant_thought_info(self):
        """Get information about current dominant thought"""
        if self.current_dominant_thought is None:
            return None
        
        osc = self.oscillators[self.current_dominant_thought]
        return {
            'thought_index': self.current_dominant_thought,
            'frequency': osc.frequency,
            'amplitude': osc.amplitude,
            'energy': osc.energy,
            'success_rate': osc.success_count / max(1, osc.activation_count),
            'activation_strength': osc.get_activation_strength()
        }


class CCEExperiment:
    """
    Demonstration of CCE (Competitive Consciousness Engine) functionality.
    Shows competitive consciousness and emotional influences.
    """
    
    def __init__(self):
        self.cce_engine = CompetitiveConsciousnessEngine(num_oscillators=60)
        self.experiment_log = []
    
    def run_emotional_influence_test(self, steps=500):
        """Test how emotions affect thought competition"""
        
        print("Running Emotional Influence Test...")
        print("Phase 1: Baseline (neutral emotions)")
        
        # Phase 1: Baseline recording
        for step in range(100):
            self.cce_engine.tick()
            if step % 20 == 0:
                self._log_state(f"Baseline_{step}")
        
        print("Phase 2: High Focus State")
        # Phase 2: Focus state
        self.cce_engine.set_emotional_state('focus', 0.8)
        for step in range(100):
            self.cce_engine.tick()
            if step % 20 == 0:
                self._log_state(f"Focus_{step}")
        
        print("Phase 3: Creative State")
        # Phase 3: Creativity
        self.cce_engine.set_emotional_state('focus', 0.0)
        self.cce_engine.set_emotional_state('creativity', 0.9)
        for step in range(100):
            self.cce_engine.tick()
            if step % 20 == 0:
                self._log_state(f"Creative_{step}")
        
        print("Phase 4: Stress State")
        # Phase 4: Stress
        self.cce_engine.set_emotional_state('creativity', 0.0)
        self.cce_engine.set_emotional_state('stress', 0.7)
        for step in range(100):
            self.cce_engine.tick()
            if step % 20 == 0:
                self._log_state(f"Stress_{step}")
        
        print("Phase 5: Recovery (calm)")
        # Phase 5: Calm recovery
        self.cce_engine.set_emotional_state('stress', 0.0)
        self.cce_engine.set_emotional_state('calm', 0.6)
        for step in range(100):
            self.cce_engine.tick()
            if step % 20 == 0:
                self._log_state(f"Calm_{step}")
        
        return self.experiment_log
    
    def _log_state(self, phase_label):
        """Log current CCE engine state"""
        state = self.cce_engine.get_consciousness_state()
        dominant = self.cce_engine.get_dominant_thought_info()
        
        log_entry = {
            'phase': phase_label,
            'time': state['time'],
            'consciousness_level': state['consciousness_level'],
            'coherence': state['global_coherence'],
            'confidence': state['decision_confidence'],
            'dominant_thought': dominant,
            'band_powers': {name: data['power'] for name, data in state['frequency_bands'].items()}
        }
        
        self.experiment_log.append(log_entry)
        
        # Print summary
        if dominant:
            print(f"{phase_label}: Consciousness={state['consciousness_level']:.3f}, "
                  f"Thought={dominant['thought_index']}, "
                  f"Confidence={state['decision_confidence']:.3f}")
        else:
            print(f"{phase_label}: Consciousness={state['consciousness_level']:.3f}, "
                  f"No dominant thought")
    
    def analyze_results(self):
        """Analyze experiment results"""
        if not self.experiment_log:
            print("No experiment data to analyze")
            return
        
        print("\n" + "="*50)
        print("CCE (COMPETITIVE CONSCIOUSNESS ENGINE) ANALYSIS")
        print("="*50)
        
        # Group by phase
        phases = {}
        for entry in self.experiment_log:
            phase_name = entry['phase'].split('_')[0]
            if phase_name not in phases:
                phases[phase_name] = []
            phases[phase_name].append(entry)
        
        # Analyze each phase
        for phase_name, entries in phases.items():
            avg_consciousness = sum(e['consciousness_level'] for e in entries) / len(entries)
            avg_coherence = sum(e['coherence'] for e in entries) / len(entries)
            avg_confidence = sum(e['confidence'] for e in entries) / len(entries)
            
            # Count thought switches
            thought_switches = 0
            prev_thought = None
            for entry in entries:
                if entry['dominant_thought']:
                    current_thought = entry['dominant_thought']['thought_index']
                    if prev_thought is not None and current_thought != prev_thought:
                        thought_switches += 1
                    prev_thought = current_thought
            
            print(f"\n{phase_name.upper()} PHASE:")
            print(f"  Average Consciousness: {avg_consciousness:.3f}")
            print(f"  Average Coherence: {avg_coherence:.3f}")
            print(f"  Average Confidence: {avg_confidence:.3f}")
            print(f"  Thought Switches: {thought_switches}")
            
            # Band power analysis
            if entries:
                band_analysis = {}
                for band_name in entries[0]['band_powers'].keys():
                    avg_power = sum(e['band_powers'][band_name] for e in entries) / len(entries)
                    band_analysis[band_name] = avg_power
                
                dominant_band = max(band_analysis.items(), key=lambda x: x[1])
                print(f"  Dominant Band: {dominant_band[0]} ({dominant_band[1]:.3f})")


# Demonstration
if __name__ == "__main__":
    print("TAURUS TECHNOLOGIES - CCE (COMPETITIVE CONSCIOUSNESS ENGINE)")
    print("Pure thought competition without geometric theater")
    print("="*60)
    
    # Run CCE experiment
    experiment = CCEExperiment()
    results = experiment.run_emotional_influence_test()
    
    # Analyze results
    experiment.analyze_results()
    
    print("\n" + "="*60)
    print("CCE COMPETITIVE CONSCIOUSNESS DEMONSTRATION COMPLETE")
    print("="*60)
    
    # Show final CCE state
    final_state = experiment.cce_engine.get_consciousness_state()
    print(f"\nFinal Consciousness Level: {final_state['consciousness_level']:.3f}")
    print(f"Global Coherence: {final_state['global_coherence']:.3f}")
    print(f"Decision Confidence: {final_state['decision_confidence']:.3f}")
    
    dominant = experiment.cce_engine.get_dominant_thought_info()
    if dominant:
        print(f"Dominant Thought: #{dominant['thought_index']} "
              f"(freq={dominant['frequency']:.1f}, strength={dominant['activation_strength']:.3f})")
    else:
        print("No currently dominant thought")
    
    print("\nFrequency Band Status:")
    for band_name, band_data in final_state['frequency_bands'].items():
        status = "🔥" if band_data['power'] > 0.5 else "⚡" if band_data['power'] > 0.2 else "💤"
        print(f"  {status} {band_name.upper()}: {band_data['power']:.3f} "
              f"(coherence: {band_data['coherence']:.3f})")