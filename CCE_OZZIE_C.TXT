/*
//---------------------------------------------------
// cce_phase2.h -- CCE PHASE2 Enhanced Consciousness Engine -- Aussie AI Base Library  
// Created June 2025
// Copyright (c) 2025 Taurus Technologies / Aussie AI Labs Pty Ltd
//---------------------------------------------------

TAURUS INDUSTRIES - CCE PHASE2 UPGRADE - C++ IMPLEMENTATION
Enhanced Competitive Consciousness Engine with Neurological Abstractions

PHASE2 Features:
- Hebbian Learning Network
- Thalamic Gating System
- Binding Group Formation
- Neural Fatigue & Refractory Periods
- Memory Consolidation
- Aussie AI Integration
*/

#ifndef CCE_PHASE2_INCLUDE_HEADER_H
#define CCE_PHASE2_INCLUDE_HEADER_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>

// Aussie AI Base Library includes
#include "aport.h"
#include "aussieai.h"
#include "aassert.h"

//---------------------------------------------------
// CCE PHASE2 Constants and Configuration
//---------------------------------------------------

#define CCE_MAX_OSCILLATORS 100
#define CCE_MAX_FREQUENCY_BANDS 5
#define CCE_MAX_BINDING_GROUPS 20
#define CCE_MAX_COUPLING_CONNECTIONS 50
#define CCE_MAX_EXPERIMENT_LOG 1000
#define CCE_PERFORMANCE_VECTOR_SIZE 10

// Frequency band ranges
#define CCE_DELTA_MIN 0.5f
#define CCE_DELTA_MAX 4.0f
#define CCE_THETA_MIN 4.0f
#define CCE_THETA_MAX 8.0f
#define CCE_ALPHA_MIN 8.0f
#define CCE_ALPHA_MAX 13.0f
#define CCE_BETA_MIN 13.0f
#define CCE_BETA_MAX 30.0f
#define CCE_GAMMA_MIN 30.0f
#define CCE_GAMMA_MAX 100.0f

// PHASE2 enhancement parameters
#define CCE_HEBBIAN_LEARNING_RATE 0.01f
#define CCE_REFRACTORY_MAX_DURATION 2.0f
#define CCE_BINDING_THRESHOLD 0.3f
#define CCE_MEMORY_CONSOLIDATION_THRESHOLD 0.5f
#define CCE_GATING_ADAPTATION_RATE 0.1f

//---------------------------------------------------
// Enhanced Oscillator Structure with Neurological Features
//---------------------------------------------------

typedef struct {
    int oscillator_id;
    float frequency;
    float base_frequency;        // For memory persistence
    float amplitude;
    float phase;
    float energy;
    
    // PHASE2 Neurological Enhancements
    float refractory_period;     // Neural fatigue counter
    float max_refractory;        // Maximum fatigue duration
    float fatigue_rate;          // How quickly fatigue accumulates
    float recovery_rate;         // How quickly we recover
    
    // Hebbian learning properties
    int coupling_oscillator_ids[CCE_MAX_COUPLING_CONNECTIONS];
    float coupling_weights[CCE_MAX_COUPLING_CONNECTIONS];
    int num_couplings;
    float hebbian_strength;      // Learning rate for Hebbian updates
    
    // Memory and adaptation
    float baseline_shift;        // Persistent memory adjustment
    float usage_decay_rate;      // Unused connection decay
    float activation_threshold;
    
    // Performance tracking
    int success_count;
    int failure_count;
    int activation_count;
    float recent_performance[20]; // Ring buffer for recent performance
    int performance_index;
    
    // Aussie AI integration
    float performance_vector[CCE_PERFORMANCE_VECTOR_SIZE];
    bool aussie_enhanced;
    
} CCE_EnhancedOscillator;

//---------------------------------------------------
// Binding Group Structure for Concept Formation
//---------------------------------------------------

typedef struct {
    int group_id;
    int member_oscillator_ids[10];
    int num_members;
    float binding_strength;
    int formation_time;
    bool is_cross_band;          // Cross-frequency band binding
    
} CCE_BindingGroup;

//---------------------------------------------------
// Enhanced Frequency Band with Binding Capabilities
//---------------------------------------------------

typedef struct {
    char name[16];
    float freq_min;
    float freq_max;
    
    int oscillator_ids[CCE_MAX_OSCILLATORS];
    int num_oscillators;
    
    // Enhanced properties
    float band_coherence;
    float band_power;
    
    // Binding groups within this band
    CCE_BindingGroup binding_groups[CCE_MAX_BINDING_GROUPS];
    int num_binding_groups;
    
    // Thalamic gating mechanism
    float gating_factor;         // Attention-based gating (1.0 = neutral)
    float gating_decay;          // How quickly gating returns to baseline
    
    // Cross-band coherence tracking
    float cross_band_coherence[CCE_MAX_FREQUENCY_BANDS];
    
    // Emotional multipliers for this band
    float focus_multiplier;
    float creativity_multiplier;
    float stress_multiplier;
    float calm_multiplier;
    float anxiety_multiplier;
    float curiosity_multiplier;
    
} CCE_EnhancedFrequencyBand;

//---------------------------------------------------
// Hebbian Learning Network Structure
//---------------------------------------------------

typedef struct {
    float global_coupling_matrix[CCE_MAX_OSCILLATORS][CCE_MAX_OSCILLATORS];
    float learning_rate;
    int total_connections;
    float average_coupling_strength;
    float max_coupling_strength;
    float network_density;
    
    // Learning history for analysis
    int learning_events;
    float connection_formation_rate;
    
} CCE_HebbianNetwork;

//---------------------------------------------------
// Thalamic Gating System Structure
//---------------------------------------------------

typedef struct {
    float attention_vector[CCE_MAX_FREQUENCY_BANDS];
    float baseline_gating;
    int recent_adjustments;
    
    // Gating patterns for different emotions
    float focus_pattern[CCE_MAX_FREQUENCY_BANDS];
    float creativity_pattern[CCE_MAX_FREQUENCY_BANDS];
    float stress_pattern[CCE_MAX_FREQUENCY_BANDS];
    float calm_pattern[CCE_MAX_FREQUENCY_BANDS];
    float anxiety_pattern[CCE_MAX_FREQUENCY_BANDS];
    float curiosity_pattern[CCE_MAX_FREQUENCY_BANDS];
    
} CCE_ThalamicGatingSystem;

//---------------------------------------------------
// Neuromodulation System Structure
//---------------------------------------------------

typedef struct {
    // Neurotransmitter levels calculated from emotional state
    float dopamine_level;        // Focus/reward effects
    float serotonin_level;       // Calm/mood effects
    float acetylcholine_level;   // Attention effects
    float norepinephrine_level;  // Stress/arousal effects
    
    // Adaptation rates for different neurotransmitters
    float adaptation_rates[4];
    
    // Modulation history
    int modulation_events;
    
} CCE_NeuromodulationSystem;

//---------------------------------------------------
// Aussie AI Processor Structure
//---------------------------------------------------

typedef struct {
    int num_oscillators;
    int vector_operations;
    int pattern_recognitions;
    int optimizations_applied;
    
    // Pattern recognition data
    float correlation_matrix[CCE_MAX_OSCILLATORS][CCE_MAX_OSCILLATORS];
    float dominant_pattern_strength;
    
    // Optimization metrics
    float optimization_rate;
    int recent_operations;
    
} CCE_AussieAIProcessor;

//---------------------------------------------------
// Emotional State Structure
//---------------------------------------------------

typedef struct {
    float focus;
    float creativity;
    float stress;
    float calm;
    float anxiety;
    float curiosity;
    
} CCE_EmotionalState;

//---------------------------------------------------
// PHASE2 Enhanced Consciousness Metrics
//---------------------------------------------------

typedef struct {
    float consciousness_level;
    float global_coherence;
    float decision_confidence;
    
    // PHASE2 new metrics
    float binding_integration;   // Concept integration level
    float attention_focus;       // Thalamic gating focus level
    float competition_intensity; // How intense thought competition is
    
} CCE_ConsciousnessMetrics;

//---------------------------------------------------
// PHASE2 Performance Metrics
//---------------------------------------------------

typedef struct {
    int hebbian_activations;
    int binding_formations;
    int gating_adjustments;
    int memory_consolidations;
    int refractory_periods;
    
} CCE_Phase2Metrics;

//---------------------------------------------------
// Enhanced CCE Engine Main Structure
//---------------------------------------------------

typedef struct {
    // Core oscillators with neurological enhancements
    CCE_EnhancedOscillator oscillators[CCE_MAX_OSCILLATORS];
    int num_oscillators;
    
    // Enhanced frequency bands with binding
    CCE_EnhancedFrequencyBand frequency_bands[CCE_MAX_FREQUENCY_BANDS];
    int num_frequency_bands;
    
    // PHASE2 neurological subsystems
    CCE_HebbianNetwork hebbian_network;
    CCE_ThalamicGatingSystem thalamic_gate;
    CCE_NeuromodulationSystem neuromodulation;
    CCE_AussieAIProcessor aussie_processor;
    
    // State and metrics
    CCE_EmotionalState emotional_state;
    CCE_ConsciousnessMetrics consciousness_metrics;
    CCE_Phase2Metrics phase2_metrics;
    
    // Competition and dominance tracking
    int current_dominant_thought;
    float dominance_duration;
    int dominant_thought_history[100];
    int history_index;
    
    // Timing
    float time;
    float dt;
    
    // Configuration
    bool aussie_ai_enabled;
    bool hebbian_learning_enabled;
    bool binding_groups_enabled;
    bool thalamic_gating_enabled;
    bool memory_consolidation_enabled;
    
} CCE_EnhancedEngine;

//---------------------------------------------------
// Function Declarations - Enhanced Oscillator Operations
//---------------------------------------------------

void cce_enhanced_oscillator_init(CCE_EnhancedOscillator* osc, int id, float frequency, float energy);
void cce_enhanced_oscillator_tick(CCE_EnhancedOscillator* osc, float dt);
void cce_enhanced_oscillator_strengthen_coupling(CCE_EnhancedOscillator* osc, int other_id, float co_activation);
void cce_enhanced_oscillator_decay_unused_connections(CCE_EnhancedOscillator* osc);
void cce_enhanced_oscillator_enter_refractory(CCE_EnhancedOscillator* osc, float dominance_duration);
void cce_enhanced_oscillator_update_baseline_memory(CCE_EnhancedOscillator* osc, bool success);
float cce_enhanced_oscillator_get_activation_strength(const CCE_EnhancedOscillator* osc);
void cce_enhanced_oscillator_update_performance_vector(CCE_EnhancedOscillator* osc);

//---------------------------------------------------
// Function Declarations - Enhanced Frequency Band Operations
//---------------------------------------------------

void cce_enhanced_band_init(CCE_EnhancedFrequencyBand* band, const char* name, float freq_min, float freq_max);
void cce_enhanced_band_add_oscillator(CCE_EnhancedFrequencyBand* band, int oscillator_id);
void cce_enhanced_band_update_metrics(CCE_EnhancedFrequencyBand* band, CCE_EnhancedOscillator* oscillators);
void cce_enhanced_band_check_for_binding(CCE_EnhancedFrequencyBand* band, CCE_EnhancedOscillator* oscillators);
void cce_enhanced_band_apply_thalamic_gating(CCE_EnhancedFrequencyBand* band, float attention_signal);
float cce_enhanced_band_calculate_phase_coherence(CCE_EnhancedFrequencyBand* band, CCE_EnhancedOscillator* oscillators);

//---------------------------------------------------
// Function Declarations - Hebbian Learning Network
//---------------------------------------------------

void cce_hebbian_network_init(CCE_HebbianNetwork* network);
void cce_hebbian_network_process_learning_cycle(CCE_HebbianNetwork* network, CCE_EnhancedOscillator* oscillators, int num_oscillators, float dt);
void cce_hebbian_network_normalize_coupling_strengths(CCE_HebbianNetwork* network, CCE_EnhancedOscillator* oscillators, int num_oscillators);
void cce_hebbian_network_update_global_matrix(CCE_HebbianNetwork* network, CCE_EnhancedOscillator* oscillators, int num_oscillators);

//---------------------------------------------------
// Function Declarations - Thalamic Gating System
//---------------------------------------------------

void cce_thalamic_gating_init(CCE_ThalamicGatingSystem* gating);
void cce_thalamic_gating_update(CCE_ThalamicGatingSystem* gating, CCE_EnhancedFrequencyBand* bands, int num_bands, float global_attention_focus, float dt);
void cce_thalamic_gating_apply_emotional_pattern(CCE_ThalamicGatingSystem* gating, CCE_EnhancedFrequencyBand* bands, int num_bands, const char* emotion, float intensity);

//---------------------------------------------------
// Function Declarations - Neuromodulation System
//---------------------------------------------------

void cce_neuromodulation_init(CCE_NeuromodulationSystem* neuromod);
void cce_neuromodulation_calculate_neurotransmitters(CCE_NeuromodulationSystem* neuromod, const CCE_EmotionalState* emotional_state);
void cce_neuromodulation_apply_modulation(CCE_NeuromodulationSystem* neuromod, CCE_EnhancedOscillator* oscillators, int num_oscillators, float dt);

//---------------------------------------------------
// Function Declarations - Aussie AI Processor
//---------------------------------------------------

void cce_aussie_processor_init(CCE_AussieAIProcessor* processor, int num_oscillators);
void cce_aussie_processor_process_consciousness_state(CCE_AussieAIProcessor* processor, CCE_EnhancedEngine* engine, float dt);
float cce_aussie_processor_recognize_oscillator_patterns(CCE_AussieAIProcessor* processor, CCE_EnhancedOscillator* oscillators, int num_oscillators);
void cce_aussie_processor_apply_optimizations(CCE_AussieAIProcessor* processor, CCE_EnhancedEngine* engine, float pattern_strength);

//---------------------------------------------------
// Function Declarations - Enhanced CCE Engine Core
//---------------------------------------------------

void cce_enhanced_engine_init(CCE_EnhancedEngine* engine, int num_oscillators, bool aussie_ai_enabled);
void cce_enhanced_engine_create_oscillators(CCE_EnhancedEngine* engine);
void cce_enhanced_engine_create_frequency_bands(CCE_EnhancedEngine* engine);
void cce_enhanced_engine_tick(CCE_EnhancedEngine* engine);
int cce_enhanced_engine_process_competition(CCE_EnhancedEngine* engine);
void cce_enhanced_engine_update_consciousness_metrics(CCE_EnhancedEngine* engine);
void cce_enhanced_engine_set_emotional_state(CCE_EnhancedEngine* engine, const char* emotion, float intensity, float target_attention);

//---------------------------------------------------
// Function Declarations - Utility and Analysis
//---------------------------------------------------

void cce_enhanced_engine_print_state(const CCE_EnhancedEngine* engine);
void cce_enhanced_engine_print_dominant_thought_info(const CCE_EnhancedEngine* engine);
void cce_enhanced_engine_print_frequency_bands(const CCE_EnhancedEngine* engine);
void cce_enhanced_engine_print_phase2_metrics(const CCE_EnhancedEngine* engine);
void cce_enhanced_engine_print_hebbian_network_state(const CCE_EnhancedEngine* engine);

//---------------------------------------------------
// Math utility functions
//---------------------------------------------------

float cce_random_float(float min, float max);
float cce_clamp(float value, float min, float max);
float cce_sigmoid(float x);
int cce_find_oscillator_band(const CCE_EnhancedEngine* engine, int oscillator_id);

#endif // CCE_PHASE2_INCLUDE_HEADER_H

//---------------------------------------------------
// End of Section 1: Core Headers and Structures
// Ready for Section 2: Enhanced Oscillator Implementation
//---------------------------------------------------


/*
//---------------------------------------------------
// cce_phase2_oscillators.cpp -- Enhanced Oscillator Implementation -- Aussie AI Base Library  
// Created June 2025
// Copyright (c) 2025 Taurus Technologies / Aussie AI Labs Pty Ltd
//---------------------------------------------------

TAURUS TECHNOLOGIES - CCE PHASE2 UPGRADE - C++ IMPLEMENTATION
Section 2: Enhanced Oscillator Implementation with Neurological Features

Features implemented:
- Refractory periods (neural fatigue)
- Hebbian coupling with other oscillators  
- Memory persistence through baseline adjustments
- Performance vector tracking for Aussie AI
- Adaptive decay rates based on usage
*/

#include "cce_phase2.h"

//---------------------------------------------------
// Enhanced Oscillator Initialization
//---------------------------------------------------

void cce_enhanced_oscillator_init(CCE_EnhancedOscillator* osc, int id, float frequency, float energy)
{
    yassert(osc != NULL);
    yassert(frequency > 0.0f);
    yassert(energy > 0.0f);
    
    // Core oscillator properties
    osc->oscillator_id = id;
    osc->frequency = frequency;
    osc->base_frequency = frequency;  // For memory persistence
    osc->amplitude = 1.0f;
    osc->phase = cce_random_float(0.0f, 2.0f * M_PI);
    osc->energy = energy;
    
    // Enhanced neurological properties
    osc->refractory_period = 0.0f;
    osc->max_refractory = CCE_REFRACTORY_MAX_DURATION;
    osc->fatigue_rate = 0.1f;
    osc->recovery_rate = 0.05f;
    
    // Hebbian learning properties
    osc->num_couplings = 0;
    osc->hebbian_strength = CCE_HEBBIAN_LEARNING_RATE;
    for (int i = 0; i < CCE_MAX_COUPLING_CONNECTIONS; i++) {
        osc->coupling_oscillator_ids[i] = -1;
        osc->coupling_weights[i] = 0.0f;
    }
    
    // Memory and adaptation
    osc->baseline_shift = 0.0f;
    osc->usage_decay_rate = 0.001f;
    osc->activation_threshold = 0.5f;
    
    // Success tracking
    osc->success_count = 0;
    osc->failure_count = 0;
    osc->activation_count = 0;
    osc->performance_index = 0;
    for (int i = 0; i < 20; i++) {
        osc->recent_performance[i] = 0.5f;  // Neutral start
    }
    
    // Aussie AI integration
    osc->aussie_enhanced = true;
    for (int i = 0; i < CCE_PERFORMANCE_VECTOR_SIZE; i++) {
        osc->performance_vector[i] = 0.0f;
    }
    
    fprintf(stdout, "Enhanced Oscillator %d initialized: freq=%.2fHz, energy=%.3f\n", 
            id, frequency, energy);
}

//---------------------------------------------------
// Enhanced Oscillator Tick - Core Update Function
//---------------------------------------------------

void cce_enhanced_oscillator_tick(CCE_EnhancedOscillator* osc, float dt)
{
    yassert(osc != NULL);
    yassert(dt > 0.0f);
    
    // Update phase with memory-adjusted frequency
    float effective_frequency = osc->frequency + osc->baseline_shift;
    osc->phase += 2.0f * M_PI * effective_frequency * dt;
    if (osc->phase > 2.0f * M_PI) {
        osc->phase -= 2.0f * M_PI;
    }
    
    // Calculate base amplitude with refractory consideration
    if (osc->refractory_period > 0.0f) {
        // Reduced amplitude during refractory period
        float refractory_factor = fmaxf(0.1f, 1.0f - (osc->refractory_period / osc->max_refractory));
        osc->amplitude = osc->energy * sinf(osc->phase) * refractory_factor;
        
        // Recovery from refractory period
        osc->refractory_period = fmaxf(0.0f, osc->refractory_period - osc->recovery_rate * dt);
    } else {
        osc->amplitude = osc->energy * sinf(osc->phase);
    }
    
    // Apply Hebbian coupling influences
    cce_enhanced_oscillator_apply_hebbian_coupling(osc);
    
    // Decay unused connections
    cce_enhanced_oscillator_decay_unused_connections(osc);
    
    // Update performance vector for Aussie AI
    cce_enhanced_oscillator_update_performance_vector(osc);
}

//---------------------------------------------------
// Apply Hebbian Coupling Effects
//---------------------------------------------------

void cce_enhanced_oscillator_apply_hebbian_coupling(CCE_EnhancedOscillator* osc)
{
    yassert(osc != NULL);
    
    if (osc->num_couplings == 0) return;
    
    float coupling_influence = 0.0f;
    for (int i = 0; i < osc->num_couplings; i++) {
        if (osc->coupling_weights[i] > 0.01f) {  // Only consider significant connections
            // Coupling influence proportional to connection strength
            coupling_influence += osc->coupling_weights[i] * 0.1f;
        }
    }
    
    // Apply coupling influence to amplitude
    osc->amplitude += coupling_influence;
    
    // Cap amplitude to prevent runaway amplification
    osc->amplitude = cce_clamp(osc->amplitude, -2.0f, 2.0f);
}

//---------------------------------------------------
// Strengthen Hebbian Coupling with Another Oscillator
//---------------------------------------------------

void cce_enhanced_oscillator_strengthen_coupling(CCE_EnhancedOscillator* osc, int other_id, float co_activation)
{
    yassert(osc != NULL);
    yassert(other_id >= 0);
    yassert(co_activation >= 0.0f);
    
    // Find existing connection or create new one
    int connection_index = -1;
    for (int i = 0; i < osc->num_couplings; i++) {
        if (osc->coupling_oscillator_ids[i] == other_id) {
            connection_index = i;
            break;
        }
    }
    
    // Create new connection if not found and space available
    if (connection_index == -1 && osc->num_couplings < CCE_MAX_COUPLING_CONNECTIONS) {
        connection_index = osc->num_couplings;
        osc->coupling_oscillator_ids[connection_index] = other_id;
        osc->coupling_weights[connection_index] = 0.0f;
        osc->num_couplings++;
    }
    
    // Strengthen connection using Hebbian rule
    if (connection_index >= 0) {
        float learning_amount = osc->hebbian_strength * co_activation;
        osc->coupling_weights[connection_index] += learning_amount;
        
        // Cap maximum coupling strength
        osc->coupling_weights[connection_index] = fminf(1.0f, osc->coupling_weights[connection_index]);
        
        // Count global multiplications for Aussie AI tracking
        g_aussie_multiplications_count++;
    }
}

//---------------------------------------------------
// Decay Unused Hebbian Connections
//---------------------------------------------------

void cce_enhanced_oscillator_decay_unused_connections(CCE_EnhancedOscillator* osc)
{
    yassert(osc != NULL);
    
    for (int i = 0; i < osc->num_couplings; i++) {
        // Decay all connections slowly
        osc->coupling_weights[i] *= (1.0f - osc->usage_decay_rate);
        
        // Remove very weak connections by compacting array
        if (osc->coupling_weights[i] < 0.01f) {
            // Shift remaining connections down
            for (int j = i; j < osc->num_couplings - 1; j++) {
                osc->coupling_oscillator_ids[j] = osc->coupling_oscillator_ids[j + 1];
                osc->coupling_weights[j] = osc->coupling_weights[j + 1];
            }
            osc->num_couplings--;
            i--; // Check this position again after shift
        }
    }
}

//---------------------------------------------------
// Enter Refractory Period After Dominating
//---------------------------------------------------

void cce_enhanced_oscillator_enter_refractory(CCE_EnhancedOscillator* osc, float dominance_duration)
{
    yassert(osc != NULL);
    yassert(dominance_duration >= 0.0f);
    
    // Longer dominance = longer refractory period
    float fatigue_amount = osc->fatigue_rate * dominance_duration;
    osc->refractory_period = fminf(osc->max_refractory, fatigue_amount);
    
    fprintf(stdout, "Oscillator %d entering refractory period: %.3fs (dominance=%.3fs)\n", 
            osc->oscillator_id, osc->refractory_period, dominance_duration);
}

//---------------------------------------------------
// Update Baseline Memory Based on Success
//---------------------------------------------------

void cce_enhanced_oscillator_update_baseline_memory(CCE_EnhancedOscillator* osc, bool success)
{
    yassert(osc != NULL);
    
    if (success) {
        // Successful oscillators get slight frequency boost
        osc->baseline_shift += 0.001f;
        osc->success_count++;
        osc->recent_performance[osc->performance_index] = 1.0f;
    } else {
        // Failed oscillators get slight frequency reduction
        osc->baseline_shift -= 0.0005f;
        osc->failure_count++;
        osc->recent_performance[osc->performance_index] = 0.0f;
    }
    
    // Cap baseline shifts to prevent runaway changes
    osc->baseline_shift = cce_clamp(osc->baseline_shift, -2.0f, 2.0f);
    
    osc->activation_count++;
    osc->performance_index = (osc->performance_index + 1) % 20;
    
    if (success) {
        fprintf(stdout, "Memory consolidation: Oscillator %d baseline shift: %+.4f (success)\n", 
                osc->oscillator_id, osc->baseline_shift);
    }
}

//---------------------------------------------------
// Get Enhanced Activation Strength
//---------------------------------------------------

float cce_enhanced_oscillator_get_activation_strength(const CCE_EnhancedOscillator* osc)
{
    yassert(osc != NULL);
    
    float base_strength = fabsf(osc->amplitude) * osc->energy;
    
    // Reduce strength during refractory period
    if (osc->refractory_period > 0.0f) {
        float refractory_penalty = osc->refractory_period / osc->max_refractory;
        base_strength *= (1.0f - 0.7f * refractory_penalty);
    }
    
    // Boost from Hebbian coupling
    float coupling_boost = 0.0f;
    for (int i = 0; i < osc->num_couplings; i++) {
        coupling_boost += fabsf(osc->coupling_weights[i]);
    }
    coupling_boost *= 0.1f; // Scale coupling influence
    
    // Memory boost from baseline
    float memory_boost = fmaxf(0.0f, osc->baseline_shift * 0.1f);
    
    return base_strength + coupling_boost + memory_boost;
}

//---------------------------------------------------
// Update Performance Vector for Aussie AI Processing
//---------------------------------------------------

void cce_enhanced_oscillator_update_performance_vector(CCE_EnhancedOscillator* osc)
{
    yassert(osc != NULL);
    
    // Calculate recent success rate
    float recent_success_rate = 0.5f; // Default neutral
    if (osc->activation_count > 0) {
        float success_sum = 0.0f;
        int count = 0;
        for (int i = 0; i < 20; i++) {
            success_sum += osc->recent_performance[i];
            count++;
        }
        recent_success_rate = success_sum / (float)count;
    }
    
    // Calculate total coupling strength
    float total_coupling = 0.0f;
    for (int i = 0; i < osc->num_couplings; i++) {
        total_coupling += fabsf(osc->coupling_weights[i]);
    }
    
    // Update 10-dimensional performance vector for Aussie AI
    osc->performance_vector[0] = osc->frequency / 100.0f;          // Normalized frequency
    osc->performance_vector[1] = fabsf(osc->amplitude);            // Amplitude magnitude
    osc->performance_vector[2] = osc->energy;                      // Energy level
    osc->performance_vector
   
  
 


/*
//---------------------------------------------------
// cce_phase2_engine.cpp -- Enhanced CCE Engine Core -- Aussie AI Base Library  
// Created June 2025
// Copyright (c) 2025 Taurus Technologies / Aussie AI Labs Pty Ltd
//---------------------------------------------------

TAURUS TECHNOLOGIES - CCE PHASE2 UPGRADE - C++ IMPLEMENTATION
Section 3: Enhanced Engine Core and Neurological Systems

Systems implemented:
- Enhanced Frequency Bands with Binding Groups
- Hebbian Learning Network
- Thalamic Gating System
- Neuromodulation System
- Aussie AI Processor
- Enhanced CCE Engine Core
*/

#include "cce_phase2.h"

//---------------------------------------------------
// Enhanced Frequency Band Implementation
//---------------------------------------------------

void cce_enhanced_band_init(CCE_EnhancedFrequencyBand* band, const char* name, float freq_min, float freq_max)
{
    yassert(band != NULL);
    yassert(name != NULL);
    yassert(freq_min < freq_max);
    
    strncpy(band->name, name, sizeof(band->name) - 1);
    band->name[sizeof(band->name) - 1] = '\0';
    band->freq_min = freq_min;
    band->freq_max = freq_max;
    
    band->num_oscillators = 0;
    band->band_coherence = 0.0f;
    band->band_power = 0.0f;
    band->num_binding_groups = 0;
    
    // Initialize thalamic gating
    band->gating_factor = 1.0f;
    band->gating_decay = 0.02f;
    
    // Initialize cross-band coherence
    for (int i = 0; i < CCE_MAX_FREQUENCY_BANDS; i++) {
        band->cross_band_coherence[i] = 0.0f;
    }
    
    // Set emotional multipliers based on band type
    if (strcmp(name, "delta") == 0) {
        band->calm_multiplier = 1.5f;
        band->stress_multiplier = 0.7f;
        band->focus_multiplier = 0.8f;
        band->creativity_multiplier = 1.0f;
        band->anxiety_multiplier = 0.8f;
        band->curiosity_multiplier = 0.9f;
    } else if (strcmp(name, "theta") == 0) {
        band->creativity_multiplier = 2.0f;
        band->curiosity_multiplier = 1.8f;
        band->calm_multiplier = 1.3f;
        band->focus_multiplier = 1.0f;
        band->stress_multiplier = 0.8f;
        band->anxiety_multiplier = 0.9f;
    } else if (strcmp(name, "alpha") == 0) {
        band->calm_multiplier = 1.8f;
        band->focus_multiplier = 1.2f;
        band->stress_multiplier = 0.6f;
        band->creativity_multiplier = 1.1f;
        band->anxiety_multiplier = 0.7f;
        band->curiosity_multiplier = 1.0f;
    } else if (strcmp(name, "beta") == 0) {
        band->focus_multiplier = 2.2f;
        band->anxiety_multiplier = 1.5f;
        band->stress_multiplier = 1.8f;
        band->calm_multiplier = 0.8f;
        band->creativity_multiplier = 0.9f;
        band->curiosity_multiplier = 1.2f;
    } else if (strcmp(name, "gamma") == 0) {
        band->focus_multiplier = 1.8f;
        band->creativity_multiplier = 1.5f;
        band->curiosity_multiplier = 2.0f;
        band->calm_multiplier = 0.9f;
        band->stress_multiplier = 1.0f;
        band->anxiety_multiplier = 1.1f;
    }
    
    fprintf(stdout, "Enhanced Frequency Band '%s' initialized: %.1f-%.1fHz\n", 
            name, freq_min, freq_max);
}

void cce_enhanced_band_add_oscillator(CCE_EnhancedFrequencyBand* band, int oscillator_id)
{
    yassert(band != NULL);
    yassert(oscillator_id >= 0);
    
    if (band->num_oscillators < CCE_MAX_OSCILLATORS) {
        band->oscillator_ids[band->num_oscillators] = oscillator_id;
        band->num_oscillators++;
    }
}

void cce_enhanced_band_update_metrics(CCE_EnhancedFrequencyBand* band, CCE_EnhancedOscillator* oscillators)
{
    yassert(band != NULL);
    yassert(oscillators != NULL);
    
    if (band->num_oscillators == 0) {
        band->band_coherence = 0.0f;
        band->band_power = 0.0f;
        return;
    }
    
    // Calculate band power
    float total_power = 0.0f;
    for (int i = 0; i < band->num_oscillators; i++) {
        int osc_id = band->oscillator_ids[i];
        float activation = cce_enhanced_oscillator_get_activation_strength(&oscillators[osc_id]);
        total_power += activation * oscillators[osc_id].energy;
    }
    band->band_power = total_power / (float)band->num_oscillators;
    
    // Apply thalamic gating factor
    band->band_power *= band->gating_factor;
    
    // Calculate phase coherence
    band->band_coherence = cce_enhanced_band_calculate_phase_coherence(band, oscillators);
    
    // Update binding groups
    cce_enhanced_band_check_for_binding(band, oscillators);
}

float cce_enhanced_band_calculate_phase_coherence(CCE_EnhancedFrequencyBand* band, CCE_EnhancedOscillator* oscillators)
{
    yassert(band != NULL);
    yassert(oscillators != NULL);
    
    if (band->num_oscillators < 2) return 1.0f;
    
    // Calculate coherence using circular statistics
    float sum_cos = 0.0f;
    float sum_sin = 0.0f;
    
    for (int i = 0; i < band->num_oscillators; i++) {
        int osc_id = band->oscillator_ids[i];
        float phase = oscillators[osc_id].phase;
        sum_cos += cosf(phase);
        sum_sin += sinf(phase);
    }
    
    // Mean vector magnitude = coherence
    float mean_magnitude = sqrtf(sum_cos * sum_cos + sum_sin * sum_sin) / (float)band->num_oscillators;
    return mean_magnitude;
}

void cce_enhanced_band_check_for_binding(CCE_EnhancedFrequencyBand* band, CCE_EnhancedOscillator* oscillators)
{
    yassert(band != NULL);
    yassert(oscillators != NULL);
    
    // Look for oscillators with strong mutual coupling
    for (int i = 0; i < band->num_oscillators; i++) {
        for (int j = i + 1; j < band->num_oscillators; j++) {
            int osc1_id = band->oscillator_ids[i];
            int osc2_id = band->oscillator_ids[j];
            
            float coupling_potential = cce_calculate_coupling_potential(&oscillators[osc1_id], &oscillators[osc2_id]);
            
            if (coupling_potential > CCE_BINDING_THRESHOLD && band->num_binding_groups < CCE_MAX_BINDING_GROUPS) {
                // Check if they're already in a binding group together
                bool already_bound = false;
                for (int g = 0; g < band->num_binding_groups; g++) {
                    CCE_BindingGroup* group = &band->binding_groups[g];
                    bool has_osc1 = false, has_osc2 = false;
                    for (int m = 0; m < group->num_members; m++) {
                        if (group->member_oscillator_ids[m] == osc1_id) has_osc1 = true;
                        if (group->member_oscillator_ids[m] == osc2_id) has_osc2 = true;
                    }
                    if (has_osc1 && has_osc2) {
                        already_bound = true;
                        break;
                    }
                }
                
                if (!already_bound) {
                    // Create new binding group
                    CCE_BindingGroup* new_group = &band->binding_groups[band->num_binding_groups];
                    new_group->group_id = band->num_binding_groups;
                    new_group->member_oscillator_ids[0] = osc1_id;
                    new_group->member_oscillator_ids[1] = osc2_id;
                    new_group->num_members = 2;
                    new_group->binding_strength = coupling_potential;
                    new_group->formation_time = 0;
                    new_group->is_cross_band = false;
                    
                    band->num_binding_groups++;
                    
                    fprintf(stdout, "Binding group formed in %s band: oscillators %d-%d (strength=%.3f)\n",
                            band->name, osc1_id, osc2_id, coupling_potential);
                }
            }
        }
    }
    
    // Update existing binding groups
    for (int g = 0; g < band->num_binding_groups; g++) {
        CCE_BindingGroup* group = &band->binding_groups[g];
        group->formation_time++;
        
        // Decay binding strength over time if not reinforced
        if (group->formation_time % 10 == 0) {
            group->binding_strength *= 0.98f;
        }
    }
    
    // Remove weak binding groups
    for (int g = band->num_binding_groups - 1; g >= 0; g--) {
        if (band->binding_groups[g].binding_strength < 0.1f) {
            // Shift remaining groups down
            for (int k = g; k < band->num_binding_groups - 1; k++) {
                band->binding_groups[k] = band->binding_groups[k + 1];
            }
            band->num_binding_groups--;
        }
    }
}

void cce_enhanced_band_apply_thalamic_gating(CCE_EnhancedFrequencyBand* band, float attention_signal)
{
    yassert(band != NULL);
    
    float target_gating = 1.0f + attention_signal;
    
    // Smooth transition to target gating
    float gating_diff = target_gating - band->gating_factor;
    band->gating_factor += gating_diff * CCE_GATING_ADAPTATION_RATE;
    
    // Ensure gating stays in reasonable bounds
    band->gating_factor = cce_clamp(band->gating_factor, 0.1f, 3.0f);
}

//---------------------------------------------------
// Hebbian Learning Network Implementation
//---------------------------------------------------

void cce_hebbian_network_init(CCE_HebbianNetwork* network)
{
    yassert(network != NULL);
    
    network->learning_rate = CCE_HEBBIAN_LEARNING_RATE;
    network->total_connections = 0;
    network->average_coupling_strength = 0.0f;
    network->max_coupling_strength = 0.0f;
    network->network_density = 0.0f;
    network->learning_events = 0;
    network->connection_formation_rate = 0.0f;
    
    // Initialize global coupling matrix
    for (int i = 0; i < CCE_MAX_OSCILLATORS; i++) {
        for (int j = 0; j < CCE_MAX_OSCILLATORS; j++) {
            network->global_coupling_matrix[i][j] = 0.0f;
        }
    }
    
    fprintf(stdout, "Hebbian Learning Network initialized\n");
}

void cce_hebbian_network_process_learning_cycle(CCE_HebbianNetwork* network, CCE_EnhancedOscillator* oscillators, int num_oscillators, float dt)
{
    yassert(network != NULL);
    yassert(oscillators != NULL);
    yassert(num_oscillators > 0);
    
    // Update global coupling matrix from oscillator connections
    cce_hebbian_network_update_global_matrix(network, oscillators, num_oscillators);
    
    // Apply global normalization to prevent runaway strengthening
    cce_hebbian_network_normalize_coupling_strengths(network, oscillators, num_oscillators);
    
    // Calculate network statistics
    network->total_connections = 0;
    float strength_sum = 0.0f;
    network->max_coupling_strength = 0.0f;
    
    for (int i = 0; i < num_oscillators; i++) {
        for (int j = 0; j < num_oscillators; j++) {
            if (i != j && network->global_coupling_matrix[i][j] > 0.01f) {
                network->total_connections++;
                strength_sum += network->global_coupling_matrix[i][j];
                if (network->global_coupling_matrix[i][j] > network->max_coupling_strength) {
                    network->max_coupling_strength = network->global_coupling_matrix[i][j];
                }
            }
        }
    }
    
    network->average_coupling_strength = (network->total_connections > 0) ? 
        strength_sum / (float)network->total_connections : 0.0f;
    
    int max_possible_connections = num_oscillators * (num_oscillators - 1);
    network->network_density = (max_possible_connections > 0) ? 
        (float)network->total_connections / (float)max_possible_connections : 0.0f;
    
    network->learning_events++;
}

void cce_hebbian_network_update_global_matrix(CCE_HebbianNetwork* network, CCE_EnhancedOscillator* oscillators, int num_oscillators)
{
    yassert(network != NULL);
    yassert(oscillators != NULL);
    
    for (int i = 0; i < num_oscillators; i++) {
        CCE_EnhancedOscillator* osc = &oscillators[i];
        
        // Clear this oscillator's row in the matrix
        for (int j = 0; j < num_oscillators; j++) {
            network->global_coupling_matrix[i][j] = 0.0f;
        }
        
        // Update from oscillator's coupling weights
        for (int c = 0; c < osc->num_couplings; c++) {
            int target_id = osc->coupling_oscillator_ids[c];
            if (target_id < num_oscillators) {
                network->global_coupling_matrix[i][target_id] = osc->coupling_weights[c];
            }
        }
    }
}

void cce_hebbian_network_normalize_coupling_strengths(CCE_HebbianNetwork* network, CCE_EnhancedOscillator* oscillators, int num_oscillators)
{
    yassert(network != NULL);
    yassert(oscillators != NULL);
    
    // Find maximum coupling strength
    float max_coupling = 0.0f;
    for (int i = 0; i < num_oscillators; i++) {
        for (int j = 0; j < num_oscillators; j++) {
            if (network->global_coupling_matrix[i][j] > max_coupling) {
                max_coupling = network->global_coupling_matrix[i][j];
            }
        }
    }
    
    // Normalize if any coupling is too strong
    if (max_coupling > 2.0f) {
        float normalization_factor = 1.5f / max_coupling;
        
        for (int i = 0; i < num_oscillators; i++) {
            for (int j = 0; j < num_oscillators; j++) {
                network->global_coupling_matrix[i][j] *= normalization_factor;
            }
        }
        
        // Update oscillator coupling weights
        for (int i = 0; i < num_oscillators; i++) {
            CCE_EnhancedOscillator* osc = &oscillators[i];
            for (int c = 0; c < osc->num_couplings; c++) {
                int target_id = osc->coupling_oscillator_ids[c];
                if (target_id < num_oscillators) {
                    osc->coupling_weights[c] = network->global_coupling_matrix[i][target_id];
                }
            }
        }
        
        fprintf(stdout, "Hebbian network normalized: factor=%.3f\n", normalization_factor);
    }
}

//---------------------------------------------------
// Thalamic Gating System Implementation
//---------------------------------------------------

void cce_thalamic_gating_init(CCE_ThalamicGatingSystem* gating)
{
    yassert(gating != NULL);
    
    gating->baseline_gating = 1.0f;
    gating->recent_adjustments = 0;
    
    // Initialize attention vector
    for (int i = 0; i < CCE_MAX_FREQUENCY_BANDS; i++) {
        gating->attention_vector[i] = 0.0f;
    }
    
    // Initialize gating patterns for different emotions
    // Focus pattern: enhance beta and gamma, reduce alpha
    gating->focus_pattern[0] = 0.9f;  // delta
    gating->focus_pattern[1] = 0.8f;  // theta
    gating->focus_pattern[2] = 0.7f;  // alpha
    gating->focus_pattern[3] = 1.5f;  // beta
    gating->focus_pattern[4] = 1.3f;  // gamma
    
    // Creativity pattern: enhance theta and gamma
    gating->creativity_pattern[0] = 0.9f;  // delta
    gating->creativity_pattern[1] = 2.0f
