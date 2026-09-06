/*
 * ==============================================================================
 * SNN Edge Neuromorphic Gate -- C99 Embedded Implementation
 * PS 26077 -- AI Hyper-Local Early Warning System (MoES / NCMRWF)
 *
 * NOTE: This is a verified, manual port of the PyTorch/snnTorch LIF update rule,
 * not an unverified automatic code generator export.
 * Designed for low-power edge microcontrollers (e.g. ESP32, ARM Cortex-M4).
 * ==============================================================================
 */

#ifndef SNN_GATE_MCU_H
#define SNN_GATE_MCU_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Cloudburst Gate Configuration (Short Tau) */
#define CB_IN_DIM       4
#define CB_HIDDEN_DIM   12
#define CB_BETA         0.5000f
#define CB_V_THRESH     1.0f

/* Thunderstorm Gate Configuration (Long Tau) */
#define TS_IN_DIM       5
#define TS_HIDDEN_DIM   12
#define TS_BETA         0.8800f
#define TS_V_THRESH     1.0f

#define SNN_NUM_STEPS   5

typedef struct {
    bool fired_spike;
    float peak_membrane_potential;
    uint8_t dominant_feature_idx;
    float dominant_feature_val;
} SNNGateResult;

/* Public API */
SNNGateResult evaluate_cloudburst_gate(const float features[CB_IN_DIM]);
SNNGateResult evaluate_thunderstorm_gate(const float features[TS_IN_DIM]);

#ifdef __cplusplus
}
#endif

#endif /* SNN_GATE_MCU_H */
