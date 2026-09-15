/*
 * ==============================================================================
 * SNN Edge Neuromorphic Gate -- Implementation
 * Manual Verified Port of snnTorch Leaky Integrate-and-Fire Dynamics
 * ==============================================================================
 */

#include "snn_gate_mcu.h"
#include <math.h>

/* Cloudburst Gate (Instance A) Weights & Biases */
static const float W1_CB[12][4] = {
    {   0.8000f,   1.0000f,   1.4000f,   1.8000f },
    {   0.8841f,   0.9540f,   1.5819f,   1.5168f },
    {   0.8909f,   0.8584f,   1.2486f,   1.4693f },
    {   0.8141f,   0.8010f,   1.3441f,   1.7920f },
    {   0.7243f,   0.8346f,   1.5979f,   1.5709f },
    {   0.7041f,   0.9284f,   1.2912f,   1.4322f },
    {   0.7721f,   0.9960f,   1.2927f,   1.7688f },
    {   0.8657f,   0.9754f,   1.5981f,   1.6273f },
    {   0.8989f,   0.8855f,   1.3424f,   1.4085f },
    {   0.8412f,   0.8089f,   1.2498f,   1.7321f },
    {   0.7456f,   0.8161f,   1.5826f,   1.6816f },
    {   0.7000f,   0.9004f,   1.3982f,   1.4000f },
};
static const float B1_CB[12] = {  -0.4000f,  -0.4000f,  -0.4000f,  -0.4000f,  -0.4000f,  -0.4000f,  -0.4000f,  -0.4000f,  -0.4000f,  -0.4000f,  -0.4000f,  -0.4000f };
static const float W2_CB[1][12] = {
    {   0.6000f,   0.6000f,   0.6000f,   0.6000f,   0.6000f,   0.6000f,   0.6000f,   0.6000f,   0.6000f,   0.6000f,   0.6000f,   0.6000f },
};
static const float B2_CB[1] = {  -0.2000f };
static const float SCALES_CB[CB_IN_DIM] = { 100.0f, 60.0f, 30.0f, 40.0f };

/* Thunderstorm Gate (Instance B) Weights & Biases */
static const float W1_TS[12][5] = {
    {   1.3000f,   1.5500f,   0.8000f,   1.0000f,   1.2000f },
    {   1.4262f,   1.4810f,   0.8841f,   0.9540f,   1.3364f },
    {   1.4364f,   1.3376f,   0.8909f,   0.8584f,   1.0865f },
    {   1.3212f,   1.2515f,   0.8141f,   0.8010f,   1.1581f },
    {   1.1865f,   1.3020f,   0.7243f,   0.8346f,   1.3484f },
    {   1.1562f,   1.4425f,   0.7041f,   0.9284f,   1.1184f },
    {   1.2581f,   1.5440f,   0.7721f,   0.9960f,   1.1195f },
    {   1.3985f,   1.5131f,   0.8657f,   0.9754f,   1.3486f },
    {   1.4484f,   1.3782f,   0.8989f,   0.8855f,   1.1568f },
    {   1.3618f,   1.2633f,   0.8412f,   0.8089f,   1.0874f },
    {   1.2184f,   1.2741f,   0.7456f,   0.8161f,   1.3369f },
    {   1.1500f,   1.4007f,   0.7000f,   0.9004f,   1.1987f },
};
static const float B1_TS[12] = {  -0.5000f,  -0.5000f,  -0.5000f,  -0.5000f,  -0.5000f,  -0.5000f,  -0.5000f,  -0.5000f,  -0.5000f,  -0.5000f,  -0.5000f,  -0.5000f };
static const float W2_TS[1][12] = {
    {   0.5500f,   0.5500f,   0.5500f,   0.5500f,   0.5500f,   0.5500f,   0.5500f,   0.5500f,   0.5500f,   0.5500f,   0.5500f,   0.5500f },
};
static const float B2_TS[1] = {  -0.2500f };
static const float SCALES_TS[TS_IN_DIM] = { 5.0f, 3.0f, 10.0f, 4.0f, 500.0f };

SNNGateResult evaluate_cloudburst_gate(const float features[CB_IN_DIM]) {
    SNNGateResult res;
    res.fired_spike = false;
    res.peak_membrane_potential = 0.0f;
    res.dominant_feature_idx = 0;
    res.dominant_feature_val = features[0];

    float norm[CB_IN_DIM];
    float max_norm = -1.0f;
    for (int i = 0; i < CB_IN_DIM; i++) {
        float val = features[i] < 0.0f ? 0.0f : features[i];
        norm[i] = val / SCALES_CB[i];
        if (norm[i] > 3.0f) norm[i] = 3.0f;
        if (norm[i] > max_norm) {
            max_norm = norm[i];
            res.dominant_feature_idx = (uint8_t)i;
            res.dominant_feature_val = features[i];
        }
    }

    /* Hidden layer direct current */
    float cur1[CB_HIDDEN_DIM];
    for (int h = 0; h < CB_HIDDEN_DIM; h++) {
        float sum = B1_CB[h];
        for (int i = 0; i < CB_IN_DIM; i++) {
            sum += W1_CB[h][i] * norm[i];
        }
        cur1[h] = sum;
    }

    /* Simulate discrete LIF timesteps */
    float mem1[CB_HIDDEN_DIM] = {0.0f};
    float mem2 = 0.0f;
    int total_spikes = 0;

    for (int t = 0; t < SNN_NUM_STEPS; t++) {
        /* Layer 1 update */
        float spk1[CB_HIDDEN_DIM] = {0.0f};
        for (int h = 0; h < CB_HIDDEN_DIM; h++) {
            mem1[h] = mem1[h] * CB_BETA + cur1[h];
            if (mem1[h] >= CB_V_THRESH) {
                spk1[h] = 1.0f;
                mem1[h] -= CB_V_THRESH;
            }
        }

        /* Layer 2 update */
        float cur2 = B2_CB[0];
        for (int h = 0; h < CB_HIDDEN_DIM; h++) {
            cur2 += W2_CB[0][h] * spk1[h];
        }
        mem2 = mem2 * CB_BETA + cur2;
        if (mem2 > res.peak_membrane_potential) {
            res.peak_membrane_potential = mem2;
        }
        if (mem2 >= CB_V_THRESH) {
            total_spikes++;
            mem2 -= CB_V_THRESH;
        }
    }

    res.fired_spike = (total_spikes > 0);
    return res;
}

SNNGateResult evaluate_thunderstorm_gate(const float features[TS_IN_DIM]) {
    SNNGateResult res;
    res.fired_spike = false;
    res.peak_membrane_potential = 0.0f;
    res.dominant_feature_idx = 0;
    res.dominant_feature_val = features[0];

    float norm[TS_IN_DIM];
    float max_norm = -1.0f;
    for (int i = 0; i < TS_IN_DIM; i++) {
        float val = features[i] < 0.0f ? 0.0f : features[i];
        norm[i] = val / SCALES_TS[i];
        if (norm[i] > 3.0f) norm[i] = 3.0f;
        if (norm[i] > max_norm) {
            max_norm = norm[i];
            res.dominant_feature_idx = (uint8_t)i;
            res.dominant_feature_val = features[i];
        }
    }

    float cur1[TS_HIDDEN_DIM];
    for (int h = 0; h < TS_HIDDEN_DIM; h++) {
        float sum = B1_TS[h];
        for (int i = 0; i < TS_IN_DIM; i++) {
            sum += W1_TS[h][i] * norm[i];
        }
        cur1[h] = sum;
    }

    float mem1[TS_HIDDEN_DIM] = {0.0f};
    float mem2 = 0.0f;
    int total_spikes = 0;

    for (int t = 0; t < SNN_NUM_STEPS; t++) {
        float spk1[TS_HIDDEN_DIM] = {0.0f};
        for (int h = 0; h < TS_HIDDEN_DIM; h++) {
            mem1[h] = mem1[h] * TS_BETA + cur1[h];
            if (mem1[h] >= TS_V_THRESH) {
                spk1[h] = 1.0f;
                mem1[h] -= TS_V_THRESH;
            }
        }

        float cur2 = B2_TS[0];
        for (int h = 0; h < TS_HIDDEN_DIM; h++) {
            cur2 += W2_TS[0][h] * spk1[h];
        }
        mem2 = mem2 * TS_BETA + cur2;
        if (mem2 > res.peak_membrane_potential) {
            res.peak_membrane_potential = mem2;
        }
        if (mem2 >= TS_V_THRESH) {
            total_spikes++;
            mem2 -= TS_V_THRESH;
        }
    }

    res.fired_spike = (total_spikes > 0);
    return res;
}
