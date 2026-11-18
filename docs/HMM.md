# HMM and Viterbi route reconstruction

Code: `src/hmm.py` (model), `src/hmm_experiments.py` (tuning + comparison) · results: `results/hmm/` · parameters: `artifacts/decode_config.json`

1. **Hidden states**: for GPS fix *i*, the states are that fix's candidate arcs (the same ≤12 arcs within 60 m the
   classifier scored). Each state carries its `arc_id`, so the candidate → road-segment link is kept.
2. **Emission**: `alpha · log P(arc | fix)` from the final LightGBM (floored at 1e-4). alpha = 0.5 down-weights over-confident
   probabilities relative to the road-graph prior.
3. **Transition** (cost = −log transition score):

| Move between fix i−1 and i | Cost |
|---|---|
| same arc | 0 |
| successor arc (starts where previous ends) | c_next = 1.0 |
| two arcs ahead | c_2hop = 3.0 |
| twin arc (U-turn on the same road) | c_twin = 20.0 |
| no connection ("reset") | c_jump = 100.0 |

4. **Missing candidates / GPS gaps**: if dt > gap_s = 5.0 s, the successor test is replaced by a BFS of up to
   max_hops = 12 arcs, with cost c_next + c_hop·(hops−1), c_hop = 1.0. A fix with no candidate (nothing within
   300 m) gets no state column; it is returned as arc −1 and decoding continues from the previous fix.
5. **Disconnected segments**: c_jump is finite, so the decoder can always continue. It pays a large penalty and starts a
   new route piece, and these "route breaks" are counted.
6. **Viterbi recurrence**: V_i(s) = em_i(s) + max_r [V_{i−1}(r) − cost(r→s)], with back-pointers, then back-tracking from argmax V_N.

**Tuning** (training data only): coordinate search over the costs on 5-fold out-of-fold probabilities of the final
classifier on the 80 training trips (16 configurations, `results/hmm/decoder_tuning.csv`); train Top-1
92.69% (start) → 92.94% (best).

## Did HMM decoding improve the route? (student-test, 19,160 fixes)
| Method | Top-1 | Continuity | Route breaks | Route precision | Route recall | Route F1 | LCS ratio |
|---|---|---|---|---|---|---|---|
| classifier_only | 92.32% | 96.61% | 646 | 91.81% | 90.10% | 90.80% | 89.67% |
| hmm_no_gap | 95.18% | 99.96% | 7 | 98.17% | 91.77% | 94.78% | 91.61% |
| hmm_full | 95.16% | 99.97% | 7 | 98.14% | 91.75% | 94.75% | 91.58% |

Yes. Pointwise Top-1 rises by about 2.8 points, route breaks drop from 646 to 7, continuity becomes almost 100%, and
route precision rises from about 92% to 98% (the classifier-only route contains many spurious side arcs).
Gap handling (hmm_full vs hmm_no_gap) makes no measurable difference on student-test, which has only 44 gaps > 5 s.
It is kept because it is principled for the larger gaps seen in training (up to 42 h).

| Trip | Classifier only | HMM | Δ (points) | gaps > 5 s |
|---|---|---|---|---|
| 00000000 | 95.25% | 96.84% | +1.60 | 0 |
| 00000007 | 90.32% | 86.34% | -3.99 | 0 |
| 00000009 | 89.32% | 92.66% | +3.34 | 0 |
| 00000040 | 96.13% | 97.56% | +1.44 | 0 |
| 00000052 | 98.09% | 98.85% | +0.76 | 0 |
| 00000056 | 92.50% | 95.78% | +3.28 | 0 |
| 00000065 | 85.94% | 87.37% | +1.43 | 5 |
| 00000076 | 94.35% | 97.78% | +3.43 | 39 |
| 00000077 | 88.98% | 96.01% | +7.03 | 0 |
| 00000091 | 92.22% | 93.48% | +1.26 | 0 |

**Where HMM did not help**: 00000007. Trip 00000007 ends with the vehicle parked or crawling for about 150 s next to a
two-way road. The decoder keeps one consistent direction for the whole stationary stretch, and the twin/U-turn cost stops it
from switching, so if that direction is wrong all those fixes are wrong together. The classifier alone flips between the
twins and gets about half of them right by chance.
