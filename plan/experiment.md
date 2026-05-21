# experiment.md

## 1. 요약

목표는 `meta-llama/Llama-3.2-1B-Instruct`의 공식 reference score가 존재하는 benchmark를 기준으로, Original LM Head와 Proposed Coarse-Fine LM Head의 정확도 및 latency를 비교하는 것이다.

우선순위는 빠르게 끝나는 순서로 `ARC-Challenge → HellaSwag → IFEval → GSM8K`로 둔다. ARC-Challenge와 HellaSwag는 객관식이므로 가장 먼저 수행한다. GSM8K는 CoT 생성이 포함되어 후순위로 둔다.

공식 reference score는 Meta Llama 3.2 model card의 `Llama 3.2 1B bf16` 값을 사용한다. 세부 prompt, `num_shots`, `max_gen_len`, `max_prompt_len`, `return_logprobs` 등은 `meta-llama/Llama-3.2-1B-Instruct-evals`의 각 task detail dataset에서 `eval_config`를 읽어 동일하게 맞춘다.

| Benchmark | 공식 설정 | Metric | Reference BF16 |
|---|---:|---|---:|
| ARC-Challenge | 0-shot | acc | 59.4 |
| HellaSwag | 0-shot | acc | 41.2 |
| IFEval | 0-shot | Avg(Prompt/Instruction acc Loose/Strict) | 59.5 |
| GSM8K CoT | 8-shot | em_maj1@1 | 44.4 |

공통 비교 대상은 다음으로 고정한다.

- `Original`: 원본 BF16 LM Head.
- `Proposed`: Coarse-Fine LM Head.
- `Baseline-1`: Weight-only binarized coarse LM Head.
- `Baseline-2`: Weight-and-activation binarized coarse LM Head.
- `Optional`: SVDSoftmax 또는 FlashHead reproduced implementation.

공통 보고 지표는 다음으로 고정한다.

- Task accuracy 또는 exact match.
- Token agreement with Original LM Head.
- Containment@k: original top-1 token이 coarse top-k 후보 안에 포함되는 비율.
- LM Head latency.
- End-to-end TPOT.
- top-k sweep: 64, 128, 256, 512, 1024, 2048.

---

## 2. 실험방법

### 2.1 ARC-Challenge

- 목적: 가장 빠른 1차 정확도 검증.
- 공식 reference: `0-shot`, `acc`, `59.4`.
- Dataset config: `Llama-3.2-1B-Instruct-evals__arc_challenge__details`.
- 실행 방식:
  - `eval_config`에서 `input_final_prompt`, `num_shots`, `max_gen_len`, `max_prompt_len`, `return_logprobs`를 읽는다.
  - 가능한 경우 공식 prompt를 그대로 사용한다.
  - `Original` 결과가 reference 59.4 근처로 재현되는지 먼저 확인한다.
  - 이후 동일 prompt에서 `Proposed`와 baselines를 비교한다.
- 보고:
  - Accuracy.
  - Containment@k.
  - Token agreement.
  - LM Head latency.
  - E2E TPOT.
- 우선순위: 1순위.

### 2.2 HellaSwag

- 목적: commonsense sentence-completion 계열에서 빠른 객관식 검증.
- 공식 reference: `0-shot`, `acc`, `41.2`.
- Dataset config: `Llama-3.2-1B-Instruct-evals__hellaswag_chat__details`.
- 실행 방식:
  - `eval_config`와 official final prompt를 그대로 사용한다.
  - `Original`이 reference 41.2 근처로 재현되는지 확인한다.
  - 이후 `Proposed`와 baselines를 동일 조건에서 비교한다.
- 보고:
  - Accuracy.
  - Containment@k.
  - Token agreement.
  - LM Head latency.
  - E2E TPOT.
- 우선순위: 2순위.

### 2.3 IFEval

- 목적: instruction-following 성능 보존 확인.
- 공식 reference: `0-shot`, `Avg(Prompt/Instruction acc Loose/Strict)`, `59.5`.
- Dataset configs:
  - `Llama-3.2-1B-Instruct-evals__ifeval__loose__details`.
  - `Llama-3.2-1B-Instruct-evals__ifeval__strict__details`.
- 실행 방식:
  - loose와 strict를 모두 수행한다.
  - 공식 metric처럼 Prompt-level 및 Instruction-level accuracy를 계산한 뒤 평균을 보고한다.
  - 생성형 평가이므로 `max_gen_len`을 반드시 `eval_config`와 동일하게 둔다.
- 보고:
  - Loose/Strict Prompt accuracy.
  - Loose/Strict Instruction accuracy.
  - Average score.
  - Containment@k over generated tokens.
  - Token agreement.
  - E2E TPOT.
- 우선순위: 3순위.

### 2.4 GSM8K CoT

- 목적: 수학 reasoning에서 CoT 생성 중 top-1 보존성 확인.
- 공식 reference: `8-shot`, `em_maj1@1`, `44.4`.
- Dataset config: `Llama-3.2-1B-Instruct-evals__gsm8k__details`.
- 실행 방식:
  - 8-shot CoT prompt를 사용한다.
  - 생성 길이가 길어 시간이 많이 걸리므로, 먼저 100개 sample subset으로 빠르게 확인한다.
  - 최종 논문용 결과는 full test set 기준으로 수행한다.
  - 답 추출은 공식 parsed answer 방식과 동일하게 맞춘다.
- 보고:
  - Exact match.
  - Containment@k over all generated tokens.
  - Token agreement.
  - Average generated tokens.
  - E2E TPOT.
- 우선순위: 4순위.

---

## 3. 공통 실행 조건

- Model: `meta-llama/Llama-3.2-1B-Instruct`.
- Precision: BF16.
- Device: 우선 RTX 4090 또는 DGX Spark에서 sanity check, 최종은 A100 포함.
- Batch size:
  - 우선 `batch_size = 1`.
  - 추가 실험으로 `batch_size = 4, 8`을 수행해 batching 민감도 확인.
- Decoding:
  - Greedy decoding.
  - `temperature = 0`.
  - Sampling 미사용.
- Proposed setting:
  - Tile size: 기존 논문 설정과 동일하게 시작.
  - Cluster size: 기존 논문 설정과 동일하게 시작.
  - top-k: 64, 128, 256, 512, 1024, 2048 sweep.
- 재현 순서:
  1. Original BF16로 공식 reference score가 재현되는지 확인한다.
  2. Proposed를 적용해 accuracy 변화가 없는지 확인한다.
  3. Containment@k와 token agreement를 동시에 기록한다.
  4. LM Head latency와 E2E TPOT를 기록한다.
  5. Baseline-1, Baseline-2와 비교한다.
  6. 가능하면 SVDSoftmax 또는 FlashHead reproduced를 추가한다.

---

## 4. 기록 파일 형식

각 task마다 다음 파일을 저장한다.

- `{task}_original.jsonl`
- `{task}_proposed_top{k}.jsonl`
- `{task}_baseline_binW_top{k}.jsonl`
- `{task}_baseline_binWA_top{k}.jsonl`
- `{task}_summary.csv`

각 sample에는 다음 항목을 기록한다.

- `task`
- `sample_id`
- `prompt_hash`
- `target`
- `original_output`
- `method_output`
- `original_answer`
- `method_answer`
- `is_correct_original`
- `is_correct_method`
- `token_agreement`
- `containment_count`
- `generated_token_count`
- `lm_head_time_ms`
- `e2e_time_ms`
- `tpot_ms`

---

## 5. 논문용 표 구성

최종 논문에는 다음 두 표를 우선 사용한다.

### Table A. Accuracy preservation

| Task | Official BF16 | Original reproduced | Proposed top-k | Delta |
|---|---:|---:|---:|---:|

### Table B. Latency and containment

| Task | top-k | Containment@k | Token agreement | LM Head speedup | E2E speedup | TPOT |
|---|---:|---:|---:|---:|---:|---:|
