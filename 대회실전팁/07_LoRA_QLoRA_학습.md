# LMSYS Chatbot Arena - 16위 솔루션: LoRA/QLoRA 완벽 가이드

**대회**: LMSYS - Chatbot Arena Human Preference Predictions  
**순위**: 16위  
**작성자**: Chris Deotte (cdeotte)  
**URL**: https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527596

---

## 특징

이 글은 단순한 대회 솔루션이 아니라 **LLM 파인튜닝의 완전한 가이드**입니다. 매우 상세하고 교육적인 내용을 담고 있습니다.

---

## LoRA/QLoRA 효율적 학습 원리

- Gemma2-9B 전체 파라미터: 90억 개
- LoRA rank=64 사용 시 업데이트 파라미터: **약 200k개** (전체의 0.002%)
- DeBERTa-v3-base는 90k 파라미터 → LoRA가 DeBERTa 수준 파라미터만 업데이트!

### LoRA 작동 원리
```
기존 가중치 행렬 W (m×n) 고정
LoRA 행렬 A (m×r) + B (r×n) 만 학습
최종 가중치 = W + A × B
```

---

## LoRA 하이퍼파라미터 튜닝 가이드

### 1. 모듈 선택
모든 선형 레이어 사용이 최선:
```python
target_modules = ["q_proj", "k_proj", "v_proj", 
                  "down_proj", "up_proj", "o_proj", "gate_proj"]
```

### 2. 학습률 설정
- 2e-4 또는 2e-5 권장
- 배치 크기 8 유지 (gradient accumulation 활용)

### 3. Alpha 최적화
- LR=2e-4 → alpha=4가 최적
- 규칙: `alpha/rank × LR`이 backbone 학습률이 됨

### 4. Rank 최적화
- rank < 16: 성능 저하
- rank ≥ 16: 양호 (이 대회에서 rank=1024가 +0.002 향상)

---

## 실버 메달 → 골드 메달 단계별 가이드

### 공개 노트북 → 실버 메달
| 변경 사항 | LB 향상 |
|----------|---------|
| TTA 활성화 | 0.941 → 0.926 |
| r=64, a=16, freeze=0 | → 0.913 |
| 모든 모듈 추가 | → 0.903 |
| 외부 33k 데이터 추가 | → 0.899 (실버) |

### 실버 → 골드 시도
| 변경 사항 | LB 향상 |
|----------|---------|
| max_len 1024→2048 | 0.899 → 0.895 |
| r=1024 | → 0.894 |
| LoRA fp16 + 8bit 추론 | → 0.893 |
| 두 Gemma2 TTA | → 0.891 |
| 3개 분류 헤드 | → 0.890 |
| **왼쪽 자르기** | → 0.885 |

---

## 핵심 기법: 왼쪽 텍스트 자르기

Decoder LLM은 **마지막 토큰**을 분류에 사용 → 오른쪽이 중요!
```python
def prepare_text(self, prompts, responses_a, responses_b):
    rounds = [...]
    for k in range(len(rounds)):
        text = "\n".join(rounds[k:])
        if len(tokenizer(text)["input_ids"]) < 3072:
            break
    return text  # 앞부분 자르기 (왼쪽 truncation)
```

인코더 모델(DeBERTa)은 첫 토큰 사용 → 오른쪽 자르기가 맞음

---

## 다중 GPU 학습 방법

| 방법 | 설명 |
|------|------|
| 데이터 병렬 (DP) | 각 GPU가 LLM 복사본 보유 |
| 모델 병렬 (MP) | LLM을 GPU에 분산 |
| 하이브리드 병렬 | DP+MP (가장 효율적) |

HuggingFace Trainer는 DP/MP만 지원. 하이브리드는 Axolotl/DeepSpeed 필요.

---

## Auxiliary Learning (보조 학습)

모델A, 모델B를 예측하는 보조 헤드 추가 → LLM이 더 많은 관련 작업 학습 → 성능 향상
