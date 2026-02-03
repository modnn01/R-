# 유방암 생존율 예측 웹앱

## 프로젝트 개요
이 프로젝트는 미국 유방암 환자 데이터를 기반으로 환자의 나이와 종양 크기를 입력받아 생존율을 예측하는 Streamlit 웹 애플리케이션입니다.

## 주요 기능

### 1. 생존율 예측
- **입력 변수**: 나이, 종양 크기
- **출력**: 특정 시점(12, 24, 36, 60, 84개월)의 생존 확률
- **모델**: Cox 비례위험모형 (Cox Proportional Hazards Model)

### 2. 시각화
- **생존 곡선**: 시간에 따른 생존 확률 변화
- **데이터 분포**: 입력값과 전체 데이터셋 비교
- **인터랙티브 UI**: 슬라이더를 통한 실시간 예측 업데이트

### 3. 결과 해석
- 생존율에 따른 자동 해석 제공
- 색상 코드를 통한 위험도 표시
- 모델 계수 및 위험비 정보 제공

## 기술 스택
- **Frontend**: Streamlit
- **Backend**: Python
- **생존분석**: scikit-survival
- **데이터 처리**: pandas, numpy
- **시각화**: matplotlib, seaborn

## 설치 방법

### 필수 패키지 설치
```bash
pip install streamlit pandas numpy scikit-survival matplotlib seaborn
```

## 실행 방법

### 로컬 환경에서 실행
```bash
streamlit run breast_cancer_app.py
```

실행 후 브라우저에서 `http://localhost:8501`로 접속하세요.

### Streamlit Cloud에 배포
1. GitHub 저장소에 `breast_cancer_app.py`, `Breast_Cancer.csv`, `requirements.txt` 업로드
2. [Streamlit Cloud](https://streamlit.io/cloud)에서 저장소 연결
3. 자동으로 배포 완료!

## 데이터셋 정보
- **출처**: 미국 유방암 환자 데이터
- **환자 수**: 4,024명
- **주요 변수**: 
  - Age (나이)
  - Tumor Size (종양 크기)
  - Survival Months (생존 개월)
  - Status (생존 여부)

## 모델 설명

### Cox 비례위험모형
생존분석에서 가장 널리 사용되는 준모수적 방법으로, 다음과 같은 특징이 있습니다:

- **비례위험 가정**: 시간에 관계없이 위험비가 일정
- **준모수적**: 기저 위험함수에 대한 가정이 없음
- **해석 가능성**: 계수의 지수값이 위험비(Hazard Ratio)

### 모델 수식
```
h(t|X) = h₀(t) × exp(β₁ × Age + β₂ × TumorSize)
```

여기서:
- h(t|X): 시점 t에서의 위험률
- h₀(t): 기저 위험함수
- β: 회귀 계수

## 사용 예시

1. **환자 정보 입력**
   - 나이: 50세
   - 종양 크기: 30mm
   - 예측 시점: 36개월

2. **결과 확인**
   - 36개월 생존율: XX.X%
   - 생존 곡선 그래프
   - 데이터셋 내 위치 확인

3. **해석**
   - 생존율 수준에 따른 자동 해석
   - 위험비 및 모델 계수 확인

## 주의사항

⚠️ **중요**: 이 시스템은 교육 및 연구 목적으로 개발되었습니다.

- 예측 결과는 통계적 모델에 기반하며 참고용입니다
- 실제 치료 결정은 반드시 전문 의료진과 상담하시기 바랍니다
- 모델은 나이와 종양크기만 고려하며, 다른 중요한 임상 변수는 미포함
- 개별 환자의 실제 예후와 차이가 있을 수 있습니다

## 향후 개선 사항
1. 추가 임상 변수 포함 (Stage, Grade, Hormone Status 등)
2. 다른 생존분석 모델과 비교 (Random Survival Forest, DeepSurv 등)
3. 신뢰구간 표시
4. 환자 데이터 업로드 기능
5. 결과 리포트 다운로드 기능

## 파일 구조
```
.
├── breast_cancer_app.py      # 메인 Streamlit 앱
├── Breast_Cancer.csv          # 데이터셋 (필수!)
├── requirements.txt           # 패키지 의존성
└── README.md                  # 프로젝트 문서
```

**중요**: `Breast_Cancer.csv` 파일은 반드시 `breast_cancer_app.py`와 같은 폴더에 있어야 합니다!

## 라이선스
이 프로젝트는 교육 및 연구 목적으로 자유롭게 사용할 수 있습니다.

## 문의
프로젝트 관련 문의사항이 있으시면 이슈를 등록해주세요.

---
Made with ❤️ for Medical Research and Education
