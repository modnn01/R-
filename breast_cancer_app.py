import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.util import Surv
import warnings
warnings.filterwarnings('ignore')

# 페이지 설정
st.set_page_config(
    page_title="유방암 생존율 예측 시스템",
    page_icon="🎗️",
    layout="wide"
)

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# 제목
st.title("🎗️ 유방암 환자 생존율 예측 시스템")
st.markdown("---")

# 데이터 로드 및 전처리
@st.cache_data
def load_and_preprocess_data():
    try:
        # 데이터 로드
        df = pd.read_csv('Breast_Cancer.csv')
        
        # 결측치 제거
        df = df.dropna()
        
        # 데이터 타입 변환
        df['Age'] = df['Age'].astype(float)
        df['Tumor Size'] = df['Tumor Size'].astype(float)
        df['Survival Months'] = df['Survival Months'].astype(float)
        
        # Status를 이진 변수로 변환 (Dead=True, Alive=False)
        df['event'] = (df['Status'] == 'Dead').astype(bool)
        
        return df
    except FileNotFoundError:
        st.error("❌ 데이터 파일(Breast_Cancer.csv)을 찾을 수 없습니다. 앱과 같은 폴더에 위치시켜주세요.")
        st.stop()
        return None
    except Exception as e:
        st.error(f"❌ 데이터 로드 중 오류가 발생했습니다: {str(e)}")
        st.stop()
        return None

# 모델 학습
@st.cache_resource
def train_model(df):
    try:
        # 특징 선택
        X = df[['Age', 'Tumor Size']].values.astype(float)
        
        # 생존 객체 생성 - 구조화된 배열로 직접 생성
        y = np.array(
            [(event, time) for event, time in zip(df['event'].values, df['Survival Months'].values)],
            dtype=[('event', bool), ('time', float)]
        )
        
        # Cox 모델 학습
        cox_model = CoxPHSurvivalAnalysis()
        cox_model.fit(X, y)
        
        return cox_model
    except Exception as e:
        st.error(f"❌ 모델 학습 중 오류가 발생했습니다: {str(e)}")
        st.stop()
        return None

# 데이터 로드
try:
    df = load_and_preprocess_data()
    
    # 사이드바에 데이터 통계 표시
    st.sidebar.header("📊 데이터셋 정보")
    st.sidebar.metric("전체 환자 수", len(df))
    st.sidebar.metric("생존 환자", len(df[df['Status'] == 'Alive']))
    st.sidebar.metric("사망 환자", len(df[df['Status'] == 'Dead']))
    st.sidebar.metric("평균 추적기간", f"{df['Survival Months'].mean():.1f}개월")
    
    # 모델 학습
    cox_model = train_model(df)
    
except Exception as e:
    st.error(f"❌ 초기화 중 오류가 발생했습니다: {str(e)}")
    st.stop()

# 메인 영역을 두 개의 컬럼으로 분할
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🔍 환자 정보 입력")
    
    # 사용자 입력
    age = st.slider(
        "나이 (세)", 
        min_value=int(df['Age'].min()), 
        max_value=int(df['Age'].max()), 
        value=50,
        help="환자의 나이를 선택하세요"
    )
    
    tumor_size = st.slider(
        "종양 크기 (mm)", 
        min_value=int(df['Tumor Size'].min()), 
        max_value=int(df['Tumor Size'].max()), 
        value=30,
        help="종양의 크기를 선택하세요"
    )
    
    # 예측 시점 선택
    prediction_months = st.selectbox(
        "예측 시점 선택",
        [12, 24, 36, 60, 84],
        index=2,
        help="몇 개월 후의 생존율을 예측할지 선택하세요"
    )
    
    st.markdown("---")
    
    # 입력값 요약
    st.markdown("### 입력된 환자 정보")
    info_df = pd.DataFrame({
        '항목': ['나이', '종양 크기', '예측 시점'],
        '값': [f"{age}세", f"{tumor_size}mm", f"{prediction_months}개월"]
    })
    st.table(info_df)

with col2:
    st.subheader("📈 생존율 예측 결과")
    
    try:
        # 예측 수행
        X_new = np.array([[float(age), float(tumor_size)]])
        
        # 생존 함수 예측
        surv_funcs = cox_model.predict_survival_function(X_new)
        
        # 시간 포인트 생성
        max_time = int(df['Survival Months'].max())
        time_points = np.arange(0, max_time + 1, 1)
        
        # 각 시점의 생존 확률 계산
        survival_probs = np.array([surv_funcs[0](t) for t in time_points])
        
        # 선택한 시점의 생존율
        if prediction_months < len(survival_probs):
            survival_rate = survival_probs[prediction_months] * 100
        else:
            survival_rate = survival_probs[-1] * 100
        
        # 결과 표시
        st.metric(
            label=f"{prediction_months}개월 생존율",
            value=f"{survival_rate:.1f}%",
            delta=None
        )
        
        # 생존율에 따른 해석 제공
        st.markdown("### 결과 해석")
        if survival_rate >= 90:
            st.success("✅ 매우 높은 생존율이 예측됩니다.")
        elif survival_rate >= 70:
            st.info("ℹ️ 양호한 생존율이 예측됩니다.")
        elif survival_rate >= 50:
            st.warning("⚠️ 중등도의 생존율이 예측됩니다.")
        else:
            st.error("❗ 주의가 필요한 생존율입니다.")
        
        # 생존 곡선 그리기
        st.markdown("### 생존 곡선")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(time_points, survival_probs * 100, 'b-', linewidth=2)
        ax.axvline(x=prediction_months, color='r', linestyle='--', linewidth=2, label=f'{prediction_months} months')
        ax.axhline(y=survival_rate, color='r', linestyle='--', linewidth=1, alpha=0.5)
        ax.scatter([prediction_months], [survival_rate], color='red', s=100, zorder=5)
        ax.set_xlabel('Survival Time (Months)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Survival Probability (%)', fontsize=12, fontweight='bold')
        ax.set_title(f'Survival Curve: Age={age}, Tumor Size={tumor_size}mm', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_ylim([0, 105])
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"❌ 예측 중 오류가 발생했습니다: {str(e)}")

# 하단에 추가 정보 표시
st.markdown("---")
st.subheader("📊 데이터 분포 비교")

col3, col4 = st.columns(2)

with col3:
    # 나이 분포
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.hist(df['Age'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.axvline(x=age, color='red', linestyle='--', linewidth=2, label='Your Input')
    ax1.set_xlabel('Age', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax1.set_title('Age Distribution in Dataset', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    st.pyplot(fig1)

with col4:
    # 종양 크기 분포
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.hist(df['Tumor Size'], bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
    ax2.axvline(x=tumor_size, color='red', linestyle='--', linewidth=2, label='Your Input')
    ax2.set_xlabel('Tumor Size (mm)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax2.set_title('Tumor Size Distribution in Dataset', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    st.pyplot(fig2)

# 모델 정보
st.markdown("---")
with st.expander("ℹ️ 모델 정보 및 주의사항"):
    st.markdown("""
    ### 사용된 모델
    - **Cox 비례위험모형 (Cox Proportional Hazards Model)**
    - 생존분석에서 가장 널리 사용되는 준모수적 방법
    - 나이와 종양크기를 독립변수로 사용
    
    ### 모델 계수 (Coefficients)
    """)
    
    coef_df = pd.DataFrame({
        '변수': ['Age', 'Tumor Size'],
        '계수 (Coefficient)': cox_model.coef_,
        '위험비 (Hazard Ratio)': np.exp(cox_model.coef_)
    })
    st.table(coef_df)
    
    st.markdown("""
    ### 주의사항
    1. 이 예측은 통계적 모델에 기반하며, 개별 환자의 실제 예후와 다를 수 있습니다.
    2. 예측 결과는 참고용이며, 실제 치료 결정은 반드시 전문 의료진과 상담하시기 바랍니다.
    3. 모델은 나이와 종양크기만을 고려하며, 다른 중요한 임상 변수들은 포함되지 않았습니다.
    4. 데이터는 미국 유방암 환자 데이터셋 (N=4,024)을 기반으로 학습되었습니다.
    """)

# 푸터
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>🎗️ Breast Cancer Survival Prediction System</p>
        <p style='font-size: 12px; color: gray;'>
            이 시스템은 교육 및 연구 목적으로 개발되었습니다.<br>
            실제 임상 결정에는 전문 의료진과 상담하시기 바랍니다.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
