#결측치 50% 이상 컬럼 제거, Missing 처리
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import seaborn as sns
from scipy.stats import ks_2samp, anderson_ksamp, chisquare
from sklearn.model_selection import learning_curve  # 올바른 임포트 위치
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
import tempfile, os
import warnings
import matplotlib
matplotlib.use('Qt5Agg')  # 또는 'Qt5Agg'를 사용할 수도 있습니다.
import matplotlib.pyplot as plt
import gc

warnings.filterwarnings("ignore")

# 임시 폴더 지정 (메모리맵 오류 방지)
os.environ['JOBLIB_TEMP_FOLDER'] = tempfile.mkdtemp()

# 데이터 전처리
file_path = "C:/Users/KIM/AppData/Local/JetBrains/PyCharm2024.1/cpython-cache/Users/KIM/PycharmProjects/250409_jpgstructralanly/250428/250517_1024최종데이터셋.csv"
df = pd.read_csv(file_path)
df['is_AI_image'] = df['Subfolder Name'].apply(lambda x: 1 if not str(x).startswith('n') else 0)

X = df.drop(columns=['is_AI_image'])
y = df['is_AI_image']

cols_to_drop = [col for col in X.columns if col.endswith('_seq') or col.endswith('_info')]
cols_to_drop += ['Subfolder Name', 'File Name']
X = X.drop(columns=cols_to_drop)
print(f"제외된 컬럼들: {cols_to_drop}")

# 결측치 처리 함수
def handle_missing_data(X):
    missing_ratios = X.isnull().mean()  # 각 컬럼의 결측치 비율 계산
    cols_to_drop = missing_ratios[missing_ratios >= 0.5].index  # 결측치 비율이 50% 이상인 컬럼들
    X = X.drop(columns=cols_to_drop)  # 해당 컬럼들 제거
    print(f"제거된 컬럼들 (결측치 비율 50% 이상): {cols_to_drop.tolist()}")

    cols_to_fill = missing_ratios[missing_ratios < 0.5].index  # 결측치 비율이 50% 미만인 컬럼들
    X[cols_to_fill] = X[cols_to_fill].fillna('Missing')  # 결측치는 'Missing'으로 처리
    return X

# 결측치 처리 적용
X = handle_missing_data(X)

# 레이블 인코딩
cols_to_encode = X.select_dtypes(include=['object']).columns.tolist()
for col in cols_to_encode:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    print(f"Label Encoding for column '{col}': {le.classes_}")

# 특성 이름을 일관되게 정리하는 함수
def clean_feature_names(columns):
    return [col.replace(" ", "_").replace("<", "").replace(">", "").replace(",", "").replace("[", "_").replace("]", "") for col in columns]

# 모델 정의 및 하이퍼파라미터 튜닝
models = {
    "Logistic Regression": {
        "model": LogisticRegression(max_iter=10000, solver='saga'),
        "params": {
            'C': [0.1, 1, 10],
            'penalty': ['l2', 'l1'],
            'solver': ['liblinear', 'saga']
        }
    },
    "Random Forest": {
        "model": RandomForestClassifier(random_state=42),
        "params": {
            'n_estimators': [50, 100],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'class_weight': ['balanced', None]
        }
    },
    "XGBoost": {
        "model": XGBClassifier(eval_metric='logloss'),
        "params": {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 6, 10],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }
    },
    "Decision Tree": {
        "model": DecisionTreeClassifier(random_state=42),
        "params": {
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    },
    "SVM": {
        "model": SVC(probability=True),
        "params": {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf'], #
            'gamma': ['scale', 'auto']
        }
    },
    "KNN": {
        "model": KNeighborsClassifier(),
        "params": {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
        }
    },
}

model_results = {}

# 데이터 분할
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)
print(f"\nTrain size: {X_train.shape[0]}, Validation size: {X_val.shape[0]}, Test size: {X_test.shape[0]}")

# 모델 튜닝 및 평가 함수
def tune_and_evaluate_model(name, model_info):
    print(f"\n🚀 모델: {name}")
    model = model_info["model"]
    params = model_info["params"]

    # 특성 이름을 일관되게 정리
    X_train.columns = clean_feature_names(X_train.columns)
    X_val.columns = clean_feature_names(X_val.columns)
    X_test.columns = clean_feature_names(X_test.columns)

    grid_search = GridSearchCV(estimator=model, param_grid=params, cv=5, scoring='accuracy', n_jobs=1, verbose=1)
    grid_search.fit(X_train, y_train)

    print(f" - 최적 하이퍼파라미터: {grid_search.best_params_}")
    print(f" - 최적 교차검증 정확도: {grid_search.best_score_:.4f}")

    best_model = grid_search.best_estimator_
    y_val_pred = best_model.predict(X_val)
    val_acc = accuracy_score(y_val, y_val_pred)
    print(f" - 검증 정확도: {val_acc:.4f}")
    print(classification_report(y_val, y_val_pred))

    y_test_pred = best_model.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    print(f" - 테스트 정확도: {test_acc:.4f}")
    print(classification_report(y_test, y_test_pred))

    train_class_counts = y_train.value_counts().sort_index()
    test_class_counts = y_test.value_counts().sort_index()
    expected = (test_class_counts.sum() * train_class_counts) / train_class_counts.sum()
    chi2_stat, p_val = chisquare(test_class_counts, expected)
    print(f"\nChi-square test: Stat={chi2_stat}, P={p_val}")

    # KS 검정
    train_probs = best_model.predict_proba(X_train)[:, 1]
    test_probs = best_model.predict_proba(X_test)[:, 1]
    ks_stat, ks_p_val = ks_2samp(train_probs, test_probs)
    print(f"KS 검정: 통계량={ks_stat:.4f}, p-value={ks_p_val:.4f}")

    print("Train_probs unique:", np.unique(train_probs))
    print("Test_probs unique:", np.unique(test_probs))
    print("Train_probs n unique:", len(np.unique(train_probs)))
    print("Test_probs n unique:", len(np.unique(test_probs)))

    # AD 검정
    ad_stat, _, sig_level = anderson_ksamp([train_probs, test_probs])
    print(f"Anderson-Darling 검정: 통계량={ad_stat:.4f}, 유의수준={sig_level}")

    # 혼동 행렬
    cm = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues") #, xticklabels=[], yticklabels=[])
    plt.xlabel("Predicted label")
    plt.ylabel("Actual label")
    plt.title(f"Confusion Matrix: {name}")
    plt.show()

    # 학습 곡선
    train_sizes, train_scores, val_scores = learning_curve(best_model, X_train, y_train, cv=5, n_jobs=1)
    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Training score')
    plt.plot(train_sizes, np.mean(val_scores, axis=1), label='Validation score')
    plt.title(f"Learning Curve for {name}")
    plt.xlabel('Training Size')
    plt.ylabel('Score')
    plt.legend()
    plt.show()

    # 오분류 샘플 출력
    misclassified = pd.DataFrame({
        'Subfolder Name': df.loc[y_test.index, 'Subfolder Name'],
        'File Name': df.loc[y_test.index, 'File Name'],
        'Actual': y_test,
        'Predicted': y_test_pred
    })
    misclassified = misclassified[misclassified['Actual'] != misclassified['Predicted']]
    print(f"\n오분류 샘플 ({name}):")
    print(misclassified[['Subfolder Name', 'File Name', 'Actual', 'Predicted']])

    print(" - 🔍 해석 정보: ")
    try:
        if hasattr(best_model, "coef_"):
            coef_df = pd.DataFrame({'Feature': X_train.columns, 'Coefficient': best_model.coef_[0]})
            print(coef_df.sort_values(by='Coefficient', key=abs, ascending=False).head(10))
        elif hasattr(best_model, "feature_importances_"):
            fi_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': best_model.feature_importances_})
            print(fi_df.sort_values(by='Importance', ascending=False).head(10))
        else:
            print("  > SHAP 사용 시도 중...")  # SHAP 생략
    except Exception as e:
        print(f"  > SHAP 해석 예외 발생: {e}")

    model_results[name] = {
        "Best Params": grid_search.best_params_,
        "CV Mean": grid_search.best_score_,
        "Validation Accuracy": val_acc,
        "Test Accuracy": test_acc
    }

    if name == "Random Forest":
        global final_model, y_pred_ai
        final_model = best_model
        y_pred_ai = y_test_pred

    # 모델 학습 후 불필요한 객체들 메모리 해제
    del grid_search
    del best_model
    gc.collect()  # 불필요한 객체들 메모리에서 해제

# 모델 학습 및 평가
for name, model_info in models.items():
    tune_and_evaluate_model(name, model_info)

# 모델 성능 비교표 출력
results_df = pd.DataFrame(model_results).T.sort_values(by="Validation Accuracy", ascending=False)
print("\n📊 모델 성능 비교표:")
print(results_df)
best_model_name = results_df.index[0]
print(f"\n✅ 최적 모델: {best_model_name}")

# 시각화 및 통계 테스트
for label, data in zip(["Train", "Validation", "Test"], [y_train, y_val, y_test]):
    plt.figure(figsize=(8, 4))
    sns.countplot(x=data, hue=data, palette="Set2", legend=False)
    plt.title(f"{label} Dataset Distribution")
    plt.xlabel("Class (0: AI Image, 1: Human Image)")
    plt.ylabel("Count")
    plt.show()

# 전체 코드 끝나고
gc.collect()  # 전체 작업 끝난 후 가비지 컬렉션 실행
