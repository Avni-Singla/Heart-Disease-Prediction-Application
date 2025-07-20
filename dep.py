import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Page config
st.set_page_config(page_title="Heart Disease Predictor", layout="centered")
st.title("🫀 Heart Disease Prediction Web App")

# File uploader
uploaded_file = st.sidebar.file_uploader("📂 Upload Heart Disease CSV File", type=["csv"])

# Load data if uploaded
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("✅ File successfully loaded!")

        # Show data preview
        with st.expander("🔍 Preview Dataset"):
            st.dataframe(df.head())

        # Separate features and target
        target_col = "target" if "target" in df.columns else df.columns[-1]
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Train model (DO NOT use caching)
        def train_model(X, y):
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model.fit(X_train, y_train)
            return model, X_test, y_test

        model, X_test, y_test = train_model(X, y)

        # Sidebar input form
        st.sidebar.header("📝 Enter Patient Data")
        user_input = {}
        for col in X.columns:
            if df[col].dtype in [np.float64, np.int64]:
                user_input[col] = st.sidebar.number_input(
                    f"{col}", float(df[col].min()), float(df[col].max()), float(df[col].mean())
                )
            else:
                user_input[col] = st.sidebar.text_input(f"{col}", "")

        input_df = pd.DataFrame([user_input])

        # Prediction
        if st.sidebar.button("🔮 Predict"):
            prediction = model.predict(input_df)[0]
            proba = model.predict_proba(input_df)[0]

            st.subheader("📊 Prediction Result")
            st.write(f"**Prediction:** {'Heart Disease' if prediction == 1 else 'No Heart Disease'}")
            st.write(f"**Confidence:** {round(np.max(proba)*100, 2)}%")

            # Probability chart
            fig, ax = plt.subplots()
            sns.barplot(x=model.classes_, y=proba, ax=ax)
            ax.set_xlabel("Class")
            ax.set_ylabel("Probability")
            st.pyplot(fig)

        # Feature importance
        with st.expander("📌 Feature Importance"):
            importances = model.feature_importances_
            feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=True)
            fig, ax = plt.subplots(figsize=(8, 6))
            feat_imp.plot(kind='barh', ax=ax)
            ax.set_title("Feature Importance")
            st.pyplot(fig)

        # Confusion matrix
        with st.expander("🧮 Confusion Matrix on Test Data"):
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
            fig, ax = plt.subplots()
            disp.plot(ax=ax)
            st.pyplot(fig)

        st.markdown("---")
        st.caption("Made with ❤️ using Streamlit")

    except Exception as e:
        st.error(f"❌ Error reading or processing the file: {e}")
        st.stop()

else:
    st.warning("⚠️ Please upload a CSV file to get started.")
