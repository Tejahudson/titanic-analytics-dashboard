import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# =====================================================
# Load Dataset
# =====================================================
df = pd.read_csv("Titanic-Dataset.csv")

# =====================================================
# Welcome Section
# =====================================================
st.markdown("""
### Welcome 👋  
This dashboard allows users to interactively explore the Titanic dataset.

**Small Tutorial:**
- Select the type of analysis from the dropdown
- Choose specific options to view insights
- Only selected analyses will be displayed
""")

# =====================================================
# Main Analysis Dropdown
# =====================================================
analysis_type = st.selectbox(
    "Select Analysis Type",
    [
        "Select an option",
        "Survival Analysis",
        "Exploratory Data Analysis",
        "Predictive Modeling"
    ],
    index=0
)

# =====================================================
# PHASE 1: SURVIVAL ANALYSIS
# =====================================================
if analysis_type == "Survival Analysis":
    st.subheader("Survival Analysis")

    survival_option = st.selectbox(
        "Select Survival Analysis Option",
        [
            "Select an option",
            "Overall Survival Rate",
            "Survival by Gender",
            "Survival by Passenger Class",
            "Survival by Age Group"
        ],
        index=0
    )

    if survival_option == "Overall Survival Rate":
        survival_rate = df["Survived"].mean() * 100
        st.metric("Overall Survival Rate", f"{survival_rate:.2f}%")

    elif survival_option == "Survival by Gender":
        gender_survival = df.groupby("Sex")["Survived"].mean() * 100

        fig, ax = plt.subplots()
        gender_survival.plot(kind="bar", ax=ax)
        ax.set_xlabel("Gender")
        ax.set_ylabel("Survival Rate (%)")
        ax.set_title("Survival Rate by Gender")
        st.pyplot(fig)

        st.write(
            "Insight: Female passengers had a significantly higher survival rate compared to male passengers."
        )

    elif survival_option == "Survival by Passenger Class":
        class_survival = df.groupby("Pclass")["Survived"].mean() * 100

        fig, ax = plt.subplots()
        class_survival.plot(kind="bar", ax=ax)
        ax.set_xlabel("Passenger Class")
        ax.set_ylabel("Survival Rate (%)")
        ax.set_title("Survival Rate by Passenger Class")
        st.pyplot(fig)

        st.write(
            "Insight: Third-class passengers experienced the lowest survival rate."
        )

    elif survival_option == "Survival by Age Group":

        def age_group(age):
            if age < 18:
                return "Child"
            elif age <= 60:
                return "Adult"
            else:
                return "Senior"

        df_age = df.dropna(subset=["Age"]).copy()
        df_age["AgeGroup"] = df_age["Age"].apply(age_group)

        age_survival = df_age.groupby("AgeGroup")["Survived"].mean() * 100

        fig, ax = plt.subplots()
        age_survival.plot(kind="bar", ax=ax)
        ax.set_xlabel("Age Group")
        ax.set_ylabel("Survival Rate (%)")
        ax.set_title("Survival Rate by Age Group")
        st.pyplot(fig)

        st.write(
            "Insight: Children had a higher survival rate, aligning with evacuation priority policies."
        )

# =====================================================
# PHASE 2: EXPLORATORY DATA ANALYSIS (EDA)
# =====================================================
if analysis_type == "Exploratory Data Analysis":
    st.subheader("Exploratory Data Analysis")

    eda_column = st.selectbox(
        "Select Column",
        df.columns
    )

    plot_type = st.selectbox(
        "Select Plot Type",
        ["Histogram", "Bar Chart", "Scatter Plot"]
    )

    fig, ax = plt.subplots()

    if plot_type == "Histogram":
        ax.hist(df[eda_column].dropna(), bins=20)
        ax.set_xlabel(eda_column)
        ax.set_ylabel("Frequency")
        ax.set_title(f"Histogram of {eda_column}")

    elif plot_type == "Bar Chart":
        df[eda_column].value_counts().plot(kind="bar", ax=ax)
        ax.set_xlabel(eda_column)
        ax.set_ylabel("Count")
        ax.set_title(f"Bar Chart of {eda_column}")

    elif plot_type == "Scatter Plot":
        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

        y_col = st.selectbox(
            "Select Y-axis Column",
            numeric_cols
        )

        ax.scatter(df[eda_column], df[y_col])
        ax.set_xlabel(eda_column)
        ax.set_ylabel(y_col)
        ax.set_title(f"{eda_column} vs {y_col}")

    st.pyplot(fig)

# =====================================================
# PHASE 3: PREDICTIVE MODELING (ML)
# =====================================================
if analysis_type == "Predictive Modeling":
    st.subheader("Predictive Modeling")

    model_type = st.selectbox(
        "Select Model",
        [
            "Linear Regression",
            "Support Vector Machine",
            "Decision Tree",
            "Random Forest"
        ]
    )

    # ---------------------------
    # Preprocessing
    # ---------------------------
    data = df[["Survived", "Pclass", "Sex", "Age", "Fare"]].dropna()
    data["Sex"] = data["Sex"].map({"male": 0, "female": 1})

    X = data.drop("Survived", axis=1)
    y = data["Survived"]

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ---------------------------
    # Model Training & Evaluation
    # ---------------------------
    if model_type == "Linear Regression":
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_absolute_error

        model = LinearRegression()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        st.write(f"Mean Absolute Error (MAE): {mae:.4f}")
        st.write(
            "Note: Linear Regression is used for comparison even though survival is a binary outcome."
        )

    elif model_type == "Support Vector Machine":
        from sklearn.svm import SVC
        from sklearn.metrics import accuracy_score

        model = SVC()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        st.write(f"Accuracy: {acc:.4f}")

    elif model_type == "Decision Tree":
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.metrics import accuracy_score

        model = DecisionTreeClassifier(random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        st.write(f"Accuracy: {acc:.4f}")

    elif model_type == "Random Forest":
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score

        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        st.write(f"Accuracy: {acc:.4f}")
