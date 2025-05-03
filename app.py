import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# DATA
st.title("Wine Data Dashboard")
df = pd.read_csv("wine.csv")  

# hvis type er numerisk, lav den til tekst
if df['type'].dtype != object:
    df['type'] = df['type'].map({0: "white", 1: "red"})

# SIDEBAR
st.sidebar.title("Vælg visualisering")
choice = st.sidebar.radio("Visualiseringstype", [
    "Alcohol Histogram",
    "Boxplot: Quality by Wine Type",
    "Correlation Heatmap",
    "Alcohol vs Quality Scatter",
    "PCA (2D projection)",
    "PCA (3D projection)",
    "Additional info"

])

# VISUALS

if choice == "Alcohol Histogram":
    st.subheader("1. Alcohol Distribution")
    fig, ax = plt.subplots()
    ax.hist(df['alcohol'], bins=20, color='skyblue', edgecolor='black')
    ax.set_xlabel("Alcohol %")
    ax.set_ylabel("Antal vine")
    st.pyplot(fig)

elif choice == "Boxplot: Quality by Wine Type":
    st.subheader("2. Wine Quality by Type")
    fig, ax = plt.subplots()
    sns.boxplot(data=df, x='type', y='quality', ax=ax)
    st.pyplot(fig)

elif choice == "Correlation Heatmap":
    st.subheader("3. Correlation Matrix")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap( df.select_dtypes(include='number').corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

elif choice == "Alcohol vs Quality Scatter":
    st.subheader("4. Alcohol vs Quality")
    fig = px.scatter(df, x='alcohol', y='quality', color='type', title="Alcohol vs Quality")
    st.plotly_chart(fig)

elif choice == "PCA (2D projection)":
    st.subheader("5. PCA: 2D Projection")
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    X = StandardScaler().fit_transform(numeric_df.drop(columns=['quality'], errors='ignore'))
    pca = PCA(n_components=2)
    components = pca.fit_transform(X)

    pca_df = pd.DataFrame(data=components, columns=["PC1", "PC2"])
    pca_df["type"] = df["type"]

    fig = px.scatter(pca_df, x="PC1", y="PC2", color="type", title="PCA - 2D Projection")
    st.plotly_chart(fig)

elif choice == "PCA (3D projection)":
    st.subheader("6. PCA fra CSV – 3D")
    pca_df = pd.read_csv("pca.csv")

    fig = px.scatter_3d(
        pca_df, x='pc 1', y='pc 2', z='pc 3',
        color=pca_df['type'] if 'type' in pca_df.columns else None,
        title="PCA - 3D Projection from CSV"
    )
    st.plotly_chart(fig)
elif choice == "Additional info":
    st.subheader("📖 Ekstern viden")
    st.markdown("""
### 🔗 Links:
- [Wikipedia: Wine Quality](https://en.wikipedia.org/wiki/Wine_quality)  
- [YouTube: What makes a wine great?](https://www.youtube.com/results?search_query=wine+quality+factors)

### 💡 Anbefalinger:
- Vin med højere alkoholindhold har ofte højere vurdering.
- Residual sugar har begrænset indflydelse på kvalitet.
- PCA kan bruges til at reducere støj og visualisere vinens kemiske profil.

""")