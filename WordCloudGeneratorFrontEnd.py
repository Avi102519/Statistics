import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Set Streamlit page config
st.set_page_config(page_title="WordCloud Viewer", layout="centered")

st.title("☁️ WordCloud Viewer")
st.markdown("Designed by **Avi**")

# Default hardcoded text
default_text = (
    "Python Python Python Matplotlib Matplotlib Seaborn Network Plot Violin Chart Pandas "
    "Datascience Wordcloud Spider Radar Parrallel Alpha Color Brewer Density Scatter Barplot "
    "Barplot Boxplot Violinplot Treemap Stacked Area Chart Chart Visualization Dataviz Donut Pie "
    "Time-Series Wordcloud Wordcloud Sankey Bubble"
)

# Option to edit text
text = st.text_area("✍️ Text for WordCloud:", value=default_text, height=200)

# Generate and display the WordCloud
wordcloud = WordCloud(
    width=400,
    height=200,
    margin=2,
    background_color='black',
    colormap='Accent',
    mode='RGBA'
).generate(text)

fig, ax = plt.subplots()
ax.imshow(wordcloud, interpolation='bilinear')  # 'quadric' isn't a valid interpolation
ax.axis("off")
plt.margins(x=0, y=0)

st.pyplot(fig)
