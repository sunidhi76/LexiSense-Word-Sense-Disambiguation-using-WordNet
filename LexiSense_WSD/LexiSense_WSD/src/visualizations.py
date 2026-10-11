import plotly.express as px


def score_chart(candidates):
    labels = [x["synset"] for x in candidates]
    scores = [x["score"] for x in candidates]
    fig = px.bar(x=scores, y=labels, orientation="h", labels={"x":"Context Match Score", "y":"Candidate Sense"})
    fig.update_layout(height=max(300, 48 * len(labels)), margin=dict(l=10,r=20,t=20,b=20), template="plotly_white")
    return fig


def dataset_word_chart(df):
    counts = df["Target Word"].value_counts().reset_index()
    counts.columns = ["Target Word", "Count"]
    return px.bar(counts, x="Target Word", y="Count", labels={"Count":"Occurrences"}, template="plotly_white")
