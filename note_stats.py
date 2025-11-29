import marimo

__generated_with = "0.12.10"
app = marimo.App(width="medium")


@app.cell
def _(alt, dt, mo):
    chart_width = 450

    mo.vstack([
        mo.md("# Training progress"),
        alt.Chart(dt).mark_line().encode(
            x='epoch',
            y='policy_acc',
            tooltip=['epoch', 'policy_acc']
        ).properties(title="Policy Accuracy over Epochs", width=chart_width) |
        alt.Chart(dt).mark_line(color='red').encode(
            x='epoch',
            y=alt.Y('policy_acc_change_pct', 
                    scale=alt.Scale(domain=[-1, 2], clamp=True)),
            tooltip=['epoch', 'policy_acc_change_pct'],
        ).properties(title="Policy Accuracy Change", width=chart_width),

        alt.Chart(dt).mark_line().encode(
            x='epoch',
            y='loss',
            tooltip=['epoch', 'loss']
        ).properties(title="Loss over Epochs", width=chart_width) |
        alt.Chart(dt).mark_line(color='green').encode(
            x='epoch',
            y='value_mae',
            tooltip=['epoch', 'value_mae']
        ).properties(title="Value MAE over Epochs", width=chart_width),
    ])
    return (chart_width,)


@app.cell
def _(dt):
    # Enrich data
    dt['policy_acc_change_pct'] = dt['policy_acc'].pct_change() * 100
    dt['policy_acc_change_pct'] = dt['policy_acc_change_pct'].fillna(0)  # Fill NaN for first row
    dt
    return


@app.cell
def _(pd):
    dt = pd.read_csv("/Users/cheshir/Documents/projects/ml/alpahgomoku/checkpoints/training_metrics.csv")
    return (dt,)


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import altair as alt
    return alt, mo, pd


if __name__ == "__main__":
    app.run()
