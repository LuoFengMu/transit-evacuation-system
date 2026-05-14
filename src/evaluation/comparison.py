"""Three-scenario comparison for v0.4.1."""
from dataclasses import dataclass, field
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from src.evaluation.metrics import EvacuationMetrics, metrics_to_dict


@dataclass
class ComparisonResult:
    scenario_labels: list[str] = field(default_factory=list)
    metrics: list[EvacuationMetrics] = field(default_factory=list)

    def add(self, label: str, m: EvacuationMetrics):
        self.scenario_labels.append(label)
        self.metrics.append(m)

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for label, m in zip(self.scenario_labels, self.metrics):
            d = metrics_to_dict(m)
            d["scenario"] = label
            rows.append(d)
        return pd.DataFrame(rows)

    def render_chart(self, key: str = "comp"):
        if len(self.metrics) < 2:
            st.info("需要至少两个方案进行对比")
            return

        df = self.to_dataframe()
        scenarios = df["scenario"].tolist()

        # Split into two groups by scale
        rate_metrics = {
            "completion_rate": ("疏散完成率", "成功率"),
            "max_station_pressure": ("最大站点压力", "压力指数"),
        }
        dist_metrics = {
            "avg_walk_distance_m": ("平均步行距离(m)", "距离(m)"),
        }

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("比率指标 (0-1)", "距离指标 (m)"),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]],
        )

        colors = ["#2980b9", "#e67e22", "#27ae60"]
        for i, (col, (title, unit)) in enumerate(rate_metrics.items()):
            if col in df.columns:
                fig.add_trace(go.Bar(
                    name=title, x=scenarios, y=df[col],
                    text=[f"{v:.1%}" if v <= 1 else f"{v:.2f}" for v in df[col]],
                    textposition="outside",
                    marker_color=colors[i % len(colors)],
                ), row=1, col=1)

        for i, (col, (title, unit)) in enumerate(dist_metrics.items()):
            if col in df.columns:
                fig.add_trace(go.Bar(
                    name=title, x=scenarios, y=df[col],
                    text=[f"{v:.0f}" for v in df[col]],
                    textposition="outside",
                    marker_color=colors[(len(rate_metrics) + i) % len(colors)],
                ), row=1, col=2)

        fig.update_layout(
            height=350, margin=dict(l=10, r=10, t=40, b=10),
            legend=dict(orientation="h", y=1.15),
            barmode="group",
        )
        fig.update_yaxes(range=[0, 1.3], row=1, col=1)
        st.plotly_chart(fig, use_container_width=True, key=f"comp_chart_{key}")

    def render_table(self):
        df = self.to_dataframe()
        # Format for display
        display_df = df.copy()
        for col in ["completion_rate", "rail_share"]:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda v: f"{v:.1%}")
        if "max_station_pressure" in display_df.columns:
            display_df["max_station_pressure"] = display_df["max_station_pressure"].apply(lambda v: f"{v:.2f}")
        if "avg_walk_distance_m" in display_df.columns:
            display_df["avg_walk_distance_m"] = display_df["avg_walk_distance_m"].apply(lambda v: f"{v:.0f}")
        st.dataframe(display_df.set_index("scenario"), use_container_width=True)
