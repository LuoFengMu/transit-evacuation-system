"""Three-scenario comparison for v0.4.0.

Scenario A: Bus direct to shelters (baseline)
Scenario B: Rail priority (bus → rail for most)
Scenario C: Hybrid cooperative (walk / bus / rail mix)
"""
from dataclasses import dataclass, field
import pandas as pd
import plotly.graph_objects as go
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

    def render_chart(self):
        if len(self.metrics) < 2:
            st.info("需要至少两个方案进行对比")
            return
        df = self.to_dataframe()
        metrics_to_plot = ["completion_rate", "rail_share", "max_station_pressure", "avg_walk_distance_m"]
        labels_cn = {"completion_rate": "疏散完成率", "rail_share": "轨道分担率",
                      "max_station_pressure": "最大站点压力", "avg_walk_distance_m": "平均步行距离(m)"}

        fig = go.Figure()
        for col in metrics_to_plot:
            if col in df.columns:
                fig.add_trace(go.Bar(
                    name=labels_cn.get(col, col),
                    x=df["scenario"], y=df[col],
                    text=df[col].apply(lambda v: f"{v:.2f}" if v < 1 else f"{v:.0f}"),
                    textposition="outside",
                ))
        fig.update_layout(
            barmode="group", height=350,
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", y=1.1),
        )
        st.plotly_chart(fig, use_container_width=True)

    def render_table(self):
        df = self.to_dataframe()
        st.dataframe(df.set_index("scenario"), use_container_width=True)
