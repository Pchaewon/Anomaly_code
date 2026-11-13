"""Tkinter GUI for Ridge + Robbins-Monro recipe tuning simulation.

Run:
    pip install numpy pandas scikit-learn matplotlib
    python bow_recipe_tkinter_simulator.py

This is a synthetic closed-loop simulator, NOT a validated digital twin.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

X_COLS = [f"frame in temp {i}pct" for i in range(0, 101, 10)]
Y_COL = "avg bow bf total"
PCTS = list(range(0, 101, 10))


def make_demo_data(seed: int = 8) -> pd.DataFrame:
    """Generate correlated temperature profiles and Bow for GUI testing."""
    rng = np.random.default_rng(seed)
    n = 220
    base = np.linspace(350, 450, len(X_COLS))
    common = rng.normal(0, 1.6, (n, 1))
    local = rng.normal(0, 0.65, (n, len(X_COLS)))
    x = base + common + local
    y = 2.0 - 0.06 * np.sum(x - base, axis=1) + rng.normal(0, 0.16, n)
    return pd.DataFrame(x, columns=X_COLS).assign(**{Y_COL: y})


def load_process_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [col for col in X_COLS + [Y_COL] if col not in df.columns]
    if missing:
        raise ValueError("CSV에 필요한 컬럼이 없습니다:\n" + ", ".join(missing))
    df = df.copy()
    for col in X_COLS + [Y_COL]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=X_COLS + [Y_COL]).reset_index(drop=True)
    if len(df) < 15:
        raise ValueError("유효한 행이 최소 15개 필요합니다.")
    return df


@dataclass
class Config:
    target: float
    deadband: float
    step: float
    max_change: float
    max_total: float
    delay: int
    run_count: int
    process_hours: float
    start_bow: float | None
    true_slope: float
    noise_std: float
    drift: float
    seed: int
    alpha: float
    auto: bool

    def validate(self) -> None:
        if self.deadband < 0 or self.step <= 0:
            raise ValueError("Deadband는 0 이상, Step Size는 0 초과여야 합니다.")
        if self.max_change <= 0 or self.max_total <= 0:
            raise ValueError("최대 온도 변경량은 0 초과여야 합니다.")
        if self.delay not in (1, 2):
            raise ValueError("적용 지연은 N+1 또는 N+2여야 합니다.")
        if not 1 <= self.run_count <= 5000:
            raise ValueError("Run 수는 1~5000 범위여야 합니다.")
        if self.process_hours <= 0 or self.noise_std < 0 or self.alpha < 0:
            raise ValueError("공정시간>0, Noise>=0, Ridge Alpha>=0이어야 합니다.")
        if self.true_slope > 0:
            raise ValueError("온도 상승 → Bow 감소 가정이므로 실제 가상 민감도는 0 이하로 설정하세요.")


class Simulator:
    def __init__(self, data: pd.DataFrame, cfg: Config):
        cfg.validate()
        self.cfg = cfg
        self.data = data
        self.x0 = data.iloc[-1][X_COLS].to_numpy(dtype=float)
        self.y0 = float(data.iloc[-1][Y_COL]) if cfg.start_bow is None else cfg.start_bow

        x_train = data.iloc[:-1][X_COLS].to_numpy(dtype=float)
        y_train = data.iloc[:-1][Y_COL].to_numpy(dtype=float)
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x_train)
        ridge = Ridge(alpha=cfg.alpha)
        ridge.fit(x_scaled, y_train)
        self.sensitivity = ridge.coef_ / scaler.scale_

        # Physical knowledge: higher temperature lowers Bow.
        negative_weights = np.maximum(-self.sensitivity, 0.0)
        self.fallback = negative_weights.sum() < 1e-8
        if self.fallback:
            negative_weights = np.ones(len(X_COLS))
        self.weights = negative_weights / negative_weights.sum()
        self.direction = -self.weights

        self.x_active = self.x0.copy()
        self.pending = None  # {apply_run, recipe, estimated_delta, origin}
        self.completed = 0
        self.tuning_count = 0
        self.history: list[dict] = []
        self.next_idx = 0
        self.last_proposed = None
        rng = np.random.default_rng(cfg.seed)
        self.noise = rng.normal(0, cfg.noise_std, cfg.run_count)
        self.noise[0] = 0.0  # first simulated measurement equals START_BOW

    def propose(self, measured_bow: float) -> np.ndarray:
        error = measured_bow - self.cfg.target
        step = self.cfg.step / ((self.completed + 1) ** 0.6)
        delta = -step * error * self.direction
        delta = np.clip(delta, -self.cfg.max_change, self.cfg.max_change)
        return np.clip(
            self.x_active + delta,
            self.x0 - self.cfg.max_total,
            self.x0 + self.cfg.max_total,
        )

    def schedule(self, recipe: np.ndarray, origin: str) -> bool:
        if self.pending is not None:
            raise ValueError("이전 변경 레시피의 결과를 기다리는 중입니다.")
        if self.next_idx < 1:
            raise ValueError("먼저 Run을 실행하여 Bow를 측정하세요.")
        recipe = np.asarray(recipe, dtype=float)
        if len(recipe) != len(X_COLS) or not np.all(np.isfinite(recipe)):
            raise ValueError("온도 입력값 11개를 확인하세요.")
        change = np.clip(recipe - self.x_active, -self.cfg.max_change, self.cfg.max_change)
        proposal = np.clip(
            self.x_active + change,
            self.x0 - self.cfg.max_total,
            self.x0 + self.cfg.max_total,
        )
        if np.allclose(proposal, self.x_active, atol=1e-9, rtol=0):
            return False
        measured_idx = self.next_idx - 1
        if measured_idx + self.cfg.delay >= self.cfg.run_count:
            raise ValueError("시뮬레이션 종료 전 적용할 Run이 없습니다. Run 수를 늘리세요.")
        self.pending = {
            "apply_run": measured_idx + self.cfg.delay,
            "recipe": proposal,
            "estimated_delta": float(np.dot(self.sensitivity, proposal - self.x_active)),
            "origin": origin,
        }
        self.last_proposed = proposal.copy()
        self.tuning_count += 1
        self.history[-1]["decision"] = origin
        self.history[-1]["apply_to"] = self.pending["apply_run"]
        return True

    def next_run(self) -> dict | None:
        if self.next_idx >= self.cfg.run_count:
            return None
        i = self.next_idx
        applied = False
        applied_origin = ""
        if self.pending is not None and i == self.pending["apply_run"]:
            self.x_active = self.pending["recipe"].copy()
            applied_origin = self.pending["origin"]
            self.pending = None
            applied = True
            self.completed += 1

        disturbance = self.cfg.drift * i + self.noise[i]
        bow = (
            self.y0
            + self.cfg.true_slope * np.sum(self.x_active - self.x0)
            + disturbance
        )
        bow_keep = self.y0 + disturbance  # same random noise, no temp change
        error = bow - self.cfg.target

        row = {
            "run": i,
            "hours": (i + 1) * self.cfg.process_hours,
            "bow_tuned": float(bow),
            "bow_keep": float(bow_keep),
            "error": float(error),
            "decision": "KEEP",
            "applied": applied,
            "applied_origin": applied_origin,
            "apply_to": None,
            "mean_temp": float(np.mean(self.x_active)),
            "total_temp_change": float(np.sum(self.x_active - self.x0)),
        }
        self.history.append(row)
        self.next_idx += 1

        if self.pending is not None:
            row["decision"] = "WAIT"
        elif abs(error) <= self.cfg.deadband:
            row["decision"] = "KEEP"
        elif self.cfg.auto:
            if i + self.cfg.delay >= self.cfg.run_count:
                row["decision"] = "END"
            else:
                proposal = self.propose(bow)
                row["decision"] = "LIMIT"
                if self.schedule(proposal, "TUNE"):
                    row["decision"] = "TUNE"
        else:
            row["decision"] = "REVIEW"  # manual mode waits for engineer

        return row

    def run_all(self) -> None:
        while self.next_idx < self.cfg.run_count:
            self.next_run()

    def history_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.history)

    def metrics(self) -> str:
        if not self.history:
            return "Run을 실행하면 성능 지표가 표시됩니다."
        df = self.history_frame()
        tuned_err = np.abs(df["bow_tuned"] - self.cfg.target)
        keep_err = np.abs(df["bow_keep"] - self.cfg.target)
        return (
            f"RM MAE {tuned_err.mean():.3f}  |  유지 MAE {keep_err.mean():.3f}  |  "
            f"허용범위 만족률 {np.mean(tuned_err <= self.cfg.deadband):.1%} / "
            f"{np.mean(keep_err <= self.cfg.deadband):.1%}  |  "
            f"튜닝 예약 {self.tuning_count}회 / 적용 {self.completed}회"
        )


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Bow Recipe Simulator | Ridge + Robbins-Monro")
        self.geometry("1390x875")
        self.minsize(1130, 720)
        self.df = make_demo_data()
        self.source_name = "내장 데모 데이터 (실제 공정 아님)"
        self.sim: Simulator | None = None
        self.fields: dict[str, tk.StringVar] = {}
        self.manual_entries: list[tk.StringVar] = []
        self._make_ui()
        self.reset_simulator()

    def _field(self, parent, row, label, key, default, width=13):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=3, padx=(5, 8))
        self.fields[key] = tk.StringVar(value=str(default))
        ttk.Entry(parent, textvariable=self.fields[key], width=width).grid(
            row=row, column=1, sticky="ew", pady=3, padx=5
        )

    def _make_ui(self):
        split = ttk.Panedwindow(self, orient="horizontal")
        split.pack(fill="both", expand=True, padx=10, pady=8)
        left = ttk.Frame(split, padding=6)
        right = ttk.Frame(split, padding=6)
        split.add(left, weight=0)
        split.add(right, weight=1)

        ttk.Label(left, text="시뮬레이션 제어", font=("", 15, "bold")).pack(anchor="w", pady=(0, 8))
        ttk.Button(left, text="실제 CSV 불러오기", command=self.load_csv).pack(fill="x", pady=2)
        ttk.Button(left, text="데모 데이터로 전환", command=self.use_demo).pack(fill="x", pady=2)
        self.source_label = ttk.Label(left, text="", wraplength=310, foreground="#566173")
        self.source_label.pack(anchor="w", pady=(3, 9))

        params = ttk.LabelFrame(left, text="품질 및 제어 파라미터", padding=7)
        params.pack(fill="x")
        params.columnconfigure(1, weight=1)
        entries = [
            ("Target Bow", "target", 0.0),
            ("Deadband (±)", "deadband", 1.5),
            ("Step Size", "step", 0.5),
            ("1회 최대 ΔT (℃)", "max_change", 2.0),
            ("누적 최대 ΔT (℃)", "max_total", 5.0),
            ("Run 수", "runs", 120),
            ("초기 Bow (빈칸=CSV)", "start_bow", 2.5),
            ("가상 실제 민감도 (음수)", "true_slope", -0.06),
            ("Bow Noise 표준편차", "noise", 0.05),
            ("Run당 Bow Drift", "drift", 0.0),
            ("Ridge alpha", "alpha", 10.0),
            ("난수 Seed", "seed", 42),
        ]
        for row, (label, key, value) in enumerate(entries):
            self._field(params, row, label, key, value)
        row = len(entries)
        ttk.Label(params, text="적용 시점").grid(row=row, column=0, sticky="w", padx=5, pady=3)
        self.fields["delay"] = tk.StringVar(value="N+2")
        ttk.Combobox(params, textvariable=self.fields["delay"], values=["N+1", "N+2"],
                     state="readonly", width=10).grid(row=row, column=1, sticky="ew", padx=5)
        row += 1
        ttk.Label(params, text="가공시간 (h)").grid(row=row, column=0, sticky="w", padx=5, pady=3)
        self.fields["hours"] = tk.StringVar(value="13.3")
        ttk.Combobox(params, textvariable=self.fields["hours"], values=["13.3", "18.5"],
                     state="readonly", width=10).grid(row=row, column=1, sticky="ew", padx=5)
        row += 1
        ttk.Label(params, text="제어 방식").grid(row=row, column=0, sticky="w", padx=5, pady=3)
        self.fields["mode"] = tk.StringVar(value="자동 추천")
        ttk.Combobox(params, textvariable=self.fields["mode"], values=["자동 추천", "수동 조정"],
                     state="readonly", width=10).grid(row=row, column=1, sticky="ew", padx=5)

        btn = ttk.Frame(left)
        btn.pack(fill="x", pady=(10, 5))
        ttk.Button(btn, text="초기화 / 설정 적용", command=self.reset_simulator).pack(fill="x", pady=2)
        ttk.Button(btn, text="다음 Run 1회", command=self.step_once).pack(fill="x", pady=2)
        ttk.Button(btn, text="남은 Run 전체 실행", command=self.run_all).pack(fill="x", pady=2)
        ttk.Button(btn, text="CSV / 그래프 저장", command=self.export).pack(fill="x", pady=2)
        ttk.Label(left, text="※ 값 변경 후 '초기화 / 설정 적용'을 누르세요.",
                  foreground="#697386", wraplength=310).pack(anchor="w", pady=5)

        self.notebook = ttk.Notebook(right)
        self.notebook.pack(fill="both", expand=True)
        tab_graph = ttk.Frame(self.notebook)
        tab_recipe = ttk.Frame(self.notebook)
        tab_history = ttk.Frame(self.notebook)
        self.notebook.add(tab_graph, text="Bow / 온도 추이")
        self.notebook.add(tab_recipe, text="11구간 레시피 / 수동 조정")
        self.notebook.add(tab_history, text="Run 이력")

        self.figure = Figure(figsize=(9.4, 6.5), dpi=100)
        self.ax_bow, self.ax_temp = self.figure.subplots(2, 1, gridspec_kw={"height_ratios": [2, 1]})
        self.figure.subplots_adjust(hspace=0.35, left=0.09, right=0.97, top=0.95, bottom=0.09)
        self.canvas = FigureCanvasTkAgg(self.figure, master=tab_graph)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.metric_label = ttk.Label(tab_graph, text="", font=("", 10, "bold"), wraplength=900)
        self.metric_label.pack(fill="x", padx=10, pady=8)

        ttk.Label(tab_recipe, text="현재 레시피 / 예약 레시피 (예약 없을 때는 현재 값)",
                  font=("", 12, "bold")).pack(anchor="w", padx=12, pady=(12, 5))
        cols = ("profile", "active", "next", "delta", "ridge")
        self.recipe_table = ttk.Treeview(tab_recipe, columns=cols, show="headings", height=11)
        labels = {"profile": "구간", "active": "현재 온도", "next": "예약/추천 온도",
                  "delta": "변경량", "ridge": "Ridge 민감도"}
        for col in cols:
            self.recipe_table.heading(col, text=labels[col])
            self.recipe_table.column(col, width=120, anchor="center")
        self.recipe_table.pack(fill="x", padx=12, pady=(0, 10))

        manual = ttk.LabelFrame(tab_recipe, text="수동 시나리오: 각 프로파일 Δ℃ 입력 후 N+1/N+2 예약", padding=8)
        manual.pack(fill="x", padx=12, pady=8)
        for j, pct in enumerate(PCTS):
            r, c = divmod(j, 6)
            frame = ttk.Frame(manual)
            frame.grid(row=r, column=c, padx=5, pady=5)
            ttk.Label(frame, text=f"{pct}%").pack()
            v = tk.StringVar(value="0.0")
            self.manual_entries.append(v)
            ttk.Entry(frame, textvariable=v, width=8, justify="center").pack()
        preset = ttk.Frame(manual)
        preset.grid(row=2, column=0, columnspan=6, sticky="w", pady=(10, 2))
        ttk.Button(preset, text="모두 +0.2℃", command=lambda: self.manual_preset(0.2)).pack(side="left", padx=3)
        ttk.Button(preset, text="모두 -0.2℃", command=lambda: self.manual_preset(-0.2)).pack(side="left", padx=3)
        ttk.Button(preset, text="모두 0℃", command=lambda: self.manual_preset(0)).pack(side="left", padx=3)
        ttk.Button(preset, text="수동 레시피 예약", command=self.schedule_manual).pack(side="left", padx=12)
        ttk.Label(tab_recipe, text="수동 조정 모드에서 '다음 Run 1회'로 Bow를 측정한 후 수동 레시피를 예약하세요.\n"
                  "변경 폭은 1회/누적 제한으로 자동 제한되며, 예약된 변경은 결과 확인 전까지 중복 예약되지 않습니다.",
                  wraplength=850, foreground="#697386").pack(anchor="w", padx=13, pady=8)

        hist_cols = ("run", "hours", "bow_tuned", "bow_keep", "decision", "applied", "apply_to")
        self.hist_table = ttk.Treeview(tab_history, columns=hist_cols, show="headings")
        heads = ["Run", "누적 시간(h)", "튜닝 Bow", "유지 Bow", "판단", "적용", "예약 Run"]
        for col, name in zip(hist_cols, heads):
            self.hist_table.heading(col, text=name)
            self.hist_table.column(col, width=110, anchor="center")
        scrollbar = ttk.Scrollbar(tab_history, orient="vertical", command=self.hist_table.yview)
        self.hist_table.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.hist_table.pack(fill="both", expand=True)

        self.status_var = tk.StringVar()
        ttk.Label(self, textvariable=self.status_var, padding=7, relief="sunken", anchor="w").pack(fill="x")

    def _config(self) -> Config:
        f = lambda key: float(self.fields[key].get().strip())
        bow_text = self.fields["start_bow"].get().strip()
        return Config(
            target=f("target"), deadband=f("deadband"), step=f("step"),
            max_change=f("max_change"), max_total=f("max_total"),
            delay=1 if self.fields["delay"].get() == "N+1" else 2,
            run_count=int(self.fields["runs"].get()),
            process_hours=f("hours"), start_bow=None if bow_text == "" else float(bow_text),
            true_slope=f("true_slope"), noise_std=f("noise"),
            drift=f("drift"), seed=int(self.fields["seed"].get()),
            alpha=f("alpha"), auto=self.fields["mode"].get() == "자동 추천",
        )

    def load_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV", "*.csv"), ("모든 파일", "*.*")])
        if not path:
            return
        try:
            data = load_process_csv(path)
        except Exception as exc:
            messagebox.showerror("CSV 오류", str(exc))
            return
        self.df = data
        self.source_name = f"실제 CSV: {Path(path).name} ({len(data)}행)"
        self.fields["start_bow"].set("")  # actual last measured Bow
        self.reset_simulator()

    def use_demo(self):
        self.df = make_demo_data()
        self.source_name = "내장 데모 데이터 (실제 공정 아님)"
        self.fields["start_bow"].set("2.5")
        self.reset_simulator()

    def reset_simulator(self):
        try:
            self.sim = Simulator(self.df, self._config())
        except Exception as exc:
            messagebox.showerror("설정 오류", str(exc))
            return
        self.source_label.configure(
            text=f"{self.source_name}\nRidge 민감도 범위: "
                 f"{self.sim.sensitivity.min():+.4f} ~ {self.sim.sensitivity.max():+.4f}"
                 + (" (균등 가중치 대체)" if self.sim.fallback else "")
        )
        self.refresh()
        self.status_var.set("시뮬레이션 준비 완료. '다음 Run 1회' 또는 '남은 Run 전체 실행' 선택")

    def step_once(self):
        if self.sim is None:
            return
        row = self.sim.next_run()
        if row is None:
            messagebox.showinfo("완료", "설정한 모든 Run을 실행했습니다.")
            return
        self.refresh()
        pending = self.sim.pending
        pending_text = (
            f" / Run {pending['apply_run']} 적용 예약 (Ridge 예상 ΔBow {pending['estimated_delta']:+.3f})"
            if pending else ""
        )
        self.status_var.set(
            f"Run {row['run']} | Bow {row['bow_tuned']:+.3f} | "
            f"판단 {row['decision']} | 적용={row['applied']}{pending_text}"
        )

    def run_all(self):
        if self.sim is None:
            return
        self.sim.run_all()
        self.refresh()
        self.status_var.set(f"전체 실행 완료: {self.sim.next_idx} Run | {self.sim.metrics()}")

    def manual_preset(self, value: float):
        for v in self.manual_entries:
            v.set(f"{value:+.2f}")

    def schedule_manual(self):
        if self.sim is None:
            return
        try:
            delta = np.array([float(v.get()) for v in self.manual_entries])
            ok = self.sim.schedule(self.sim.x_active + delta, "MANUAL")
            if not ok:
                messagebox.showinfo("변경 없음", "적용할 유효한 온도 변경량이 없습니다.")
                return
        except Exception as exc:
            messagebox.showerror("수동 레시피 오류", str(exc))
            return
        self.refresh()
        self.status_var.set(f"수동 레시피를 Run {self.sim.pending['apply_run']}에 예약했습니다.")

    def refresh(self):
        s = self.sim
        if s is None:
            return
        df = s.history_frame()
        self.ax_bow.clear()
        self.ax_temp.clear()
        c = s.cfg
        self.ax_bow.axhspan(c.target - c.deadband, c.target + c.deadband,
                            color="#3ca370", alpha=0.12, label="Deadband")
        self.ax_bow.axhline(c.target, color="black", ls="--", lw=1, label="Target")
        if not df.empty:
            self.ax_bow.plot(df["run"], df["bow_keep"], color="#999999", lw=1.3, label="Keep")
            self.ax_bow.plot(df["run"], df["bow_tuned"], color="#1767ad", lw=2, label="RM / Manual")
            applied = df[df["applied"]]
            if not applied.empty:
                self.ax_bow.scatter(applied["run"], applied["bow_tuned"],
                                    color="#ed8b31", marker="D", s=24, label="Applied", zorder=3)
        self.ax_bow.set_title("Bow vs. Run (same disturbance for both scenarios)")
        self.ax_bow.set_xlabel("Run")
        self.ax_bow.set_ylabel("Bow")
        self.ax_bow.grid(alpha=0.25)
        self.ax_bow.legend(loc="best", fontsize=8)

        self.ax_temp.plot(PCTS, s.x0, color="#999999", ls="--", label="Initial")
        self.ax_temp.plot(PCTS, s.x_active, marker="o", color="#1767ad", label="Active")
        if s.pending is not None:
            self.ax_temp.plot(PCTS, s.pending["recipe"], marker="s", color="#e37b29",
                              label=f"Planned Run {s.pending['apply_run']}")
        self.ax_temp.set_xlabel("Profile (%)")
        self.ax_temp.set_ylabel("Temp (C)")
        self.ax_temp.set_title("Temperature Recipe")
        self.ax_temp.grid(alpha=0.25)
        self.ax_temp.legend(loc="best", fontsize=8)
        self.canvas.draw_idle()
        self.metric_label.configure(text=s.metrics())

        for item in self.recipe_table.get_children():
            self.recipe_table.delete(item)
        next_recipe = s.pending["recipe"] if s.pending is not None else s.x_active
        for pct, cur, nxt, sens in zip(PCTS, s.x_active, next_recipe, s.sensitivity):
            self.recipe_table.insert("", "end", values=(
                f"{pct}%", f"{cur:.3f}", f"{nxt:.3f}", f"{nxt-cur:+.3f}", f"{sens:+.5f}"
            ))
        for item in self.hist_table.get_children():
            self.hist_table.delete(item)
        if not df.empty:
            for _, row in df.tail(400).iloc[::-1].iterrows():
                to_run = "-" if pd.isna(row["apply_to"]) else str(int(row["apply_to"]))
                self.hist_table.insert("", "end", values=(
                    int(row["run"]), f"{row['hours']:.1f}", f"{row['bow_tuned']:+.3f}",
                    f"{row['bow_keep']:+.3f}", row["decision"],
                    "Y" if row["applied"] else "-", to_run
                ))

    def export(self):
        if self.sim is None or not self.sim.history:
            messagebox.showinfo("저장 불가", "먼저 1회 이상 Run을 실행하세요.")
            return
        folder = filedialog.askdirectory(title="결과 저장 폴더")
        if not folder:
            return
        root = Path(folder)
        try:
            self.sim.history_frame().to_csv(root / "simulation_history.csv", index=False, encoding="utf-8-sig")
            recipe = self.sim.pending["recipe"] if self.sim.pending else self.sim.x_active
            pd.DataFrame({
                "Profile": X_COLS,
                "Current_Temp": self.sim.x_active,
                "Planned_Temp": recipe,
                "Change_Temp": recipe - self.sim.x_active,
                "Sensitivity": self.sim.sensitivity,
                "Weight": self.sim.weights,
            }).to_csv(root / "simulation_recipe.csv", index=False, encoding="utf-8-sig")
            self.figure.savefig(root / "simulation_plot.png", dpi=160)
        except Exception as exc:
            messagebox.showerror("저장 실패", str(exc))
            return
        messagebox.showinfo("저장 완료", f"저장 위치: {root}\nCSV 2개 + PNG 1개")


if __name__ == "__main__":
    App().mainloop()
