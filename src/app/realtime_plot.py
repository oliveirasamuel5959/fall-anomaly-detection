import csv
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider

# ── Configuration ────────────────────────────────────────────────────────────
CSV_FILE       = Path.cwd() / "sensor_output" / "sensor_data_normal.csv"
WINDOW_SECS    = 5.0      # seconds of data visible in the scrollable window
INTERVAL_MS    = 50       # animation refresh rate (ms)
ROWS_PER_FRAME = 20       # max new CSV rows consumed per frame

# ── Full data store (grows as CSV is read) ───────────────────────────────────
data: dict[str, list] = {
    col: [] for col in ("timestamp", "AccX", "AccY", "AccZ", "GyroX", "GyroY", "GyroZ")
}

# Shared mutable state
state = {
    "file":        None,
    "reader":      None,
    "user_scroll": False,   # True while the user drags the slider
}

# ── Plot setup — white / light theme ─────────────────────────────────────────
plt.rcParams.update({
    "axes.facecolor":   "#f5f5f5",
    "figure.facecolor": "white",
    "axes.edgecolor":   "#cccccc",
    "axes.labelcolor":  "#333333",
    "xtick.color":      "#555555",
    "ytick.color":      "#555555",
    "grid.color":       "#e2e2e2",
    "grid.linestyle":   "--",
    "grid.linewidth":   0.6,
    "axes.grid":        True,
    "text.color":       "#333333",
})

fig = plt.figure(figsize=(14, 8), facecolor="white")
fig.suptitle("IMU Sensor – Live Feed", fontsize=13, color="#222222",
             fontweight="bold", y=0.98)

# Reserve bottom space for the slider
gs = gridspec.GridSpec(2, 1, hspace=0.45,
                       left=0.07, right=0.97,
                       top=0.92, bottom=0.17)

ax_acc  = fig.add_subplot(gs[0])
ax_gyro = fig.add_subplot(gs[1])

for ax, title in ((ax_acc,  "Accelerometer (m/s²)"),
                  (ax_gyro, "Gyroscope (rad/s)")):
    ax.set_title(title, color="#444444", fontsize=10, pad=6)

ACC_COLORS  = {"AccX": "#e63946", "AccY": "#2a9d8f", "AccZ": "#457b9d"}
GYRO_COLORS = {"GyroX": "#e9822c", "GyroY": "#7b2d8b", "GyroZ": "#f4a261"}

lines: dict[str, plt.Line2D] = {}
for key, color in ACC_COLORS.items():
    (lines[key],) = ax_acc.plot([], [], color=color, lw=1.4, label=key)
for key, color in GYRO_COLORS.items():
    (lines[key],) = ax_gyro.plot([], [], color=color, lw=1.4, label=key)

ax_acc.legend(loc="upper right",  fontsize=8, framealpha=0.7, edgecolor="#cccccc")
ax_gyro.legend(loc="upper right", fontsize=8, framealpha=0.7, edgecolor="#cccccc")

status_text = fig.text(0.5, 0.01, "", ha="center", fontsize=8, color="#888888")

# ── Horizontal scroll slider ──────────────────────────────────────────────────
ax_slider = fig.add_axes([0.07, 0.07, 0.88, 0.03], facecolor="#eeeeee")
slider = Slider(
    ax=ax_slider,
    label="◀  Scroll  ▶",
    valmin=0.0,
    valmax=max(WINDOW_SECS, 1.0),
    valinit=WINDOW_SECS,
    color="#457b9d",
    initcolor="none",
)
slider.label.set_color("#555555")
slider.label.set_fontsize(8)
slider.valtext.set_visible(False)

# Detect drag start / end so we can pause live-follow
def _on_press(event):
    if event.inaxes == ax_slider:
        state["user_scroll"] = True

def _on_release(_event):
    state["user_scroll"] = False

fig.canvas.mpl_connect("button_press_event",   _on_press)
fig.canvas.mpl_connect("button_release_event", _on_release)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _open_csv():
    f = open(CSV_FILE, newline="")
    return f, csv.DictReader(f)


def _read_new_rows(reader) -> int:
    read = 0
    for row in reader:
        try:
            data["timestamp"].append(float(row["timestamp"]))
            for col in ("AccX", "AccY", "AccZ", "GyroX", "GyroY", "GyroZ"):
                data[col].append(float(row[col]))
            read += 1
            if read >= ROWS_PER_FRAME:
                break
        except (ValueError, KeyError):
            continue
    return read


def _window_slice(t_start: float, t_end: float) -> slice:
    """Binary-search for the slice of timestamps within [t_start, t_end]."""
    ts = data["timestamp"]
    lo, hi = 0, len(ts)
    for i, t in enumerate(ts):
        if t >= t_start:
            lo = i
            break
    for i in range(len(ts) - 1, -1, -1):
        if ts[i] <= t_end:
            hi = i + 1
            break
    return slice(lo, hi)


# ── Animation callback ────────────────────────────────────────────────────────

def update(_frame):
    # 1. Lazy-open CSV
    if state["file"] is None:
        if not CSV_FILE.exists():
            status_text.set_text(f"Waiting for {CSV_FILE} …")
            return list(lines.values())
        state["file"], state["reader"] = _open_csv()

    # 2. Pull new rows
    rows_read = _read_new_rows(state["reader"])

    ts = data["timestamp"]
    if len(ts) < 2:
        status_text.set_text("Buffering …")
        return list(lines.values())

    t_first  = ts[0]
    t_last   = ts[-1]
    duration = t_last - t_first

    # 3. Expand slider range to cover all data
    new_max = max(duration, WINDOW_SECS)
    if slider.valmax != new_max:
        slider.valmax = new_max
        slider.ax.set_xlim(0, new_max)

    # 4. Auto-follow live end unless the user is scrolling
    if not state["user_scroll"]:
        slider.set_val(new_max)

    # 5. Compute visible window from slider (slider value = right edge in secs)
    right_edge  = slider.val                     # seconds from t_first
    t_win_end   = t_first + right_edge
    t_win_start = t_win_end - WINDOW_SECS
    sl = _window_slice(t_win_start, t_win_end)

    ts_win = ts[sl]
    if not ts_win:
        return list(lines.values())

    xrel = [t - t_first for t in ts_win]        # x axis: elapsed seconds

    # 6. Paint lines
    for key in ("AccX", "AccY", "AccZ", "GyroX", "GyroY", "GyroZ"):
        lines[key].set_data(xrel, data[key][sl])

    for ax in (ax_acc, ax_gyro):
        ax.set_xlim(xrel[0], xrel[-1] + 0.001)
        ax.relim()
        ax.autoscale_view(scalex=False, scaley=True)

    # 7. Status bar
    pct = 100.0 * right_edge / new_max if new_max > 0 else 100.0
    status_text.set_text(
        f"{len(ts):,} rows total  |  "
        f"window: {right_edge - WINDOW_SECS:.1f}s – {right_edge:.1f}s  |  "
        f"duration: {duration:.1f}s  |  "
        f"position: {pct:.0f}%"
        + ("  [live]" if not state["user_scroll"] else "  [scrolling]")
    )

    return list(lines.values())


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ani = FuncAnimation(
        fig,
        update,
        interval=INTERVAL_MS,
        blit=False,
        cache_frame_data=False,
    )
    plt.show()