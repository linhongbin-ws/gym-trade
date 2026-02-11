import os
import signal
import time
from lightweight_charts import Chart
from pynput import keyboard
from pynput.keyboard import Key
from threading import Lock

class ChartMod(Chart):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._exit_now = False

        self._keyboard = keyboard.Listener(
            on_press=self.on_key,
            daemon=True
        )
        self._keyboard.start()
        self._lock = Lock()
        self.press_n = False

    def on_key(self, key):
        if key == Key.esc:
            print("ESC pressed → force exit")
            self._exit_now = True
            self._keyboard.stop()
        elif key.char == 'n':
            print("n pressed")
            with self._lock:
                self.press_n = True


    def maybe_exit(self):
        if self._exit_now:
            try:
                self.exit()          # close GUI
                time.sleep(0.1)      # let Chromium start shutdown
            finally:
                # 🔴 KILL THE PROCESS GROUP — NO BLOCKING POSSIBLE
                os.kill(os.getpid(), signal.SIGKILL)


if __name__ == "__main__":
    import pandas as pd

    df = pd.DataFrame({
        "time": pd.date_range("2024-01-01", periods=50, freq="D"),
        "open": range(50),
        "high": [x + 1 for x in range(50)],
        "low": [x - 1 for x in range(50)],
        "close": range(50),
        "volume": [100] * 50,
    })

    chart = ChartMod(toolbox=True, inner_width=1)
    chart.set(df)
    chart.show(block=False)

    while True:
        chart.maybe_exit()
        time.sleep(0.05)