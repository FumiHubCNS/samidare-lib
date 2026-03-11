from multiprocessing import Process, Queue
import queue
import numpy as np

def push_board32_point(data_queue, x, board, values_32ch):
    """
    x: スカラー
    board: 0..3
    values_32ch: 長さ32の配列/リスト
    """
    values_32ch = np.asarray(values_32ch, dtype=float)
    if values_32ch.shape != (32,):
        raise ValueError(f"values_32ch must have shape (32,), got {values_32ch.shape}")

    data_queue.put({
        "type": "board32_point",
        "x": float(x),
        "board": int(board),
        "values": values_32ch,
    })

def _run_pyqtgraph_stream_4x32_from_queue_independent_x(
    data_queue,
    max_points: int = 2000,
    interval_ms: int = 50,
):
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtCore, QtWidgets

    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])

    win = pg.GraphicsLayoutWidget(show=True, title="PyQtGraph 4x32 Streaming")
    win.resize(1400, 900)

    plots = []
    curve_groups = []

    for board in range(4):
        row = board // 2
        col = board % 2
        p = win.addPlot(row=row, col=col)
        p.setTitle(f"board {board}")
        p.setLabel("left", "adc value")
        p.setLabel("bottom", "timestamp")
        p.showGrid(x=True, y=True)
        p.setYRange(0, 1024, padding=0)

        legend = p.addLegend(offset=(-10, 10), colCount=2, labelTextSize="8pt")
        legend.setBrush((0, 0, 0, 180))
        legend.setPen((180, 180, 180, 200))

        curves = []
        for ch in range(32):
            color = pg.intColor(ch, hues=32)
            curve = p.plot(
                pen=None,
                symbol="o",
                symbolSize=4,
                symbolBrush=color,
                symbolPen=None,
                name=f"ch{ch}",
            )
            # curve = p.plot(
            #     pen=pg.mkPen(color=color, width=1),
            #     antialias=False,
            #     name=f"ch{ch}",
            # )
            curves.append(curve)

        plots.append(p)
        curve_groups.append(curves)

    x_data_by_board = [np.array([], dtype=float) for _ in range(4)]
    y_data_by_board = [np.empty((32, 0), dtype=float) for _ in range(4)]

    def append_board32_point(x, board, values_32ch):
        nonlocal x_data_by_board, y_data_by_board

        if not (0 <= board < 4):
            print(f"invalid board: {board}")
            return False

        values_32ch = np.asarray(values_32ch, dtype=float)
        if values_32ch.shape != (32,):
            print(f"invalid values shape: {values_32ch.shape}, expected (32,)")
            return False

        x_data_by_board[board] = np.concatenate([
            x_data_by_board[board],
            np.array([x], dtype=float)
        ])[-max_points:]

        y_new = values_32ch.reshape(32, 1)
        y_data_by_board[board] = np.concatenate([
            y_data_by_board[board],
            y_new
        ], axis=1)[:, -max_points:]

        return True

    def redraw_board(board):
        x_data = x_data_by_board[board]
        y_data = y_data_by_board[board]

        for ch in range(32):
            curve_groups[board][ch].setData(x=x_data, y=y_data[ch])

        if len(x_data) > 0:
            xmin = float(x_data[0])
            xmax = float(x_data[-1])
            if xmax <= xmin:
                xmax = xmin + 1.0
            plots[board].setXRange(xmin, xmax, padding=0)

    def update():
        updated_boards = set()

        while True:
            try:
                item = data_queue.get_nowait()
            except queue.Empty:
                break

            if item is None:
                app.quit()
                return

            if isinstance(item, dict) and item.get("type") == "board32_point":
                ok = append_board32_point(
                    x=item["x"],
                    board=item["board"],
                    values_32ch=item["values"],
                )
                if ok:
                    updated_boards.add(item["board"])
            else:
                print(f"unknown item: {type(item)}")

        for board in updated_boards:
            redraw_board(board)

    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(interval_ms)

    pg.exec()

def launch_pyqtgraph_stream_4x32_queue(
    max_points: int = 2000,
    interval_ms: int = 50,
):
    data_queue = Queue()

    proc = Process(
        target=_run_pyqtgraph_stream_4x32_from_queue_independent_x,
        kwargs={
            "data_queue": data_queue,
            "max_points": max_points,
            "interval_ms": interval_ms,
        },
        daemon=True,
    )
    proc.start()
    return proc, data_queue