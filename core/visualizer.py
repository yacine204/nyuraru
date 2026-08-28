import ctypes
import os
import numpy as np

_LIB_CANDIDATES = ["libnnui.so", "libnnui.dylib", "nnui.dll"]


def _find_lib():
    for name in _LIB_CANDIDATES:
        path = os.path.join(os.path.dirname(__file__), "..", name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "Could not find libnnui.so / .dylib / .dll next to your project root. "
        "Build it with: gcc -shared -fPIC -O2 -o libnnui.so nn_ui.c -lraylib -lm"
    )


class Visualizer:
    def __init__(self, layer_sizes: list[int], output_activation: str, screen_w=1680, screen_h=800):
        self.lib = ctypes.CDLL(_find_lib())
        self._configure_signatures()

        self.layer_sizes = layer_sizes
        i_n_nodes = layer_sizes[0]
        hidden_sizes = layer_sizes[1:-1]
        o_n_nodes = layer_sizes[-1]

        hidden_arr = (ctypes.c_int * len(hidden_sizes))(*hidden_sizes) if hidden_sizes else (ctypes.c_int * 0)()

        self.lib.ui_init(
            screen_w, screen_h,
            i_n_nodes,
            hidden_arr, len(hidden_sizes),
            o_n_nodes,
            output_activation.encode(),
        )

    def _configure_signatures(self):
        self.lib.ui_init.argtypes = [
            ctypes.c_int, ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int), ctypes.c_int,
            ctypes.c_int, ctypes.c_char_p,
        ]
        self.lib.ui_set_layer_nodes.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
        self.lib.ui_should_close.restype = ctypes.c_int
        self.lib.ui_close.argtypes = []

        self.lib.ui_set_input_grid.argtypes = [ctypes.c_int, ctypes.c_int]

        self.lib.ui_predict_pressed.restype = ctypes.c_int
        self.lib.ui_clear_pressed.restype = ctypes.c_int
        self.lib.ui_get_board_pixels.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_float)]
        self.lib.ui_clear_board.argtypes = []
        self.lib.ui_reset_display.argtypes = []

        self.lib.ui_frame_with_prediction.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_float]
        self.lib.ui_board_modified.restype = ctypes.c_int
        self.lib.ui_reset_board_modified.argtypes = []

    def should_close(self) -> bool:
        return bool(self.lib.ui_should_close())

    def set_input_grid(self, grid_w: int, grid_h: int):
        self.lib.ui_set_input_grid(grid_w, grid_h)

    def push_activations(self, cache_a: list[np.ndarray]):
        for idx, layer_vals in enumerate(cache_a):
            sample = layer_vals[0] if layer_vals.ndim > 1 else layer_vals
            v = np.ascontiguousarray(sample, dtype=np.float32)
            buf = (ctypes.c_float * len(v))(*v)
            self.lib.ui_set_layer_nodes(idx, buf, len(v))

    def predict_pressed(self) -> bool:
        return bool(self.lib.ui_predict_pressed())

    def clear_pressed(self) -> bool:
        return bool(self.lib.ui_clear_pressed())

    def get_board_pixels(self, grid_w: int, grid_h: int) -> np.ndarray:
        n = grid_w * grid_h
        buf = (ctypes.c_float * n)()
        self.lib.ui_get_board_pixels(grid_w, grid_h, buf)
        return np.array(buf, dtype=np.float32)

    def clear_board(self):
        self.lib.ui_clear_board()

    def board_modified(self) -> bool:
        return bool(self.lib.ui_board_modified())
    
    def reset_display(self):
        self.lib.ui_reset_display()

    def frame(self, prediction=None):
        if prediction is not None:
            digit, confidence = prediction
            self.lib.ui_frame_with_prediction(1, int(digit), float(confidence))
        else:
            self.lib.ui_frame_with_prediction(0, 0, 0.0)

    def close(self):
        self.lib.ui_close()

    def reset_board_modified(self):
        if hasattr(self.lib, 'ui_reset_board_modified'):
            self.lib.ui_reset_board_modified()