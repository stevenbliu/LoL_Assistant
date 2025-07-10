import ctypes

GWL_EXSTYLE = -20
WS_EX_LAYERED = 0x80000
WS_EX_TRANSPARENT = 0x20
LWA_ALPHA = 0x2
# LWA_COLORKEY = 0x1  # Uncomment if you want to use color key transparency


def make_non_clickable(hwnd: int):
    """
    Make the window transparent to mouse events (click-through).
    """
    style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
    style |= WS_EX_LAYERED | WS_EX_TRANSPARENT  # Add layered & transparent styles
    ctypes.windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style)
    # Set full opacity (255). Using alpha for layered window.
    ctypes.windll.user32.SetLayeredWindowAttributes(hwnd, 0, 255, LWA_ALPHA)
    # Alternative with colorkey (less common, can cause flicker):
    # ctypes.windll.user32.SetLayeredWindowAttributes(hwnd, 0x00FFFFFF, 0, LWA_COLORKEY)


def make_clickable(hwnd: int):
    """
    Make the window receive mouse events again (not click-through).
    """
    style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
    style &= ~WS_EX_TRANSPARENT  # Remove transparent flag
    style |= WS_EX_LAYERED  # Keep layered style
    ctypes.windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style)
    ctypes.windll.user32.SetLayeredWindowAttributes(hwnd, 0, 255, LWA_ALPHA)
