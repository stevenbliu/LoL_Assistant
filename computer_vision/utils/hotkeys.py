import keyboard


def listen_hotkey(overlay):
    keyboard.add_hotkey("ctrl+d", lambda: overlay.toggleDrawingSignal.emit())
    keyboard.wait()
