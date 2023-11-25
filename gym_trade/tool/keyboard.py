from pynput import keyboard
class Keyboard():
    def __init__(self,):
        pass 
    def get_char(self):
        # The event listener will be running in this block
        with keyboard.Events() as events:
            for event in events:
                if isinstance(event, keyboard.Events.Press) and not isinstance(event, keyboard.Events.Release):
                    break
        return event.key.char if hasattr(event.key, "char") else event.key
