from gym_trade.env.wrapper.base import BaseWrapper
import sys
class ActionOracle(BaseWrapper):
    KEYBOARD_MAP = {
                "a": 1,
                "s": 2,
                "d": 0,
            }
    def __init__(self, env,
                 device='keyboard',
                 **kwargs):
        super().__init__(env)
        self._device_type = device
        if device in ['keyboard', 'script']:
            from gym_trade.tool.keyboard import Keyboard
            self._device = Keyboard()
        else:
            raise NotImplementedError
    
    def get_oracle_action(self):
        if self._device_type in ['keyboard', 'script']:
            while True:
                ch  = self._device.get_char()
                if ch == 'q':
                    sys.exit(0)
                elif ch in self.KEYBOARD_MAP and self._device_type == "keyboard":
                    return self.KEYBOARD_MAP[ch]
                elif self._device_type == "script":
                    return self.env.get_oracle_action()
                else:
                    continue 