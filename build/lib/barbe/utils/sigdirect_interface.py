"""
The point of this code is to handle sigdirect output in a way that would be improper to handle on
 the side of BARBE.
"""

from sigdirect import SigDirect


class SigDirectWrapper:
    def __init__(self):
        # IAIN this should have settings for sigdirect and more accurate utilities
        self._sigdirect_model = None
        pass

    def get_rules(self):
        pass

    def get_features(self):
        pass

    def get_translation(self):
        pass
