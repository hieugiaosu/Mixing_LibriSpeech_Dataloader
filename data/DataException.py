class SilenceWindow(Exception):
    def __init__(self,msg="The window is silence"):
        super().__init__()
        self.msg = msg
    def __str__(self):
        return self.msg